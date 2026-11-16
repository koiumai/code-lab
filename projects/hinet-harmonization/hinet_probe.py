#!/usr/bin/env python3
"""Research runner for official Apache-2.0 HINet. No shadow generation.
Clone upstream at 40fa55a03ec5f2f832c48f3f98b8b83176b255dc beside this script.
Model path tested on A40; baseline/output safety additions CPU-tested only.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import urllib.request

REV = 'eab44a6a08e63cb129c93d1b7022c9c31c158326'
NAME = 'Resolution_RAW_iHarmony4.pth'
SHA = 'c5475d7a2a77260f8c9503b02e46a7b456bd1a7a7d7b0fc3c85ef534b473eef8'
URL = f'https://huggingface.co/spaces/WindVChen/INR-Harmon/resolve/{REV}/pretrained_models/{NAME}'

def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''): h.update(b)
    return h.hexdigest()

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--image', required=True)
    p.add_argument('--baseline', help='Optional same-size prior result: copy ONLY pixels outside object mask, e.g. existing shadows')
    p.add_argument('--mask', required=True, help='White=foreground to harmonize; black=keep')
    p.add_argument('--output', required=True, help='PNG path, original dimensions preserved')
    p.add_argument('--upstream', default=str(Path(__file__).parent/'upstream'))
    p.add_argument('--checkpoint', default=str(Path(__file__).parent/'weights'/NAME))
    p.add_argument('--device', choices=['cuda','cpu'], default='cuda')
    p.add_argument('--patch-size', type=int, default=256)
    p.add_argument('--download', action='store_true', help='Download the pinned official checkpoint if missing')
    a = p.parse_args()
    if a.patch_size < 32: p.error('patch-size must be at least 32')
    from output_utils import refuse_overwrite, restore_outside, write_exclusive
    target = Path(a.output)
    refuse_overwrite(target)
    if target.suffix.lower() != '.png': p.error('output must be PNG for exact outside-mask invariance')
    cp = Path(a.checkpoint)
    if not cp.exists():
        if not a.download: raise FileNotFoundError(f'{cp}; use --download')
        cp.parent.mkdir(parents=True, exist_ok=True)
        tmp = cp.with_suffix('.part')
        urllib.request.urlretrieve(URL, tmp)
        if digest(tmp) != SHA: raise RuntimeError('Checkpoint SHA256 mismatch')
        tmp.replace(cp)
    if digest(cp) != SHA: raise RuntimeError('Checkpoint SHA256 mismatch')

    import numpy as np
    import cv2
    import torch
    # Legacy upstream uses np.int once during HRNet construction.
    if not hasattr(np, 'int'): np.int = int
    sys.path.insert(0, str(Path(a.upstream).resolve()))
    import inference_for_arbitrary_resolution_image as upstream
    from model.hrnetv2.hrnet_ocr import HighResolutionNet
    from model.build_model import build_model

    old_argv = sys.argv
    try:
        sys.argv = [old_argv[0]]
        opt = upstream.parse_args()
    finally: sys.argv = old_argv
    opt.transform_mean = [.5,.5,.5]
    opt.transform_var = [.5,.5,.5]
    opt.device = a.device
    opt.save_path = None
    opt.workers = 0
    opt.batch_size = 1
    bgr = cv2.imread(a.image, cv2.IMREAD_COLOR)
    mask = cv2.imread(a.mask, cv2.IMREAD_GRAYSCALE)
    if bgr is None or mask is None: raise ValueError('Failed to read image or mask')
    if bgr.shape[:2] != mask.shape: raise ValueError('Image and mask dimensions must match')
    if min(mask.shape) < 32: raise ValueError('Image dimensions must be at least 32')
    if not (mask>100).any(): raise ValueError('Mask contains no foreground')
    opt.split_num = max(1, math.ceil(min(mask.shape)/a.patch_size))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    mask3 = np.repeat(mask[:,:,None],3,axis=2)
    baseline = cv2.imread(a.baseline, cv2.IMREAD_COLOR) if a.baseline else None
    if a.baseline and (baseline is None or baseline.shape != bgr.shape):
        raise ValueError('Baseline must be readable and exactly match input dimensions')

    # Full harmonization checkpoint contains the HRNet weights. Suppress its
    # redundant ImageNet loader (also a broken Windows-relative path on Linux).
    # strict=True below prevents silently leaving random/uninitialized weights.
    loader = HighResolutionNet.load_pretrained_weights
    HighResolutionNet.load_pretrained_weights = lambda self, *args, **kwargs: None
    try: model = build_model(opt)
    finally: HighResolutionNet.load_pretrained_weights = loader
    # Strip optimizer/scheduler pickle suffix without executing any pickle.
    # Extraction verifies original SHA and exact opcode boundary each run.
    from extract_weights_only import extract
    model_only = extract(cp, cp.with_suffix('.modelonly.pth'))
    state = torch.load(model_only, map_location='cpu', weights_only=True)
    # This official checkpoint has three unused legacy root encoder entries.
    # Permit only those exact extras; strict loading still rejects all missing
    # current-model weights and all other unexpected keys.
    weights = state['model']
    legacy = {'INR_encoding.fourier_embed.input',
              'INR_encoding.predict_embed.ffm.weight',
              'INR_encoding.predict_embed.ffm.bias'}
    expected = set(model.state_dict())
    for key in legacy - expected:
        weights.pop(key, None)
    model.load_state_dict(weights, strict=True)
    del state
    model = model.to(a.device).eval()
    # Do not use AMP/BF16: initial test is ordinary FP32, compatible with 2080.
    torch.backends.cudnn.benchmark = False
    upstream.global_state[0] = 1
    # Upstream resets peak stats every patch; prevent resets so our reported
    # peak covers the entire invocation, including the warm-up forward.
    resets = {}
    if a.device == 'cuda':
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        for name in ['reset_max_memory_allocated', 'reset_max_memory_cached']:
            resets[name] = getattr(torch.cuda, name)
            setattr(torch.cuda, name, lambda *args, **kwargs: None)
    t = time.monotonic()
    try:
        with torch.inference_mode(): pred = upstream.inference(model,opt,rgb,mask3)
    finally:
        for name, method in resets.items(): setattr(torch.cuda,name,method)
    if a.device == 'cuda': torch.cuda.synchronize()
    elapsed = time.monotonic()-t
    # Upstream normalizes/de-normalizes even background pixels. Restore exact
    # original pixels outside its own foreground threshold, including any shadow.
    pred = restore_outside(pred, bgr, mask, baseline)
    target.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode('.png', pred)
    if not ok: raise RuntimeError('PNG encoding failed')
    write_exclusive(target, encoded.tobytes())
    meta = {'model':'HINet RAW_iHarmony4', 'checkpoint_sha256':SHA,
            'checkpoint_url':URL, 'seconds':elapsed, 'shape':list(pred.shape),
            'split_num':opt.split_num, 'requested_patch_size':a.patch_size,
            'outside_mask_maxdiff':int(np.abs(pred.astype(np.int16)-bgr)[mask<=100].max()) if (mask<=100).any() else 0,
            'device':a.device, 'dtype':'float32', 'shadow_generation':False,
            'baseline_used':bool(a.baseline),
            'outside_mask_reference':'baseline' if baseline is not None else 'original',
            'outside_mask_reference_maxdiff':int(np.abs(pred.astype(np.int16)-(baseline if baseline is not None else bgr))[mask<=100].max()) if (mask<=100).any() else 0,
            'peak_allocated_bytes':torch.cuda.max_memory_allocated() if a.device=='cuda' else None,
            'peak_reserved_bytes':torch.cuda.max_memory_reserved() if a.device=='cuda' else None}
    write_exclusive(target.with_suffix('.json'), json.dumps(meta,indent=2).encode())
    print(json.dumps(meta,indent=2))

if __name__=='__main__': main()
