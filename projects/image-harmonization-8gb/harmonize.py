#!/usr/bin/env python3
"""Experimental local harmonization. RTX 2080 hardware not yet validated.

Images never leave this machine. First inference downloads Hugging Face weights.
Klein4B Apache-2.0 terms apply. Allocator cap excludes CUDA/driver allocations;
6.5 GiB is a starting point, NOT a guaranteed whole-process 8 GB limit.
Use --reference for model-free composition of an existing local reference.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from PIL import Image, ImageOps
from compositing import load_mask, default_shadow_mask, compose
from prompt_profiles import build_prompt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--image', required=True)
    p.add_argument('--mask', '--object-mask', dest='object_mask', required=True)
    p.add_argument('--output', required=True, help='New lossless PNG; diagnostics go into <stem>-assets')
    p.add_argument('--prompt', help='Override the legacy model lighting instruction')
    p.add_argument('--lighting-profile', choices=('legacy','photometric'), default='legacy')
    p.add_argument('--subject', help='Unambiguous foreground subject description; photometric only')
    p.add_argument('--lighting', help='Target scene illumination; photometric only')
    p.add_argument('--shadow-vertical-scale', type=float, default=1.0, help='Automatic shadow ellipse vertical scale [1,4]')
    p.add_argument('--shadow-mask')
    p.add_argument('--no-shadow', action='store_true')
    p.add_argument('--reference', help='Local generated reference; skips GPU/model loading')
    p.add_argument('--max-side', type=int, default=768)
    p.add_argument('--memory-gib', '--budget-gib', dest='budget_gib', type=float, default=6.5)
    p.add_argument('--shadow-strength', type=float, default=.6)
    p.add_argument('--object-mode', choices=('gain','delta','blend'), default='gain')
    p.add_argument('--object-strength', type=float, default=1.0)
    p.add_argument('--feather', type=float, default=3)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--revision', help='Optional Hugging Face checkpoint commit')
    a = p.parse_args()
    try:
        prompt = build_prompt(a.lighting_profile, a.subject, a.lighting, a.no_shadow, a.prompt)
    except ValueError as exc:
        p.error(str(exc))
    if not np.isfinite(a.shadow_vertical_scale) or not 1 <= a.shadow_vertical_scale <= 4:
        p.error('--shadow-vertical-scale must be finite in [1,4]')
    if a.shadow_vertical_scale != 1 and (a.shadow_mask or a.no_shadow):
        p.error('--shadow-vertical-scale != 1 applies only to automatic shadow regions')
    if not np.isfinite(a.object_strength) or not 0 <= a.object_strength <= 1:
        p.error('--object-strength must be finite in [0,1]')
    if a.no_shadow and a.shadow_mask:
        p.error('--no-shadow and --shadow-mask are mutually exclusive')
    if not 64 <= a.max_side <= 2048 or not 0 < a.budget_gib <= 8 or not 0 <= a.shadow_strength <= 1 or not np.isfinite(a.feather) or a.feather < 0:
        p.error('Invalid size, GPU budget, shadow strength or feather')
    with Image.open(a.image) as im:
        if im.mode not in ('RGB','L'):
            p.error('Image must be opaque RGB or grayscale; flatten transparency explicitly')
        src = ImageOps.exif_transpose(im).convert('RGB')
    obj = load_mask(a.object_mask, src.size)
    if not np.any(obj > 0): p.error('Object mask is empty')
    shadow = np.zeros_like(obj) if a.no_shadow else (load_mask(a.shadow_mask, src.size) if a.shadow_mask else default_shadow_mask(obj, a.shadow_vertical_scale))
    shadow[obj > 0] = 0
    destination = Path(a.output)
    if destination.suffix.lower() != '.png': p.error('--output must be PNG for exact pixel preservation')
    if destination.exists(): p.error('Output already exists; choose a new path')
    out = destination.with_name(destination.stem + '-assets')
    out.mkdir(parents=True, exist_ok=False)
    src.save(out/'original.png')
    Image.fromarray(np.uint8(np.rint(obj*255))).save(out/'object-mask.png')
    Image.fromarray(np.uint8(np.rint(shadow*255))).save(out/'shadow-allowed.png')
    Image.fromarray(np.uint8(((obj>0)|(shadow>0))*255)).save(out/'allowed-union.png')
    report = {'args':vars(a), 'resolved_prompt':prompt, 'prompt_used_for_generation':not bool(a.reference), 'method':{'gain':'low-frequency RGB gain','delta':'low-frequency RGB difference','blend':'direct generated reference blending'}[a.object_mode] + '; shadow: luminance attenuation',
              'warnings':['RTX 2080 physical hardware untested; A5000 NF4/FP16 path tested',
                          'Automatic shadow area is a conservative heuristic; inspect shadow-allowed.png',
                          'Geometry changes in generated reference can produce incorrect lighting; inspect final.png',
                          'Allocator budget does not include driver overhead'],
              'network':'Hugging Face model download only; no input image upload',
              'model':'black-forest-labs/FLUX.2-klein-4B', 'model_license':'Apache-2.0'}
    try:
        if a.reference:
            reference_path = Path(a.reference)
        else:
            cmd = [sys.executable, str(Path(__file__).with_name('_klein_backend.py')), '--input', str(out/'original.png'),
                   '--prompt', prompt, '--out',str(out/'inference'),'--mode','nf4','--nf4-compute','fp16',
                   '--max-side',str(a.max_side),'--budget-gib',str(a.budget_gib),'--seeds',str(a.seed)]
            if a.revision: cmd += ['--revision',a.revision]
            subprocess.run(cmd, check=True)
            reference_path = out/'inference'/f'seed{a.seed}-raw.png'
        with Image.open(reference_path) as im:
            ref = im.convert('RGB').resize(src.size, Image.Resampling.LANCZOS)
        ref.save(out/'reference.png')
        result, metrics = compose(np.asarray(src), np.asarray(ref), obj, shadow, a.shadow_strength, a.feather,
                                  object_mode=a.object_mode, object_strength=a.object_strength)
        Image.fromarray(result).save(destination)
        # Audit lossless deliverable, not just pre-save array.
        saved = np.asarray(Image.open(destination))
        allowed = (obj>0)|(shadow>0)
        assert np.array_equal(saved[~allowed],np.asarray(src)[~allowed])
        report.update(metrics)
        report['size'] = src.size
        print(str(destination))
    except BaseException as exc:
        report['error'] = repr(exc)
        raise
    finally:
        (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))

if __name__ == '__main__': main()
