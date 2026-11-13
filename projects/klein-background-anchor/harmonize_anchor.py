#!/usr/bin/env python3
"""Experimental local Klein background-anchored repaint. Background copies the supplied baseline.

This is an experimental candidate, not a quality guarantee. CUDA/8GB
measurements were made on A40, not RTX2080. Inputs never leave this machine;
model weights download from Hugging Face. No automatic installs or GPU rentals.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from PIL import Image, ImageOps


def read_inputs(image,mask,baseline=None):
    with Image.open(image) as im:
        if im.mode not in ('RGB','L'): raise ValueError('Image must be opaque RGB/grayscale')
        source=ImageOps.exif_transpose(im).convert('RGB')
    with Image.open(mask) as im:
        im=ImageOps.exif_transpose(im)
        if im.mode not in ('1','L'):raise ValueError('Mask must be 1-bit or 8-bit grayscale')
        if im.size!=source.size:raise ValueError('Mask dimensions must match input')
        values=np.asarray(im).astype(np.float32)
    if values.max()<=1:values*=255
    if not np.any(values):raise ValueError('Object mask is empty')
    mask=Image.fromarray(values.astype(np.uint8),'L')
    if baseline:
        with Image.open(baseline) as im:base=ImageOps.exif_transpose(im).convert('RGB')
        if base.size!=source.size:raise ValueError('Baseline dimensions must match input')
    else:base=source.copy()
    return source,mask,base


def compose(source,mask,base,reference,blend=.5):
    if not np.isfinite(blend) or not 0<=blend<=1:raise ValueError('Blend must be finite in [0,1]')
    src=np.asarray(source).astype(np.float32);m=np.asarray(mask).astype(np.float32)/255*blend
    ref=np.asarray(reference.resize(source.size,Image.Resampling.LANCZOS)).astype(np.float32)
    result=np.asarray(base).copy();edit=np.asarray(mask)>0
    pixels=src*(1-m[:,:,None])+ref*m[:,:,None]
    result[edit]=np.uint8(np.rint(np.clip(pixels[edit],0,255)))
    assert np.array_equal(result[~edit],np.asarray(base)[~edit])
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--image',required=True);p.add_argument('--mask',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--prompt',required=True,help='Describe foreground appearance and scene illumination, not a global restyling')
    p.add_argument('--blend',type=float,default=.5,help='Object mix strength [0,1]; .5 tempers excessive darkening')
    p.add_argument('--max-side',type=int,default=768);p.add_argument('--memory-gib',type=float,default=6.5)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--baseline',help='Optional existing shadow composite; pixels outside object are copied from this image')
    p.add_argument('--reference',help='Existing local model reference; skips GPU/download, for composition testing')
    p.add_argument('--revision')
    a=p.parse_args()
    if not a.prompt.strip():p.error('Prompt cannot be blank')
    if not np.isfinite(a.blend) or not 0<=a.blend<=1:p.error('Blend must be finite in [0,1]')
    if not 64<=a.max_side<=1024 or not 0<a.memory_gib<=8:p.error('Invalid size/budget')
    assets=a.output.with_name(a.output.stem+'-assets')
    if a.output.suffix.lower()!='.png':p.error('Output must be PNG')
    if a.output.exists() or assets.exists():p.error('Output/assets already exist; choose a new name')
    try:source,mask,base=read_inputs(a.image,a.mask,a.baseline)
    except ValueError as e:p.error(str(e))
    assets.mkdir(parents=True)
    source.save(assets/'source.png');base.save(assets/'baseline.png');mask.save(assets/'object-mask.png')
    report={'args':{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
            'model':'black-forest-labs/FLUX.2-klein-4B','model_license':'Apache-2.0',
            'note':'Experimental next-sigma background latent projection, not LanPaint; no quality guarantee. Background exactly copies baseline, or source if baseline omitted.'}
    try:
        if a.reference:
            raw_path=Path(a.reference)
        else:
            cmd=[sys.executable,str(Path(__file__).with_name('_anchor_backend.py')),
                 '--input',str(assets/'source.png'),'--anchor-mask',str(assets/'object-mask.png'),
                 '--out',str(assets/'inference'),'--prompt',a.prompt,'--mode','nf4','--nf4-compute','fp16',
                 '--steps','4','--max-side',str(a.max_side),'--budget-gib',str(a.memory_gib),'--seeds',str(a.seed)]
            if a.revision:cmd+=['--revision',a.revision]
            subprocess.run(cmd,check=True)
            raw_path=assets/'inference'/f'seed{a.seed}-anchor1-raw.png'
        with Image.open(raw_path) as im:reference=im.convert('RGB')
        reference.save(assets/'reference.png')
        result=compose(source,mask,base,reference,a.blend)
        Image.fromarray(result).save(a.output)
        saved=np.asarray(Image.open(a.output));outside=np.asarray(mask)==0
        assert np.array_equal(saved[outside],np.asarray(base)[outside])
        report.update({'outside_object_changes_vs_baseline':0,'size':source.size})
        print(str(a.output))
    except BaseException as exc:
        report['error']=repr(exc);raise
    finally:(assets/'report.json').write_text(json.dumps(report,indent=2))

if __name__=='__main__':main()
