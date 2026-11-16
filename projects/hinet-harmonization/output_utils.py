"""CPU-only validation / exact mask compositing; no generative postprocessing."""
from pathlib import Path
import numpy as np

def output_paths(path):
    path=Path(path)
    if path.suffix.lower()!='.png': raise ValueError('Output must be PNG')
    return path,path.with_suffix('.json')

def refuse_overwrite(path):
    paths=output_paths(path)
    for p in paths:
        if p.exists() or p.is_symlink():raise FileExistsError(f'Refusing to overwrite: {p}')
    return paths

def restore_outside(pred, original, mask, baseline=None):
    if pred.shape!=original.shape or mask.shape!=original.shape[:2]:raise ValueError('Image/mask dimensions differ')
    if baseline is not None and baseline.shape!=original.shape:raise ValueError('Baseline dimensions differ; no resizing is performed')
    background=original if baseline is None else baseline
    result=background.copy()
    result[mask>100]=pred[mask>100]
    return result

def write_exclusive(path, payload):
    """Exclusive create also rejects a file created after the preflight check."""
    with Path(path).open('xb') as f:f.write(payload)
