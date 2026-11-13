"""Masked composition with gain, low-frequency delta, or reference blending."""
import numpy as np
from PIL import Image, ImageOps
from scipy.ndimage import gaussian_filter, distance_transform_edt


def load_mask(path, size):
    """Accept grayscale 8-bit coverage (0..255), binary 0/1, or Pillow 1-bit."""
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        if im.size != size:
            raise ValueError(f'Mask dimensions {im.size} must equal image {size}')
        if im.mode not in ('L', '1'):
            raise ValueError('Mask must be grayscale 8-bit or 1-bit, not RGB/palette/16-bit')
        x = np.asarray(im).astype(np.float32)
    if not np.isfinite(x).all() or x.min() < 0 or x.max() > 255:
        raise ValueError('Invalid mask values')
    return x if x.max() <= 1 else x / 255


def default_shadow_mask(obj, vertical_scale=1.0):
    """Conservative contact area, NOT an inferred physically correct shadow extent."""
    if not np.isfinite(vertical_scale) or not 1 <= vertical_scale <= 4:
        raise ValueError('shadow vertical scale must be finite in [1,4]')
    ys, xs = np.nonzero(obj > 0)
    if not len(xs):
        raise ValueError('Object mask is empty')
    h, w = obj.shape
    bw, bh = xs.max()-xs.min()+1, ys.max()-ys.min()+1
    yy, xx = np.mgrid[:h, :w]
    cx = (xs.max()+xs.min())/2
    cy = ys.max() + .08*bh
    rx, ry = max(3, .7*bw), max(3, .18*bh) * vertical_scale
    ellipse = ((xx-cx)/rx)**2 + ((yy-cy)/ry)**2 <= 1
    return (ellipse & (obj == 0)).astype(np.float32)


def _inner_alpha(mask, feather):
    support = mask > 0
    if feather <= 0:
        return mask.copy()
    # Padding makes even image-border edges explicit; never expands support.
    distance = distance_transform_edt(np.pad(support, 1))[1:-1, 1:-1]
    return mask * np.minimum(distance / feather, 1)


def compose(source, reference, obj, shadow, strength=.6, feather=3,
            object_mode='gain', object_strength=1.0):
    """Return uint8 RGB and audit metrics, with exact-copy outside mask union."""
    if source.dtype != np.uint8 or source.ndim != 3 or source.shape[2] != 3:
        raise ValueError('source must be uint8 RGB')
    if reference.shape != source.shape or obj.shape != source.shape[:2] or shadow.shape != obj.shape:
        raise ValueError('Image and mask shapes must match')
    for m in (obj, shadow):
        if not np.isfinite(m).all() or m.min() < 0 or m.max() > 1:
            raise ValueError('Masks must be finite in [0,1]')
    if not np.any(obj > 0):
        raise ValueError('Object mask is empty')
    if not np.isfinite(reference).all() or not 0 <= strength <= 1 or not np.isfinite(feather) or feather < 0:
        raise ValueError('Invalid reference, strength or feather')
    if object_mode not in ('gain', 'delta', 'blend'):
        raise ValueError('object_mode must be gain, delta or blend')
    if not np.isfinite(object_strength) or not 0 <= object_strength <= 1:
        raise ValueError('object_strength must be finite in [0,1]')
    s, ref = source.astype(np.float32), reference.astype(np.float32)
    radius = max(2, min(obj.shape)*.015)
    weight = gaussian_filter(obj, radius) + 1e-6
    def smooth(x):
        return np.stack([gaussian_filter(x[:,:,c]*obj, radius)/weight for c in range(3)], -1)
    base, light = smooth(s), smooth(ref)
    gain = np.clip((light+16)/(base+16), .5, 1.7)
    # Preserve supplied coverage; inward fading leaves bright rims after darkening.
    alpha = obj * object_strength
    if object_mode == 'gain':
        # Keep the old arithmetic for bit-identical default behavior.
        out = s * (1 + alpha[:,:,None]*(gain-1))
    elif object_mode == 'delta':
        out = s + alpha[:,:,None]*(light-base)
    else:
        # Intentionally allows synthesized texture/geometry inside the original mask.
        # This is not registration: displaced generated boundaries may create seams.
        out = s + alpha[:,:,None]*(ref-s)
    # Object coverage always takes priority, including antialiased edge pixels.
    shadow = shadow.copy(); shadow[obj > 0] = 0
    luma = lambda x: x @ np.array([.2126,.7152,.0722], np.float32)
    ratio = np.clip((gaussian_filter(luma(ref), max(1,radius/3))+8) /
                    (gaussian_filter(luma(s), max(1,radius/3))+8), .3, 1)
    # Fill object holes for distance calculation: contact shadows must reach the object edge.
    shadow_edge = _inner_alpha(((shadow>0)|(obj>0)).astype(np.float32), feather)
    shadow_alpha = shadow * shadow_edge
    attenuation = 1 - strength*shadow_alpha*(1-ratio)
    out *= attenuation[:,:,None]
    allowed = (obj > 0) | (shadow > 0)
    result = source.copy()
    result[allowed] = np.uint8(np.rint(np.clip(out[allowed],0,255)))
    assert np.array_equal(result[~allowed], source[~allowed])
    return result, {'object_mode':object_mode, 'object_strength':float(object_strength),
                    'outside_allowed_changed':0, 'allowed_pixels':int(allowed.sum()),
                    'gain_min':float(gain[obj>0].min()), 'gain_max':float(gain[obj>0].max()),
                    'shadow_min_attenuation':float(attenuation.min())}
