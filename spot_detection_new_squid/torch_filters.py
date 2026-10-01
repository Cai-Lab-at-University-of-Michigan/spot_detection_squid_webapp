"""
GPU filter primitives (PyTorch: CUDA / MPS / CPU) backing the smoothed
local-maxima detector.

Every filter here is built from *separable* 1-D primitives (reflect-padded
conv1d for the Gaussian; unfold + amax/amin for the cubic max/min filters)
so that:

  (a) they avoid the less-reliable 3-D pooling kernels on Apple's MPS
      backend, and
  (b) border handling reproduces scipy's default mode='reflect' exactly
      (including even filter sizes via scipy's origin=0 window placement),
      so results are bit-identical to the scipy backend.
"""
from __future__ import annotations

import numpy as np


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


def gpu_available() -> bool:
    """True if torch is importable AND a GPU device (cuda or mps) is usable."""
    try:
        import torch
        if torch.cuda.is_available():
            return True
        mps = getattr(torch.backends, 'mps', None)
        return bool(mps and mps.is_available())
    except Exception:
        return False


def select_device(prefer: str | None = None):
    import torch
    if prefer is not None:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device('cuda')
    mps = getattr(torch.backends, 'mps', None)
    if mps and mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def _reflect_index(n, pad_left, pad_right, device):
    """Indices (length n + pad_left + pad_right) that reflect-pad a length-n
    axis, matching scipy's default `mode='reflect'` (half-sample symmetric,
    i.e. the edge sample is duplicated: ... c b a | a b c d | d c b ...).

    Asymmetric padding is needed for even-sized filter windows, where scipy
    centres the window with `lo = k//2` on the left and `hi = k-1-k//2` on the
    right (origin=0 convention)."""
    import torch
    rng = torch.arange(-pad_left, n + pad_right, device=device)
    period = 2 * n
    idx = torch.remainder(rng, period)
    idx = torch.where(idx >= n, period - 1 - idx, idx)
    return idx.to(torch.long)


def _along_axis(t, axis, fn):
    """Apply a 1-D op `fn: (M, L) -> (M, Lout)` along `axis` of tensor `t`."""
    t2 = t.movedim(axis, -1).contiguous()
    shape = t2.shape
    flat = t2.reshape(-1, shape[-1])
    out = fn(flat)
    out = out.reshape(*shape[:-1], out.shape[-1])
    return out.movedim(-1, axis)


def _gaussian_kernel1d(sigma, truncate=4.0):
    """1-D Gaussian weights matching scipy.ndimage.gaussian_filter."""
    radius = int(truncate * sigma + 0.5)
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    phi = np.exp(-0.5 * (x / sigma) ** 2)
    phi /= phi.sum()
    return phi.astype(np.float32), radius


def gaussian_filter_torch(t, sigma, truncate=4.0):
    import torch
    import torch.nn.functional as F
    w_np, r = _gaussian_kernel1d(sigma, truncate)
    w = torch.as_tensor(w_np, device=t.device, dtype=t.dtype).view(1, 1, -1)

    def fn(flat):
        idx = _reflect_index(flat.shape[-1], r, r, flat.device)
        padded = flat.index_select(-1, idx).unsqueeze(1)   # (M, 1, L+2r)
        return F.conv1d(padded, w).squeeze(1)              # (M, L)

    for axis in range(t.ndim):
        t = _along_axis(t, axis, fn)
    return t


def max_filter_torch(t, size):
    """Separable cubic maximum filter (scipy maximum_filter, size=size).
    Handles even sizes with scipy's origin=0 window placement."""
    lo, hi = size // 2, size - 1 - size // 2     # left/right window extent

    def fn(flat):
        idx = _reflect_index(flat.shape[-1], lo, hi, flat.device)
        padded = flat.index_select(-1, idx)
        return padded.unfold(-1, size, 1).amax(-1)

    for axis in range(t.ndim):
        t = _along_axis(t, axis, fn)
    return t


def min_filter_torch(t, size):
    """Separable cubic minimum filter (scipy minimum_filter, size=size).
    Handles even sizes with scipy's origin=0 window placement."""
    lo, hi = size // 2, size - 1 - size // 2

    def fn(flat):
        idx = _reflect_index(flat.shape[-1], lo, hi, flat.device)
        padded = flat.index_select(-1, idx)
        return padded.unfold(-1, size, 1).amin(-1)

    for axis in range(t.ndim):
        t = _along_axis(t, axis, fn)
    return t


def nonzero_coords(mask):
    import torch
    try:
        idx = torch.nonzero(mask, as_tuple=False)
    except Exception:
        # Some older MPS builds lack nonzero; fall back to a host-side scan.
        idx = torch.nonzero(mask.cpu(), as_tuple=False)
    return idx.to('cpu').numpy().astype(np.float64)


def core_torch_block(vol, footprint_size, threshold, smooth_sigma,
                      tolerance, prominence_footprint_size, device):
    """Run the full filter pipeline on one in-memory block, on `device`."""
    import torch
    arr = np.ascontiguousarray(vol, dtype=np.float32)
    raw = torch.from_numpy(arr).to(device)

    t = raw
    if smooth_sigma and smooth_sigma > 0:
        t = gaussian_filter_torch(raw, smooth_sigma)

    # Peaks located on the smoothed volume, thresholds applied to RAW values
    # -- see detection._smoothed_local_max_core, which this must match exactly.
    mx = max_filter_torch(t, footprint_size)
    is_peak = (mx == t) & (raw >= float(threshold))

    if tolerance and tolerance > 0:
        mn = min_filter_torch(t, prominence_footprint_size)
        is_peak = is_peak & ((raw - mn) >= float(tolerance))

    return nonzero_coords(is_peak)


def required_halo(footprint_size, smooth_sigma, prominence_footprint_size,
                   tolerance, truncate=4.0):
    """Per-side Z-halo (in voxels) that makes chunked processing bit-identical
    to whole-volume processing at internal seams.

    The prominence check is min_filter(gaussian(v)), so the supports add: a
    Gaussian radius r plus the larger of the two morphological half-widths.
    """
    r = int(truncate * smooth_sigma + 0.5) if (smooth_sigma and smooth_sigma > 0) else 0
    morph = footprint_size // 2
    if tolerance and tolerance > 0:
        morph = max(morph, prominence_footprint_size // 2)
    return r + morph


def smoothed_local_max_core_torch(vol, footprint_size, threshold, smooth_sigma,
                                   tolerance, prominence_footprint_size,
                                   device, z_chunk=None):
    """GPU core. Processes the whole volume at once (z_chunk=None) or in Z
    chunks with a halo so large volumes fit in GPU memory without changing the
    result."""
    Z = vol.shape[0]
    if z_chunk is None or z_chunk >= Z:
        return core_torch_block(
            vol, footprint_size, threshold, smooth_sigma,
            tolerance, prominence_footprint_size, device)

    halo = required_halo(footprint_size, smooth_sigma,
                          prominence_footprint_size, tolerance)
    out = []
    for z0 in range(0, Z, z_chunk):
        z1 = min(z0 + z_chunk, Z)
        a, b = max(0, z0 - halo), min(Z, z1 + halo)
        coords = core_torch_block(
            vol[a:b], footprint_size, threshold, smooth_sigma,
            tolerance, prominence_footprint_size, device)
        if len(coords):
            zc = coords[:, 0] + a                       # global Z
            keep = (zc >= z0) & (zc < z1)               # discard halo-only peaks
            coords = coords[keep]
            coords[:, 0] = zc[keep]
            out.append(coords)
    return np.concatenate(out, axis=0) if out else np.empty((0, 3))
