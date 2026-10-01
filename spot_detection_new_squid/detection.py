"""
Noise-robust 3-D spot detector (smoothed local maxima with a prominence
check). Two interchangeable backends, selectable at runtime:

* backend='scipy' : the original CPU implementation (scipy.ndimage). Always
  available; also used as the fallback.
* backend='torch' : a GPU-accelerated implementation built on PyTorch. The
  same code runs on CUDA (NVIDIA), MPS (Apple Silicon / Metal) or CPU, chosen
  via the `device` argument or auto-detected.
* backend='auto'  : use torch on a GPU if both torch and a GPU (cuda/mps) are
  available, otherwise fall back to scipy.

1. Gaussian smoothing (smooth_sigma) suppresses single-voxel noise spikes
   before peak detection, the way blob_dog/blob_log do implicitly via their
   scale-space filtering.
2. A prominence check requires each peak to exceed a local background
   estimate (a minimum filter over prominence_footprint_size) by at least
   `tolerance` -- a cheap, h_maxima-like criterion computed with a second
   separable filter pass instead of morphological reconstruction + DBSCAN
   merging.

`threshold` and `tolerance` are both expressed in RAW pixel units. Smoothing
is used only to *locate* peaks; the two accept/reject tests are then applied
to the original image, so both parameters are numbers you can find by opening
the image and hovering over a spot. (Before, they were compared against the
smoothed volume, where a spot's peak sits ~1.36x lower than its raw value --
so values tuned under the old behaviour are not comparable.)

The GPU and CPU backends are numerically identical -- see selftest.py for the
scipy-vs-torch parity check.
"""
from __future__ import annotations

import multiprocessing as mp
from typing import NamedTuple

import numpy as np
from scipy.ndimage import maximum_filter, minimum_filter, gaussian_filter

from . import torch_filters as tf


# --------------------------------------------------------------------------- #
#  CPU backend (scipy)                                                        #
# --------------------------------------------------------------------------- #
def _smoothed_local_max_core(vol, footprint_size, threshold, smooth_sigma,
                              tolerance, prominence_footprint_size):
    """The actual filter pipeline, on whatever volume it's given (one tile or
    a whole composite blob).

    Note: the cubic neighbourhoods use scipy's `size=` argument (a flat box
    structuring element) instead of an explicit `footprint=np.ones(...)`. The
    result is identical, but `size=` lets scipy take its fast *separable*
    code path instead of the generic per-footprint one.
    """
    raw = vol.astype(np.float32)
    v = raw
    if smooth_sigma and smooth_sigma > 0:
        v = gaussian_filter(raw, sigma=smooth_sigma)

    # Peaks are *located* on the smoothed volume (noise-robust), but both
    # thresholds are applied to RAW pixel values, so they mean what you read
    # off the image in ImageJ rather than a post-filter quantity.
    is_peak = (maximum_filter(v, size=footprint_size) == v) & (raw >= threshold)

    if tolerance and tolerance > 0:
        # Local background from the smoothed volume: a min-filter on raw data
        # latches onto the darkest noise voxel in the window and overstates
        # prominence by ~8%, whereas the smoothed min-filter tracks the
        # background level you would read off the image.
        local_min = minimum_filter(v, size=prominence_footprint_size)
        is_peak &= (raw - local_min) >= tolerance

    coords = np.column_stack(np.where(is_peak))   # columns: Z, Y, X
    return coords.astype(np.float64)


def _smoothed_local_max_tile_worker(args):
    """One tile's full filter pipeline, run in its own worker process --
    coordinates are offset back into the composite volume's frame before
    returning."""
    (tile_vol, y0, x0, footprint_size, threshold, smooth_sigma,
     tolerance, prominence_footprint_size) = args
    coords = _smoothed_local_max_core(
        tile_vol, footprint_size, threshold, smooth_sigma,
        tolerance, prominence_footprint_size,
    )
    if len(coords):
        coords[:, 1] += y0
        coords[:, 2] += x0
    return coords


# --------------------------------------------------------------------------- #
#  Reported intensity                                                         #
# --------------------------------------------------------------------------- #
DEFAULT_INTENSITY_TOP_K = 10


class TopKStats(NamedTuple):
    """Per-spot summary of the `n_used` brightest voxels around each spot.
    Every field is a raw-unit array with one entry per detected spot."""
    median: np.ndarray      # median of those voxels -- the reported intensity
    maximum: np.ndarray     # brightest of them, i.e. the brightest in the box
    minimum: np.ndarray     # dimmest of them, i.e. the k-th brightest in the box
    n_used: np.ndarray      # how many voxels the three above were taken over


def top_k_intensity_stats(vol, coords, box_size, top_k=DEFAULT_INTENSITY_TOP_K):
    """Median / max / min of the `top_k` brightest voxels around each spot.

    The peak voxel on its own is a biased, noisy intensity: it is by
    construction the largest value in its neighbourhood, so whatever noise sits
    on the spot is sampled at its maximum rather than averaged out. Taking the
    median of the spot's brightest few voxels instead measures the core of the
    PSF, and the median (rather than a mean) keeps one hot voxel from carrying
    the estimate. The max and min of that same set bracket it: the max is the
    brightest voxel in the box, the min the `top_k`-th brightest, so the gap
    between them is how fast the spot falls off across its core. A min sitting
    at the background level means that spot has fewer than `top_k` voxels of
    signal and the tail of the set has reached into background -- the median is
    still signal there, but only just.

    `box_size` is the side of the cubic neighbourhood, placed exactly as
    scipy's `maximum_filter(size=box_size)` places it (``size // 2`` voxels
    back, ``size - 1 - size // 2`` forward, so even sizes are asymmetric). Pass
    the detector's `footprint_size` and the box is the very window the spot won
    its peak search in. Note that search runs on the SMOOTHED volume, so the
    raw peak voxel is not necessarily the largest raw value in the box -- which
    is itself part of why it is a poor intensity estimate. Boxes are clipped at
    the volume edges; a box holding fewer than `top_k` voxels uses all of them.

    Returns a TopKStats of four arrays, one entry per row of `coords`.
    """
    lo, hi = box_size // 2, box_size - 1 - box_size // 2
    nz, ny, nx = vol.shape
    n = len(coords)
    median = np.empty(n, dtype=np.float64)
    maximum = np.empty(n, dtype=np.float64)
    minimum = np.empty(n, dtype=np.float64)
    n_used = np.empty(n, dtype=np.int64)

    # A per-spot loop rather than one gathered index array: the boxes overlap
    # the volume edges by different amounts, and gathering with clipped indices
    # would repeat edge voxels into the top-k instead of shrinking the box.
    for i, (z, y, x) in enumerate(np.asarray(coords)[:, :3].astype(int)):
        box = vol[max(z - lo, 0):min(z + hi + 1, nz),
                  max(y - lo, 0):min(y + hi + 1, ny),
                  max(x - lo, 0):min(x + hi + 1, nx)].reshape(-1)
        k = min(top_k, box.size)
        top = box if k == box.size else np.partition(box, box.size - k)[box.size - k:]
        median[i] = np.median(top)
        maximum[i] = top.max()
        minimum[i] = top.min()
        n_used[i] = k

    return TopKStats(median, maximum, minimum, n_used)


# --------------------------------------------------------------------------- #
#  Public dispatcher                                                          #
# --------------------------------------------------------------------------- #
def alg_smoothed_local_max_3d(vol, footprint_size=5, threshold=1000,
                               smooth_sigma=1.0, tolerance=200,
                               prominence_footprint_size=9,
                               tile_mode='combined', processes=6,
                               n_rows=1, n_cols=1, tile_ny=None, tile_nx=None,
                               backend='auto', device=None, z_chunk=None):
    """
    Detect smoothed local maxima in a 3-D volume.

    Backends
    --------
    backend='auto'  : run the GPU (torch) path if torch and a GPU (cuda/mps)
                      are both available, else the scipy path. (default)
    backend='torch' : force the GPU path; `device` chooses 'cuda'/'mps'/'cpu'
                      (None = auto-detect). The whole volume is processed in
                      one pass; set `z_chunk` to cap GPU memory on large stacks.
    backend='scipy' : force the original CPU path (supports tile_mode='tiles'
                      for multiprocessing).

    The tile_mode / processes / n_rows / n_cols / tile_ny / tile_nx arguments
    only apply to the scipy backend; the GPU backend replaces process-level
    tiling with on-device parallelism (and optional `z_chunk` for memory).

    footprint_size            : side length (pixels) of the cubic neighbourhood for peak search
    threshold                 : minimum RAW pixel value at the peak to accept it -- the number
                                 you read off the image in ImageJ, not a post-filter quantity
    smooth_sigma               : Gaussian sigma (pixels) for pre-smoothing; 0 disables smoothing
    tolerance                  : minimum prominence in RAW units -- raw peak value minus the local
                                 background level, i.e. the spot's height above background by eye; 0 disables the check
    prominence_footprint_size  : side length (pixels) of the cubic neighbourhood for the background estimate
    processes                  : worker processes when tile_mode='tiles'; 1 (or None) runs tiles sequentially
    device                     : torch device for backend in {'auto','torch'}; None auto-detects
    z_chunk                    : torch backend only -- process this many Z-slices per pass (None = all at once)
    """
    if backend == 'torch':
        use_torch = True
    elif backend == 'auto':
        use_torch = tf.gpu_available()
    elif backend == 'scipy':
        use_torch = False
    else:
        raise ValueError(f"unknown backend {backend!r}; "
                          f"expected 'auto', 'torch' or 'scipy'")

    if use_torch:
        dev = tf.select_device(device)
        return tf.smoothed_local_max_core_torch(
            vol, footprint_size, threshold, smooth_sigma,
            tolerance, prominence_footprint_size, dev, z_chunk)

    # ---- scipy path ----
    if tile_mode != 'tiles' or n_rows * n_cols <= 1:
        return _smoothed_local_max_core(
            vol, footprint_size, threshold, smooth_sigma,
            tolerance, prominence_footprint_size,
        )

    tasks = []
    for tile_idx in range(n_rows * n_cols):
        row, col = divmod(tile_idx, n_cols)
        y0, x0 = row * tile_ny, col * tile_nx
        tile_vol = vol[:, y0:y0 + tile_ny, x0:x0 + tile_nx]
        tasks.append((tile_vol, y0, x0, footprint_size, threshold, smooth_sigma,
                      tolerance, prominence_footprint_size))

    if processes and processes > 1:
        with mp.Pool(processes=processes) as pool:
            results = pool.map(_smoothed_local_max_tile_worker, tasks)
    else:
        results = [_smoothed_local_max_tile_worker(t) for t in tasks]

    nonempty = [r for r in results if len(r)]
    return np.concatenate(nonempty, axis=0) if nonempty else np.empty((0, 3))
