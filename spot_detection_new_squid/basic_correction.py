"""BaSiC (Peng et al. 2017) illumination correction.

Fitting and correction are both done per (tile, channel), independently --
`fit_basic_fields()` fits a flatfield/darkfield model from that tile's own
z-stack (randomly subsampled down to `max_fit_slices` slices if the stack is
deeper than that, so a single fit never has to hold an oversized stack in
memory), and `apply_basic_correction()` applies those fields to that same
tile's volume. Nothing is shared or cached across tiles.

`fit_and_correct_tile()` is the two together, kept as the standalone entry
point the calibration scripts use. In the pipeline itself BaSiC is step 1 of
the correction chain in `corrections.py`, which calls `fit_basic_fields`
directly so the correction stays in float32 across the remaining steps rather
than round-tripping through the image dtype in between.

Only this module touches the `basicpy` fitting machinery (the import is
deferred so the rest of the pipeline doesn't require `basicpy` to be installed
unless `use_basic=True`).
"""
from __future__ import annotations

import threading
import warnings

import numpy as np

DEFAULT_MAX_FIT_SLICES = 150

# Only one BaSiC fit at a time per process. On the GPU detection path the
# tile loader runs in `io_workers` prefetch threads (see tasks.prefetch), so
# without this several fits would run concurrently -- each moving a whole
# tile to the compute device (~4.3 GB for a 145x2720x2720 stack in float32)
# while the detector is using it too. That is enough to exhaust an 11 GB card
# and stall the run. Serialising costs nothing real: the expensive part of
# prefetching is reading the tile off disk, which still overlaps. On the CPU
# multiprocessing path each worker is a separate process with its own lock,
# so those fits stay parallel as before.
_FIT_LOCK = threading.Lock()


def _fit_stack(vol, max_fit_slices, rng):
    """The slices used to fit one tile's BaSiC model: the whole volume, or a
    random subsample of `max_fit_slices` slices if it's deeper than that."""
    n = vol.shape[0]
    if n <= max_fit_slices:
        return vol
    idx = np.sort(rng.choice(n, size=max_fit_slices, replace=False))
    return vol[idx]


def fit_basic_fields(vol, device='cpu', get_darkfield=True,
                      smoothness_flatfield=None, smoothness_darkfield=None,
                      max_fit_slices=DEFAULT_MAX_FIT_SLICES, rng=None):
    """Fit a BaSiC model on (a sample of) this tile's own z-stack.

    Returns ``(flatfield, darkfield)`` as float32 (Y, X) arrays, ready for
    `apply_basic_correction`.
    """
    from basicpy import BaSiC

    rng = rng if rng is not None else np.random.default_rng()
    fit_stack = _fit_stack(vol, max_fit_slices, rng)

    # basicpy's smoothness_* fields are typed `float`, not `Optional[float]`
    # -- they default to None (auto-estimated) but only if the keyword is
    # left unset entirely; passing smoothness_*=None explicitly fails
    # pydantic validation.
    basic_kwargs = {}
    if smoothness_flatfield is not None:
        basic_kwargs['smoothness_flatfield'] = smoothness_flatfield
    if smoothness_darkfield is not None:
        basic_kwargs['smoothness_darkfield'] = smoothness_darkfield
    with warnings.catch_warnings():
        # basicpy warns that get_darkfield=True "may lead to unstable results"
        # from a pydantic validator, i.e. once per BaSiC() construction. The
        # caveat is real, but darkfield estimation is a deliberate setting
        # here (basic_get_darkfield) and a model is fit per tile+channel, so
        # the warning is re-emitted dozens of times per run -- and its
        # stacklevel=3 points at a different caller each time, which defeats
        # Python's once-per-location de-duplication. Suppress only this one
        # message; every other basicpy warning still comes through.
        warnings.filterwarnings(
            'ignore',
            message=r'.*get_darkfield=True.*',
            category=UserWarning,
        )
        basic = BaSiC(get_darkfield=get_darkfield, device=device, **basic_kwargs)

    # One fit at a time per process -- see _FIT_LOCK. Only the fit is guarded;
    # applying the correction is plain numpy and runs unserialised.
    with _FIT_LOCK:
        basic.fit(fit_stack)
        return (np.asarray(basic.flatfield, dtype=np.float32),
                np.asarray(basic.darkfield, dtype=np.float32))


def fit_and_correct_tile(vol, device='cpu', get_darkfield=True,
                          smoothness_flatfield=None, smoothness_darkfield=None,
                          max_fit_slices=DEFAULT_MAX_FIT_SLICES, rng=None):
    """Fit a BaSiC model on (a sample of) this tile's own z-stack and apply
    it to the full volume. Returns the corrected volume (same dtype)."""
    flatfield, darkfield = fit_basic_fields(
        vol, device=device, get_darkfield=get_darkfield,
        smoothness_flatfield=smoothness_flatfield,
        smoothness_darkfield=smoothness_darkfield,
        max_fit_slices=max_fit_slices, rng=rng)
    return apply_basic_correction(vol, flatfield, darkfield, vol.dtype)


def apply_basic_correction(vol, flatfield, darkfield, dtype):
    """(image - darkfield) / flatfield, clipped back into `dtype`'s range."""
    corrected = (vol.astype(np.float32) - darkfield[None]) / flatfield[None]
    hi = np.iinfo(dtype).max if np.issubdtype(dtype, np.integer) else None
    corrected = np.clip(corrected, 0, hi)
    return np.round(corrected).astype(dtype)
