"""Automatic per-channel `threshold` / `tolerance` calibration.

Why this can be cheap
---------------------
The detector's peak mask is::

    v        = gaussian(raw, smooth_sigma)
    is_peak  = (maximum_filter(v, footprint_size) == v) & (raw >= threshold)
    is_peak &= (raw - minimum_filter(v, prominence_footprint_size)) >= tolerance

The `maximum_filter(v) == v` term depends on neither `threshold` nor
`tolerance`. So *one* filter pass over a chunk yields every candidate maximum
together with the two statistics the thresholds act on -- its raw intensity
and its raw prominence -- and any (threshold, tolerance) pair is then pure
post-filtering on two per-candidate arrays. Sweeping the whole parameter
surface costs nothing beyond that single pass, and detection never has to be
re-run.

Raw units in, raw units out
---------------------------
`threshold` and `tolerance` are in RAW pixel units, so the emitted values are
numbers you can find by hovering over a spot in ImageJ. The *estimates*,
however, are made on the smoothed statistics and only then carried into raw
units by quantile matching (`_match_quantile`). Estimating directly in raw
units is measurably worse: raw candidate intensities keep the pixel noise
that smoothing removes, broadening the noise mode until Otsu returns almost
the same split for every channel -- 219/220/225 on three gel channels whose
hand-tuned thresholds were 400/250/250 -- which throws away the
channel-to-channel discrimination the smoothed-domain estimate gets right.

How the two parameters are chosen
---------------------------------
They behave very differently and are estimated differently:

* **threshold** -- Otsu's split on ``log10(smoothed candidate intensity)``,
  mapped to raw. Smoothed candidate intensities are strongly bimodal (a tight
  noise mode plus a sparse bright tail), which is exactly what Otsu wants.
  Validated against hand-tuned values on a calibration gel: it recovers the
  channel ordering and sits a near-constant 1.5x below the human picks
  (1.42-1.60 across four channels, 5% spread), with a 3.5% coefficient of
  variation across 12 chunks. Hence `strictness`, default 1.5.

* **tolerance** -- an explicit false-positive budget against a fitted null,
  *not* Otsu. On sparse samples real spots can be under 0.1% of candidates,
  leaving no second mode for Otsu to find; it then splits the noise mode
  itself. The null is modelled as a Gumbel: a background local maximum's
  prominence is ``max - min`` of noise over a window, i.e. an extreme-value
  statistic, and a Gaussian null underestimates its tail.

Guarding against empty chunks
-----------------------------
On sparse tissue many chunks contain no signal at all. There Otsu bisects the
noise mode and the resulting threshold produces a ~500x spot-density blowout.
This is detectable: expressed as a z-score against a null fitted to the lower
95% of candidates, failing chunks scored 0.0-0.4 while good chunks scored
17-123 -- no overlap. Chunks below `z_gate` are dropped before aggregation.
A second, empirical pass then drops any surviving chunk whose spot density at
the consensus parameters is `outlier_factor` above the median, and
re-aggregates. That -- deciding which chunks to trust -- is what the
iteration is for; the parameter sweep itself needs no iteration.

Output is a `channel_parameters.yaml` in exactly the format the pipeline
already consumes, plus a per-chunk diagnostic CSV and the count-vs-threshold
curves, so a bad calibration is visible rather than silent.
"""
from __future__ import annotations

import os
from typing import NamedTuple

import numpy as np
import pandas as pd
import yaml

from .io_utils import (
    configure_single_tiff_channels,
    extract_wavelength_from_channel,
    get_image_dir,
    get_resolution_nm,
    is_reference_channel,
    list_ome_tiles,
    load_tile_slab,
    load_tile_volume,
    read_ome_tile_info,
    tile_path,
)

DEFAULT_CHUNK_SHAPE = (32, 512, 512)
# Empirical: hand-tuned thresholds sat this far above Otsu's split, with a
# 6% spread across four channels of a calibration gel.
DEFAULT_STRICTNESS = 1.5


# ── statistics ───────────────────────────────────────────────────────────────
def _otsu(x, nbins=1024):
    """Otsu's threshold on a 1-D sample (histogram formulation)."""
    x = np.asarray(x, dtype=np.float64)
    finite = x[np.isfinite(x)]
    if finite.size == 0 or finite.min() == finite.max():
        return float(finite[0]) if finite.size else 0.0
    hist, edges = np.histogram(finite, bins=nbins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    hist = hist.astype(np.float64)

    w0 = np.cumsum(hist)
    w1 = np.cumsum(hist[::-1])[::-1]
    csum = np.cumsum(hist * centers)
    m0 = csum / np.maximum(w0, 1e-12)
    m1 = (np.cumsum((hist * centers)[::-1])[::-1]) / np.maximum(w1, 1e-12)

    between = w0[:-1] * w1[1:] * (m0[:-1] - m1[1:]) ** 2
    if not np.any(np.isfinite(between)) or between.max() <= 0:
        return float(np.median(finite))
    return float(centers[int(np.argmax(between))])


def _otsu_log(x):
    """Otsu's split of a positive sample, computed in log10 space."""
    return float(10 ** _otsu(np.log10(np.clip(np.asarray(x, float), 1e-3, None))))


def _null_fit(x, keep_frac=0.95):
    """Robust (median, sigma) of the noise mode.

    The MAD is taken over the lower `keep_frac` of the sample so a sparse
    signal tail cannot inflate it.
    """
    x = np.asarray(x, float)
    core = x[x <= np.quantile(x, keep_frac)]
    if core.size < 2:
        return float(np.median(x)), 0.0
    med = float(np.median(core))
    return med, float(1.4826 * np.median(np.abs(core - med)))


def _gumbel_null(x, qlo=50.0, qhi=90.0):
    """Fit a Gumbel null by matching two low quantiles of the sample.

    Only quantiles inside the noise mode are used, so the signal tail does
    not drag the fit.
    """
    x = np.asarray(x, float)
    a, b = np.percentile(x, qlo), np.percentile(x, qhi)
    ca = -np.log(-np.log(qlo / 100.0))
    cb = -np.log(-np.log(qhi / 100.0))
    beta = (b - a) / (cb - ca) if cb > ca else 0.0
    if not np.isfinite(beta) or beta <= 0:
        med, sig = _null_fit(x)
        beta = max(sig, 1e-6)
        return med, beta
    return float(a - beta * ca), float(beta)


def _gumbel_quantile(mu, beta, p):
    p = float(np.clip(p, 1e-12, 1 - 1e-12))
    return float(mu - beta * np.log(-np.log(p)))


# ── candidate extraction ─────────────────────────────────────────────────────
class Candidates(NamedTuple):
    """Per-candidate statistics in both domains.

    `raw_*` are what the detector thresholds against (real pixel values, so
    the emitted parameters are readable off the image). `smoothed_*` are what
    the estimators run on -- see `estimate_chunk` for why the estimate is not
    made directly in raw units.
    """
    raw_intensity: np.ndarray
    raw_prominence: np.ndarray
    smoothed_intensity: np.ndarray
    smoothed_prominence: np.ndarray
    # (N, 3) voxel positions (z, y, x) within the chunk. Only the gene
    # objective needs these -- it has to co-localize candidates across
    # channels -- so they are None unless `with_coords` was requested.
    coords: np.ndarray = None


def candidate_statistics(vol, footprint_size=10, smooth_sigma=1.0,
                          prominence_footprint_size=9, backend='scipy', device=None,
                          with_coords=False):
    """One filter pass -> `Candidates` for every candidate local maximum.

    The raw arrays mirror `detection._smoothed_local_max_core` exactly, minus
    the two threshold comparisons, so post-filtering them reproduces the
    detector's output for any (threshold, tolerance).
    """
    if backend == 'torch':
        import torch
        from . import torch_filters as tf
        dev = tf.select_device(device)
        raw = torch.from_numpy(np.ascontiguousarray(vol, dtype=np.float32)).to(dev)
        t = raw
        if smooth_sigma and smooth_sigma > 0:
            t = tf.gaussian_filter_torch(raw, smooth_sigma)
        is_max = tf.max_filter_torch(t, footprint_size) == t
        min_t = tf.min_filter_torch(t, prominence_footprint_size)
        cols = [raw[is_max], (raw - min_t)[is_max], t[is_max], (t - min_t)[is_max]]
        arrays = [c.to('cpu').numpy().astype(np.float64) for c in cols]
        coords = tf.nonzero_coords(is_max).astype(np.int64) if with_coords else None
        return Candidates(*arrays, coords=coords)

    from scipy.ndimage import gaussian_filter, maximum_filter, minimum_filter
    raw = vol.astype(np.float32)
    v = raw
    if smooth_sigma and smooth_sigma > 0:
        v = gaussian_filter(raw, sigma=smooth_sigma)
    is_max = maximum_filter(v, size=footprint_size) == v
    local_min = minimum_filter(v, size=prominence_footprint_size)
    return Candidates(
        raw[is_max].astype(np.float64),
        (raw - local_min)[is_max].astype(np.float64),
        v[is_max].astype(np.float64),
        (v - local_min)[is_max].astype(np.float64),
        coords=np.argwhere(is_max).astype(np.int64) if with_coords else None,
    )


def spot_count(inten, prom, threshold, tolerance):
    """Spots the detector would return for this (threshold, tolerance).

    Call with the RAW arrays -- that is what the detector compares against.
    """
    return int(np.count_nonzero((inten >= threshold) & (prom >= tolerance)))


def _match_quantile(source, source_cut, target):
    """The value in `target` that keeps the same upper-tail fraction as
    `source_cut` does in `source`.

    Used to carry a cut estimated on smoothed statistics over to raw units
    without assuming a global scale factor -- the raw/smoothed ratio varies
    with brightness (1.36x at a typical spot's peak, ~1.2x for the dim ones
    near the cut), so a single multiplier would bias the operating point.
    """
    frac = float(np.mean(source >= source_cut))
    if frac <= 0:
        return float(np.max(target))
    if frac >= 1:
        return float(np.min(target))
    return float(np.quantile(target, 1.0 - frac))


# ── per-chunk estimate ───────────────────────────────────────────────────────
def estimate_chunk(cand, chunk_voxels, tile_voxels, expected_fp_per_tile=10.0):
    """Threshold/tolerance estimate and gate diagnostics for one chunk.

    Both cuts are *estimated* on the smoothed statistics and only then mapped
    into raw units by quantile matching. Estimating directly in raw units is
    measurably worse: raw candidate intensities carry the pixel noise that
    smoothing removes, which broadens the noise mode until Otsu returns nearly
    the same split for every channel (219/220/225 on three gel channels whose
    hand-tuned thresholds differed), destroying the channel-to-channel
    discrimination the smoothed-domain estimate gets right. The returned
    values are raw, so they still mean what you read off the image.
    """
    n = len(cand.raw_intensity)
    if n < 50:
        return {'n_candidates': n, 'threshold': np.nan, 'tolerance': np.nan,
                'z_threshold': 0.0, 'null_median': np.nan, 'null_sigma': np.nan,
                'reason': 'too few candidates'}

    thr_sm = _otsu_log(cand.smoothed_intensity)
    imed, isig = _null_fit(cand.smoothed_intensity)
    z_thr = (thr_sm - imed) / isig if isig > 0 else 0.0
    thr = _match_quantile(cand.smoothed_intensity, thr_sm, cand.raw_intensity)

    # Tolerance: the prominence at which the fitted null is expected to throw
    # up `expected_fp_per_tile` false maxima across a whole tile.
    mu, beta = _gumbel_null(cand.smoothed_prominence)
    candidates_per_tile = n * (tile_voxels / max(chunk_voxels, 1))
    p_tail = expected_fp_per_tile / max(candidates_per_tile, 1.0)
    tol_sm = _gumbel_quantile(mu, beta, 1.0 - p_tail)
    tol = _match_quantile(cand.smoothed_prominence, tol_sm, cand.raw_prominence)

    return {'n_candidates': n, 'threshold': thr, 'tolerance': tol,
            'z_threshold': z_thr, 'null_median': imed, 'null_sigma': isig,
            'raw_background': float(np.median(cand.raw_intensity)), 'reason': ''}


def _solve_threshold_for_density(chunks, tolerance, target_density, thr0,
                                  chunk_voxels, thr_floor=None, hi=8.0, iters=60):
    """Scale factor on `thr0` whose pooled spot density hits `target_density`
    (spots per megavoxel). Bisection -- the sweep is free, so this is cheap.

    Returns (scale, achieved_density). The threshold is never driven below
    `thr_floor` (the background level): past that point it no longer rejects
    anything and only `tolerance` is still filtering, so a lower value would
    be meaningless rather than more sensitive.
    """
    def density(scale):
        total = sum(spot_count(c['cand'].raw_intensity, c['cand'].raw_prominence,
                               thr0 * scale, tolerance) for c in chunks)
        return total / max(len(chunks) * chunk_voxels / 1e6, 1e-12)

    lo = max(thr_floor / thr0, 1e-6) if (thr_floor and thr0 > 0) else 0.2
    lo = min(lo, hi)
    if density(hi) > target_density:
        return hi, density(hi)
    if density(lo) < target_density:
        return lo, density(lo)
    for _ in range(iters):
        mid = np.sqrt(lo * hi)
        if density(mid) > target_density:
            lo = mid
        else:
            hi = mid
    scale = float(np.sqrt(lo * hi))
    return scale, density(scale)


# ── chunk sampling ───────────────────────────────────────────────────────────
def _score_positions(slab, chunk_shape, n_xy):
    """Rank xy positions in a slab by bright-tail contrast, keeping them
    non-overlapping. Biases sampling toward regions that contain signal --
    on sparse tissue most positions are empty."""
    _, cy, cx = chunk_shape
    H, W = slab.shape[1], slab.shape[2]
    scored = []
    for y0 in range(0, H - cy + 1, cy):
        for x0 in range(0, W - cx + 1, cx):
            sub = slab[::4, y0:y0 + cy:4, x0:x0 + cx:4]
            scored.append((float(np.percentile(sub, 99.9) - np.median(sub)), y0, x0))
    scored.sort(reverse=True)
    return [(y0, x0) for _, y0, x0 in scored[:n_xy]]


def _iter_chunks(image_dir, fov, channel, n_z_slabs, n_xy_per_slab, chunk_shape,
                  correction_kwargs=None):
    """Yield (z0, y0, x0, chunk) for one tile+channel.

    Without corrections each z-slab is read once and several xy chunks are cut
    from it. With them the BaSiC model must be fit on the tile's full z-stack,
    so the whole tile is loaded and corrected first, then chunked in memory.
    """
    info = read_ome_tile_info(tile_path(image_dir, fov))
    cz, cy, cx = chunk_shape
    cz = min(cz, info.n_z)
    nz = info.n_z

    # Slab starts spread over the stack, avoiding the outermost planes, which
    # are commonly outside the sample.
    margin = min(nz // 10, max(nz - cz, 0))
    span = max(nz - cz - 2 * margin, 0)
    if n_z_slabs == 1:
        starts = [margin + span // 2]
    else:
        starts = [int(margin + round(i * span / (n_z_slabs - 1))) for i in range(n_z_slabs)]
    starts = sorted(set(min(max(s, 0), max(nz - cz, 0)) for s in starts))

    if correction_kwargs is not None:
        from .corrections import apply_corrections
        vol = apply_corrections(load_tile_volume(image_dir, fov, channel), **correction_kwargs)
        for z0 in starts:
            slab = vol[z0:z0 + cz]
            for y0, x0 in _score_positions(slab, chunk_shape, n_xy_per_slab):
                yield z0, y0, x0, slab[:, y0:y0 + cy, x0:x0 + cx]
        return

    for z0 in starts:
        slab = load_tile_slab(image_dir, fov, channel, z0, z0 + cz)
        for y0, x0 in _score_positions(slab, chunk_shape, n_xy_per_slab):
            yield z0, y0, x0, slab[:, y0:y0 + cy, x0:x0 + cx]
        del slab


def _slab_starts(n_z, cz, n_z_slabs):
    """z offsets of the sampled slabs, avoiding the outermost planes."""
    margin = min(n_z // 10, max(n_z - cz, 0))
    span = max(n_z - cz - 2 * margin, 0)
    if n_z_slabs == 1:
        starts = [margin + span // 2]
    else:
        starts = [int(margin + round(i * span / (n_z_slabs - 1))) for i in range(n_z_slabs)]
    return sorted(set(min(max(s, 0), max(n_z - cz, 0)) for s in starts))


def iter_multichannel_chunks(image_dir, fov, channels, n_z_slabs, n_xy_per_slab,
                              chunk_shape, footprint_size, smooth_sigma,
                              prominence_footprint_size, backend, device,
                              correction_kwargs=None):
    """Yield {channel: (Candidates, coords)} for chunks sampled at the *same*
    (z, y, x) in every channel.

    Co-registration is the whole point: the gene objective has to co-localize
    candidates across channels, so unlike the per-channel sampler these chunks
    cannot be chosen independently per channel. Positions are ranked on a
    strided preview summed over all channels, so a region only has to be bright
    in some channel to be sampled.
    """
    info = read_ome_tile_info(tile_path(image_dir, fov))
    cz, cy, cx = chunk_shape
    cz = min(cz, info.n_z)
    H, W = info.height, info.width

    for z0 in _slab_starts(info.n_z, cz, n_z_slabs):
        # Rank xy positions on a cheap strided preview, combined over channels.
        combined = None
        for ch in channels:
            prev = load_tile_slab(image_dir, fov, ch, z0, z0 + cz, step=4).astype(np.float32)
            med = np.median(prev)
            score = np.zeros(((H - cy) // cy + 1, (W - cx) // cx + 1), dtype=np.float32)
            for iy, y0 in enumerate(range(0, H - cy + 1, cy)):
                for ix, x0 in enumerate(range(0, W - cx + 1, cx)):
                    sub = prev[:, y0:y0 + cy:4, x0:x0 + cx:4]
                    score[iy, ix] = np.percentile(sub, 99.9) - med
            # Normalise so one bright channel cannot dominate the ranking.
            rng = float(score.max() - score.min())
            combined = (score / rng if rng > 0 else score) if combined is None else \
                combined + (score / rng if rng > 0 else score)
            del prev

        flat = np.argsort(combined, axis=None)[::-1][:n_xy_per_slab]
        positions = [(int(i // combined.shape[1]) * cy, int(i % combined.shape[1]) * cx)
                     for i in flat]

        # One full read per (slab, channel); cut every chosen position from it.
        per_position = {pos: {} for pos in positions}
        for ch in channels:
            slab = load_tile_slab(image_dir, fov, ch, z0, z0 + cz)
            if correction_kwargs is not None:
                from .corrections import apply_corrections
                slab = apply_corrections(slab, **correction_kwargs)
            for (y0, x0) in positions:
                chunk = slab[:, y0:y0 + cy, x0:x0 + cx]
                cand = candidate_statistics(chunk, footprint_size, smooth_sigma,
                                            prominence_footprint_size, backend, device,
                                            with_coords=True)
                per_position[(y0, x0)][ch] = (cand, cand.coords)
            del slab

        for pos in positions:
            yield z0, pos[0], pos[1], per_position[pos]


# ── per-channel calibration ──────────────────────────────────────────────────
def calibrate_channel(image_dir, channel, tiles, *, n_z_slabs=4, n_xy_per_slab=3,
                       chunk_shape=DEFAULT_CHUNK_SHAPE, footprint_size=10,
                       smooth_sigma=1.0, prominence_footprint_size=9,
                       strictness=DEFAULT_STRICTNESS, target_density=None,
                       expected_fp_per_tile=10.0, z_gate=5.0, outlier_factor=10.0,
                       max_refine_iters=3, backend='scipy', device=None,
                       correction_kwargs=None, verbose=True):
    """Calibrate one channel. Returns (threshold, tolerance, rows) where
    `rows` are the per-chunk diagnostics."""
    info = read_ome_tile_info(tile_path(image_dir, tiles[0]))
    tile_voxels = info.n_z * info.height * info.width
    chunk_voxels = int(np.prod(chunk_shape))

    rows, kept = [], []
    for fov in tiles:
        for z0, y0, x0, chunk in _iter_chunks(image_dir, fov, channel, n_z_slabs,
                                               n_xy_per_slab, chunk_shape, correction_kwargs):
            cand = candidate_statistics(
                chunk, footprint_size, smooth_sigma, prominence_footprint_size,
                backend, device)
            est = estimate_chunk(cand, chunk.size, tile_voxels, expected_fp_per_tile)
            est.update({'channel': channel, 'tile': fov, 'z0': z0, 'y0': y0, 'x0': x0,
                        'chunk_voxels': int(chunk.size)})
            passed = bool(np.isfinite(est['threshold'])) and est['z_threshold'] >= z_gate
            est['gate'] = 'pass' if passed else 'reject'
            if not passed and not est['reason']:
                est['reason'] = f'z_threshold {est["z_threshold"]:.1f} < {z_gate}'
            rows.append(est)
            if passed:
                kept.append({'cand': cand, 'est': est})

    if not kept:
        # Nothing passed: fall back to the median of every finite estimate so
        # the run can proceed, but say so loudly.
        finite = [r for r in rows if np.isfinite(r['threshold'])]
        if not finite:
            raise RuntimeError(
                f'Calibration failed for {channel}: no chunk produced a usable estimate. '
                f'Check that the channel contains signal, or widen chunk_shape.')
        thr = float(np.median([r['threshold'] for r in finite])) * strictness
        tol = float(np.median([r['tolerance'] for r in finite]))
        print(f'  WARNING [{channel}]: no chunk passed the z_gate={z_gate} signal test. '
              f'Falling back to the ungated median -- inspect the calibration report '
              f'before trusting these values.')
        for r in rows:
            r['used'] = True
        return int(round(thr)), int(round(tol)), rows

    for r in rows:
        r['used'] = r['gate'] == 'pass'

    # Consensus, then drop density outliers among the survivors and repeat.
    for _ in range(max_refine_iters):
        thr_base = float(np.median([k['est']['threshold'] for k in kept]))
        tol = float(np.median([k['est']['tolerance'] for k in kept]))

        if target_density is not None:
            # Floor in RAW units, matching the threshold being solved for.
            # (`null_median` is the smoothed-domain noise mode; the two are
            # near-identical for flat background, but keep the units honest.)
            thr_floor = float(np.median([k['est']['raw_background'] for k in kept]))
            scale, achieved = _solve_threshold_for_density(
                kept, tol, target_density, thr_base, chunk_voxels, thr_floor=thr_floor)
            if achieved < 0.8 * target_density:
                print(f'  WARNING [{channel}]: target_density={target_density} spots/Mvoxel '
                      f'is unreachable (best {achieved:.2f} at the background floor). '
                      f'`tolerance` is the binding constraint -- raise '
                      f'expected_fp_per_tile to admit dimmer spots.')
            elif achieved > 1.25 * target_density:
                print(f'  WARNING [{channel}]: target_density={target_density} spots/Mvoxel '
                      f'not reached from above (best {achieved:.2f} at the highest '
                      f'threshold tried). Either the spots are too uniform in '
                      f'brightness to separate at that density, or the threshold '
                      f'ceiling is too low -- check the calibration report.')
        else:
            scale = strictness
        thr = thr_base * scale

        densities = np.array([
            spot_count(k['cand'].raw_intensity, k['cand'].raw_prominence, thr, tol)
            / (k['est']['chunk_voxels'] / 1e6)
            for k in kept])
        med = float(np.median(densities))
        keep_mask = densities <= max(med * outlier_factor, 1e-9)
        if keep_mask.all() or keep_mask.sum() < 3:
            break
        for i, (k, keep_it) in enumerate(zip(kept, keep_mask)):
            if not keep_it:
                k['est']['used'] = False
                k['est']['reason'] = (f'density outlier '
                                       f'({densities[i]:.1f} vs median {med:.1f})')
        kept = [k for k, keep_it in zip(kept, keep_mask) if keep_it]

    for k in kept:
        k['est']['density_at_consensus'] = (
            spot_count(k['cand'].raw_intensity, k['cand'].raw_prominence, thr, tol)
            / (k['est']['chunk_voxels'] / 1e6))

    if verbose:
        dens = [k['est'].get('density_at_consensus', float('nan')) for k in kept]
        print(f'  {channel}: threshold={thr:.0f} tolerance={tol:.0f}  '
              f'({len(kept)}/{len(rows)} chunks used, '
              f'density {np.median(dens):.2f} spots/Mvoxel)')

    return int(round(thr)), int(round(tol)), rows


def _validate_calibrated_params(params, rows, report_csv=None):
    """Refuse to hand back a threshold of 0, whichever objective produced it.

    A zero threshold is not a permissive setting, it is a broken one: detection
    keeps every candidate with `intensity >= threshold`, so at 0 every local
    maximum in the volume qualifies and the maxima table grows with the image
    rather than with the spots. The run this guard was written for reached
    ~7.5M candidates per calibration chunk against the ~6k a healthy channel
    gives, and the detection that followed was OOM-killed at ~57 GB.

    It means the corrected pixels the calibrator sampled carried no signal at
    all -- nearly always because the correction chain over-subtracted them to a
    flat zero, which is exactly what `constant_offset` applied after a
    background subtraction used to do. Failing here costs one calibration;
    failing in detection costs the run and, on a shared machine, whatever else
    was resident.
    """
    bad = sorted(wl for wl, v in params.items() if v['threshold'] <= 0)
    if not bad:
        return params

    by_wl = {}
    for r in rows:
        by_wl.setdefault(extract_wavelength_from_channel(r['channel']), []).append(r)

    detail = []
    for wl in bad:
        chan_rows = by_wl.get(wl, [])
        backgrounds = [r.get('raw_background', np.nan) for r in chan_rows]
        flat = bool(backgrounds) and all(b == 0 for b in backgrounds if np.isfinite(b))
        n_cand = int(np.median([r.get('n_candidates', 0) for r in chan_rows])) if chan_rows else 0
        detail.append(
            f'  {wl} nm: threshold={params[wl]["threshold"]} '
            f'tolerance={params[wl]["tolerance"]}, '
            f'median {n_cand:,} candidates/chunk'
            + (', every sampled chunk was uniformly zero' if flat else ''))

    raise RuntimeError(
        'Calibration returned a threshold of 0, which would make detection keep '
        'every local maximum in the tile and exhaust memory:\n'
        + '\n'.join(detail)
        + '\n\nThe corrected pixels carry no signal. Check the correction chain '
          'first -- a `constant_offset` larger than what is left after BaSiC or a '
          'rolling-ball subtraction will floor the whole volume at zero.'
        + (f'\nPer-chunk diagnostics: {report_csv}' if report_csv else ''))


def calibrate_for_genes(squid_folder, genes_file, radius, depth, *, resolution=None,
                         reference_channels=(), channels=None, tiles=None,
                         n_z_slabs=4, n_xy_per_slab=3, chunk_shape=DEFAULT_CHUNK_SHAPE,
                         footprint_size=10, smooth_sigma=1.0,
                         prominence_footprint_size=9, expected_fp_per_tile=10.0,
                         chance_weight=1.0, unassigned_weight=1.0, passes=3, z_gate=5.0,
                         backend='scipy', device=None, correction_kwargs=None,
                         output_path=None, report_dir=None, verbose=True):
    """Choose per-channel thresholds that maximise chance-corrected gene calls.

    Unlike `target_density`, this does not assume every channel carries the
    same number of spots -- it optimises the coupled outcome directly. See
    gene_objective.py for the score and why it is chance-corrected.

    Per-channel Otsu estimates seed the search, so this only has to find a
    multiplier per channel rather than an absolute value from scratch.

    Note: with `correction_kwargs`, the correction chain (and so any BaSiC fit
    in it) is run on each sampled z-slab rather than the full tile, so it is a
    close approximation of -- not identical to -- the per-tile correction
    detection will apply.
    """
    from .gene_objective import evaluate, optimize

    image_dir = get_image_dir(squid_folder)
    all_tiles = [fov for fov, _ in list_ome_tiles(image_dir)]
    tiles = all_tiles[:1] if tiles is None else [t for t in tiles if t in all_tiles]
    if not tiles:
        raise ValueError(f'None of the requested tiles exist in {image_dir}')

    info = read_ome_tile_info(tile_path(image_dir, tiles[0]))
    if channels is None:
        channels = [c for c in info.channels
                    if not is_reference_channel(c, reference_channels)]
    wl_of = {c: extract_wavelength_from_channel(c) for c in channels}
    detected_wls = set(wl_of.values())

    with open(genes_file, 'r') as f:
        genes_data = yaml.safe_load(f) or {}
    gene_keys = {frozenset(v) for v in genes_data.values()}
    missing = {wl for k in gene_keys for wl in k} - detected_wls
    if missing:
        raise ValueError(
            f'genes.yaml references channel(s) {sorted(missing)} that are not being '
            f'detected (detected: {sorted(detected_wls)}). They are probably listed in '
            f'reference_channels; gene calibration cannot optimise for them.')

    if resolution is None:
        resolution = get_resolution_nm(squid_folder)
        if resolution is None:
            raise ValueError('resolution=None but the OME metadata has no PhysicalSize; '
                             'pass resolution=[x_nm, y_nm, z_nm].')

    tile_voxels = info.n_z * info.height * info.width
    if verbose:
        print(f'Gene-objective calibration: {len(gene_keys)} barcode(s) over '
              f'channels {sorted(detected_wls)}')

    # Collect co-registered chunks once; every optimiser evaluation reuses them.
    chunk_candidates, rows = [], []
    for fov in tiles:
        for z0, y0, x0, per_channel in iter_multichannel_chunks(
                image_dir, fov, channels, n_z_slabs, n_xy_per_slab, chunk_shape,
                footprint_size, smooth_sigma, prominence_footprint_size,
                backend, device, correction_kwargs):
            by_wl = {wl_of[ch]: v for ch, v in per_channel.items()}
            chunk_candidates.append(by_wl)
            for ch, (cand, _) in per_channel.items():
                est = estimate_chunk(cand, int(np.prod(chunk_shape)), tile_voxels,
                                     expected_fp_per_tile)
                est.update({'channel': ch, 'tile': fov, 'z0': z0, 'y0': y0, 'x0': x0,
                            'chunk_voxels': int(np.prod(chunk_shape))})
                est['gate'] = 'pass' if (np.isfinite(est['threshold'])
                                          and est['z_threshold'] >= z_gate) else 'reject'
                rows.append(est)

    if not chunk_candidates:
        raise RuntimeError('Gene calibration sampled no chunks.')

    # Seed: per-channel median of the gated Otsu estimates (raw units).
    base_thresholds, tolerances = {}, {}
    for ch in channels:
        wl = wl_of[ch]
        good = [r for r in rows if r['channel'] == ch and r['gate'] == 'pass'
                and np.isfinite(r['threshold'])]
        pool = good or [r for r in rows if r['channel'] == ch and np.isfinite(r['threshold'])]
        if not pool:
            raise RuntimeError(f'No usable estimate for {ch}; widen chunk_shape or lower z_gate.')
        if not good and verbose:
            print(f'  WARNING [{ch}]: no chunk passed z_gate={z_gate}; seeding from '
                  f'the ungated median -- check the calibration report.')
        base_thresholds[wl] = float(np.median([r['threshold'] for r in pool]))
        tolerances[wl] = float(np.median([r['tolerance'] for r in pool]))

    if verbose:
        print(f'  seeded from per-channel Otsu estimates: '
              f'{ {wl: round(t) for wl, t in base_thresholds.items()} }')

    thresholds, scales, history = optimize(
        chunk_candidates, base_thresholds, tolerances, gene_keys, sorted(detected_wls),
        radius, depth, resolution, chance_weight=chance_weight,
        unassigned_weight=unassigned_weight, passes=passes, verbose=verbose)

    final = evaluate(chunk_candidates, thresholds, tolerances, gene_keys,
                     sorted(detected_wls), radius, depth, resolution, chance_weight,
                     unassigned_weight)
    if verbose:
        score, assigned, unassigned, chance = final
        frac = 100.0 * assigned / max(assigned + unassigned, 1)
        print(f'  final: score={score:.1f}  assigned={assigned}  unassigned={unassigned} '
              f'({frac:.1f}% of groups assigned)  expected-by-chance={chance:.1f}')

    params = {wl: {'threshold': int(round(thresholds[wl])),
                   'tolerance': int(round(tolerances[wl]))}
              for wl in sorted(thresholds)}

    # Diagnostics first, then the guard, then the parameter file: a degenerate
    # calibration should leave the report that explains it behind, but must not
    # leave a channel_parameters.yaml a later run could silently be pinned to.
    report_csv = None
    if report_dir is not None:
        os.makedirs(report_dir, exist_ok=True)
        report_csv = os.path.join(report_dir, 'calibration_report.csv')
        pd.DataFrame(rows).to_csv(report_csv, index=False)
        pd.DataFrame(history).to_csv(
            os.path.join(report_dir, 'gene_optimization_trace.csv'), index=False)
        if verbose:
            print(f'Per-chunk diagnostics and optimiser trace written to {report_dir}')

    _validate_calibrated_params(params, rows, report_csv)

    if output_path is not None:
        write_channel_parameters(params, output_path)
        if verbose:
            print(f'Calibrated parameters written to {output_path}')

    return params


# ── top level ────────────────────────────────────────────────────────────────
def calibrate_channel_parameters(
    squid_folder,
    *,
    output_path=None,
    report_dir=None,
    channels=None,
    reference_channels=(),
    single_tiff_channels_by_index=True,
    tiles=None,
    n_z_slabs=4,
    n_xy_per_slab=3,
    chunk_shape=DEFAULT_CHUNK_SHAPE,
    footprint_size=10,
    smooth_sigma=1.0,
    prominence_footprint_size=9,
    strictness=DEFAULT_STRICTNESS,
    target_density=None,
    objective='strictness',
    genes_file=None,
    radius=None,
    depth=None,
    resolution=None,
    chance_weight=1.0,
    unassigned_weight=1.0,
    gene_passes=3,
    expected_fp_per_tile=10.0,
    z_gate=5.0,
    outlier_factor=10.0,
    max_refine_iters=3,
    backend='scipy',
    device=None,
    correction_kwargs=None,
    verbose=True,
):
    """Estimate `threshold` and `tolerance` for every channel of an acquisition.

    Returns ``{wavelength_int: {'threshold': int, 'tolerance': int}}`` -- the
    same structure `run_pipeline` reads from channel_parameters.yaml. If
    `output_path` is given the mapping is written there in that format.

    strictness           : multiplier on the Otsu threshold. 1.0 is the raw
                            split; the 1.35 default reproduces hand-tuned
                            values on a calibration gel. Higher = fewer, brighter spots.
    target_density       : if set (spots per megavoxel), the threshold is
                            solved for this density instead and `strictness`
                            is ignored. Assumes every channel carries the same
                            spot density, which is wrong when genes use
                            different colour combinations -- prefer
                            objective='genes' in that case.
    objective            : 'strictness' (default), or 'genes' to instead
                            maximise the chance-corrected number of spot
                            groups matching a barcode in `genes_file`. The
                            gene objective needs genes_file/radius/depth and
                            optimises the channels jointly, since whether a
                            spot is assigned depends on the other channels.
    chance_weight        : weight on the decoy-based chance correction of the
                            gene score. Higher = stricter.
    expected_fp_per_tile : false-positive budget setting `tolerance`, in
                            expected noise maxima per tile.
    z_gate               : minimum signal-presence z-score for a chunk to vote.
    single_tiff_channels_by_index : when `squid_folder` is a single OME-TIFF,
                            name its channels by C index from the hardcoded
                            CHANNEL_INDEX_TO_WAVELENGTH map (default) so the
                            wavelengths in genes_file / reference_channels
                            address them in order. False trusts the file's own
                            names. Ignored for an acquisition folder.
    """
    # Before any tile is read, so channel names are settled for the whole run.
    configure_single_tiff_channels(squid_folder, single_tiff_channels_by_index)

    if objective == 'genes':
        if genes_file is None or radius is None or depth is None:
            raise ValueError(
                "objective='genes' requires genes_file, radius and depth.")
        return calibrate_for_genes(
            squid_folder, genes_file, radius, depth, resolution=resolution,
            reference_channels=reference_channels, channels=channels, tiles=tiles,
            n_z_slabs=n_z_slabs, n_xy_per_slab=n_xy_per_slab, chunk_shape=chunk_shape,
            footprint_size=footprint_size, smooth_sigma=smooth_sigma,
            prominence_footprint_size=prominence_footprint_size,
            expected_fp_per_tile=expected_fp_per_tile, chance_weight=chance_weight,
            unassigned_weight=unassigned_weight, passes=gene_passes, z_gate=z_gate, backend=backend, device=device,
            correction_kwargs=correction_kwargs, output_path=output_path, report_dir=report_dir,
            verbose=verbose)
    if objective != 'strictness':
        raise ValueError(f"unknown objective {objective!r}; expected 'strictness' or 'genes'")

    image_dir = get_image_dir(squid_folder)
    all_tiles = [fov for fov, _ in list_ome_tiles(image_dir)]
    if not all_tiles:
        raise FileNotFoundError(f'No per-tile OME-TIFFs found in {image_dir}')
    if tiles is None:
        tiles = all_tiles[:1]
    tiles = [t for t in tiles if t in all_tiles]
    if not tiles:
        raise ValueError(f'None of the requested tiles exist in {image_dir}')

    info = read_ome_tile_info(tile_path(image_dir, tiles[0]))
    if channels is None:
        channels = [c for c in info.channels
                    if not is_reference_channel(c, reference_channels)]

    if verbose:
        print(f'Calibrating {len(channels)} channel(s) from {image_dir}')
        print(f'  tiles={tiles}  chunks={n_z_slabs}x{n_xy_per_slab} of {chunk_shape}  '
              f'backend={backend}')
        if target_density is not None:
            print(f'  mode: target_density={target_density} spots/Mvoxel')
        else:
            print(f'  mode: strictness={strictness}')

    params, all_rows = {}, []
    for channel in channels:
        thr, tol, rows = calibrate_channel(
            image_dir, channel, tiles, n_z_slabs=n_z_slabs, n_xy_per_slab=n_xy_per_slab,
            chunk_shape=chunk_shape, footprint_size=footprint_size,
            smooth_sigma=smooth_sigma, prominence_footprint_size=prominence_footprint_size,
            strictness=strictness, target_density=target_density,
            expected_fp_per_tile=expected_fp_per_tile, z_gate=z_gate,
            outlier_factor=outlier_factor, max_refine_iters=max_refine_iters,
            backend=backend, device=device, correction_kwargs=correction_kwargs, verbose=verbose)
        params[extract_wavelength_from_channel(channel)] = {
            'threshold': thr, 'tolerance': tol}
        all_rows.extend(rows)

    # Diagnostics first, then the guard, then the parameter file -- see the note
    # on the same ordering in calibrate_for_genes.
    report_csv = None
    if report_dir is not None:
        os.makedirs(report_dir, exist_ok=True)
        report_csv = os.path.join(report_dir, 'calibration_report.csv')
        cols = ['channel', 'tile', 'z0', 'y0', 'x0', 'n_candidates', 'threshold',
                'tolerance', 'z_threshold', 'raw_background', 'null_median',
                'null_sigma', 'gate',
                'used', 'density_at_consensus', 'reason']
        df = pd.DataFrame(all_rows)
        for c in cols:
            if c not in df.columns:
                df[c] = np.nan
        df[cols].to_csv(report_csv, index=False)
        if verbose:
            print(f'Per-chunk calibration diagnostics written to {report_csv}')

    _validate_calibrated_params(params, all_rows, report_csv)

    if output_path is not None:
        write_channel_parameters(params, output_path)
        if verbose:
            print(f'Calibrated parameters written to {output_path}')

    return params


def write_channel_parameters(params, path):
    """Write `{wavelength: {threshold, tolerance}}` as channel_parameters.yaml."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or '.', exist_ok=True)
    plain = {int(wl): {'threshold': int(v['threshold']), 'tolerance': int(v['tolerance'])}
             for wl, v in sorted(params.items())}
    with open(path, 'w') as f:
        yaml.safe_dump(plain, f, default_flow_style=False, sort_keys=True)
