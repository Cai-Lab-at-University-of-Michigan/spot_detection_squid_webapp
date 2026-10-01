"""
Standalone local-FDR calibration of per-channel threshold / tolerance.

One tunable -- `q`, the fraction of CALLED spots you are willing to have be
noise. Everything else is estimated from the image. Self-contained: no imports
from any other calibration script, and it degrades gracefully to plain
tifffile loading if the spot_detection_new_squid package is not importable.

WHY NOT A THRESHOLD, A PERCENTILE, OR A SPOT COUNT
--------------------------------------------------
A raw threshold has to be re-guessed per sample. A percentile of the noise
fixes the false-POSITIVE rate, which controls nothing: the detector screens
every local maximum in ~1e9 voxels, so ~1e5-1e6 candidates per tile, and even
a 0.5% FPR is hundreds of false spots. An expected-false-spots-per-tile budget
is a count, so it too changes sample to sample.

The false-DISCOVERY rate does not. Fdr(t) = N_cand * P_null(S >= t) / N_det(t)
carries the candidate count in the numerator and the detection count in the
denominator, so both sides move with the sample and q means the same thing
every time.

METHOD (per channel, per statistic)
-----------------------------------
 1. Screen a few sampled chunks with the SAME filter chain the detector uses,
    collecting (peak, prominence) for every local maximum -- the candidate set.
 2. Fit the null's right TAIL, not its body, by peaks-over-threshold: a
    straight line through log S_emp between p90 and p99 of the candidates,
    where the null still dominates and extreme-value theory says the excess is
    asymptotically exponential.

    Fitting the null's BODY and extrapolating does not work here, and this is
    the whole crux: the same mode and width fitted with a Gaussian, a
    lognormal, or an IQR-scaled Gumbel move the q=0.05 cutoff by more than an
    order of magnitude. The choice of family silently becomes the tuning knob
    you were trying to remove. Only the tail is identifiable.
 3. Fdr(t) = S_null(t) / S_emp(t). S_emp at the window foot already carries
    pi0, so no separate pi0 estimate enters. Cut at the smallest t >= p90 with
    Fdr <= q.

Applied to prominence (-> tolerance) and peak (-> threshold). The detector
ANDs the two tests, so the realised FDR is <= q.

WHAT THIS DOES NOT CONTROL
--------------------------
Spectral bleedthrough. Local FDR separates signal from NOISE; a partner dye
leaking into this channel is real optical signal and will pass. If a channel
sits in a bright neighbour's tail, check the crosstalk floor separately (probe
this channel at loci matching a barcode that excludes it) and take the
stricter of the two cutoffs.

Set the values in USER PARAMETERS at the bottom, then run this file directly.
"""
from __future__ import annotations

import json
import os
import re
import sys

import numpy as np
import pandas as pd
import yaml
from scipy.ndimage import gaussian_filter, maximum_filter, minimum_filter

# --------------------------------------------------------------------------- #
#  optional pipeline integration                                               #
# --------------------------------------------------------------------------- #
_PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

try:
    from spot_detection_new_squid.io_utils import load_tile_volume as _pkg_load
except Exception:
    _pkg_load = None
try:
    from spot_detection_new_squid.basic_correction import fit_and_correct_tile as _pkg_basic
except Exception:
    _pkg_basic = None
try:
    from spot_detection_new_squid.gene_filter import group_points as _pkg_group
except Exception:
    _pkg_group = None


CHANNEL_INDEX_TO_WAVELENGTH = {0: 405, 1: 488, 2: 561, 3: 638, 4: 730}

# Fraction of CALLED spots allowed to be noise. Scale-free, so it is a genuine
# default rather than a per-sample guess -- see the module docstring.
DEFAULT_Q = 0.1


def load_volume(image, wavelength, channels_by_index=True):
    """(Z, Y, X) for one channel. Uses the package loader when available."""
    if _pkg_load is not None:
        try:
            return _pkg_load(os.path.dirname(image), _fov(image),
                             f'Fluorescence_{int(wavelength)}_nm_Ex')
        except Exception as exc:
            print(f'  package loader failed ({exc}); falling back to tifffile')

    import tifffile
    with tifffile.TiffFile(image) as tf:
        series = tf.series[0]
        axes, shape = series.axes, series.shape
        n_z = shape[axes.index('Z')]
        n_c = shape[axes.index('C')]
        if channels_by_index:
            inv = {v: k for k, v in CHANNEL_INDEX_TO_WAVELENGTH.items()}
            c_idx = inv[int(wavelength)]
        else:
            names = [c.get('Name', '') for c in []] or []
            raise SystemExit('channels_by_index=False needs the package loader')
        if c_idx >= n_c:
            raise SystemExit(f'channel index {c_idx} not in file with {n_c} channels')
        pages = series.pages
        first = pages[c_idx].asarray()
        vol = np.empty((n_z,) + first.shape, dtype=first.dtype)
        for z in range(n_z):
            vol[z] = pages[z * n_c + c_idx].asarray()
    return vol


def apply_basic(vol, device, max_fit_slices):
    if _pkg_basic is None:
        print('  BaSiC unavailable -- calibrating on uncorrected pixels. '
              'The detector must then also run with use_basic=False.')
        return vol
    return _pkg_basic(vol, device=device, get_darkfield=False,
                      max_fit_slices=max_fit_slices)


def _fov(image_path):
    m = re.search(r'R\d+_(\d+)\.ome\.tiff?$', os.path.basename(image_path))
    if not m:
        raise SystemExit(f'cannot parse FOV index from {image_path}')
    return int(m.group(1))


# --------------------------------------------------------------------------- #
#  candidate enumeration -- exactly what the detector screens                  #
# --------------------------------------------------------------------------- #
def candidate_stats(vol, footprint_size, smooth_sigma, prominence_footprint_size):
    """(peak, prominence) for every local maximum, matching detection.py."""
    raw = vol.astype(np.float32)
    sm = gaussian_filter(raw, sigma=smooth_sigma) if smooth_sigma else raw
    is_peak = maximum_filter(sm, size=footprint_size) == sm
    local_min = minimum_filter(sm, size=prominence_footprint_size)
    zz, yy, xx = np.where(is_peak)
    peak = raw[zz, yy, xx]
    return np.column_stack([peak, peak - local_min[zz, yy, xx]]).astype(float)


def sample_chunks(shape, n_chunks, chunk_xy, rng):
    nz, ny, nx = shape
    out = []
    for _ in range(n_chunks):
        y0 = int(rng.integers(0, max(1, ny - chunk_xy)))
        x0 = int(rng.integers(0, max(1, nx - chunk_xy)))
        out.append((slice(0, nz), slice(y0, y0 + chunk_xy), slice(x0, x0 + chunk_xy)))
    return out


# --------------------------------------------------------------------------- #
#  the calibration itself                                                      #
# --------------------------------------------------------------------------- #
def fdr_cutoff(values, q, tail_fit_pct=(90.0, 99.0), n_grid=4000):
    """Smallest cutoff on `values` whose tail-area Fdr is <= q.

    Returns (cutoff, diagnostics DataFrame, info dict).
    """
    s = np.sort(np.asarray(values, float)[np.isfinite(values)])
    n = len(s)
    if n < 100:
        raise ValueError(f'only {n} candidates -- sample more/larger chunks')

    def surv(x):
        """Empirical P(S >= x), floored at 1/n so ratios stay finite."""
        return np.maximum(1.0 - np.searchsorted(s, x, side='left') / n, 1.0 / n)

    # peaks-over-threshold fit on DATA VALUES (never on histogram bins: for a
    # channel with a narrow null the p90-p99 window spans only a couple of bins
    # on any grid sized for the full range, and a bin-based fit silently has
    # too few points).
    u_lo, u_hi = np.percentile(s, tail_fit_pct)
    xs = np.unique(s[(s >= u_lo) & (s <= u_hi)])
    if len(xs) < 4:                              # heavily quantised statistic
        u_lo, u_hi = np.percentile(s, (50.0, 99.5))
        xs = np.unique(s[(s >= u_lo) & (s <= u_hi)])
    slope, intercept = np.polyfit(xs, np.log(surv(xs)), 1)
    beta = -1.0 / slope if slope < 0 else np.inf

    grid = np.linspace(u_lo, float(s[-1]), n_grid)
    surv_emp = surv(grid)
    surv_null = np.exp(intercept + slope * grid)
    Fdr = np.clip(surv_null / surv_emp, 0, 1)

    hit = np.where(Fdr <= q)[0]
    cut_idx = int(hit[0]) if len(hit) else int(len(grid) - 1)
    cutoff = float(grid[cut_idx])

    diag = pd.DataFrame({'value': grid, 'tail_Fdr': Fdr, 'surv_emp': surv_emp,
                         'surv_null': surv_null})
    info = {'n_candidates': n, 'tail_u_lo': float(u_lo), 'tail_u_hi': float(u_hi),
            'tail_beta': float(beta), 'n_tail_points': int(len(xs)),
            'cutoff': cutoff, 'Fdr_at_cutoff': float(Fdr[cut_idx]),
            'frac_above': float((s >= cutoff).mean())}
    return cutoff, diag, info


# --------------------------------------------------------------------------- #
#  optional held-out check: consensus loci                                     #
# --------------------------------------------------------------------------- #
def consensus_positive_keys(codebook, channel):
    """Group keys that are evidence a spot of `channel` belongs at that locus:
    a subset of some barcode containing `channel`, missing `channel`, at least
    two channels wide, and not itself a barcode. Codebook-agnostic."""
    from itertools import combinations
    barcodes = set(codebook.values())
    out = set()
    for b in codebook.values():
        if channel not in b:
            continue
        rest = sorted(b - {channel})
        for r in range(len(rest), 1, -1):
            for c in combinations(rest, r):
                k = frozenset(c)
                if k not in barcodes:
                    out.add(k)
    return out


def probe_loci(vol, loci, smooth_sigma, prominence_footprint_size,
               half_xy=2, half_z=2, pad=12):
    """(peak, prominence) at given loci, using detection.py's criteria."""
    nz, ny, nx = vol.shape
    out = []
    for x, y, z in loci:
        xi, yi, zi = int(round(x)), int(round(y)), int(round(z))
        z0, z1 = max(0, zi - half_z - pad), min(nz, zi + half_z + pad + 1)
        y0, y1 = max(0, yi - pad), min(ny, yi + pad + 1)
        x0, x1 = max(0, xi - pad), min(nx, xi + pad + 1)
        if (z1 - z0) < 2 * half_z + 1 or (y1 - y0) < 2 * pad or (x1 - x0) < 2 * pad:
            continue
        raw = vol[z0:z1, y0:y1, x0:x1].astype(np.float32)
        sm = gaussian_filter(raw, sigma=smooth_sigma) if smooth_sigma else raw
        lmin = minimum_filter(sm, size=prominence_footprint_size)
        cz, cy, cx = zi - z0, yi - y0, xi - x0
        box = raw[max(0, cz - half_z):cz + half_z + 1,
                  max(0, cy - half_xy):cy + half_xy + 1,
                  max(0, cx - half_xy):cx + half_xy + 1]
        if box.size == 0:
            continue
        dz, dy, dx = np.unravel_index(int(np.argmax(box)), box.shape)
        pz, py, px = (max(0, cz - half_z) + dz, max(0, cy - half_xy) + dy,
                      max(0, cx - half_xy) + dx)
        pk = float(raw[pz, py, px])
        out.append((pk, pk - float(lmin[pz, py, px])))
    return np.asarray(out, float).reshape(-1, 2)


def load_bootstrap_groups(bootstrap, radius_px, depth_slices):
    """Group an existing run's detections, for the held-out consensus check."""
    if _pkg_group is None or not bootstrap:
        return None
    pat = re.compile(r'^Fluorescence_(.+)_nm_Ex$')
    chunks = []
    for fn in sorted(os.listdir(bootstrap)):
        m = pat.match(fn)
        if m:
            p = os.path.join(bootstrap, fn, 'maxima_results.csv')
            if os.path.exists(p):
                t = pd.read_csv(p)
                t['c'] = int(m.group(1))
                chunks.append(t)
    if not chunks:
        return None
    fl = pd.concat(chunks, ignore_index=True)
    return _pkg_group(fl[['X', 'Y', 'Slice']].to_numpy(float),
                      fl['c'].to_numpy(), radius_px, depth_slices)


# --------------------------------------------------------------------------- #
#  the per-channel driver, shared by both entry points                         #
# --------------------------------------------------------------------------- #
def calibrate_channels(volumes_for, channels, q, *, param_path=None, report_dir=None,
                       codebook=None, groups=None, tail_fit_pct=(90.0, 99.0),
                       n_chunks=4, chunk_xy=512, footprint_size=10, smooth_sigma=1.0,
                       prominence_footprint_size=9, seed=0):
    """Local-FDR threshold/tolerance for every channel in `channels`.

    `volumes_for(channel)` yields (label, (Z, Y, X) volume) for every tile that
    channel should be measured on -- one for a single image, several for an
    acquisition folder. It is a generator so only one tile is held at a time,
    and it is where the caller decides what corrections the measured pixels
    have had applied.

    Returns ``{wavelength_int: {'threshold': int, 'tolerance': int}}`` -- the
    structure run_pipeline reads from channel_parameters.yaml -- written to
    `param_path` when given, with localfdr_report.csv / localfdr_curves.csv
    going to `report_dir`.
    """
    rng = np.random.default_rng(seed)
    params, rows, diags = {}, [], []

    for ch in channels:
        print(f'=== channel {ch}')
        keys = (consensus_positive_keys(codebook, ch)
                if groups is not None and codebook else None)
        cands, probes = [], []
        n_tiles, sampled_voxels, tile_voxels = 0, 0, 0

        for label, vol in volumes_for(ch):
            for sl in sample_chunks(vol.shape, n_chunks, chunk_xy, rng):
                chunk = vol[sl]
                sampled_voxels += chunk.size
                cands.append(candidate_stats(chunk, footprint_size, smooth_sigma,
                                             prominence_footprint_size))
            n_tiles += 1
            tile_voxels += vol.size
            # Probed here, while this tile is still in hand: the (peak,
            # prominence) pairs do not depend on the cutoffs, only the recovery
            # fraction computed from them below does.
            if keys is not None:
                pos = groups[groups['key'].isin(keys)]
                probes.append(probe_loci(vol, pos[['x', 'y', 'z']].to_numpy(float),
                                         smooth_sigma, prominence_footprint_size))
            del vol

        if not n_tiles:
            raise ValueError(f'no volume was yielded for channel {ch}')
        cand = np.vstack(cands)
        # sampled candidates -> candidates in one whole tile
        scale = (tile_voxels / n_tiles) / sampled_voxels
        print(f'  {len(cand)} candidate maxima in {n_tiles * n_chunks} chunks '
              f'-> ~{int(len(cand) * scale)} per tile')

        thr, d_thr, i_thr = fdr_cutoff(cand[:, 0], q, tail_fit_pct)
        tol, d_tol, i_tol = fdr_cutoff(cand[:, 1], q, tail_fit_pct)
        n_pass = int(((cand[:, 0] >= thr) & (cand[:, 1] >= tol)).sum())
        print(f'  peak       tail beta={i_thr["tail_beta"]:.1f} -> threshold={thr:.0f}')
        print(f'  prominence tail beta={i_tol["tail_beta"]:.1f} -> tolerance={tol:.0f}')
        print(f'  ~{int(n_pass * scale)} spots/tile at this operating point')

        rec, n_pos = float('nan'), 0
        if probes:
            P = np.vstack(probes)
            n_pos = len(P)
            if n_pos:
                rec = float(((P[:, 0] >= thr) & (P[:, 1] >= tol)).mean())
                print(f'  held-out consensus recovery (n={n_pos}): {100 * rec:.0f}%')
        print()

        params[int(ch)] = {'threshold': int(round(thr)), 'tolerance': int(round(tol))}
        rows.append({'channel': ch, 'q': q, 'threshold': int(round(thr)),
                     'tolerance': int(round(tol)),
                     'est_spots_per_tile': int(n_pass * scale),
                     'peak_tail_beta': i_thr['tail_beta'],
                     'prom_tail_beta': i_tol['tail_beta'],
                     'consensus_n': n_pos, 'consensus_recovery': rec})
        for nm, d in (('peak', d_thr), ('prominence', d_tol)):
            d = d.copy()
            d.insert(0, 'statistic', nm)
            d.insert(0, 'channel', ch)
            diags.append(d)

    if report_dir is None and param_path is not None:
        report_dir = os.path.dirname(param_path) or '.'
    if report_dir is not None:
        os.makedirs(report_dir, exist_ok=True)
        pd.DataFrame(rows).to_csv(os.path.join(report_dir, 'localfdr_report.csv'),
                                  index=False)
        pd.concat(diags).to_csv(os.path.join(report_dir, 'localfdr_curves.csv'),
                                index=False)
    if param_path is not None:
        os.makedirs(os.path.dirname(param_path) or '.', exist_ok=True)
        with open(param_path, 'w') as f:
            yaml.safe_dump({int(k): v for k, v in params.items()}, f, sort_keys=True)
        print(f'wrote {param_path}')
    print(json.dumps(params, indent=2))
    return params


# --------------------------------------------------------------------------- #
#  entry point used by the pipeline (orchestrator.run_pipeline)                #
# --------------------------------------------------------------------------- #
def calibrate_channel_parameters_localfdr(
        squid_folder, *, q=DEFAULT_Q, output_path=None, report_dir=None,
        channels=None, reference_channels=(), single_tiff_channels_by_index=True,
        tiles=None, correction_kwargs=None, n_chunks=4, chunk_xy=512,
        tail_fit_pct=(90.0, 99.0), footprint_size=10, smooth_sigma=1.0,
        prominence_footprint_size=9, genes_file=None, bootstrap=None,
        radius=None, depth=None, resolution=None, seed=0):
    """Calibrate an acquisition (or a single OME-TIFF) from inside the pipeline.

    Same contract as calibration.calibrate_channel_parameters, so run_pipeline
    can call either: takes `squid_folder`, returns
    ``{wavelength_int: {'threshold': int, 'tolerance': int}}`` and writes it to
    `output_path`. The one tunable is `q`, the fraction of CALLED spots allowed
    to be noise.

    Unlike the standalone entry point below, the volumes it measures go through
    `correction_kwargs` -- the same chain run_pipeline applies per tile -- so
    the cutoffs are expressed on exactly the pixel scale the detector will see.

    tiles     : fovs to measure on. None picks the middle tile of the
                acquisition. The cutoffs are properties of a channel's noise
                rather than of a field of view, so one tile is normally enough,
                and each extra one costs a full tile load per channel.
    channels  : integer wavelengths to calibrate. None takes every non-reference
                channel present in the tile.
    genes_file / bootstrap / radius / depth / resolution : only the optional
                held-out consensus check uses these, and only when `bootstrap`
                points at an existing run's output folder. They never influence
                the chosen cutoffs.
    """
    # Imported here, absolutely rather than relatively, so this module still
    # runs standalone (`python localfdr_calibration.py`), where it is __main__
    # and has no package to import from.
    from spot_detection_new_squid.corrections import apply_corrections
    from spot_detection_new_squid.io_utils import (
        configure_single_tiff_channels, extract_wavelength_from_channel,
        get_image_dir, is_reference_channel, list_ome_tiles, load_tile_volume,
        read_ome_tile_info, tile_path,
    )

    # Settles the channel names before any tile is read, exactly as the other
    # calibrator does -- run_pipeline has already done it, but a direct caller
    # may not have.
    configure_single_tiff_channels(squid_folder, single_tiff_channels_by_index)
    image_dir = get_image_dir(squid_folder)
    fovs = [fov for fov, _ in list_ome_tiles(image_dir)]
    if not fovs:
        raise FileNotFoundError(f'No per-tile OME-TIFFs found in {image_dir}')
    if tiles is None:
        tiles = [fovs[len(fovs) // 2]]
    else:
        missing = sorted(set(tiles) - set(fovs))
        if missing:
            raise ValueError(f'tiles {missing} are not in {image_dir} '
                             f'(available: {fovs})')
    print(f'Local-FDR calibration (q = {q}) on tile(s) '
          f'{list(tiles)} of {len(fovs)}')

    info = read_ome_tile_info(tile_path(image_dir, tiles[0]))
    name_by_wl = {extract_wavelength_from_channel(c): c for c in info.channels
                  if not is_reference_channel(c, reference_channels)}
    if channels is None:
        channels = sorted(name_by_wl)
    else:
        channels = [int(c) for c in channels]
        unknown = [c for c in channels if c not in name_by_wl]
        if unknown:
            raise ValueError(f'channel(s) {unknown} are not detectable channels of '
                             f'{info.source} (available: {sorted(name_by_wl)})')
    print(f'channels: {channels}')

    codebook = None
    if genes_file:
        codebook = {k: frozenset(v) for k, v in yaml.safe_load(open(genes_file)).items()}
    groups = None
    if bootstrap:
        if None in (radius, depth, resolution):
            raise ValueError('the held-out consensus check needs radius, depth '
                             'and resolution alongside bootstrap.')
        groups = load_bootstrap_groups(bootstrap, radius / resolution[0],
                                       depth / resolution[2])
        if groups is None:
            print('bootstrap groups unavailable -- skipping the held-out check')

    def volumes_for(ch):
        for fov in tiles:
            vol = load_tile_volume(image_dir, fov, name_by_wl[ch])
            if correction_kwargs is not None:
                vol = apply_corrections(vol, **correction_kwargs)
            yield f'tile {fov}', vol
            del vol      # so the next tile is loaded with this one already freed

    return calibrate_channels(
        volumes_for, channels, q, param_path=output_path, report_dir=report_dir,
        codebook=codebook, groups=groups, tail_fit_pct=tail_fit_pct,
        n_chunks=n_chunks, chunk_xy=chunk_xy, footprint_size=footprint_size,
        smooth_sigma=smooth_sigma,
        prominence_footprint_size=prominence_footprint_size, seed=seed)


# --------------------------------------------------------------------------- #
#  standalone entry point (this file run directly)                             #
# --------------------------------------------------------------------------- #
def calibrate(image, genes, out, q, channels=None, bootstrap=None,
              tail_fit_pct=(90.0, 99.0), n_chunks=4, chunk_xy=512,
              footprint_size=10, smooth_sigma=1.0, prominence_footprint_size=9,
              radius=750, depth=4500, resolution=(260, 260, 1500),
              use_basic=True, basic_device='cuda', basic_max_fit_slices=150,
              param_filename='channel_parameters_localfdr.yaml', seed=0):
    os.makedirs(out, exist_ok=True)
    codebook = {k: frozenset(v) for k, v in yaml.safe_load(open(genes)).items()}
    if channels is None:
        channels = sorted({c for b in codebook.values() for c in b})
    print(f'codebook: { {k: sorted(v) for k, v in codebook.items()} }')
    print(f'channels: {channels}\nq = {q}\n')

    groups = load_bootstrap_groups(bootstrap, radius / resolution[0],
                                   depth / resolution[2])
    if groups is None and bootstrap:
        print('bootstrap groups unavailable -- skipping the held-out check\n')

    def volumes_for(ch):
        vol = load_volume(image, ch)
        if use_basic:
            vol = apply_basic(vol, basic_device, basic_max_fit_slices)
        yield os.path.basename(image), vol

    return calibrate_channels(
        volumes_for, channels, q, param_path=os.path.join(out, param_filename),
        report_dir=out, codebook=codebook, groups=groups,
        tail_fit_pct=tail_fit_pct, n_chunks=n_chunks, chunk_xy=chunk_xy,
        footprint_size=footprint_size, smooth_sigma=smooth_sigma,
        prominence_footprint_size=prominence_footprint_size, seed=seed)


if __name__ == '__main__':

    ########## USER PARAMETERS ##############
    q = 0.1   # fraction of CALLED spots allowed to be noise. Scale-free: it
              # means the same thing in every sample. Determined by sweep on
              # 0727C tile 5 (q in 0.01..0.2); the cutoffs are very insensitive
              # to it -- 561/638/730 move under 10 counts across that range.

    image = ('')
    genes = ('')
    out = ('')
    param_filename = ''

    channels = None   # None = every channel appearing in the codebook

    # Optional: an existing run's output directory (holding
    # Fluorescence_*_nm_Ex/maxima_results.csv). Used ONLY to score the chosen
    # cutoffs against consensus loci -- it never influences the choice. None
    # skips the check.
    bootstrap = None

    n_chunks = 4          # random XY chunks sampled per channel (full z each)
    chunk_xy = 512        # side length in px of each chunk
    tail_fit_pct = (90.0, 99.0)   # null-dominated window for the tail fit
    seed = 0

    # MUST match the detection settings these parameters will be used with --
    # the cutoffs are defined against this exact filter chain.
    footprint_size = 10
    smooth_sigma = 1.0
    prominence_footprint_size = 9

    # only used for the held-out consensus check
    radius = 750
    depth = 4500
    resolution = (260, 260, 1500)

    # Must match the detector's setting: the cutoffs refer to whatever pixel
    # scale they were measured on.
    use_basic = True
    basic_device = 'cuda'
    basic_max_fit_slices = 150
    ##########################################

    calibrate(image=image, genes=genes, out=out, q=q, channels=channels,
              bootstrap=bootstrap, tail_fit_pct=tail_fit_pct, n_chunks=n_chunks,
              chunk_xy=chunk_xy, footprint_size=footprint_size,
              smooth_sigma=smooth_sigma,
              prominence_footprint_size=prominence_footprint_size,
              radius=radius, depth=depth, resolution=resolution,
              use_basic=use_basic, basic_device=basic_device,
              basic_max_fit_slices=basic_max_fit_slices,
              param_filename=param_filename, seed=seed)
