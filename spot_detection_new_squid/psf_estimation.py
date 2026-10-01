"""Measure spot size from the spatial autocorrelation, and derive the three
filter parameters from it.

`footprint_size`, `smooth_sigma` and `prominence_footprint_size` all describe
neighbourhoods that should scale with how big a spot actually is. Historically
they were hardcoded (10 / 1.0 / 9) and never checked against the data.

The spot size is measured here without detecting anything first. For an image
of randomly placed spots, the spatial autocorrelation of a Gaussian spot is
itself Gaussian with width ``sigma * sqrt(2)``, so fitting the ACF's central
peak recovers the spot width. Detector noise is white, so it contributes a
delta at exactly zero lag and nothing elsewhere -- excluding that single point
from the fit is what removes it. No threshold is involved anywhere, which
matters because the obvious alternative (measure the spots you detected) needs
thresholds to find spots and so cannot be used to choose the parameters that
set those thresholds.

Validated against a direct measurement on detected spots in the 0615 gel:

    ACF fit      lateral FWHM 6.5-7.8 px,  axial 4.4-4.8 px
    direct       lateral FWHM 6.0 px,      axial 4.0 px

consistent across all four channels. On synthetic data with a known PSF it
recovered sigma = (1.57, 2.49, 2.49) against a true (1.70, 2.50, 2.50).

Note what this module deliberately does *not* do. The same autocorrelation can
in principle yield the spot *density* (Image Correlation Spectroscopy), which
would give a per-channel `target_density`. That was implemented and tested and
does not work at these spot densities: on real gel data it was wrong by 3-24x
and, worse, inconsistently between channels, so the error cannot be calibrated
away. ICS needs the particle signal to dominate the image statistics, but at
~2 spots per megavoxel the spots occupy ~0.1% of voxels and the
autocorrelation is dominated by background structure instead. Spot *size* is
recoverable there; spot *count* is not. Use `calibration.py`'s gene objective
for the count-like question.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from .io_utils import (
    configure_single_tiff_channels,
    extract_wavelength_from_channel,
    get_image_dir,
    is_reference_channel,
    list_ome_tiles,
    load_tile_slab,
    read_ome_tile_info,
    tile_path,
)

# Neighbourhood sizes as multiples of the measured lateral FWHM. Both factors
# are set so that the measured spot size on this microscope (lateral FWHM
# ~6.5 px) reproduces the previously hand-set values -- footprint 10,
# prominence window 9 -- which were separately validated: shrinking the
# footprint changed the spot count by 1 in 159, and at a prominence window of
# 9 the measured prominence already recovers 96% of a spot's true excess over
# background while larger windows raise the noise floor faster than the
# signal. The point of deriving them is that they now track the optics and
# sampling instead of being fixed.
FOOTPRINT_PER_FWHM = 1.5
PROMINENCE_PER_FWHM = 1.4

# Guard rails: a bad ACF fit should not produce an absurd filter.
MIN_FOOTPRINT, MAX_FOOTPRINT = 3, 31
MIN_SMOOTH_SIGMA, MAX_SMOOTH_SIGMA = 0.5, 3.0


def spatial_acf(vol, max_lag=(5, 10, 10), background=None):
    """Normalised spatial autocorrelation, cropped to +/- max_lag.

    Normalised by the *particle* mean (image mean minus background), not the
    raw mean. Only the shape matters for the size fit, but keeping the
    normalisation correct makes the fitted amplitude meaningful too.
    """
    f = np.asarray(vol, dtype=np.float64)
    if background is None:
        background = float(np.median(f))
    mean_particle = f.mean() - background
    if mean_particle <= 0:
        return None
    d = f - f.mean()
    power = np.abs(np.fft.fftn(d)) ** 2
    acf = np.real(np.fft.ifftn(power)) / (f.size * mean_particle ** 2)
    acf = np.fft.fftshift(acf)
    centre = [s // 2 for s in acf.shape]
    sl = tuple(slice(c - m, c + m + 1) for c, m in zip(centre, max_lag))
    return acf[sl]


def fit_acf_gaussian(vol, max_lag=(5, 10, 10)):
    """Fit ``amplitude * gaussian + offset`` to the ACF, excluding zero lag.

    Returns a dict with the spot sigma in voxels (z, y, x), or None if the fit
    fails. The ACF of a Gaussian spot has width sigma*sqrt(2), so the fitted
    widths are divided by sqrt(2).
    """
    from scipy.optimize import curve_fit

    acf = spatial_acf(vol, max_lag)
    if acf is None:
        return None
    grids = np.meshgrid(*[np.arange(-m, m + 1) for m in max_lag], indexing='ij')
    mask = np.ones(acf.shape, dtype=bool)
    mask[max_lag[0], max_lag[1], max_lag[2]] = False   # the white-noise delta

    def model(coords, amp, sz, sy, sx, off):
        z, y, x = coords
        return amp * np.exp(-(z ** 2 / (2 * sz ** 2) + y ** 2 / (2 * sy ** 2)
                              + x ** 2 / (2 * sx ** 2))) + off

    p0 = [max(float(acf[mask].max()), 1e-9), 2.0, 3.0, 3.0, float(np.median(acf))]
    try:
        popt, _ = curve_fit(model, tuple(g[mask] for g in grids), acf[mask],
                            p0=p0, maxfev=20000)
    except Exception:
        return None

    amp, sz, sy, sx, off = popt
    if not np.all(np.isfinite(popt)) or amp <= 0:
        return None
    sigma = tuple(abs(s) / np.sqrt(2) for s in (sz, sy, sx))
    if not all(0.2 < s < 20 for s in sigma):
        return None

    resid = acf[mask] - model(tuple(g[mask] for g in grids), *popt)
    return {'sigma': sigma, 'amplitude': float(amp), 'offset': float(off),
            'rms_residual': float(np.sqrt(np.mean(resid ** 2)))}


def derive_filter_parameters(sigma_zyx):
    """Map a measured spot sigma (voxels, z/y/x) to the three filter sizes."""
    sz, sy, sx = sigma_zyx
    lateral_fwhm = 2.355 * 0.5 * (sy + sx)

    footprint = int(round(FOOTPRINT_PER_FWHM * lateral_fwhm))
    prominence = int(round(PROMINENCE_PER_FWHM * lateral_fwhm))
    # Matched filtering wants smoothing at the spot's own sigma, but the
    # detector smooths isotropically in voxels, so the narrowest axis sets the
    # cap -- anything larger over-smooths that axis. With 260x260x1500 nm
    # voxels the narrow axis is z.
    smooth = float(min(sz, sy, sx))

    return {
        'footprint_size': int(np.clip(footprint, MIN_FOOTPRINT, MAX_FOOTPRINT)),
        'prominence_footprint_size': int(np.clip(prominence, MIN_FOOTPRINT, MAX_FOOTPRINT)),
        'smooth_sigma': float(np.clip(round(smooth, 2), MIN_SMOOTH_SIGMA, MAX_SMOOTH_SIGMA)),
        'lateral_fwhm_px': float(lateral_fwhm),
        'axial_fwhm_px': float(2.355 * sz),
    }


def _sample_chunks(image_dir, fov, channel, n_z_slabs, n_xy, chunk_shape):
    """Signal-bearing chunks for one tile+channel, chosen by bright-tail contrast."""
    info = read_ome_tile_info(tile_path(image_dir, fov))
    cz, cy, cx = chunk_shape
    cz = min(cz, info.n_z)
    H, W = info.height, info.width
    margin = min(info.n_z // 10, max(info.n_z - cz, 0))
    span = max(info.n_z - cz - 2 * margin, 0)
    starts = ([margin + span // 2] if n_z_slabs == 1 else
              [int(margin + round(i * span / (n_z_slabs - 1))) for i in range(n_z_slabs)])
    for z0 in sorted(set(min(max(s, 0), max(info.n_z - cz, 0)) for s in starts)):
        slab = load_tile_slab(image_dir, fov, channel, z0, z0 + cz)
        scored = []
        for y0 in range(0, H - cy + 1, cy):
            for x0 in range(0, W - cx + 1, cx):
                sub = slab[::2, y0:y0 + cy:4, x0:x0 + cx:4]
                scored.append((float(np.percentile(sub, 99.9) - np.median(sub)), y0, x0))
        scored.sort(reverse=True)
        for _, y0, x0 in scored[:n_xy]:
            yield z0, y0, x0, slab[:, y0:y0 + cy, x0:x0 + cx]
        del slab


def estimate_filter_parameters(squid_folder, *, channels=None, reference_channels=(),
                                single_tiff_channels_by_index=True,
                                tiles=None, n_z_slabs=2, n_xy_per_slab=2,
                                chunk_shape=(24, 512, 512), max_lag=(5, 10, 10),
                                report_dir=None, verbose=True):
    """Measure spot size per channel and derive one shared set of filter sizes.

    The pipeline applies a single `footprint_size` / `smooth_sigma` /
    `prominence_footprint_size` to every channel, so the per-channel
    measurements are aggregated by median. Returns
    ``(params, per_channel_sigma, rows)``.

    single_tiff_channels_by_index has the same meaning as in `run_pipeline`:
    for a single-OME-TIFF input, name its channels by C index off the hardcoded
    map rather than from the file (see `configure_single_tiff_channels`).
    """
    configure_single_tiff_channels(squid_folder, single_tiff_channels_by_index)
    image_dir = get_image_dir(squid_folder)
    all_tiles = [f for f, _ in list_ome_tiles(image_dir)]
    tiles = all_tiles[:1] if tiles is None else [t for t in tiles if t in all_tiles]
    if not tiles:
        raise ValueError(f'None of the requested tiles exist in {image_dir}')

    info = read_ome_tile_info(tile_path(image_dir, tiles[0]))
    if channels is None:
        channels = [c for c in info.channels
                    if not is_reference_channel(c, reference_channels)]

    rows = []
    for fov in tiles:
        for channel in channels:
            for z0, y0, x0, chunk in _sample_chunks(image_dir, fov, channel,
                                                     n_z_slabs, n_xy_per_slab, chunk_shape):
                fit = fit_acf_gaussian(chunk, max_lag)
                row = {'channel': channel, 'tile': fov, 'z0': z0, 'y0': y0, 'x0': x0}
                if fit is None:
                    row.update({'sigma_z': np.nan, 'sigma_y': np.nan, 'sigma_x': np.nan,
                                'rms_residual': np.nan, 'ok': False})
                else:
                    sz, sy, sx = fit['sigma']
                    row.update({'sigma_z': sz, 'sigma_y': sy, 'sigma_x': sx,
                                'rms_residual': fit['rms_residual'], 'ok': True})
                rows.append(row)

    df = pd.DataFrame(rows)
    good = df[df['ok']]
    if good.empty:
        raise RuntimeError('Spot-size estimation failed: no ACF fit converged. '
                           'Check that the sampled chunks contain spots.')

    per_channel = (good.groupby('channel')[['sigma_z', 'sigma_y', 'sigma_x']]
                       .median().to_dict('index'))
    sigma = (float(good['sigma_z'].median()),
             float(good['sigma_y'].median()),
             float(good['sigma_x'].median()))
    params = derive_filter_parameters(sigma)

    if verbose:
        print(f'Spot size from the autocorrelation ({len(good)}/{len(df)} chunk fits ok):')
        for ch, s in sorted(per_channel.items()):
            wl = extract_wavelength_from_channel(ch)
            print(f'  {wl:>4} nm: sigma (z,y,x) = '
                  f'({s["sigma_z"]:.2f}, {s["sigma_y"]:.2f}, {s["sigma_x"]:.2f}) px  '
                  f'lateral FWHM {2.355 * 0.5 * (s["sigma_y"] + s["sigma_x"]):.2f} px')
        print(f'  median sigma (z,y,x) = ({sigma[0]:.2f}, {sigma[1]:.2f}, {sigma[2]:.2f}) px'
              f'  -> lateral FWHM {params["lateral_fwhm_px"]:.2f} px, '
              f'axial FWHM {params["axial_fwhm_px"]:.2f} px')
        print(f'  derived: footprint_size={params["footprint_size"]}  '
              f'smooth_sigma={params["smooth_sigma"]}  '
              f'prominence_footprint_size={params["prominence_footprint_size"]}')

    if report_dir is not None:
        os.makedirs(report_dir, exist_ok=True)
        out = os.path.join(report_dir, 'psf_estimation_report.csv')
        df.to_csv(out, index=False)
        if verbose:
            print(f'  per-chunk spot-size fits written to {out}')

    return params, per_channel, df
