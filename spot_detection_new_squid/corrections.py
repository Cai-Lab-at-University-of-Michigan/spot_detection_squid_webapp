"""Per-tile intensity corrections applied before spot detection.

Four independent steps, each switched on by its own parameter and applied --
when more than one is enabled -- in this fixed order:

  1. Constant offset subtraction (`constant_offset`)
  2. BaSiC flatfield/darkfield  (`use_basic`)            -> basic_correction.py
  3. Rolling-ball background subtraction (`rolling_ball_radius`)
  4. Gaussian blur (`gaussian_blur_sigma`)

The whole chain runs in float32 on one working copy of the tile and is
rounded/clipped back into the image's own dtype only once, at the end, so
intermediate steps never lose precision to a uint16 round-trip and a
background subtraction that dips below zero is not truncated until everything
has been applied.

Why the offset goes first
-------------------------
`constant_offset` is the camera's pedestal -- a detector property, present
before any photon lands -- so it is removed before anything that models the
sample. Every later step either estimates an additive background itself
(BaSiC's darkfield, the rolling ball) or is linear (the blur), and subtracting
a constant *after* one of those double-subtracts the pedestal the estimator has
already taken out. With `rolling_ball_radius=50` and `constant_offset=107` that
was enough to floor an entire tile at zero, which in turn made calibration
return `threshold=0` and detection treat every voxel as a candidate.

The corollary is that a background estimator downstream will absorb most of
what the offset does: a rolling ball is equivariant under a constant shift, so
with BaSiC off, `constant_offset` + `rolling_ball_radius` gives the same pixels
as the rolling ball alone (to within float32 rounding -- measured at 1 count on
0.001% of pixels of a synthetic tile). That is the correct answer
-- the pedestal is part of the background the ball already removes -- and it is
why the two can now be enabled together without either fighting the other.
With BaSiC on, the offset does survive, because the flatfield division between
the two makes the pedestal's residue spatially varying rather than constant.

Like the BaSiC fit, the chain is used either:

* once per tile+channel while baking corrected copies of the whole image
  directory to disk (`write_corrected_image_dir`, used when
  `output_corrected=True`), or
* on the fly, right after a tile volume is loaded into memory for detection
  (called from `tasks.load_tile_channel`, used when `output_corrected=False`
  -- the corrected volume is discarded once detection on that tile+channel
  finishes, nothing extra is written to disk).

Both routes go through `apply_corrections()`, so the pixels detection sees are
the same either way. `calibration.py` calls it too, so calibrated
threshold/tolerance values refer to corrected pixel values.

IMPORTANT: every step here changes the intensity scale that `threshold` and
`tolerance` are expressed in. Turning one on or off invalidates per-channel
parameters calibrated under the old settings -- recalibrate.

Geometry note
-------------
Steps 3 and 4 are 2-D, applied plane by plane: a rolling ball is a model of
per-plane illumination/autofluorescence background, and z is sampled ~6x
coarser than xy here (1500 nm vs 260 nm), so a z-isotropic version of either
would mix planes that are physically far apart. `gaussian_blur_sigma` accepts
a 3-tuple ``(sigma_z, sigma_y, sigma_x)`` if a genuine 3-D blur is wanted.

New-layout note
---------------
The corrected copies are written back out as per-tile OME-TIFFs, mirroring the
input `ome_tiff/` folder one file per tile. They are written with axes
``CZYX`` (channel-major) rather than the acquisition's ``ZCYX`` so each
channel can be corrected, streamed to disk and freed one at a time -- holding
a whole ZCYX tile in memory would be ~10 GB for a 5-channel, 145-slice,
2720x2720 acquisition. `io_utils.load_tile_volume` reads either order.

An old-layout acquisition (one TIFF per tile/z/channel) is corrected the same
way and comes out in this new layout too, one `R0_<fov>.ome.tiff` per tile --
so detection then reads back per-tile OME-TIFFs whichever layout went in.
"""
from __future__ import annotations

import os
import shutil

import numpy as np
import tifffile
from scipy.ndimage import gaussian_filter, zoom

from .basic_correction import DEFAULT_MAX_FIT_SLICES, fit_basic_fields
from .io_utils import (
    is_single_tiff,
    is_tile_image_file,
    list_ome_tiles,
    load_tile_volume,
    read_ome_tile_info,
    tile_path,
)


# --------------------------------------------------------------------------- #
#  parameter bundle                                                           #
# --------------------------------------------------------------------------- #
def make_correction_kwargs(*, use_basic=False, basic_device='cpu',
                            basic_get_darkfield=True,
                            basic_smoothness_flatfield=None,
                            basic_smoothness_darkfield=None,
                            basic_max_fit_slices=DEFAULT_MAX_FIT_SLICES,
                            rolling_ball_radius=None, rolling_ball_shrink=None,
                            gaussian_blur_sigma=None, constant_offset=None):
    """Bundle the enabled corrections into the kwargs dict that
    `apply_corrections(vol, **kwargs)` takes, or None if nothing is enabled.

    None is the "no corrections" signal every call site tests for, so a caller
    never has to check four parameters itself.
    """
    basic = None
    if use_basic:
        basic = {
            'device': basic_device,
            'get_darkfield': basic_get_darkfield,
            'smoothness_flatfield': basic_smoothness_flatfield,
            'smoothness_darkfield': basic_smoothness_darkfield,
            'max_fit_slices': basic_max_fit_slices,
        }

    kwargs = {
        'basic': basic,
        'rolling_ball_radius': rolling_ball_radius or None,
        'rolling_ball_shrink': rolling_ball_shrink,
        'gaussian_blur_sigma': gaussian_blur_sigma or None,
        'constant_offset': constant_offset or None,
    }
    if basic is None and not any(kwargs[k] is not None for k in
                                 ('rolling_ball_radius', 'gaussian_blur_sigma',
                                  'constant_offset')):
        return None
    return kwargs


def describe_corrections(correction_kwargs):
    """One line per enabled correction, in application order, for logging."""
    if correction_kwargs is None:
        return []
    lines = []
    basic = correction_kwargs.get('basic')
    radius = correction_kwargs.get('rolling_ball_radius')
    offset = correction_kwargs.get('constant_offset')
    if offset:
        line = f'constant offset subtraction ({offset} counts)'
        if radius and basic is None:
            # Not a warning: the rolling ball is equivariant under a constant
            # shift, so this is a no-op by construction rather than a mistake.
            line += ' -- absorbed by the rolling ball below, no net effect'
        lines.append(line)
    if basic is not None:
        lines.append(
            f"BaSiC flatfield{'/darkfield' if basic['get_darkfield'] else ''} "
            f"(device={basic['device']}, fit independently per tile, "
            f"max {basic['max_fit_slices']} slices/fit)")
    if radius:
        shrink = _resolve_shrink(radius, correction_kwargs.get('rolling_ball_shrink'))
        lines.append(f'rolling-ball background subtraction '
                     f'(radius={radius} px, 2-D per z-slice, shrink={shrink}x)')
    sigma = correction_kwargs.get('gaussian_blur_sigma')
    if sigma:
        how = 'per z-slice' if np.isscalar(sigma) else '3-D (sigma_z, sigma_y, sigma_x)'
        lines.append(f'Gaussian blur (sigma={sigma} px, {how})')
    return lines


def uses_basic(correction_kwargs):
    """True if the chain includes a BaSiC fit -- the only torch-based step, and
    so the only one that constrains device sharing and fork-safety."""
    return correction_kwargs is not None and correction_kwargs.get('basic') is not None


# --------------------------------------------------------------------------- #
#  individual steps                                                           #
# --------------------------------------------------------------------------- #
def _resolve_shrink(radius, shrink=None):
    """Downsample factor for the rolling ball. Follows ImageJ's Subtract
    Background: the ball is rolled on a shrunken image and the resulting
    background is interpolated back up, which is what makes the operation
    affordable at all -- a direct roll of a radius-50 ball over one
    2720x2720 plane takes ~27 s, i.e. over an hour per 145-slice tile."""
    if shrink is not None:
        return max(1, int(shrink))
    if radius <= 10:
        return 1
    if radius <= 30:
        return 2
    if radius <= 100:
        return 4
    return 8


def _block_mean(img, shrink):
    """Downsample by `shrink` with a block mean, edge-padding to a whole
    number of blocks first. Averaging (rather than sampling) is what keeps the
    ball from following per-pixel noise into the background estimate."""
    h, w = img.shape
    ph, pw = (-h) % shrink, (-w) % shrink
    if ph or pw:
        img = np.pad(img, ((0, ph), (0, pw)), mode='edge')
    hh, ww = img.shape
    return img.reshape(hh // shrink, shrink, ww // shrink, shrink).mean(axis=(1, 3))


def rolling_ball_background_2d(plane, radius, shrink=None):
    """Rolling-ball background of one 2-D plane (float32).

    The ball has spatial radius `radius` and the intensity extent of a
    radius-`radius` ball in raw counts. When shrunk, only the spatial radius
    is divided by the shrink factor -- the intensity axis is not resampled --
    which is how ImageJ preserves the ball's "stiffness" against bright spots.
    """
    from skimage.restoration import ellipsoid_kernel, rolling_ball

    shrink = _resolve_shrink(radius, shrink)
    if shrink <= 1:
        return rolling_ball(plane, radius=radius).astype(np.float32, copy=False)

    small = _block_mean(plane, shrink).astype(np.float32, copy=False)
    r_small = max(1, int(round(radius / shrink)))
    kernel = ellipsoid_kernel((2 * r_small + 1, 2 * r_small + 1), 2 * radius)
    background = rolling_ball(small, kernel=kernel)
    background = zoom(background, shrink, order=1, mode='nearest', grid_mode=True)
    return background[:plane.shape[0], :plane.shape[1]].astype(np.float32, copy=False)


def subtract_rolling_ball(work, radius, shrink=None):
    """Subtract a per-z-slice rolling-ball background from a float32 (Z, Y, X)
    volume, in place."""
    for z in range(work.shape[0]):
        work[z] -= rolling_ball_background_2d(work[z], radius, shrink)
    return work


def apply_gaussian_blur(work, sigma):
    """Blur a float32 (Z, Y, X) volume in place. A scalar `sigma` blurs each
    z-slice independently in xy; a 3-tuple (sigma_z, sigma_y, sigma_x) blurs
    the volume in 3-D."""
    if np.isscalar(sigma):
        for z in range(work.shape[0]):
            work[z] = gaussian_filter(work[z], sigma=float(sigma))
        return work
    gaussian_filter(work, sigma=tuple(float(s) for s in sigma), output=work)
    return work


# --------------------------------------------------------------------------- #
#  the chain                                                                  #
# --------------------------------------------------------------------------- #
def apply_corrections(vol, *, basic=None, rolling_ball_radius=None,
                       rolling_ball_shrink=None, gaussian_blur_sigma=None,
                       constant_offset=None, rng=None):
    """Apply every enabled correction to one tile+channel volume, in the order
    documented at the top of this module. Returns a new volume of the same
    dtype (or `vol` itself when nothing is enabled)."""
    dtype = vol.dtype
    work = None

    def working_copy():
        return vol.astype(np.float32) if work is None else work

    if constant_offset:
        work = working_copy()
        work -= float(constant_offset)

    if basic is not None:
        # Fit on the pedestal-free volume: with get_darkfield=True the fit is
        # estimating that same additive term, and handing it data the offset has
        # already been taken out of is what keeps the two from subtracting it twice.
        src = working_copy()
        flatfield, darkfield = fit_basic_fields(src, rng=rng, **basic)
        work = (src - darkfield[None]) / flatfield[None]

    if rolling_ball_radius:
        work = subtract_rolling_ball(working_copy(), rolling_ball_radius,
                                     rolling_ball_shrink)

    if gaussian_blur_sigma:
        work = apply_gaussian_blur(working_copy(), gaussian_blur_sigma)

    if work is None:
        return vol

    if np.issubdtype(dtype, np.integer):
        np.clip(work, 0, np.iinfo(dtype).max, out=work)
        return np.round(work).astype(dtype)
    np.clip(work, 0, None, out=work)
    return work.astype(dtype, copy=False)


# --------------------------------------------------------------------------- #
#  baking corrected copies to disk                                            #
# --------------------------------------------------------------------------- #
def _ome_metadata(info):
    """OME metadata for a corrected copy: original channel names + pixel sizes."""
    metadata = {'axes': 'CZYX', 'Channel': {'Name': info.channel_names_raw}}
    px_x, px_y, px_z = info.physical_size_um
    if px_x is not None:
        metadata['PhysicalSizeX'] = px_x
        metadata['PhysicalSizeXUnit'] = 'µm'
    if px_y is not None:
        metadata['PhysicalSizeY'] = px_y
        metadata['PhysicalSizeYUnit'] = 'µm'
    if px_z is not None:
        metadata['PhysicalSizeZ'] = px_z
        metadata['PhysicalSizeZUnit'] = 'µm'
    return metadata


def write_corrected_tile(image_dir, fov, dest_path, channels_to_correct,
                          correction_kwargs):
    """Write one tile's corrected OME-TIFF to `dest_path`, from either layout.

    Every channel of the tile is written out; channels in
    `channels_to_correct` get the full correction chain (per
    `apply_corrections`, with its own independent BaSiC fit if BaSiC is
    enabled), the rest are copied through unchanged. Channels are processed one
    at a time and streamed straight to the writer, so peak memory stays at a
    single channel's z-stack.
    """
    info = read_ome_tile_info(tile_path(image_dir, fov))

    def planes():
        for channel in info.channels:
            vol = load_tile_volume(image_dir, fov, channel)
            if channel in channels_to_correct:
                vol = apply_corrections(vol, **correction_kwargs)
            for z in range(vol.shape[0]):
                yield vol[z]
            del vol

    tifffile.imwrite(
        dest_path,
        planes(),
        shape=(info.n_channels, info.n_z, info.height, info.width),
        dtype=info.dtype,
        ome=True,
        bigtiff=True,
        metadata=_ome_metadata(info),
    )


def write_corrected_image_dir(image_dir, dest_dir, channels_to_correct,
                               correction_kwargs):
    """Recreate `image_dir` under `dest_dir`: every tile is rewritten as one
    OME-TIFF with the channels in `channels_to_correct` corrected (each
    tile+channel processed independently) and all other channels copied
    through; anything else in `image_dir` (stray metadata files) is copied
    unchanged.

    `image_dir` may be in either layout -- an old layout's per-slice folder
    comes out as one `R0_<fov>.ome.tiff` per tile, the name its tiles are
    already addressed by (see legacy_io.LegacyTileRef).

    Returns the path detection should read from afterwards: `dest_dir`, or the
    single corrected file inside it when `image_dir` is a single OME-TIFF (its
    name need not match the `R0_<fov>` tile pattern, so the directory alone
    would not be enough to find it again).
    """
    os.makedirs(dest_dir, exist_ok=True)

    if is_single_tiff(image_dir):
        fov, src = list_ome_tiles(image_dir)[0]
        dest_path = os.path.join(dest_dir, os.path.basename(src))
        print(f'  Tile {fov}: correcting -> {dest_path}')
        write_corrected_tile(image_dir, fov, dest_path, channels_to_correct,
                             correction_kwargs)
        return dest_path

    tiles = list_ome_tiles(image_dir)

    for fname in os.listdir(image_dir):
        # Skips the tile images themselves -- the per-tile OME-TIFFs, or an old
        # layout's thousands of per-slice TIFFs -- which the loop below rewrites
        # tile by tile. Everything else is a sidecar file and is copied.
        if is_tile_image_file(image_dir, fname):
            continue
        src = os.path.join(image_dir, fname)
        if os.path.isfile(src):
            shutil.copyfile(src, os.path.join(dest_dir, fname))

    for fov, src in tiles:
        dest_path = os.path.join(dest_dir, os.path.basename(src))
        print(f'  Tile {fov}: correcting -> {dest_path}')
        write_corrected_tile(image_dir, fov, dest_path, channels_to_correct,
                             correction_kwargs)

    return dest_dir
