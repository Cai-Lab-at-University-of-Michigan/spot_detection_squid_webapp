"""
Reader for the *old* SQUID acquisition layout: one TIFF per
(tile, z-slice, channel), all of them in a `0/` folder::

    <squid_folder>/0/R0_<fov>_<z>_Fluorescence_488_nm_Ex.tiff
    <squid_folder>/0/coordinates.csv              (per-fov, per-z stage positions)
    <squid_folder>/coordinates.csv                (one row per fov)
    <squid_folder>/acquisition parameters.json    (Nx, Ny, Nz, dz ...)

This is the layout `spot_detection_functionalized_with_basic` reads. The rest
of this package is written against the new one (a single multi-page OME-TIFF
per tile, see io_utils.py), so rather than branching all over the pipeline this
module presents an old-layout tile *as if* it were one (Z, C, Y, X) OME-TIFF:

* `list_legacy_tiles` reports one `LegacyTileRef` per tile, standing in for the
  tile's file path,
* `legacy_tile_metadata` describes it with the same fields `io_utils.TileInfo`
  holds for a real OME-TIFF -- a Z axis and a wavelength-ordered C axis,
* `load_legacy_tile_volume` / `load_legacy_tile_slab` read one channel's
  z-planes out of the individual files.

`io_utils` dispatches to these for a per-slice folder and to its own OME-TIFF
readers otherwise, so every consumer downstream (detection, calibration, PSF
estimation, BaSiC/corrections, stitching, gene filtering) is unchanged and
works on either layout. Nothing else in the package imports this module.

Pixel sizes are the one thing the old layout cannot supply: its per-slice
TIFFs carry no resolution tags, so `resolution` has to be passed explicitly
(exactly as the old package required).
"""
from __future__ import annotations

import json
import os
import re
from collections import defaultdict

import numpy as np
import tifffile


# Region prefix carried by every tile filename, in this layout
# (`R0_<fov>_<z>_<channel>.tiff`) and the new one (`R0_<fov>.ome.tiff`) alike.
# io_utils imports it from here so the assumption lives in one place.
REGION = 'R0'

# Subfolder of a legacy acquisition folder holding the per-slice TIFFs.
LEGACY_SUBDIR = '0'

# R0_<fov>_<z>_<channel>.tiff -- the only naming the old pipeline ever wrote.
# `.tif` and mixed case are accepted too; the old package matched `.tiff` only.
LEGACY_SLICE_PATTERN = re.compile(rf'^{REGION}_(\d+)_(\d+)_(.+)\.tiff?$', re.IGNORECASE)

# The name a legacy tile is addressed by -- see LegacyTileRef.
LEGACY_TILE_NAME_PATTERN = re.compile(rf'^{REGION}_(\d+)\.ome\.tiff?$', re.IGNORECASE)

# TIFF ResolutionUnit -> µm per unit. Unit 1 (NONE), which is what SQUID
# writes, means the resolution tags carry no physical scale at all.
_RESOLUTION_UNIT_UM = {2: 25400.0, 3: 10000.0}   # inch, centimetre

# (abspath, dir mtime) -> {fov: {channel: {z: path}}} or None. See legacy_index.
_INDEX_CACHE: dict = {}

# (abspath, fov) already warned about for uneven z-depth, so the message does
# not repeat for every volume and slab read out of that tile.
_UNEVEN_Z_WARNED: set = set()


class LegacyTileRef(str):
    """One old-layout tile, addressed as if it were a single OME-TIFF.

    A new-layout tile *is* a file, so the package passes tile paths around:
    `tile_path` hands one to `read_ome_tile_info`, and `corrections` takes its
    basename to name the corrected copy. An old-layout tile is a *set* of
    files instead, so it is addressed by this `str` subclass. Its string value
    is the path the tile would have as one OME-TIFF --
    `<image_dir>/R0_<fov>.ome.tiff`, which is also the name its corrected copy
    is written under -- while `.image_dir` and `.fov` carry what the loaders
    actually need. No such file exists; nothing ever opens the string value.
    """

    def __new__(cls, image_dir, fov):
        name = f'{REGION}_{int(fov)}.ome.tiff'
        self = super().__new__(cls, os.path.join(image_dir, name))
        self.image_dir = image_dir
        self.fov = int(fov)
        return self

    # str subclasses pickle by value, which would drop image_dir/fov in a
    # spawned detection worker.
    def __reduce__(self):
        return (LegacyTileRef, (self.image_dir, self.fov))

    def __repr__(self):
        return f'LegacyTileRef(image_dir={self.image_dir!r}, fov={self.fov})'


def _scan_slice_files(image_dir):
    """{fov: {channel: {z: path}}} for one per-slice TIFF folder; {} if none."""
    index: dict = defaultdict(lambda: defaultdict(dict))
    with os.scandir(image_dir) as entries:
        for entry in entries:
            m = LEGACY_SLICE_PATTERN.match(entry.name)
            if m is not None and entry.is_file():
                fov, z, channel = int(m.group(1)), int(m.group(2)), m.group(3)
                index[fov][channel][z] = entry.path
    return {fov: {ch: dict(zs) for ch, zs in channels.items()}
            for fov, channels in index.items()}


def legacy_index(image_dir):
    """Cached {fov: {channel: {z: path}}} for `image_dir`, or None when it
    holds no per-slice TIFFs (i.e. is not an old-layout image folder).

    Cached because the whole package addresses a tile through `tile_path`,
    which would otherwise re-list a folder of tens of thousands of files for
    every volume loaded and every calibration slab. Keyed on the folder's
    mtime, so a folder still being written to is re-scanned.
    """
    if not isinstance(image_dir, str) or not os.path.isdir(image_dir):
        return None
    key = (os.path.abspath(image_dir), os.stat(image_dir).st_mtime_ns)
    if key in _INDEX_CACHE:
        return _INDEX_CACHE[key]
    index = _scan_slice_files(image_dir) or None
    _INDEX_CACHE[key] = index
    return index


def is_legacy_image_dir(path):
    """True when `path` is a folder of per-(tile, z, channel) TIFFs."""
    return legacy_index(path) is not None


def legacy_image_dir(squid_folder):
    """The per-slice TIFF folder of an old-layout acquisition, or None.

    Accepts the acquisition folder itself (whose TIFFs live in `0/`) or that
    `0/` folder directly -- which is how the corrected output directory of the
    old package would be passed back in.
    """
    if not isinstance(squid_folder, str) or not os.path.isdir(squid_folder):
        return None
    candidate = os.path.join(squid_folder, LEGACY_SUBDIR)
    if is_legacy_image_dir(candidate):
        return candidate
    if is_legacy_image_dir(squid_folder):
        return squid_folder
    return None


def legacy_squid_root(image_dir):
    """Acquisition folder holding the sidecar metadata for a per-slice folder:
    its parent when it is the `0/` subfolder, otherwise itself."""
    path = os.path.abspath(image_dir)
    if os.path.basename(path) == LEGACY_SUBDIR:
        return os.path.dirname(path)
    return path


def list_legacy_tiles(image_dir):
    """[(fov, LegacyTileRef), ...] for every tile in a per-slice TIFF folder,
    fov-sorted. Empty when `image_dir` is not one."""
    index = legacy_index(image_dir)
    if not index:
        return []
    return [(fov, LegacyTileRef(image_dir, fov)) for fov in sorted(index)]


def as_legacy_ref(path):
    """`path` as a LegacyTileRef, or None when it does not name a legacy tile.

    Recognises a ref that has been flattened back to a plain string (through a
    DataFrame column, a JSON round-trip) as well as the ref itself, so a tile
    identified by name alone still resolves.
    """
    if isinstance(path, LegacyTileRef):
        return path
    # A real file is a new-layout tile, whatever its name looks like.
    if not isinstance(path, str) or os.path.exists(path):
        return None
    m = LEGACY_TILE_NAME_PATTERN.match(os.path.basename(path))
    if m is None:
        return None
    image_dir = os.path.dirname(path)
    index = legacy_index(image_dir)
    fov = int(m.group(1))
    if index is None or fov not in index:
        return None
    return LegacyTileRef(image_dir, fov)


def _channel_sort_key(channel):
    """Order a tile's channels by wavelength, so its C axis comes out in the
    same order as a new-layout tile's (405, 488, 561, 638, 730) and lines up
    with io_utils.CHANNEL_INDEX_TO_WAVELENGTH. Unparseable names go last,
    alphabetically, rather than breaking the sort."""
    parts = str(channel).split('_')
    try:
        return (0, int(parts[1]), str(channel))
    except (IndexError, ValueError):
        return (1, 0, str(channel))


def _tile_channels(ref):
    """(by_channel, channels, n_z) for one tile: its {channel: {z: path}} map,
    its channels in C order, and the z-depth every channel has."""
    index = legacy_index(ref.image_dir)
    if index is None or ref.fov not in index:
        raise FileNotFoundError(
            f'No per-slice TIFFs for FOV {ref.fov} in {ref.image_dir}')
    by_channel = index[ref.fov]
    channels = sorted(by_channel, key=_channel_sort_key)
    depths = {ch: len(by_channel[ch]) for ch in channels}
    n_z = min(depths.values())
    if len(set(depths.values())) > 1:
        warned = (os.path.abspath(ref.image_dir), ref.fov)
        if warned not in _UNEVEN_Z_WARNED:
            _UNEVEN_Z_WARNED.add(warned)
            print(f'Warning: tile {ref.fov} in {ref.image_dir} has an uneven number of '
                  f'z-slices per channel ({depths}); using the first {n_z} slice(s), '
                  f'which every channel has, so the tile has one z-depth.')
    return by_channel, channels, n_z


def _pixel_size_um(page, squid_root):
    """(x, y, z) µm/pixel for a legacy tile, None where unavailable.

    The per-slice TIFFs the old SQUID software writes carry ResolutionUnit
    NONE, so x/y are normally None and the caller has to be given an explicit
    `resolution` -- just as the old package required. They are read from the
    TIFF resolution tags when those do carry a unit, and z from `dz(um)` in
    `acquisition parameters.json` when it is there.
    """
    unit = page.tags.valueof('ResolutionUnit')
    per_unit = _RESOLUTION_UNIT_UM.get(int(unit)) if unit is not None else None

    def _axis(tag):
        value = page.tags.valueof(tag)
        if per_unit is None or not value:
            return None
        num, den = value if isinstance(value, tuple) else (value, 1)
        return (per_unit * den / num) if num else None

    z = None
    acq_path = os.path.join(squid_root, 'acquisition parameters.json')
    if os.path.exists(acq_path):
        try:
            with open(acq_path, 'r') as f:
                z = float(json.load(f)['dz(um)'])
        except (KeyError, TypeError, ValueError):
            z = None

    return (_axis('XResolution'), _axis('YResolution'), z)


def legacy_tile_metadata(ref):
    """Shape/axes/channel metadata for one old-layout tile.

    Returned as the keyword arguments `io_utils.TileInfo` takes, so an
    old-layout tile is described by the very same object as a new-layout one:
    a (Z, C, Y, X) stack whose C axis is wavelength-ordered. Only one slice
    file's header is read -- Y/X and dtype are the same for every slice of an
    acquisition.
    """
    by_channel, channels, n_z = _tile_channels(ref)

    by_z = by_channel[channels[0]]
    first = by_z[min(by_z)]
    with tifffile.TiffFile(first) as tf:
        page = tf.pages[0]
        shape, dtype = tuple(page.shape), np.dtype(page.dtype)
        physical_size_um = _pixel_size_um(page, legacy_squid_root(ref.image_dir))

    return {
        'axes': 'ZCYX',
        'shape': (n_z, len(channels), shape[-2], shape[-1]),
        'dtype': dtype,
        'channel_names_raw': channels,
        'physical_size_um': physical_size_um,
    }


def _slice_paths(image_dir, fov, channel):
    """The slice files of one tile+channel, in z order, capped to the tile's
    shared z-depth (see `_tile_channels`)."""
    ref = LegacyTileRef(image_dir, fov)
    by_channel, _, n_z = _tile_channels(ref)
    if channel not in by_channel:
        raise FileNotFoundError(
            f'No {channel} files found for FOV {fov} in {image_dir} '
            f'(available channels: {sorted(by_channel)})')
    by_z = by_channel[channel]
    return [by_z[z] for z in sorted(by_z)[:n_z]]


def _read_stack(paths):
    """(Z, Y, X) from a list of single-plane TIFFs, in the order given."""
    first = tifffile.imread(paths[0])
    first = first[0] if first.ndim == 3 else first
    vol = np.empty((len(paths),) + first.shape, dtype=first.dtype)
    vol[0] = first
    for i, path in enumerate(paths[1:], 1):
        img = tifffile.imread(path)
        vol[i] = img[0] if img.ndim == 3 else img
    return vol


def load_legacy_tile_volume(image_dir, fov, channel, n_zslices=None):
    """(Z, Y, X) for one old-layout tile+channel -- all of its z-planes, or the
    middle `n_zslices` of them. The old-layout half of
    `io_utils.load_tile_volume`."""
    paths = _slice_paths(image_dir, fov, channel)
    if n_zslices is not None and n_zslices < len(paths):
        start = (len(paths) - n_zslices) // 2
        paths = paths[start:start + n_zslices]
    return _read_stack(paths)


def load_legacy_tile_slab(image_dir, fov, channel, z0, z1, step=1):
    """z-planes [z0, z1) of one old-layout tile+channel as a (z1-z0, Y, X)
    array, every step-th plane when `step` > 1. The old-layout half of
    `io_utils.load_tile_slab`."""
    paths = _slice_paths(image_dir, fov, channel)
    z0 = max(0, int(z0))
    z1 = min(len(paths), int(z1))
    if z1 <= z0:
        raise ValueError(f'Empty z range [{z0}, {z1}) for tile {fov} of {image_dir}')
    return _read_stack(paths[z0:z1:max(1, int(step))])
