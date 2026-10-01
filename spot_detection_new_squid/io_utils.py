"""
Filesystem I/O and tile/channel bookkeeping for a SQUID acquisition folder in
either layout: loading z-stacks, discovering which (tile, channel) pairs exist,
and reading the optional virtual-stitch coordinate files.

New vs. old layout
------------------
Both layouts are read. The old SQUID layout -- the one
`spot_detection_functionalized_with_basic` reads -- wrote one TIFF per
(tile, z-slice, channel) into a `0/` folder::

    <squid_folder>/0/R0_<fov>_<z>_Fluorescence_488_nm_Ex.tiff
    ...

The new layout writes one multi-page OME-TIFF per tile, with every z-slice and
channel stacked inside it::

    <squid_folder>/ome_tiff/R0_<fov>.ome.tiff     (axes ZCYX, e.g. 145x5x2720x2720)
    <squid_folder>/0/coordinates.csv              (per-fov, per-z stage positions)
    <squid_folder>/coordinates.csv                (one row per fov)
    <squid_folder>/acquisition parameters.json    (Nx, Ny, Nz, dz ...)
    <squid_folder>/acquisition.yaml               (objective, channels, z-stack ...)
    <squid_folder>/acquisition_channels.yaml      (channel table)

Channel names live in the OME-XML (``Fluorescence 405 nm Ex``, with spaces).
They are normalised to the underscore form used everywhere else in the
pipeline (``Fluorescence_405_nm_Ex``) so output folder names, the gene-filter
regex and `extract_wavelength_from_channel` all keep working unchanged.

Old (per-slice) folders
-----------------------
Anywhere this package takes a `squid_folder`, it also takes an old-layout
acquisition folder, or its `0/` subfolder directly. The per-slice files are
read by legacy_io.py, which presents each tile as if it were one (Z, C, Y, X)
OME-TIFF -- wavelength-ordered C axis, one `TileInfo` just like a real tile's
-- so every function below behaves the same on either layout and nothing
downstream (detection, calibration, PSF estimation, corrections, stitching)
has to know which it got. The one thing the old layout cannot supply is the
voxel size: its TIFFs carry no resolution tags, so `resolution` has to be
passed explicitly, as the old package also required. With
output_corrected=True the corrected copies come out in the *new* layout, one
`R0_<fov>.ome.tiff` per tile.

Single-file input
-----------------
Anywhere this package takes a `squid_folder`, it also takes the path to a
single OME-TIFF -- the one-tile case, e.g. a lone acquisition or one tile
pulled out of a bigger one::

    run_pipeline(squid_folder='/data/R0_3.ome.tiff', ...)

The file is then treated as a one-tile acquisition: `get_image_dir` returns
the file itself and `list_ome_tiles` reports exactly that tile, so every
consumer downstream (detection, calibration, BaSiC, stitching) works
unchanged. Its FOV number is parsed from the filename when it looks like
``R0_<fov>.ome.tiff`` and is 0 otherwise. Sidecar metadata
(`coordinates.csv`, `acquisition parameters.json`, the YAMLs) is looked up
next to the file and, if the file sits in an `ome_tiff/` folder, one level up
-- see `squid_root`. Stitching a single tile needs no stage position, so
missing coordinates are not an error there.

Because a lone file often carries no usable channel names -- and because the
wavelengths in genes.yaml / reference_channels have to land on the right
channels regardless -- its channels are named by C index from the hardcoded
`CHANNEL_INDEX_TO_WAVELENGTH` map (C 0 -> 405, C 1 -> 488, ...) rather than
from its own metadata. See `configure_single_tiff_channels`, which the entry
points call for you; pass single_tiff_channels_by_index=False to opt out.
"""
from __future__ import annotations

import csv
import json
import os
import re
import xml.etree.ElementTree as ET
from collections import defaultdict

import numpy as np
import pandas as pd
import tifffile
import yaml

from .legacy_io import (
    LEGACY_SLICE_PATTERN,
    LEGACY_SUBDIR,
    REGION,
    LegacyTileRef,
    as_legacy_ref,
    is_legacy_image_dir,
    legacy_image_dir,
    legacy_squid_root,
    legacy_tile_metadata,
    list_legacy_tiles,
    load_legacy_tile_slab,
    load_legacy_tile_volume,
)


# Columns of every per-tile maxima_results.csv: the median, largest and
# smallest of the N brightest voxels around the spot. See tasks.detect_and_save.
MAXIMA_COLUMNS = ['X', 'Y', 'Slice', 'Median', 'Max', 'Min', 'N']

# ── HARDCODED CHANNEL INDEX <-> WAVELENGTH MAP ────────────────────────────────
# Used as a last-resort fallback when a tile's OME-XML carries no channel names
# and acquisition_channels.yaml is missing, by chromatic.py to write channel
# indices into the z-shift JSON, and -- for a single-OME-TIFF input -- to name
# the channels outright (see `configure_single_tiff_channels`).
CHANNEL_INDEX_TO_WAVELENGTH = {
    0: 'Fluorescence_405_nm_Ex',
    1: 'Fluorescence_488_nm_Ex',
    2: 'Fluorescence_561_nm_Ex',
    3: 'Fluorescence_638_nm_Ex',
    4: 'Fluorescence_730_nm_Ex',
}
# Inverse: integer wavelength -> channel index (e.g. 488 -> 1)
WL_INT_TO_CH_IDX = {int(name.split('_')[1]): idx for idx, name in CHANNEL_INDEX_TO_WAVELENGTH.items()}
# ───────────────────────────────────────────────────────────────────────────

# ── INDEXED CHANNEL NAMING (single-OME-TIFF input) ────────────────────────────
# When on, `read_ome_tile_info` labels a tile's channels from
# CHANNEL_INDEX_TO_WAVELENGTH by C index instead of from the file's own
# metadata, so the wavelengths in genes.yaml / reference_channels line up with
# the OME-TIFF's channel order. Kept in the environment rather than in a plain
# module global so that detection workers inherit it whether the pool is forked
# or spawned (a spawned child re-imports this module from scratch).
_INDEXED_CHANNELS_ENV = 'SPOT_DETECTION_INDEXED_CHANNEL_NAMES'


def indexed_channel_names_enabled():
    """True when channels are being named by C index off the hardcoded map."""
    return os.environ.get(_INDEXED_CHANNELS_ENV) == '1'


def use_indexed_channel_names(enabled=True):
    """Turn indexed channel naming on/off for this process and its children."""
    os.environ[_INDEXED_CHANNELS_ENV] = '1' if enabled else '0'


def indexed_channel_names(n_channels):
    """The first `n_channels` names of CHANNEL_INDEX_TO_WAVELENGTH, in C order."""
    missing = [i for i in range(n_channels) if i not in CHANNEL_INDEX_TO_WAVELENGTH]
    if missing:
        raise ValueError(
            f'Channel index/indices {missing} are not in CHANNEL_INDEX_TO_WAVELENGTH, '
            f'so a {n_channels}-channel OME-TIFF cannot be named by index. '
            f'Update the hardcoded map at the top of io_utils.py.'
        )
    return [CHANNEL_INDEX_TO_WAVELENGTH[i] for i in range(n_channels)]


def configure_single_tiff_channels(squid_folder, by_index=True):
    """Correlate a single OME-TIFF's channels to wavelengths by C index.

    A lone OME-TIFF -- one tile pulled out of an acquisition, a crop exported
    from another tool -- often carries no usable channel names, or names that
    do not match the wavelengths written in genes.yaml and reference_channels.
    Its channel *order* is known, though, so when `squid_folder` is a single
    file its channels are named from CHANNEL_INDEX_TO_WAVELENGTH by index:
    C 0 -> 405, C 1 -> 488, and so on.

    Pass by_index=False to keep using the names in the file instead. A no-op
    for a normal acquisition folder, whose per-tile OME-XML names are trusted
    either way. Returns True when indexed naming was turned on.
    """
    enabled = bool(by_index) and is_single_tiff(squid_folder)
    use_indexed_channel_names(enabled)
    if not enabled:
        return False
    info = read_ome_tile_info(squid_folder)
    print(f'Single OME-TIFF input: naming its {info.n_channels} channel(s) by index '
          f'from the hardcoded CHANNEL_INDEX_TO_WAVELENGTH map --')
    for i, name in enumerate(info.channels):
        print(f'  channel index {i} -> {extract_wavelength_from_channel(name)} nm ({name})')
    print('  Set single_tiff_channels_by_index=False to use the names in the file instead.')
    return True
# ───────────────────────────────────────────────────────────────────────────

# Subfolder of the squid folder holding the per-tile OME-TIFFs.
OME_TIFF_SUBDIR = 'ome_tiff'
# `REGION` (the R0 prefix both layouts put on every tile filename) is defined
# in legacy_io and imported above, so the two layouts share the one assumption.
OME_TILE_PATTERN = re.compile(rf'^{REGION}_(\d+)\.ome\.tiff?$', re.IGNORECASE)
# A single-file input may be named anything; only its extension is checked.
# The fov is taken from the name when it carries one (`<anything>_<fov>.ome.tiff`).
TIFF_EXTENSIONS = ('.tif', '.tiff')
SINGLE_TILE_FOV_PATTERN = re.compile(r'_(\d+)(?:\.ome)?\.tiff?$', re.IGNORECASE)

_OME_NS = {'ome': 'http://www.openmicroscopy.org/Schemas/OME/2016-06'}

# path -> TileInfo, keyed by (path, mtime, size) so a rewritten file is re-read.
_TILE_INFO_CACHE: dict = {}


def normalize_channel_name(name):
    """'Fluorescence 488 nm Ex' -> 'Fluorescence_488_nm_Ex'."""
    return re.sub(r'\s+', '_', str(name).strip())


def is_single_tiff(path):
    """True when `path` points at one TIFF file rather than an acquisition folder."""
    return os.path.isfile(path) and path.lower().endswith(TIFF_EXTENSIONS)


def fov_from_path(path):
    """FOV number encoded in a tile filename, or 0 when it carries none."""
    m = SINGLE_TILE_FOV_PATTERN.search(os.path.basename(path))
    return int(m.group(1)) if m else 0


def squid_root(path):
    """Folder to look for acquisition sidecar metadata in.

    Takes a squid folder, an `ome_tiff/` folder, an old layout's per-slice `0/`
    folder or a single OME-TIFF, and returns the acquisition folder that would
    hold `coordinates.csv`, `acquisition parameters.json` and the YAMLs: the
    file's/folder's parent when it sits in one of those image subfolders,
    otherwise its own directory.
    """
    path = os.path.abspath(path)
    if os.path.isfile(path):
        path = os.path.dirname(path)
    if os.path.basename(path).lower() == OME_TIFF_SUBDIR:
        return os.path.dirname(path)
    # Only when it really holds per-slice TIFFs, so a squid folder that merely
    # happens to be named `0` is not mistaken for an image subfolder.
    if is_legacy_image_dir(path):
        return legacy_squid_root(path)
    return path


def default_output_folder(squid_folder):
    """Where results go when the caller passes output_folder=None.

    The squid folder itself, or -- for a single-file input, where writing
    `tile_<n>/` folders next to the image would litter someone else's
    directory -- a sibling folder named after the file.
    """
    if is_single_tiff(squid_folder):
        stem = re.sub(r'(\.ome)?\.tiff?$', '', os.path.basename(squid_folder), flags=re.IGNORECASE)
        return os.path.join(os.path.dirname(os.path.abspath(squid_folder)), f'{stem}_spots')
    return squid_folder


class TileInfo:
    """Cached shape/axes/channel metadata for one per-tile OME-TIFF."""

    def __init__(self, path, axes, shape, dtype, channel_names_raw,
                 physical_size_um):
        self.path = path
        self.axes = axes                      # e.g. 'ZCYX'
        self.shape = tuple(shape)
        self.dtype = np.dtype(dtype)
        self.channel_names_raw = list(channel_names_raw)      # as written in the file
        self.channels = [normalize_channel_name(c) for c in channel_names_raw]
        self.physical_size_um = physical_size_um              # (x, y, z) or None entries

    @property
    def height(self):
        return self.shape[self.axes.index('Y')]

    @property
    def width(self):
        return self.shape[self.axes.index('X')]

    @property
    def n_z(self):
        return self.shape[self.axes.index('Z')] if 'Z' in self.axes else 1

    @property
    def n_channels(self):
        return self.shape[self.axes.index('C')] if 'C' in self.axes else 1

    @property
    def source(self):
        """Where this tile's pixels come from, for error messages: the
        OME-TIFF's path, or -- since an old-layout tile has no single file --
        the per-slice folder and fov behind its `LegacyTileRef`."""
        ref = as_legacy_ref(self.path)
        if ref is not None:
            return f'FOV {ref.fov} of the per-slice TIFFs in {ref.image_dir}'
        return str(self.path)

    def channel_index(self, channel):
        """Index of `channel` (underscore form) within this tile's C axis."""
        wanted = normalize_channel_name(channel)
        for i, name in enumerate(self.channels):
            if name == wanted:
                return i
        raise KeyError(
            f'Channel {wanted!r} not present in {self.source}. '
            f'Available channels: {self.channels}'
        )


def _channel_names_from_ome_xml(xml_text):
    """Channel Name attributes from an OME-XML string, in C order."""
    if not xml_text:
        return []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return []
    names = []
    # Namespaced first, then bare, so both flavours of OME-XML parse.
    channels = root.findall('.//ome:Image/ome:Pixels/ome:Channel', _OME_NS)
    if not channels:
        channels = [el for el in root.iter() if el.tag.split('}')[-1] == 'Channel']
    for el in channels:
        name = el.get('Name')
        if name:
            names.append(name)
    return names


def _physical_size_from_ome_xml(xml_text):
    """(x, y, z) pixel size in micrometres from OME-XML, None where absent."""
    default = (None, None, None)
    if not xml_text:
        return default
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return default
    pixels = root.find('.//ome:Image/ome:Pixels', _OME_NS)
    if pixels is None:
        pixels = next((el for el in root.iter() if el.tag.split('}')[-1] == 'Pixels'), None)
    if pixels is None:
        return default

    def _size(axis):
        val = pixels.get(f'PhysicalSize{axis}')
        if val is None:
            return None
        unit = pixels.get(f'PhysicalSize{axis}Unit', 'µm')
        scale = {'nm': 1e-3, 'mm': 1e3, 'm': 1e6}.get(unit, 1.0)  # everything else: µm
        return float(val) * scale

    return (_size('X'), _size('Y'), _size('Z'))


def _channel_names_from_acquisition_yaml(squid_folder, n_channels):
    """Fallback channel names from acquisition_channels.yaml / acquisition.yaml."""
    for fname in ('acquisition_channels.yaml', 'acquisition.yaml'):
        path = os.path.join(squid_folder, fname)
        if not os.path.exists(path):
            continue
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        channels = (data or {}).get('channels')
        if not channels:
            continue
        names = [c['name'] for c in channels if c.get('enabled', True) and 'name' in c]
        if len(names) == n_channels:
            return names
    return []


def read_ome_tile_info(path):
    """Read (and cache) axes/shape/channel metadata for one tile.

    Takes a per-tile OME-TIFF's path or -- for an old layout's per-slice folder
    -- that tile's `LegacyTileRef`, and returns the same `TileInfo` either way:
    an old-layout tile is described as the (Z, C, Y, X) stack its files add up
    to, with a wavelength-ordered C axis (see legacy_io.py).
    """
    ref = as_legacy_ref(path)
    if ref is not None:
        # Keyed on the folder's mtime, like the OME branch is on the file's, so
        # a folder still being written to is re-read.
        key = ('legacy', os.path.abspath(ref.image_dir), ref.fov,
               os.stat(ref.image_dir).st_mtime_ns)
        cached = _TILE_INFO_CACHE.get(key)
        if cached is not None:
            return cached
        info = TileInfo(ref, **legacy_tile_metadata(ref))
        _TILE_INFO_CACHE[key] = info
        return info

    stat = os.stat(path)
    # The naming mode is part of the key: flipping it changes the channel names
    # this returns, so a cached entry from the other mode must not be reused.
    key = (os.path.abspath(path), stat.st_mtime_ns, stat.st_size,
           indexed_channel_names_enabled())
    cached = _TILE_INFO_CACHE.get(key)
    if cached is not None:
        return cached

    with tifffile.TiffFile(path) as tf:
        series = tf.series[0]
        axes, shape, dtype = series.axes, series.shape, series.dtype
        xml_text = tf.ome_metadata

    if 'Y' not in axes or 'X' not in axes:
        raise ValueError(
            f'{path} has axes {axes!r}, which have no Y/X image plane. '
            f'This package expects a (Z, C, Y, X)-style OME-TIFF per tile.'
        )

    names = _channel_names_from_ome_xml(xml_text)
    n_channels = shape[axes.index('C')] if 'C' in axes else 1
    if indexed_channel_names_enabled():
        # Single-file input: the file's own names are not trusted, its channel
        # order is. Say so when the two actually disagree, since that is the
        # case where the wavelengths downstream change meaning.
        in_file = [normalize_channel_name(n) for n in names] if len(names) == n_channels else []
        names = indexed_channel_names(n_channels)
        if in_file and in_file != names:
            print(f'Note: {path} names its channels {in_file}; using the '
                  f'hardcoded index map instead -> {names}.')
    else:
        if len(names) != n_channels:
            names = _channel_names_from_acquisition_yaml(squid_root(path), n_channels)
        if len(names) != n_channels:
            names = [CHANNEL_INDEX_TO_WAVELENGTH.get(i, f'channel_{i}') for i in range(n_channels)]
            print(f'Warning: no channel names found in {path} or acquisition YAML; '
                  f'falling back to the hardcoded CHANNEL_INDEX_TO_WAVELENGTH map.')

    info = TileInfo(path, axes, shape, dtype, names, _physical_size_from_ome_xml(xml_text))
    _TILE_INFO_CACHE[key] = info
    return info


def get_image_dir(squid_folder):
    """Where the images are: a directory of tiles, or one OME-TIFF.

    Accepts the squid folder itself (looks for its `ome_tiff/` subfolder, then
    for an old layout's per-slice `0/` subfolder), a directory that already
    contains the tiles in either form -- which is how the corrected output
    directory is passed back in -- or the path to a single OME-TIFF, which is
    returned unchanged and handled as a one-tile acquisition by
    `list_ome_tiles` and everything built on it.
    """
    # Checked first so a typo'd path says so. Without this, a non-existent file
    # fails os.path.isfile, falls through to the directory branch, and reports
    # "no ome_tiff folder found under <path>" about what is plainly a filename.
    if not os.path.exists(squid_folder):
        raise FileNotFoundError(
            f'{squid_folder} does not exist. Pass a squid acquisition folder, a '
            f'directory of per-tile OME-TIFFs, or the path to a single .ome.tiff.'
        )
    if os.path.isfile(squid_folder):
        if not is_single_tiff(squid_folder):
            raise ValueError(
                f'{squid_folder} is a file but not a TIFF. Pass a squid acquisition '
                f'folder, or a single .ome.tiff to process just that tile.'
            )
        return squid_folder
    candidate = os.path.join(squid_folder, OME_TIFF_SUBDIR)
    if os.path.isdir(candidate):
        return candidate
    if os.path.isdir(squid_folder) and list_ome_tiles(squid_folder):
        return squid_folder
    # New layout first, so an acquisition that has both an `ome_tiff/` folder
    # and a `0/` folder of per-slice TIFFs is read the new way, as before.
    legacy = legacy_image_dir(squid_folder)
    if legacy is not None:
        return legacy
    raise FileNotFoundError(
        f'No images found under {squid_folder}: neither an {OME_TIFF_SUBDIR!r} folder '
        f'with per-tile OME-TIFFs (the new SQUID layout, one <region>_<fov>.ome.tiff '
        f'per tile), nor a {LEGACY_SUBDIR!r} folder of per-slice '
        f'<region>_<fov>_<z>_<channel>.tiff files (the old layout). '
        f'Pass one of those, or the path to a single .ome.tiff.'
    )


def list_ome_tiles(image_dir):
    """[(fov, path), ...] for every tile in `image_dir`, fov-sorted.

    When `image_dir` is a single OME-TIFF (see `get_image_dir`) the result is
    that one tile, with the fov taken from its filename. When it is an old
    layout's per-slice folder, each tile's many files are reported as one
    `LegacyTileRef` standing in for the path a real OME-TIFF would have (see
    legacy_io.py); everything below takes either.
    """
    if os.path.isfile(image_dir):
        return [(fov_from_path(image_dir), image_dir)]
    # The old layout is checked first because its answer is cached, while the
    # OME scan below re-lists the folder on every call: a per-slice folder can
    # hold tens of thousands of files, and `tile_path` runs for every volume
    # loaded and every calibration slab. The two naming patterns are disjoint,
    # so this cannot shadow a new-layout folder.
    legacy_tiles = list_legacy_tiles(image_dir)
    if legacy_tiles:
        return legacy_tiles
    tiles = []
    for fname in os.listdir(image_dir):
        m = OME_TILE_PATTERN.match(fname)
        if m:
            tiles.append((int(m.group(1)), os.path.join(image_dir, fname)))
    return sorted(tiles)


def tile_path(image_dir, fov):
    """Path to one tile's OME-TIFF, whatever `.ome.tif`/`.ome.tiff` spelling it
    uses -- or, for an old layout's per-slice folder, that tile's
    `LegacyTileRef`."""
    for tile_fov, path in list_ome_tiles(image_dir):
        if tile_fov == fov:
            return path
    kind = 'per-slice TIFF set' if is_legacy_image_dir(image_dir) else 'OME-TIFF'
    raise FileNotFoundError(f'No {kind} for FOV {fov} in {image_dir}')


def is_tile_image_file(image_dir, fname):
    """True when `fname` inside `image_dir` is one of that layout's tile images:
    a per-tile OME-TIFF, or one per-slice TIFF of the old layout.

    Used when recreating an image directory (see
    `corrections.write_corrected_image_dir`) to tell the images, which are
    rewritten tile by tile, from sidecar files, which are copied through.
    """
    if OME_TILE_PATTERN.match(fname):
        return True
    return (is_legacy_image_dir(image_dir)
            and LEGACY_SLICE_PATTERN.match(fname) is not None)


def _plane_index(axes, shape, position):
    """Flat page index of one 2-D plane within a series.

    The series' pages are laid out in C order over its leading (non-YX) axes,
    so this works for ZCYX, CZYX, TCZYX ... without assuming an order.
    """
    idx = 0
    for ax, n in zip(axes[:-2], shape[:len(axes) - 2]):
        idx = idx * n + position.get(ax, 0)
    return idx


def load_tile_volume(image_dir, fov, channel, n_zslices=None):
    """
    Load all (or the middle n_zslices) z-planes of one tile+channel out of that
    tile's OME-TIFF into a (Z, Y, X) array.

    Only the requested channel's pages are read, so peak memory stays at one
    channel's stack rather than the whole ZCYX file. For an old layout's
    per-slice folder the channel's individual TIFFs are read instead, which
    costs the same memory.
    """
    path = tile_path(image_dir, fov)
    if isinstance(path, LegacyTileRef):
        return load_legacy_tile_volume(path.image_dir, path.fov, channel, n_zslices)
    info = read_ome_tile_info(path)
    c_idx = info.channel_index(channel)

    z_sorted = list(range(info.n_z))
    if n_zslices is not None and n_zslices < len(z_sorted):
        start = (len(z_sorted) - n_zslices) // 2
        z_sorted = z_sorted[start:start + n_zslices]

    vol = np.empty((len(z_sorted), info.height, info.width), dtype=info.dtype)
    with tifffile.TiffFile(path) as tf:
        pages = tf.series[0].pages
        for i, z in enumerate(z_sorted):
            page = pages[_plane_index(info.axes, info.shape, {'Z': z, 'C': c_idx, 'T': 0})]
            img = page.asarray()
            vol[i] = img[0] if img.ndim == 3 else img

    return vol


def load_tile_slab(image_dir, fov, channel, z0, z1, step=1):
    """Load z-planes [z0, z1) of one tile+channel as a (z1-z0, Y, X) array.

    Used by the calibration sampler, which cuts several xy chunks out of one
    z-slab so a slab's planes are read from disk only once. `step` > 1 reads
    every step-th plane instead, for a cheap preview of the slab.
    """
    path = tile_path(image_dir, fov)
    if isinstance(path, LegacyTileRef):
        return load_legacy_tile_slab(path.image_dir, path.fov, channel, z0, z1, step)
    info = read_ome_tile_info(path)
    c_idx = info.channel_index(channel)
    z0 = max(0, int(z0))
    z1 = min(info.n_z, int(z1))
    if z1 <= z0:
        raise ValueError(f'Empty z range [{z0}, {z1}) for tile {fov} of {path}')

    z_list = list(range(z0, z1, max(1, int(step))))
    slab = np.empty((len(z_list), info.height, info.width), dtype=info.dtype)
    with tifffile.TiffFile(path) as tf:
        pages = tf.series[0].pages
        for i, z in enumerate(z_list):
            page = pages[_plane_index(info.axes, info.shape, {'Z': z, 'C': c_idx, 'T': 0})]
            img = page.asarray()
            slab[i] = img[0] if img.ndim == 3 else img
    return slab


def get_tile_shape(squid_folder):
    """(height, width) of a tile, read from the first tile found."""
    image_dir = get_image_dir(squid_folder)
    tiles = list_ome_tiles(image_dir)
    if not tiles:
        raise FileNotFoundError(f'No tiles found in {image_dir}')
    info = read_ome_tile_info(tiles[0][1])
    return info.height, info.width


def get_resolution_nm(squid_folder):
    """[x, y, z] voxel size in nm from the OME-XML, or None if unavailable.

    None for an old-layout acquisition: its per-slice TIFFs are written with no
    physical pixel size, so `resolution` has to be given explicitly there.
    """
    image_dir = get_image_dir(squid_folder)
    tiles = list_ome_tiles(image_dir)
    if not tiles:
        return None
    px = read_ome_tile_info(tiles[0][1]).physical_size_um
    if any(p is None for p in px):
        return None
    return [p * 1000.0 for p in px]


def is_reference_channel(channel, reference_channel):
    if hasattr(reference_channel, '__iter__') and not isinstance(reference_channel, str):
        return any(str(r) in channel for r in reference_channel)
    return str(reference_channel) in channel


def extract_wavelength_from_channel(channel):
    """e.g. 'Fluorescence_561_nm_Ex' -> 561"""
    return int(channel.split('_')[1])


def get_squid_tile_channels(input_dir, reference_channel):
    """
    Find every (tile, channel) pair present in the squid folder's `ome_tiff`
    folder -- or, for an old-layout acquisition, its `0` folder of per-slice
    TIFFs -- skipping reference channels. Returns a dict: tile -> set of
    channels.
    """
    image_dir = get_image_dir(input_dir)
    tiles_by_channel = defaultdict(set)
    for fov, path in list_ome_tiles(image_dir):
        for channel in read_ome_tile_info(path).channels:
            if not is_reference_channel(channel, reference_channel):
                tiles_by_channel[fov].add(channel)
    return tiles_by_channel


def make_squid_folders(output_dir, tiles_by_channel):
    for n, channels in sorted(tiles_by_channel.items()):
        tile_dir = os.path.join(output_dir, f'tile_{n}')
        os.makedirs(tile_dir, exist_ok=True)
        for channel in sorted(channels):
            channel_dir = os.path.join(tile_dir, channel)
            os.makedirs(channel_dir, exist_ok=True)
            max_csv = os.path.join(channel_dir, 'maxima_results.csv')
            with open(max_csv, 'w') as f:
                # Leading '' is the unnamed index column pandas writes.
                csv.writer(f).writerow([''] + MAXIMA_COLUMNS)


def load_tile_stage_coords(squid_folder):
    """
    Per-tile stage positions as a DataFrame with columns ``fov``, ``x (mm)``,
    ``y (mm)`` -- one row per tile, sorted by fov.

    Prefers `0/coordinates.csv`, which in the new layout labels every row with
    its ``fov`` (one row per z-level, so the first row of each fov is taken).
    Falls back to the top-level `coordinates.csv`, whose rows are in fov order
    but carry no fov column.

    For a single-OME-TIFF input the coordinate files are looked up next to it
    (see `squid_root`) and only that tile's row is returned. A lone tile is
    its own origin, so if no coordinate file is there the position falls back
    to (0, 0) instead of raising -- stitching one tile never uses it.

    `squid_folder` may equally be an image subfolder (`ome_tiff/`, or an old
    layout's `0/`); `squid_root` finds the acquisition folder above it, so the
    per-slice folder's own `coordinates.csv` is never mistaken for the
    positional top-level one -- they have different columns and one row per
    z-level rather than per tile.
    """
    single_fov = fov_from_path(squid_folder) if is_single_tiff(squid_folder) else None
    root = squid_root(squid_folder)

    def _select(out):
        """Keep just the requested fov's row, for a single-tile input."""
        if single_fov is None:
            return out
        row = out[out['fov'] == single_fov]
        if row.empty:
            return None
        return row.reset_index(drop=True)

    per_fov_path = os.path.join(root, '0', 'coordinates.csv')
    if os.path.exists(per_fov_path):
        df = pd.read_csv(per_fov_path)
        if {'fov', 'x (mm)', 'y (mm)'}.issubset(df.columns):
            out = (df.groupby('fov', as_index=False)
                     .first()[['fov', 'x (mm)', 'y (mm)']]
                     .sort_values('fov')
                     .reset_index(drop=True))
            out = _select(out)
            if out is not None:
                return out

    top_path = os.path.join(root, 'coordinates.csv')
    if os.path.exists(top_path):
        df = pd.read_csv(top_path)
        if not {'x (mm)', 'y (mm)'}.issubset(df.columns):
            raise ValueError(f'{top_path} must have "x (mm)" and "y (mm)" columns')
        out = df[['x (mm)', 'y (mm)']].copy()
        out.insert(0, 'fov', range(len(out)))
        out = _select(out)
        if out is not None:
            return out

    if single_fov is not None:
        print(f'No stage coordinates for FOV {single_fov} under {root}; '
              f'placing the single tile at (0, 0) mm.')
        return pd.DataFrame({'fov': [single_fov], 'x (mm)': [0.0], 'y (mm)': [0.0]})

    raise FileNotFoundError(
        f'No stage coordinates found: neither {per_fov_path} (with a "fov" column) '
        f'nor {top_path} exists.'
    )


def load_virtual_stitch_coords_json(json_path, squid_folder):
    root = squid_root(squid_folder)
    acq_path = os.path.join(root, 'acquisition parameters.json')
    if not os.path.exists(acq_path):
        raise FileNotFoundError(
            f'acquisition parameters.json not found in {root}. '
            'This file is required to validate the virtual stitch coords grid.'
        )
    with open(acq_path, 'r') as f:
        acq = json.load(f)
    try:
        nx = int(acq['Nx'])
        ny = int(acq['Ny'])
    except KeyError as e:
        raise ValueError(f'acquisition parameters.json is missing expected key: {e}')

    with open(json_path, 'r') as f:
        data = json.load(f)

    if 'tiles' not in data:
        raise ValueError('Virtual stitch coords JSON must have a top-level "tiles" list')

    spatial: dict = {}
    for tile in data['tiles']:
        gi = tile['grid_index']
        pos_key = (gi['tx'], gi['ty'], gi['tz'])
        ch_idx = gi['channel']
        co = tile['crop_offset']
        if pos_key not in spatial:
            spatial[pos_key] = {'tx': gi['tx'], 'ty': gi['ty'], 'tz': gi['tz'], 'channels': {}}
        spatial[pos_key]['channels'][ch_idx] = [co['x'], co['y'], co['z']]

    n_spatial = len(spatial)
    expected = nx * ny
    if n_spatial != expected:
        raise ValueError(
            f'Virtual stitch coords JSON contains {n_spatial} unique spatial tile(s) '
            f'but acquisition parameters.json specifies Nx={nx} x Ny={ny} = {expected} tiles.'
        )

    ordered = sorted(spatial.values(), key=lambda t: (t['ty'], t['tx']))

    coords = []
    for t in ordered:
        ch_map = {}
        for ch_idx, offset in t['channels'].items():
            wl = CHANNEL_INDEX_TO_WAVELENGTH.get(ch_idx)
            if wl is None:
                raise ValueError(
                    f'Channel index {ch_idx} at tile ({t["tx"]},{t["ty"]},{t["tz"]}) '
                    f'found in JSON but not in CHANNEL_INDEX_TO_WAVELENGTH map. '
                    f'Update the hardcoded map at the top of io_utils.py.'
                )
            ch_map[wl] = offset
        coords.append(ch_map)

    print(f'Loaded {len(coords)} virtual stitch tile position(s) from {json_path} '
          f'(grid: {nx} x {ny})')
    for i, (t, ch_map) in enumerate(zip(ordered, coords)):
        print(f'  Tile {i:>3d}  grid({t["tx"]},{t["ty"]},{t["tz"]})')
        for wl, off in sorted(ch_map.items()):
            print(f'    wavelength {wl}: crop_offset x={off[0]} y={off[1]} z={off[2]}')
    return coords


def load_virtual_stitch_coords_csv(csv_path, squid_folder):
    root = squid_root(squid_folder)
    acq_path = os.path.join(root, 'acquisition parameters.json')
    if not os.path.exists(acq_path):
        raise FileNotFoundError(
            f'acquisition parameters.json not found in {root}. '
            'This file is required to validate the virtual stitch coords grid.'
        )
    with open(acq_path, 'r') as f:
        acq = json.load(f)
    try:
        nx = int(acq['Nx'])
        ny = int(acq['Ny'])
    except KeyError as e:
        raise ValueError(f'acquisition parameters.json is missing expected key: {e}')

    data = pd.read_csv(csv_path)
    if not all(col in data.columns for col in ['x', 'y', 'z', 'c']):
        raise ValueError('Virtual stitch coords CSV must have columns: x, y, z, c')

    spatial: dict = {}
    for i, tile in data.iterrows():
        pos_key = (i % ny, i // ny, 0)
        ch_idx = tile['c']
        if pos_key not in spatial:
            spatial[pos_key] = {'tx': i % ny, 'ty': i // ny, 'tz': 0, 'channels': {}}
        spatial[pos_key]['channels'][ch_idx] = [tile['x'], tile['y'], tile['z']]

    n_spatial = len(spatial)
    expected = nx * ny
    if n_spatial != expected:
        raise ValueError(
            f'Virtual stitch coords JSON contains {n_spatial} unique spatial tile(s) '
            f'but acquisition parameters.json specifies Nx={nx} x Ny={ny} = {expected} tiles.'
        )

    ordered = sorted(spatial.values(), key=lambda t: (t['ty'], t['tx']))

    coords = []
    for t in ordered:
        ch_map = {}
        for ch_idx, offset in t['channels'].items():
            wl = CHANNEL_INDEX_TO_WAVELENGTH.get(ch_idx)
            if wl is None:
                raise ValueError(
                    f'Channel index {ch_idx} at tile ({t["tx"]},{t["ty"]},{t["tz"]}) '
                    f'found in JSON but not in CHANNEL_INDEX_TO_WAVELENGTH map. '
                    f'Update the hardcoded map at the top of io_utils.py.'
                )
            ch_map[wl] = offset
        coords.append(ch_map)

    print(f'Loaded {len(coords)} virtual stitch tile position(s) from {csv_path} '
          f'(grid: {nx} x {ny})')
    for i, (t, ch_map) in enumerate(zip(ordered, coords)):
        print(f'  Tile {i:>3d}  grid({t["tx"]},{t["ty"]},{t["tz"]})')
        for wl, off in sorted(ch_map.items()):
            print(f'    wavelength {wl}: crop_offset x={off[0]} y={off[1]} z={off[2]}')
    return coords


def load_virtual_stitch_coords(path, squid_folder):
    """Dispatch to the JSON or CSV loader based on file extension."""
    ext = os.path.splitext(path)[1]
    if ext == '.json':
        return load_virtual_stitch_coords_json(path, squid_folder)
    if ext == '.csv':
        return load_virtual_stitch_coords_csv(path, squid_folder)
    raise ValueError(
        f'File for virtual stitching coords: {path} is not of valid types ".json" or ".csv"'
    )
