"""Streaming variant of spot_detection_new_squid.orchestrator.run_pipeline.

Used when the acquisition lives on the browser's computer and tiles are
uploaded one at a time. The steps, their order and every detection/stitching/
gene-filter call are the same as run_pipeline's (and come from the same
package); what differs is only how tiles reach the detector:

* `staging_dir` holds the sidecar metadata like a squid folder does, plus at
  any moment only the few tiles the browser has uploaded so far (see
  staging.py for the layout). New-layout tiles share one `ome_tiff/` folder;
  old-layout (per-slice TIFF) tiles each get their own `tiles/<fov>/0/` folder,
  which is what lets one appear all at once.
* Each tile is waited for when it is first needed and deleted as soon as all
  of its channels have been read into memory, so disk use stays at roughly
  (1 + prefetch) tiles regardless of acquisition size.
* Progress is reported through `status` (a StatusFile), which the web server
  reads to decide which tiles to ask the browser for next.

Not supported here (the server refuses it for uploaded data):
output_corrected=True. A single .ome.tiff is uploaded whole and run through
run_pipeline instead.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import shutil
import threading
import time

import yaml
from tqdm import tqdm

from spot_detection_new_squid.chromatic import (
    compute_chromatic_z_shift_auto,
    save_chromatic_z_shift_file,
)
from spot_detection_new_squid.corrections import (
    describe_corrections,
    make_correction_kwargs,
    uses_basic,
)
from spot_detection_new_squid.gene_filter import filter_genes
from spot_detection_new_squid.io_utils import (
    configure_single_tiff_channels,
    extract_wavelength_from_channel,
    get_resolution_nm,
    get_tile_shape,
    is_reference_channel,
    load_virtual_stitch_coords,
    make_squid_folders,
    read_ome_tile_info,
    tile_path,
)
from spot_detection_new_squid.localfdr_calibration import (
    DEFAULT_Q,
    calibrate_channel_parameters_localfdr,
)
from spot_detection_new_squid.monitoring import monitor_memory_tree
from spot_detection_new_squid.stitching import stitch_squid_tile_csvs
from spot_detection_new_squid.tasks import (
    detect_and_save,
    find_maxima_3d_tile_channel,
    load_tile_channel,
    prefetch,
)
from spot_detection_new_squid.torch_filters import gpu_available, select_device
from spot_detection_new_squid.validation import (
    validate_channel_parameters,
    validate_chromatic_z_shift,
    validate_genes_yaml,
)
from staging import (
    ACQ_PARAMS,
    LEGACY,
    copy_acq_params,
    image_dir,
    ready_path,
    remove_tile,
    wait_until_exists,
)

POLL_S = 0.5


def wait_for_tile(staging_dir, layout, fov):
    """Block until tile `fov` has fully arrived (browser upload or Globus).
    A tile is only moved out of incoming/ once every byte of every file is
    there, so its existence means it is complete."""
    path = ready_path(staging_dir, layout, fov)
    wait_until_exists(path, staging_dir, POLL_S)
    if layout == LEGACY:
        # The old-layout reader takes dz from the acquisition parameters in
        # the folder above the per-slice folder, as on a real acquisition. The
        # server normally puts a copy there before the tile appears; this
        # covers a tile that finished before the metadata was uploaded.
        copy_acq_params(staging_dir, path)
    return path


def _wait_and_detect(task):
    """CPU-pool worker: wait for the tile, then load + detect as run_pipeline does."""
    wait_for_tile(task['staging_dir'], task['layout'], task['n'])
    return find_maxima_3d_tile_channel(task)


def _calibration_folder(staging_dir, layout, fovs):
    """A squid folder the calibrator can read `fovs` from.

    New layout: the staging folder itself, whose ome_tiff/ holds them all.
    Old layout: each tile has its own folder, but the calibrator reads from
    one image folder, so the slices are hard-linked (no extra disk) into
    staging/calibration/0/, which is deleted again after calibrating.
    """
    if layout != LEGACY:
        return staging_dir
    root = os.path.join(staging_dir, 'calibration')
    shutil.rmtree(root, ignore_errors=True)
    os.makedirs(os.path.join(root, '0'))
    for fov in fovs:
        src_dir = image_dir(staging_dir, layout, fov)
        for name in os.listdir(src_dir):
            os.link(os.path.join(src_dir, name), os.path.join(root, '0', name))
    src = os.path.join(staging_dir, ACQ_PARAMS)
    if os.path.exists(src):
        shutil.copyfile(src, os.path.join(root, ACQ_PARAMS))
    return root


class _TileReleaser:
    """Deletes a staged tile once every one of its channels has been read."""

    def __init__(self, staging_dir, layout, channels_per_tile, status):
        self.staging_dir = staging_dir
        self.layout = layout
        self.remaining = dict(channels_per_tile)
        self.status = status
        self.lock = threading.Lock()

    def channel_read(self, fov):
        with self.lock:
            self.remaining[fov] -= 1
            if self.remaining[fov] > 0:
                return
        remove_tile(self.staging_dir, self.layout, fov)
        self.status.tile_released(fov)


def run_streaming_pipeline(
    *,
    staging_dir,
    tile_order,
    status,
    layout='ome',
    output_folder,
    channel_parameter_file=None,
    auto_calibrate=False,
    calibration_kwargs=None,
    reference_channels,
    single_tiff_channels_by_index=False,
    overlap_percentage=0,
    virtual_stitch_coords_path=None,
    resolution=None,
    genes_file=None,
    radius,
    depth,
    chromatic_ref_channel=None,
    chromatic_z_shift_px=None,
    footprint_size=10,
    smooth_sigma=1.0,
    prominence_footprint_size=9,
    backend='auto',
    device=None,
    z_chunk=None,
    io_workers=3,
    processes=6,
    use_basic=False,
    output_corrected=False,
    basic_get_darkfield=True,
    basic_smoothness_flatfield=None,
    basic_smoothness_darkfield=None,
    basic_max_fit_slices=150,
    rolling_ball_radius=None,
    rolling_ball_shrink=None,
    gaussian_blur_sigma=None,
    constant_offset=None,
):
    """`tile_order` is the list of fovs in the order they will be uploaded; the
    calibration tile(s), if any, come first. `layout` is 'ome' (one OME-TIFF
    per tile) or 'legacy' (per-slice TIFFs). All other parameters are
    run_pipeline's."""
    if output_corrected:
        raise ValueError('output_corrected=True is not supported for uploaded data.')
    os.makedirs(output_folder, exist_ok=True)
    memory_log = os.path.join(output_folder, 'memory_log.csv')
    squid_folder = staging_dir

    # A staging folder is never a single TIFF, so this just resets the naming
    # mode to "trust the names in the file", as run_pipeline would.
    configure_single_tiff_channels(squid_folder, single_tiff_channels_by_index)

    start_full = time.time()
    stop_and_save = monitor_memory_tree(interval=0.5, output_csv=memory_log)

    if genes_file is not None:
        if not os.path.exists(genes_file):
            raise FileNotFoundError(f'Genes file not found: {genes_file}')
        print(f'Validating genes file: {genes_file}')
        genes_data = validate_genes_yaml(genes_file)
        print(f'Genes file OK — {len(genes_data)} gene(s) defined: {", ".join(genes_data.keys())}')

    if chromatic_ref_channel is None:
        validate_chromatic_z_shift(chromatic_z_shift_px)
        if chromatic_z_shift_px is not None:
            print(f'Chromatic z-shift correction enabled (manual): {chromatic_z_shift_px}')
    else:
        if chromatic_z_shift_px is not None:
            print(f'Both chromatic_ref_channel ({chromatic_ref_channel}) and '
                  f'chromatic_z_shift_px are set. The automatically computed '
                  f'shifts will override chromatic_z_shift_px.')
        if chromatic_ref_channel in reference_channels:
            raise ValueError(
                f'chromatic_ref_channel ({chromatic_ref_channel}) is listed in '
                f'reference_channels and will not be detected. '
                f'Choose a channel that is NOT in reference_channels.')
        print(f'Automatic chromatic z-shift enabled (reference channel: {chromatic_ref_channel} nm)')

    if len(reference_channels) > 1:
        reference_channel = tuple(reference_channels)
    elif len(reference_channels) == 1:
        reference_channel = reference_channels[0]
    else:
        reference_channel = ()

    virtual_stitch_coords = None
    if virtual_stitch_coords_path is not None:
        virtual_stitch_coords = load_virtual_stitch_coords(virtual_stitch_coords_path, squid_folder)

    if not auto_calibrate and channel_parameter_file is None:
        raise ValueError('Provide channel_parameter_file, or set auto_calibrate=True to '
                         'estimate threshold/tolerance from the data.')

    # ── WAIT FOR THE FIRST TILE(S) ────────────────────────────────────────────
    # Voxel size, tile shape and the channel list are read from the first tile
    # to arrive; calibration needs all of its tiles present before it starts.
    cal_tiles = list((calibration_kwargs or {}).get('tiles') or tile_order[:1]) \
        if auto_calibrate else tile_order[:1]
    status.update(phase='waiting for first tile', waiting_for=cal_tiles)
    print(f'Waiting for tile(s) {cal_tiles} to upload...')
    for fov in cal_tiles:
        wait_for_tile(staging_dir, layout, fov)
    status.update(waiting_for=[])
    first_dir = image_dir(staging_dir, layout, cal_tiles[0])

    if resolution is None:
        resolution = get_resolution_nm(first_dir)
        if resolution is None:
            raise ValueError(
                'resolution=None was requested but the image metadata has no '
                'PhysicalSizeX/Y/Z -- an old-layout acquisition never does, its '
                'per-slice TIFFs carry no pixel size. '
                'Set resolution=[x_nm, y_nm, z_nm] explicitly.')
        print(f'Voxel size read from OME metadata: '
              f'{resolution[0]:.1f} x {resolution[1]:.1f} x {resolution[2]:.1f} nm')

    use_gpu = (backend == 'torch') or (backend == 'auto' and gpu_available())
    resolved_backend = 'torch' if use_gpu else 'scipy'
    detect_device = select_device(device) if use_gpu else None
    # output_corrected is always False here, so BaSiC fits stay on the CPU
    # (see run_pipeline for why they must not share the detector's GPU).
    basic_device = 'cpu'

    tile_shape = get_tile_shape(first_dir)
    first_info = read_ome_tile_info(tile_path(first_dir, cal_tiles[0]))
    channels = sorted(c for c in first_info.channels
                      if not is_reference_channel(c, reference_channel))
    source = 'per-slice TIFF tiles (old layout)' if layout == LEGACY else 'per-tile OME-TIFFs'
    print(f'Streaming {len(tile_order)} uploaded {source} '
          f'(tiles are {tile_shape[1]} x {tile_shape[0]} px, channels {channels})')
    make_squid_folders(output_folder, {fov: set(channels) for fov in tile_order})

    correction_kwargs = make_correction_kwargs(
        use_basic=use_basic,
        basic_device=basic_device,
        basic_get_darkfield=basic_get_darkfield,
        basic_smoothness_flatfield=basic_smoothness_flatfield,
        basic_smoothness_darkfield=basic_smoothness_darkfield,
        basic_max_fit_slices=basic_max_fit_slices,
        rolling_ball_radius=rolling_ball_radius,
        rolling_ball_shrink=rolling_ball_shrink,
        gaussian_blur_sigma=gaussian_blur_sigma,
        constant_offset=constant_offset,
    )
    if correction_kwargs is not None:
        print('\nIntensity corrections enabled (applied in this order):')
        for i, line in enumerate(describe_corrections(correction_kwargs), start=1):
            print(f'  {i}. {line}')

    # ── PER-CHANNEL DETECTION PARAMETERS ──────────────────────────────────────
    if auto_calibrate:
        status.update(phase='calibrating')
        print(f'\nLocal-FDR calibration of per-channel threshold/tolerance '
              f'(q = {(calibration_kwargs or {}).get("q", DEFAULT_Q)}) on tile(s) {cal_tiles}...')
        start_cal = time.time()
        cal_kwargs = {
            'footprint_size': footprint_size,
            'smooth_sigma': smooth_sigma,
            'prominence_footprint_size': prominence_footprint_size,
            'correction_kwargs': correction_kwargs,
            'output_path': os.path.join(output_folder, 'channel_parameters_localfdr.yaml'),
            'report_dir': output_folder,
            'reference_channels': reference_channels,
            'single_tiff_channels_by_index': single_tiff_channels_by_index,
            'genes_file': genes_file,
            'radius': radius,
            'depth': depth,
            'resolution': resolution,
        }
        cal_kwargs.update(calibration_kwargs or {})
        # Explicit, so the calibrator does not pick "the middle tile" out of
        # whichever few tiles happen to be staged right now.
        cal_kwargs['tiles'] = cal_tiles
        cal_folder = _calibration_folder(staging_dir, layout, cal_tiles)
        try:
            channel_parameters = calibrate_channel_parameters_localfdr(cal_folder, **cal_kwargs)
        finally:
            if cal_folder != staging_dir:
                shutil.rmtree(cal_folder, ignore_errors=True)
        print(f'Calibration finished in {time.time() - start_cal:.2f}s')
    else:
        print(f'Reading per channel parameters from {channel_parameter_file}')
        with open(channel_parameter_file, 'r') as file:
            channel_parameters = yaml.safe_load(file)
        print('Parameters read successfully')
    validate_channel_parameters(channel_parameters)

    # ── DETECTION, TILE BY TILE AS THEY ARRIVE ────────────────────────────────
    detect_channels = []
    for channel in channels:
        wl = extract_wavelength_from_channel(channel)
        if wl not in channel_parameters:
            print(f'No channel_parameters entry for wavelength {wl} (channel {channel}), skipping.')
            continue
        detect_channels.append(channel)
    if not detect_channels:
        raise ValueError('No channel has an entry in the channel parameters; nothing to detect.')

    detect_tasks = []
    for n in tile_order:
        for channel in detect_channels:
            wl = extract_wavelength_from_channel(channel)
            detect_tasks.append({
                'n': n, 'channel': channel, 'image_dir': image_dir(staging_dir, layout, n),
                'staging_dir': staging_dir, 'layout': layout,
                'out_csv': os.path.join(output_folder, f'tile_{n}', channel, 'maxima_results.csv'),
                'threshold': channel_parameters[wl]['threshold'],
                'tolerance': channel_parameters[wl]['tolerance'],
                'footprint_size': footprint_size, 'smooth_sigma': smooth_sigma,
                'prominence_footprint_size': prominence_footprint_size,
                'backend': resolved_backend, 'device': device, 'z_chunk': z_chunk,
                'correction_kwargs': correction_kwargs,
            })

    releaser = _TileReleaser(staging_dir, layout, {n: len(detect_channels) for n in tile_order}, status)
    status.update(phase='detecting')
    start = time.time()

    if use_gpu:
        def load(task):
            status.waiting(task['n'], True)
            wait_for_tile(staging_dir, layout, task['n'])
            status.waiting(task['n'], False)
            # Mismatched channels would otherwise surface as a confusing
            # "channel not in tile" error deep inside the loader.
            tile_channels = read_ome_tile_info(tile_path(task['image_dir'], task['n'])).channels
            if task['channel'] not in tile_channels:
                raise ValueError(f'Tile {task["n"]} has channels {tile_channels}, '
                                 f'expected {task["channel"]} like tile {cal_tiles[0]}.')
            result = load_tile_channel(task)
            releaser.channel_read(task['n'])
            return result

        print(f'Spot detection backend: GPU/torch on {detect_device} '
              f'(serial compute, {io_workers} I/O prefetch threads)')
        with tqdm(total=len(detect_tasks), desc='Detecting 3D maxima [GPU]', unit='tile-channel') as pbar:
            for task, vol in prefetch(detect_tasks, load, max_ahead=io_workers):
                n, channel, n_spots = detect_and_save(task, vol)
                status.channel_detected(n, channel, n_spots)
                pbar.set_postfix_str(f'tile {n} {channel}: {n_spots} spots')
                pbar.update()
    else:
        # On the CPU path the load happens inside the worker, so a tile is
        # released when its last channel is *detected* rather than read.
        start_method = 'spawn' if uses_basic(correction_kwargs) else None
        ctx = mp.get_context(start_method)
        print(f'Spot detection backend: CPU/scipy ({processes} worker processes'
              f'{", spawned for BaSiC fork-safety" if start_method else ""})')
        with ctx.Pool(processes=processes) as pool:
            with tqdm(total=len(detect_tasks), desc='Detecting 3D maxima [CPU]', unit='tile-channel') as pbar:
                for n, channel, n_spots in pool.imap_unordered(_wait_and_detect, detect_tasks):
                    releaser.channel_read(n)
                    status.channel_detected(n, channel, n_spots)
                    pbar.set_postfix_str(f'tile {n} {channel}: {n_spots} spots')
                    pbar.update()

    print(f'Processing finished in {time.time() - start:.2f}s')

    status.update(phase='stitching')
    start = time.time()
    print('Starting stitching...')
    stitch_squid_tile_csvs(squid_folder, output_folder, overlap_percentage, resolution,
                           virtual_stitch_coords, tile_shape=tile_shape)
    print(f'Finished stitching in {time.time() - start:.2f} seconds')

    if chromatic_ref_channel is not None:
        status.update(phase='chromatic z-shift')
        start = time.time()
        print(f'\nComputing chromatic z-shift automatically (reference: {chromatic_ref_channel} nm)...')
        chromatic_z_shift_px, z_shift_stats = compute_chromatic_z_shift_auto(
            output_folder, chromatic_ref_channel, radius, depth, resolution)
        save_chromatic_z_shift_file(
            chromatic_z_shift_px, z_shift_stats, chromatic_ref_channel, output_folder, resolution)
        print(f'Chromatic z-shift computation finished in {time.time() - start:.2f}s\n')

    if genes_file is not None:
        status.update(phase='gene filtering')
        start = time.time()
        print('Starting gene filtering...')
        filter_genes(output_folder, genes_file, radius, depth, resolution, chromatic_z_shift_px)
        print(f'Finished gene filtering in {time.time() - start:.2f} seconds')

    stop_and_save()
    print(f'Finished entire process in {time.time() - start_full:.2f} seconds')
