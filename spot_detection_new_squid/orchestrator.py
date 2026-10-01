"""Top-level pipeline: detect -> stitch -> chromatic z-shift -> gene filter.

`run_pipeline()` takes the same parameters as the original script's
USER PARAMETERS block, so a caller can just set those values directly (see
run_spot_detection.py) rather than going through a config file or CLI flags.

This is the *new-layout* variant: it reads per-tile OME-TIFFs from
`<squid_folder>/ome_tiff/` (every z-slice and channel stacked into one file
per tile). It also reads the older layout -- one TIFF per (tile, z, channel)
in `<squid_folder>/0/` -- and `squid_folder` may equally be the path to a
single OME-TIFF, which runs the whole pipeline on that one tile. The layout is
detected from the folder; nothing else about the run changes with it, except
that an old-layout acquisition carries no voxel size, so `resolution` must be
given rather than left None. See io_utils.py and legacy_io.py.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import time

import yaml
from tqdm import tqdm

from .basic_correction import DEFAULT_MAX_FIT_SLICES
from .chromatic import compute_chromatic_z_shift_auto, save_chromatic_z_shift_file
from .corrections import (
    describe_corrections,
    make_correction_kwargs,
    uses_basic,
    write_corrected_image_dir,
)
from .gene_filter import filter_genes
from .io_utils import (
    OME_TIFF_SUBDIR,
    configure_single_tiff_channels,
    default_output_folder,
    extract_wavelength_from_channel,
    fov_from_path,
    get_image_dir,
    get_resolution_nm,
    get_squid_tile_channels,
    get_tile_shape,
    is_legacy_image_dir,
    is_single_tiff,
    load_virtual_stitch_coords,
    make_squid_folders,
)
from .localfdr_calibration import DEFAULT_Q, calibrate_channel_parameters_localfdr
from .monitoring import monitor_memory_tree
from .stitching import stitch_squid_tile_csvs
from .tasks import find_maxima_3d_tile_channel, load_tile_channel, detect_and_save, prefetch
from .torch_filters import gpu_available, select_device
from .validation import (
    validate_channel_parameters,
    validate_chromatic_z_shift,
    validate_genes_yaml,
)


def run_pipeline(
    *,
    squid_folder,
    output_folder=None,
    channel_parameter_file=None,
    auto_calibrate=False,
    calibration_kwargs=None,
    reference_channels,
    single_tiff_channels_by_index=True,
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
    basic_max_fit_slices=DEFAULT_MAX_FIT_SLICES,
    rolling_ball_radius=None,
    rolling_ball_shrink=None,
    gaussian_blur_sigma=None,
    constant_offset=None,
):
    """Run the full spot-detection pipeline: detect maxima for every
    (tile, channel), stitch per-tile results into per-channel point clouds,
    optionally compute chromatic z-shifts automatically, then group points
    into gene barcodes.

    squid_folder       : the acquisition folder (with its `ome_tiff/`
                          subfolder), or the path to a single OME-TIFF to
                          process just that one tile. In the single-file case
                          sidecar metadata is looked up next to the file (and
                          one level up if it sits in an `ome_tiff/` folder),
                          and results default to a `<name>_spots` folder
                          beside it rather than to the squid folder.
    single_tiff_channels_by_index : only applies when `squid_folder` is a
                          single OME-TIFF. True (default) names that file's
                          channels by C index from the hardcoded
                          CHANNEL_INDEX_TO_WAVELENGTH map in io_utils.py
                          (C 0 -> 405, C 1 -> 488, ...), so the wavelengths in
                          genes_file and reference_channels land on the right
                          channels even when the file's own metadata names them
                          differently or not at all. Set it False to trust the
                          names in the file -- do that if the file's channels
                          are not the first N of that map, e.g. a 3-channel
                          crop holding 488/561/638.
    resolution         : [x, y, z] voxel size in nm. Set to None to read it
                          from the OME-TIFF metadata (PhysicalSizeX/Y/Z).
    channel_parameter_file : channel_parameters.yaml with per-wavelength
                          threshold/tolerance. May be None when
                          auto_calibrate is True.
    auto_calibrate     : estimate threshold/tolerance per channel from the
                          data instead of reading them from a file, by
                          local-FDR calibration (see localfdr_calibration.py):
                          each channel is cut where a fraction q of the CALLED
                          spots would be noise, q defaulting to 0.1. The
                          estimated values are written to
                          <output_folder>/channel_parameters_localfdr.yaml
                          alongside localfdr_report.csv and the Fdr curves,
                          and are used for this run.
    calibration_kwargs : dict of overrides forwarded to
                          calibrate_channel_parameters_localfdr -- e.g.
                          {'q': 0.05}, {'tiles': [3, 5]} (default: the middle
                          tile), or {'n_chunks': 8, 'chunk_xy': 768}.
    use_basic          : fit a BaSiC flatfield/darkfield model independently
                          for each tile (from that tile's own z-stack,
                          randomly subsampled to `basic_max_fit_slices`
                          slices if it's deeper than that) and correct each
                          tile's volume before detection. Fits run on the GPU
                          only when they finish before detection starts
                          (output_corrected=True); otherwise on the CPU, so
                          they do not contend with the detector for the device.
    rolling_ball_radius : radius in pixels of a rolling-ball background,
                          subtracted from every z-slice (ImageJ's Subtract
                          Background). None/0 disables it.
    rolling_ball_shrink : downsample factor the ball is rolled at; None picks
                          ImageJ's factor for the radius (1/2/4/8). Larger is
                          faster and smoother, 1 is exact and very slow.
    gaussian_blur_sigma : Gaussian blur sigma in pixels applied to the image
                          itself. A scalar blurs each z-slice in xy; a
                          (sigma_z, sigma_y, sigma_x) tuple blurs in 3-D.
                          None/0 disables it. This is *not* `smooth_sigma`,
                          which the detector applies internally to locate
                          peaks without changing the reported intensities.
    constant_offset     : counts subtracted from every voxel (e.g. a camera
                          baseline). None/0 disables it.
    output_corrected   : when at least one correction is enabled, also bake
                          the corrected images to
                          <output_folder>/corrected/ome_tiff/ (one OME-TIFF
                          per tile) and detect from there. Every enabled
                          correction is baked in, in the order listed in
                          corrections.py. When False, the same chain is
                          applied in-memory per tile as it's loaded and
                          discarded once that tile+channel is detected --
                          nothing extra is written to disk.

    Parameters mirror the original script's USER PARAMETERS block. See
    run_spot_detection.py for a typed-in-script entry point.

    Every correction changes the intensity scale `threshold` and `tolerance`
    are expressed in, so changing one invalidates per-channel parameters
    calibrated under the old settings.

    Note: on the CPU/scipy path with use_basic=True and output_corrected=False,
    the worker pool is *spawned* rather than forked (basicpy is torch-based,
    and torch's thread pools deadlock in forked children once this process has
    initialised them, which auto_calibrate does). Spawned workers re-import the
    calling module, so a script that calls run_pipeline must guard the call
    with `if __name__ == '__main__':` -- both bundled entry-point scripts do.
    The other three corrections are plain numpy/scipy and do not force a spawn.

    On the GPU path BaSiC fits are kept on the CPU unless output_corrected is
    True, so they do not compete with the detector for GPU memory, and fits are
    serialised within a process so the io_workers prefetch threads cannot run
    several at once.
    """
    if output_folder is None:
        output_folder = default_output_folder(squid_folder)
        print(f'output_folder not set; writing results to {output_folder}')
    os.makedirs(output_folder, exist_ok=True)
    memory_log = os.path.join(output_folder, 'memory_log.csv')

    # Decide how this input's channels are named before anything reads a tile:
    # for a single OME-TIFF they come from the hardcoded index map, so that
    # genes_file / reference_channels wavelengths address its C axis by order.
    configure_single_tiff_channels(squid_folder, single_tiff_channels_by_index)

    start_full = time.time()
    stop_and_save = monitor_memory_tree(interval=0.5, output_csv=memory_log)

    genes_data = None
    if genes_file is not None:
        if not os.path.exists(genes_file):
            raise FileNotFoundError(f'Genes file not found: {genes_file}')
        print(f'Validating genes file: {genes_file}')
        genes_data = validate_genes_yaml(genes_file)
        print(f'Genes file OK — {len(genes_data)} gene(s) defined: {", ".join(genes_data.keys())}')

    if resolution is None:
        resolution = get_resolution_nm(squid_folder)
        if resolution is None:
            raise ValueError(
                'resolution=None was requested but the image metadata has no '
                'PhysicalSizeX/Y/Z -- an old-layout acquisition never does, its '
                'per-slice TIFFs carry no pixel size. '
                'Set resolution=[x_nm, y_nm, z_nm] explicitly.'
            )
        print(f'Voxel size read from OME metadata: '
              f'{resolution[0]:.1f} x {resolution[1]:.1f} x {resolution[2]:.1f} nm')

    if chromatic_ref_channel is None:
        # Only validate the manual dict when it will actually be used.
        validate_chromatic_z_shift(chromatic_z_shift_px)
        if chromatic_z_shift_px is not None:
            print(f'Chromatic z-shift correction enabled (manual): {chromatic_z_shift_px}')
    else:
        if chromatic_z_shift_px is not None:
            print(
                f'Both chromatic_ref_channel ({chromatic_ref_channel}) and '
                f'chromatic_z_shift_px are set. The automatically computed '
                f'shifts will override chromatic_z_shift_px.'
            )
        if chromatic_ref_channel in reference_channels:
            raise ValueError(
                f'chromatic_ref_channel ({chromatic_ref_channel}) is listed in '
                f'reference_channels and will not be detected. '
                f'Choose a channel that is NOT in reference_channels.'
            )
        print(f'Automatic chromatic z-shift enabled (reference channel: {chromatic_ref_channel} nm)')

    # An empty list is legitimate -- it means "detect every channel". The
    # single-element case stays a bare scalar for backward compatibility with
    # is_reference_channel's substring test.
    if len(reference_channels) > 1:
        reference_channel = tuple(reference_channels)
    elif len(reference_channels) == 1:
        reference_channel = reference_channels[0]
    else:
        reference_channel = ()

    virtual_stitch_coords = None
    if virtual_stitch_coords_path is not None:
        virtual_stitch_coords = load_virtual_stitch_coords(virtual_stitch_coords_path, squid_folder)
        if is_single_tiff(squid_folder):
            # The coords file describes the whole grid, but stitching only sees
            # this one tile and indexes the list from 0 -- so hand it just this
            # tile's entry, picked by the fov its filename carries.
            fov = fov_from_path(squid_folder)
            if fov >= len(virtual_stitch_coords):
                raise ValueError(
                    f'{squid_folder} is FOV {fov} but {virtual_stitch_coords_path} only '
                    f'describes {len(virtual_stitch_coords)} tile(s).'
                )
            print(f'Single-tile input: using virtual stitch coords for FOV {fov} only.')
            virtual_stitch_coords = [virtual_stitch_coords[fov]]

    if not auto_calibrate and channel_parameter_file is None:
        raise ValueError(
            'Provide channel_parameter_file, or set auto_calibrate=True to '
            'estimate threshold/tolerance from the data.'
        )
    start = time.time()

    use_gpu = (backend == 'torch') or (backend == 'auto' and gpu_available())
    resolved_backend = 'torch' if use_gpu else 'scipy'
    detect_device = select_device(device) if use_gpu else None
    # basicpy only supports device in {'cpu', 'cuda'} (no 'mps'); mirror the
    # spot-detection device where possible, otherwise fall back to CPU.
    # basicpy is torch-based, so a BaSiC fit competes with the detector for the
    # same device. It may share the GPU only when the correction runs to
    # completion *before* detection starts (output_corrected=True). With
    # output_corrected=False the fits happen inline in the prefetch threads
    # while the main thread is detecting, and a fit moves a whole tile to the
    # device (~4.3 GB for 145x2720x2720 float32) on top of detection's ~6.3 GB
    # -- enough to exhaust an 11 GB card and stall. Keeping those fits on the
    # CPU also gives genuine CPU/GPU overlap rather than contention.
    share_gpu = use_gpu and detect_device.type == 'cuda' and output_corrected
    basic_device = 'cuda' if share_gpu else 'cpu'

    print('Starting parallel processing...')
    image_dir = get_image_dir(squid_folder)
    tile_shape = get_tile_shape(squid_folder)
    if is_single_tiff(image_dir):
        source = 'single OME-TIFF'
    elif is_legacy_image_dir(image_dir):
        source = 'per-slice TIFFs (old layout)'
    else:
        source = 'per-tile OME-TIFFs'
    print(f'Reading {source} from {image_dir} '
          f'(tiles are {tile_shape[1]} x {tile_shape[0]} px)')
    tiles_by_channel = get_squid_tile_channels(squid_folder, reference_channel)
    make_squid_folders(output_folder, tiles_by_channel)

    # ── INTENSITY CORRECTIONS ─────────────────────────────────────────────────
    # Every correction happens per tile+channel (see corrections.py) -- there
    # is no shared/global model, so there's nothing to precompute here beyond
    # the kwargs every tile will be corrected with. They are built before
    # calibration so that calibration measures the same corrected pixel values
    # detection will later see.
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
        print(f'\nLocal-FDR calibration of per-channel threshold/tolerance '
              f'(q = {(calibration_kwargs or {}).get("q", DEFAULT_Q)})...')
        start_cal = time.time()
        cal_kwargs = {
            'footprint_size': footprint_size,
            'smooth_sigma': smooth_sigma,
            'prominence_footprint_size': prominence_footprint_size,
            # The cutoffs are expressed on whatever pixel scale they are
            # measured on, so calibration sees the same corrected pixels
            # detection will.
            'correction_kwargs': correction_kwargs,
            'output_path': os.path.join(output_folder, 'channel_parameters_localfdr.yaml'),
            'report_dir': output_folder,
            'reference_channels': reference_channels,
            'single_tiff_channels_by_index': single_tiff_channels_by_index,
            # Only the optional held-out consensus check reads these, and only
            # when calibration_kwargs sets `bootstrap`; they never move the
            # chosen cutoffs.
            'genes_file': genes_file,
            'radius': radius,
            'depth': depth,
            'resolution': resolution,
        }
        cal_kwargs.update(calibration_kwargs or {})
        channel_parameters = calibrate_channel_parameters_localfdr(squid_folder, **cal_kwargs)
        print(f'Calibration finished in {time.time() - start_cal:.2f}s')
        if channel_parameter_file is not None:
            print(f'Note: auto_calibrate=True, so {channel_parameter_file} is ignored.')
    else:
        print(f'Reading per channel parameters from {channel_parameter_file}')
        with open(channel_parameter_file, 'r') as file:
            channel_parameters = yaml.safe_load(file)
        print('Parameters read successfully')
    validate_channel_parameters(channel_parameters)

    channels_to_detect = sorted({
        channel
        for channels in tiles_by_channel.values()
        for channel in channels
        if extract_wavelength_from_channel(channel) in channel_parameters
    })

    if output_corrected and correction_kwargs is None:
        print('output_corrected=True but no corrections are enabled; nothing to write.')
    elif output_corrected:
        corrected_dir = os.path.join(output_folder, 'corrected', OME_TIFF_SUBDIR)
        print(f'Writing corrected OME-TIFFs to {corrected_dir}...')
        start_write = time.time()
        # Detection now reads the pre-corrected copies (the returned path, which
        # for a single-file input is the corrected file itself); no need to
        # re-apply the corrections per tile at load time below.
        image_dir = write_corrected_image_dir(
            image_dir, corrected_dir, set(channels_to_detect), correction_kwargs)
        print(f'Finished writing corrected images in {time.time() - start_write:.2f}s\n')
        correction_kwargs = None
    # ───────────────────────────────────────────────────────────────────────────

    detect_tasks = []
    for n, channels in tiles_by_channel.items():
        for channel in channels:
            wl = extract_wavelength_from_channel(channel)
            if wl not in channel_parameters:
                print(f'No channel_parameters entry for wavelength {wl} (channel {channel}), skipping.')
                continue
            threshold = channel_parameters[wl]['threshold']
            tolerance = channel_parameters[wl]['tolerance']
            out_csv = os.path.join(output_folder, f'tile_{n}', channel, 'maxima_results.csv')
            detect_tasks.append({
                'n': n, 'channel': channel, 'image_dir': image_dir, 'out_csv': out_csv,
                'threshold': threshold, 'tolerance': tolerance,
                'footprint_size': footprint_size, 'smooth_sigma': smooth_sigma,
                'prominence_footprint_size': prominence_footprint_size,
                'backend': resolved_backend, 'device': device, 'z_chunk': z_chunk,
                'correction_kwargs': correction_kwargs,
            })

    if use_gpu:
        print(f'Spot detection backend: GPU/torch on {detect_device} '
              f'(serial compute, {io_workers} I/O prefetch threads)')
        with tqdm(total=len(detect_tasks), desc='Detecting 3D maxima [GPU]', unit='tile-channel') as pbar:
            for task, vol in prefetch(detect_tasks, load_tile_channel, max_ahead=io_workers):
                n, channel, n_spots = detect_and_save(task, vol)
                pbar.set_postfix_str(f'tile {n} {channel}: {n_spots} spots')
                pbar.update()
    else:
        # Workers that run BaSiC must be spawned, not forked. basicpy is
        # torch-based, and torch's internal thread pools do not survive
        # fork(): if anything in this process has already initialised them --
        # auto_calibrate does, and so does baking corrected images -- forked
        # children inherit that state and deadlock on first use, hanging the
        # pool indefinitely. Spawned children start clean. Fork is kept
        # otherwise since it starts faster and this path never touches BaSiC.
        # Only BaSiC forces this: the rolling ball, blur and offset are plain
        # numpy/scipy and are fork-safe.
        start_method = 'spawn' if uses_basic(correction_kwargs) else None
        ctx = mp.get_context(start_method)
        print(f'Spot detection backend: CPU/scipy ({processes} worker processes'
              f'{", spawned for BaSiC fork-safety" if start_method else ""})')
        with ctx.Pool(processes=processes) as pool:
            with tqdm(total=len(detect_tasks), desc='Detecting 3D maxima [CPU]', unit='tile-channel') as pbar:
                for n, channel, n_spots in pool.imap_unordered(find_maxima_3d_tile_channel, detect_tasks):
                    pbar.set_postfix_str(f'tile {n} {channel}: {n_spots} spots')
                    pbar.update()

    print(f'Processing finished in {time.time() - start:.2f}s')

    start = time.time()
    print('Starting stitching...')
    stitch_squid_tile_csvs(squid_folder, output_folder, overlap_percentage, resolution,
                            virtual_stitch_coords, tile_shape=tile_shape)
    print(f'Finished stitching in {time.time() - start:.2f} seconds')

    # ── AUTOMATIC CHROMATIC Z-SHIFT COMPUTATION ───────────────────────────────
    # Runs after stitching (so detected_points.csv files are ready) and before
    # gene filtering so the computed shifts can be applied.
    if chromatic_ref_channel is not None:
        start = time.time()
        print(f'\nComputing chromatic z-shift automatically '
              f'(reference: {chromatic_ref_channel} nm)...')
        chromatic_z_shift_px, z_shift_stats = compute_chromatic_z_shift_auto(
            output_folder, chromatic_ref_channel, radius, depth, resolution
        )
        save_chromatic_z_shift_file(
            chromatic_z_shift_px, z_shift_stats, chromatic_ref_channel, output_folder, resolution
        )
        print(f'Chromatic z-shift computation finished in {time.time() - start:.2f}s\n')
    # ───────────────────────────────────────────────────────────────────────────

    if genes_file is not None:
        start = time.time()
        print('Starting gene filtering...')
        filter_genes(output_folder, genes_file, radius, depth, resolution, chromatic_z_shift_px)
        print(f'Finished gene filtering in {time.time() - start:.2f} seconds')

    stop_and_save()
    print(f'Finished entire process in {time.time() - start_full:.2f} seconds')
