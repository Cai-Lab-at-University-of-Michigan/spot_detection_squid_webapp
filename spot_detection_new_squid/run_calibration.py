"""
Standalone entry point for auto-calibrating per-channel detection parameters.

Estimates `threshold` and `tolerance` for every channel from the images
themselves and writes a channel_parameters.yaml you can inspect (and edit)
before running detection. See calibration.py for how the two parameters are
estimated and why they are estimated differently.

Run it the same way as run_spot_detection.py -- edit the values below, then:

    python run_calibration.py

Outputs, both in `report_dir`:
  channel_parameters_auto.yaml  ready to pass as channel_parameter_file
  calibration_report.csv        one row per chunk: estimate, gate verdict,
                                 resulting spot density, rejection reason

To skip this step and calibrate inline as part of a pipeline run, set
auto_calibrate=True in run_spot_detection.py instead.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spot_detection_new_squid.calibration import calibrate_channel_parameters

if __name__ == '__main__':

    ########## USER PARAMETERS ##############
    squid_folder = ""  # Folder to the entire squid data. ** NOT the 'ome_tiff' (or old-layout '0') folder with the images** Either SQUID layout works. May also be the path to a single .ome.tiff to calibrate from that one tile (then leave `tiles` as None)
    report_dir = ""
    reference_channels = [405]   # channels to skip, same meaning as in the pipeline

    # Only used when squid_folder is a single .ome.tiff. True correlates the
    # wavelengths in genes.yaml and reference_channels to that file's channels
    # by C-axis order, using the hardcoded CHANNEL_INDEX_TO_WAVELENGTH map in
    # io_utils.py (channel 0 -> 405, 1 -> 488, 2 -> 561, 3 -> 638, 4 -> 730),
    # whatever the file's own metadata calls them. Set it False if the file's
    # channels are NOT the first N of that map -- e.g. a 3-channel crop holding
    # 488/561/638 -- in which case its own channel names are used instead.
    single_tiff_channels_by_index = True

    genes_file = ''  # path to the file containing the gene barcode specifications
    radius = 750  # cylindrical radius of a spot for gene grouping across channels
    depth = 10000 # depth of a spot for gene grouping across channels

    # The threshold/tolerance written out are in RAW pixel units -- the same
    # numbers you would read by hovering over a spot in ImageJ -- so they can
    # be sanity-checked against the images directly.

    # ── HOW STRICT? pick ONE of these two ───────────────────────────────────
    # strictness: multiplier on the Otsu split of candidate brightness.
    #   1.0  = the bare split (most permissive)
    #   1.5  = default; sits where hand-tuned gel values did, 1.42-1.60 across
    #          four channels (5% spread)
    #   2.0  = noticeably stricter, only clearly bright spots
    strictness = 0.85

    # target_density: if not None, solve for the threshold that yields this
    # many spots per megavoxel instead, and ignore `strictness`. Often the
    # more interpretable knob -- historical hand-tuned gel parameters worked
    # out to about 4.8 spots/Mvoxel.
    target_density = None
    # ─────────────────────────────────────────────────────────────────────────

    # False-positive budget that sets `tolerance`, in expected noise maxima
    # per tile. Lower = stricter. Raise it if the calibration warns that a
    # target_density is unreachable.
    expected_fp_per_tile = 30

    # Sampling. n_z_slabs * n_xy_per_slab chunks are taken per tile; each
    # z-slab is read once and the xy chunks are cut from it.
    tiles = [0]              # which FOVs to sample; None = the first tile
    n_z_slabs = 4
    n_xy_per_slab = 3
    chunk_shape = (32, 512, 512)   # ~8 Mvoxel; threshold is converged by this size

    # Signal-presence gate. Chunks whose estimated threshold is fewer than
    # this many robust sigma above their own noise mode contain no signal --
    # Otsu would split the noise itself there -- and are dropped.
    z_gate = 5.0

    # Must match what run_spot_detection.py uses, or the calibrated numbers
    # will not correspond to the detector that consumes them.
    footprint_size = 10
    smooth_sigma = 1.0
    prominence_footprint_size = 9

    backend = 'scipy'   # 'torch' to use the GPU
    device = None
    ##########################################

    params = calibrate_channel_parameters(
        squid_folder,
        output_path=os.path.join(report_dir, 'channel_parameters_auto.yaml'),
        report_dir=report_dir,
        reference_channels=reference_channels,
        single_tiff_channels_by_index=single_tiff_channels_by_index,
        tiles=tiles,
        n_z_slabs=n_z_slabs,
        n_xy_per_slab=n_xy_per_slab,
        chunk_shape=chunk_shape,
        footprint_size=footprint_size,
        smooth_sigma=smooth_sigma,
        prominence_footprint_size=prominence_footprint_size,
        strictness=strictness,
        objective = 'genes',
        genes_file = genes_file,
        radius=radius,
        depth=depth,
        target_density=target_density,
        expected_fp_per_tile=expected_fp_per_tile,
        z_gate=z_gate,
        backend=backend,
        device=device,
    )

    print()
    print('Calibrated parameters:')
    for wl, p in sorted(params.items()):
        print(f'  {wl}: threshold={p["threshold"]}  tolerance={p["tolerance"]}')
