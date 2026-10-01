"""
Entry point for the SQUID spot-detection pipeline (either SQUID layout).

`squid_folder` below takes either SQUID layout, detected from the folder:

* the new one -- one multi-page OME-TIFF per tile,
  `<squid_folder>/ome_tiff/R0_<fov>.ome.tiff`, all z-slices and channels
  stacked inside;
* the older one -- one TIFF per (tile, z, channel) in `<squid_folder>/0/`,
  what `spot_detection_functionalized_with_basic` reads. Set `resolution`
  explicitly for these: per-slice TIFFs carry no pixel size, so it cannot be
  read from the images.

It may equally be the path to a single .ome.tiff instead of an acquisition
folder, which runs everything on that one tile.

Edit the values in USER PARAMETERS below, then run this file directly
(e.g. `python run_spot_detection.py`, or your IDE's Run button) -- same
"type your settings directly into the script" workflow as the original
single-file version, no CLI flags.
"""
import os
import sys

# Make the package importable when this file is run directly (python
# run_spot_detection.py) rather than as `python -m
# spot_detection_new_squid.run_spot_detection` -- direct execution only puts
# this file's own folder on sys.path, not its parent.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spot_detection_new_squid.orchestrator import run_pipeline

if __name__ == '__main__':

    ########## USER PARAMETERS ##############
    squid_folder = ""  # Folder to the entire squid data. ** NOT the 'ome_tiff' (or old-layout '0') folder with the images** Either SQUID layout works. May also be the path to a single .ome.tiff to process just that one tile, e.g. ".../ome_tiff/R0_3.ome.tiff"
    output_folder = ""  # folder to output the results to, if set to None output will be in the squid folder (or, for a single .ome.tiff, in a "<name>_spots" folder next to it)
    channel_parameter_file = '' # path to the file containing the per-channel parameters ** make sure the names of the channels match the squid ones, i.e. 488 NOT 1,2,3 etc**  Ignored when auto_calibrate is True; may be None then.

    # ── AUTOMATIC PARAMETER CALIBRATION ──────────────────────────────────────
    # If True, per-channel threshold/tolerance are estimated from the images
    # instead of read from channel_parameter_file, by LOCAL-FDR calibration
    # (localfdr_calibration.py): each channel is cut where a fraction q of the
    # CALLED spots would be noise, estimated from that channel's own candidate
    # maxima. q is the only tunable and means the same thing in every sample.
    # The values used are written to
    # <output_folder>/channel_parameters_localfdr.yaml, with the per-channel
    # summary in localfdr_report.csv and the Fdr-vs-cutoff curves in
    # localfdr_curves.csv -- check the report's estimated spots/tile and tail
    # betas before trusting a run.
    # Run localfdr_calibration.py directly to calibrate without detecting
    # (e.g. on a true-black image); run_calibration.py still drives the older
    # Otsu/gene-objective calibrator in calibration.py, which the pipeline no
    # longer uses.
    auto_calibrate = False
    # Overrides forwarded to calibrate_channel_parameters_localfdr(), e.g.
    #   {'q': 0.05}                  stricter: fewer called spots may be noise.
    #                                 The cutoffs are very insensitive to q
    #                                 wherever the null tail decays fast.
    #   {'tiles': [0, 3]}            measure on these tiles instead of the
    #                                 middle one (each costs a full tile load
    #                                 per channel)
    #   {'n_chunks': 8}              more sampled chunks per tile
    #   {'chunk_xy': 768}            larger chunks (more candidates each)
    #   {'bootstrap': '<out>/tile_5'} score the chosen cutoffs against consensus
    #                                 loci from an existing run -- reported
    #                                 only, it never moves the cutoffs
    calibration_kwargs = None
    # ─────────────────────────────────────────────────────────────────────────
    reference_channels = [405]  # reference channels to ignore in processing

    # Only used when squid_folder is a single .ome.tiff. True correlates the
    # wavelengths in genes.yaml and reference_channels to that file's channels
    # by C-axis order, using the hardcoded CHANNEL_INDEX_TO_WAVELENGTH map in
    # io_utils.py (channel 0 -> 405, 1 -> 488, 2 -> 561, 3 -> 638, 4 -> 730),
    # whatever the file's own metadata calls them. Set it False if the file's
    # channels are NOT the first N of that map -- e.g. a 3-channel crop holding
    # 488/561/638 -- in which case its own channel names are used instead.
    single_tiff_channels_by_index = True

    overlap_percentage = 0  # percentage overlap set in the squid software from imaging
    virtual_stitch_coords_path = None  # path to .json file from virtual stitching in SAVAII, if set to None stitching will be done uniformly with overlap_percentage
    resolution = [260,260,1500]  # image resolution in nm as [x, y, z]; None reads it from the OME-TIFF metadata (PhysicalSizeX/Y/Z)
    genes_file = ''  # path to the file containing the gene barcode specifications
    radius = 750  # cylindrical radius of a spot for gene grouping across channels
    depth = 4500 # depth of a spot for gene grouping across channels

    # ── CHROMATIC Z-SHIFT ────────────────────────────────────────────────────
    # Set chromatic_ref_channel to the wavelength (int) of a DETECTED channel
    # (i.e. NOT listed in reference_channels) to enable automatic z-shift
    # computation. The script will:
    #   1. Detect and stitch all channels unshifted (standard pipeline).
    #   2. Find spots in each channel that co-localise with a reference spot
    #      within the same cylinder used for gene grouping (radius / depth).
    #   3. Compute the median z-offset per channel and save:
    #        <output_folder>/chromatic_z_shift.json  (virtual-stitch-coords format)
    #        <output_folder>/chromatic_z_shift.csv   (human-readable summary)
    #   4. Apply the computed shifts before gene filtering.
    #
    # Set chromatic_ref_channel = None to disable automatic computation.
    # If both chromatic_ref_channel and chromatic_z_shift_px are set, the
    # automatically computed shifts override chromatic_z_shift_px.
    chromatic_ref_channel = None# int wavelength, or None to disable

    # Manual z-shift fallback (used only when chromatic_ref_channel is None).
    # dict of {wavelength_int: z_shift_px} — positive shifts that channel later in z.
    chromatic_z_shift_px = None
    # ─────────────────────────────────────────────────────────────────────────

    # smoothed_local_max_3d parameters
    footprint_size = 10
    smooth_sigma = 1.0
    prominence_footprint_size = 9

    backend = 'auto'  # 'auto' uses the GPU/torch path when a GPU is available; anything else (incl. None) runs scipy on CPU
    device = None
    z_chunk = 16  # measured peak ~6.3 GB on a 150x2720x2720 tile at footprint_size=10 (RTX 2080 Ti, 10.75 GiB usable); z_chunk=32 OOMs
    io_workers = 3

    processes = 6

    # ── INTENSITY CORRECTIONS ───────────────────────────────────────────────
    # Four independent corrections, each off by default and each applied to a
    # tile+channel before its spots are found. When more than one is on they
    # run in this order (the whole chain is done in float32 and rounded back
    # to the image dtype only at the end):
    #
    #   1. constant offset             constant_offset
    #   2. BaSiC flatfield/darkfield   use_basic
    #   3. rolling-ball background     rolling_ball_radius
    #   4. Gaussian blur               gaussian_blur_sigma
    #
    # The offset runs first because steps 2 and 3 estimate an additive
    # background themselves -- subtracting a constant after one of them removes
    # the camera pedestal twice, which can floor the whole tile at zero.
    #
    # ALL FOUR CHANGE THE INTENSITY SCALE that `threshold` and `tolerance` in
    # channel_parameter_file are expressed in. Turning one on or off means the
    # per-channel parameters must be recalibrated.

    # If use_basic is True, a BaSiC flatfield/darkfield model is fit
    # independently for each tile (from that tile's own z-stack -- randomly
    # subsampled to basic_max_fit_slices slices if it's deeper than that)
    # and applied to that tile before its spots are found. Fitting runs on
    # whichever device (GPU/CPU) is being used for spot detection above.
    use_basic = True

    basic_get_darkfield = False  # also estimate a darkfield (dark-current offset) component
    basic_smoothness_flatfield = None  # None = auto-estimated by BaSiC
    basic_smoothness_darkfield = None  # None = auto-estimated by BaSiC
    basic_max_fit_slices = 150  # cap on z-slices used to fit each tile's model; randomly subsampled if the tile is deeper than this

    # Rolling-ball background subtraction (ImageJ's Subtract Background), run
    # in 2D on each z-slice. Set the radius in PIXELS -- comfortably larger
    # than a spot (a few times footprint_size) so spots ride on top of the
    # background rather than being flattened into it. None or 0 disables it.
    rolling_ball_radius = None
    # The ball is rolled on a shrunken copy of each plane and the background
    # interpolated back up, exactly as ImageJ does -- rolling it at full
    # resolution takes ~27 s per 2720x2720 plane (over an hour per tile).
    # None picks ImageJ's factor for the radius (<=10 -> 1, <=30 -> 2,
    # <=100 -> 4, else 8). Raise it for speed, set 1 for an exact (slow) roll.
    rolling_ball_shrink = None

    # Gaussian blur applied to the IMAGE, in pixels -- unlike smooth_sigma
    # above (which the detector uses internally to locate peaks and which does
    # not change the intensities it reports), this blur changes the pixel
    # values that threshold/tolerance are tested against and that land in the
    # output CSVs. A scalar blurs each z-slice in xy; a
    # (sigma_z, sigma_y, sigma_x) tuple blurs in 3D. None or 0 disables it.
    gaussian_blur_sigma = None

    # Flat value subtracted from every voxel, in counts (e.g. a camera
    # baseline). Applied last, after any background subtraction, and the
    # result is clipped at 0. None or 0 disables it.
    constant_offset = None

    # If True, the corrected images are also written to
    # <output_folder>/corrected/ome_tiff/ as one OME-TIFF per tile, mirroring
    # the original ome_tiff directory, and detection runs on those saved
    # copies. Every correction enabled above is baked into them. Note these
    # files are as large as the originals (~10 GB per 5-channel, 145-slice,
    # 2720x2720 tile).
    # If False, the corrections are applied in memory as each tile+channel is
    # loaded for detection and discarded afterward -- nothing is written to disk.
    output_corrected = False
    ##########################################

    run_pipeline(
        squid_folder=squid_folder,
        output_folder=output_folder,
        channel_parameter_file=channel_parameter_file,
        auto_calibrate=auto_calibrate,
        calibration_kwargs=calibration_kwargs,
        reference_channels=reference_channels,
        single_tiff_channels_by_index=single_tiff_channels_by_index,
        overlap_percentage=overlap_percentage,
        virtual_stitch_coords_path=virtual_stitch_coords_path,
        resolution=resolution,
        genes_file=genes_file,
        radius=radius,
        depth=depth,
        chromatic_ref_channel=chromatic_ref_channel,
        chromatic_z_shift_px=chromatic_z_shift_px,
        footprint_size=footprint_size,
        smooth_sigma=smooth_sigma,
        prominence_footprint_size=prominence_footprint_size,
        backend=backend,
        device=device,
        z_chunk=z_chunk,
        io_workers=io_workers,
        processes=processes,
        use_basic=use_basic,
        output_corrected=output_corrected,
        basic_get_darkfield=basic_get_darkfield,
        basic_smoothness_flatfield=basic_smoothness_flatfield,
        basic_smoothness_darkfield=basic_smoothness_darkfield,
        basic_max_fit_slices=basic_max_fit_slices,
        rolling_ball_radius=rolling_ball_radius,
        rolling_ball_shrink=rolling_ball_shrink,
        gaussian_blur_sigma=gaussian_blur_sigma,
        constant_offset=constant_offset,
    )
