"""
Entry point for the SQUID spot-detection pipeline, using per-channel
threshold/tolerance from LOCAL-FDR CALIBRATION instead of the gene-objective
auto-tuning.

Copy of run_spot_detection.py -- identical in every respect except the
per-channel parameters, which come from consensus_calibration/
localfdr_calibrate.py at q = 0.1 (see the LOCAL-FDR CALIBRATION block below).
The original run_spot_detection.py is untouched.

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
(e.g. `python run_spot_detection_localfdr.py`, or your IDE's Run button) --
same "type your settings directly into the script" workflow as the original
single-file version, no CLI flags.
"""
import os
import sys

# Make the package importable when this file is run directly (python
# run_spot_detection_localfdr.py) rather than as `python -m
# spot_detection_new_squid.run_spot_detection_localfdr` -- direct execution
# only puts this file's own folder on sys.path, not its parent.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spot_detection_new_squid.orchestrator import run_pipeline

if __name__ == '__main__':

    ########## USER PARAMETERS ##############
    squid_folder = ""  # Folder to the entire squid data. ** NOT the 'ome_tiff' (or old-layout '0') folder with the images** Either SQUID layout works. May also be the path to a single .ome.tiff to process just that one tile, e.g. ".../ome_tiff/R0_3.ome.tiff"
    output_folder = ""  # folder to output the results to, if set to None output will be in the squid folder (or, for a single .ome.tiff, in a "<name>_spots" folder next to it)

    channel_parameter_file = '' # path to the file containing the per-channel parameters ** make sure the names of the channels match the squid ones, i.e. 488 NOT 1,2,3 etc**  Ignored when auto_calibrate is True; may be None then.

    auto_calibrate = False
    calibration_kwargs = None

    reference_channels = [405]  # reference channels to ignore in processing

    single_tiff_channels_by_index = False

    overlap_percentage = 0  # percentage overlap set in the squid software from imaging
    virtual_stitch_coords_path = None  # path to .json file from virtual stitching in SAVAII, if set to None stitching will be done uniformly with overlap_percentage
    resolution = [260,260,1500]  # image resolution in nm as [x, y, z]; None reads it from the OME-TIFF metadata (PhysicalSizeX/Y/Z)
    genes_file = ''  # path to the file containing the gene barcode specifications
    radius = 940  # cylindrical radius of a spot for gene grouping across channels
    depth = 4500 # depth of a spot for gene grouping across channels

    chromatic_ref_channel = None# int wavelength, or None to disable

    chromatic_z_shift_px = None

    footprint_size = 10
    smooth_sigma = 0
    prominence_footprint_size = 9

    backend = 'auto'  # 'auto' uses the GPU/torch path when a GPU is available; anything else (incl. None) runs scipy on CPU
    device = None
    z_chunk = 16  # measured peak ~6.3 GB on a 150x2720x2720 tile at footprint_size=10 (RTX 2080 Ti, 10.75 GiB usable); z_chunk=32 OOMs
    io_workers = 3

    processes = 6

    use_basic = False

    basic_get_darkfield = False  # also estimate a darkfield (dark-current offset) component
    basic_smoothness_flatfield = None  # None = auto-estimated by BaSiC
    basic_smoothness_darkfield = None  # None = auto-estimated by BaSiC
    basic_max_fit_slices = 150  # cap on z-slices used to fit each tile's model; randomly subsampled if the tile is deeper than this

    rolling_ball_radius = 0
    rolling_ball_shrink = 0

    gaussian_blur_sigma = 0 #2.
    constant_offset = None

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
