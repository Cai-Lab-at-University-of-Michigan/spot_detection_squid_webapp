"""
3D spot-detection package for *new-layout* SQUID microscopy acquisitions --
one multi-page OME-TIFF per tile (`<squid_folder>/ome_tiff/R0_<fov>.ome.tiff`,
all z-slices and channels stacked inside).

It reads the older layout as well -- one TIFF per (tile, z-slice, channel) in
`<squid_folder>/0/`, what `spot_detection_functionalized_with_basic` takes --
and `squid_folder` may equally be the path to a single OME-TIFF, to run on just
that tile. Whichever you pass, the layout is detected from the folder and the
run is otherwise identical; see io_utils and legacy_io.

Detection, stitching, chromatic-shift and gene-filtering logic is unchanged
from `spot_detection_functionalized_with_basic`; only the I/O layer differs
(io_utils, legacy_io, basic_correction, stitching, orchestrator).

Modules
-------
detection      -- public alg_smoothed_local_max_3d() + scipy (CPU) backend
torch_filters  -- PyTorch (CUDA/MPS/CPU) backend used by detection.py
io_utils       -- OME-TIFF tile loading, channel bookkeeping, stage/virtual-stitch coords
legacy_io      -- reader for the old per-(tile, z, channel) TIFF layout, which
                  io_utils presents as per-tile OME-TIFFs
basic_correction -- per-tile BaSiC flatfield/darkfield fit
corrections    -- the pre-detection correction chain (constant offset, BaSiC,
                  rolling ball, blur) + optional corrected OME-TIFF output
localfdr_calibration -- per-channel threshold/tolerance at a fixed false-discovery
                  rate q; what run_pipeline(auto_calibrate=True) uses
calibration    -- the older Otsu / gene-objective calibrator, still driven by
                  run_calibration.py but no longer used by the pipeline
stitching      -- combines per-tile maxima CSVs into per-channel point clouds
chromatic      -- automatic chromatic z-shift computation
gene_filter    -- multi-color gene barcode grouping
validation     -- input file / parameter validation
monitoring     -- background RSS memory monitor
tasks          -- per-(tile, channel) load + detect glue, shared by CPU/GPU dispatch
orchestrator   -- run_pipeline(): the full detect -> stitch -> gene-filter flow
run_spot_detection -- typed-in-script entry point (edit params, run as __main__)
run_calibration -- standalone entry point for parameter calibration only
selftest       -- scipy-vs-torch parity / speed check

Quick use
---------
    from spot_detection_new_squid.detection import alg_smoothed_local_max_3d
    coords = alg_smoothed_local_max_3d(volume, footprint_size=10, threshold=800,
                                        tolerance=150, backend='auto')

    from spot_detection_new_squid.orchestrator import run_pipeline
    run_pipeline(squid_folder=..., channel_parameter_file=..., ...)
"""
from .calibration import calibrate_channel_parameters
from .detection import alg_smoothed_local_max_3d
from .localfdr_calibration import calibrate_channel_parameters_localfdr
from .orchestrator import run_pipeline

__all__ = ['alg_smoothed_local_max_3d', 'calibrate_channel_parameters',
           'calibrate_channel_parameters_localfdr', 'run_pipeline']
