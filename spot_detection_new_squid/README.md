# `spot_detection_new_squid` — spot detection for either SQUID layout

Same pipeline as
[`spot_detection_functionalized_with_basic`](../spot_detection_functionalized_with_basic/),
updated for acquisitions that stack every z-slice and channel of a tile into a
single OME-TIFF instead of writing one TIFF per (tile, z, channel). It still
reads the old per-slice layout too, so one package covers both.

The detection algorithm, stitching maths, chromatic z-shift computation and
gene grouping are unchanged — `detection.py`, `torch_filters.py`,
`chromatic.py`, `validation.py`, `monitoring.py`, `tasks.py` and `selftest.py`
are byte-for-byte copies. The I/O layer differs; `calibration.py` and
`gene_objective.py` are new; `detection.py`/`torch_filters.py` now threshold
raw rather than smoothed values; and `gene_filter.py` gained the
unassigned-points output and had its grouping extracted into `group_points`
(the grouping logic itself is unchanged).

## Which input do I give it?

`squid_folder` takes any of three things; the layout is detected from the
folder and nothing else about the run changes with it.

| Acquisition folder contains | Read as |
|---|---|
| `ome_tiff/R0_<fov>.ome.tiff` (one stacked file per tile) | the new layout |
| `0/R0_<fov>_<z>_Fluorescence_488_nm_Ex.tiff` (one file per slice/channel) | the old layout — see [below](#or-an-old-per-slice-folder) |
| a path to a single `.ome.tiff` | that one tile — see [below](#or-a-single-ome-tiff) |

A folder holding both is read the new way.

## Layout it expects

```
<squid_folder>/
  ome_tiff/
    R0_0.ome.tiff        axes ZCYX, e.g. 145 x 5 x 2720 x 2720, uint16
    R0_1.ome.tiff        channel names in the OME-XML ("Fluorescence 488 nm Ex")
    ...
  0/
    coordinates.csv      region, fov, z_level, x (mm), y (mm), z (um), time
    mosaic_view/
  coordinates.csv        region, x (mm), y (mm), z (mm) — one row per fov
  acquisition parameters.json    Nx, Ny, Nz, dz(um) ...
  acquisition.yaml               objective, z-stack, channels ...
  acquisition_channels.yaml      channel table
```

### Or an old per-slice folder

The layout `spot_detection_functionalized_with_basic` reads is accepted as it
stands — pass the acquisition folder, or its `0/` folder directly:

```
<squid_folder>/
  0/
    R0_0_0_Fluorescence_405_nm_Ex.tiff      one file per (tile, z, channel)
    R0_0_0_Fluorescence_488_nm_Ex.tiff
    ...
    coordinates.csv
  coordinates.csv
  acquisition parameters.json
```

[`legacy_io.py`](legacy_io.py) presents each tile as if it were one
(Z, C, Y, X) OME-TIFF — a wavelength-ordered C axis (405, 488, 561, 638, 730)
so it matches a new-layout tile's, one `TileInfo` just like a real tile's —
and [`io_utils.py`](io_utils.py) dispatches to it. Everything downstream
(detection, both calibrators, PSF estimation, corrections, stitching, gene
filtering) is untouched and gives bit-identical results on the same pixels
whichever layout they arrived in.

Two things to know:

- **Set `resolution` explicitly.** Per-slice TIFFs are written with no pixel
  size, so `resolution=None` cannot work here; pass
  `resolution=[x_nm, y_nm, z_nm]` as the old package required.
- **`output_corrected=True` writes the new layout.** Corrected copies come out
  as one `R0_<fov>.ome.tiff` per tile under
  `<output_folder>/corrected/ome_tiff/`, not as per-slice TIFFs.

Channel names come from the filenames, as before; a tile whose channels have
unequal numbers of z-slices (a truncated acquisition) is trimmed to the depth
they all share, with a warning.

### Or a single OME-TIFF

`squid_folder` also accepts the path to one `.ome.tiff` instead of an
acquisition folder, which runs the whole pipeline on that one tile:

```python
run_pipeline(squid_folder='/data/acq/ome_tiff/R0_3.ome.tiff', ...)
```

- The tile's **FOV number** comes from its filename when it has one
  (`R0_3.ome.tiff` → 3, so results land in `tile_3/`); otherwise it is 0.
- **Sidecar metadata** (`coordinates.csv`, `acquisition parameters.json`, the
  YAMLs) is looked up next to the file, and one level up when the file sits in
  an `ome_tiff/` folder — so pointing at a tile inside a real acquisition still
  picks up that acquisition's metadata.
- **Channels are named by index.** A lone file often carries no usable channel
  names, so its channels are labelled from the hardcoded
  `CHANNEL_INDEX_TO_WAVELENGTH` map at the top of
  [`io_utils.py`](io_utils.py) — C 0 → 405, C 1 → 488, C 2 → 561, C 3 → 638,
  C 4 → 730 — whatever the file's own metadata calls them. That is what makes
  the wavelengths in `genes.yaml` and `reference_channels` line up with its
  channels. The mapping is printed at the start of every run. Set
  `single_tiff_channels_by_index=False` to use the file's own names instead —
  do that if its channels are **not** the first N of that map (e.g. a
  3-channel crop holding 488/561/638, which would otherwise be read as
  405/488/561). Acquisition folders are unaffected: their per-tile OME-XML
  names are trusted either way.
- **Stitching** one tile needs no stage position: if no coordinate file is
  found the tile is placed at (0, 0) rather than erroring, and coordinates are
  relative to the tile itself either way.
- **`output_folder=None`** writes to a `<filename>_spots` folder next to the
  image, rather than into whatever directory the image happens to live in.

## Running the pipeline

Open [`run_spot_detection.py`](run_spot_detection.py), edit the values in the
`USER PARAMETERS` block (paths, channel/gene config, detection thresholds,
backend), then run the file directly:

```bash
python run_spot_detection.py
```

Same "type your settings into the script" workflow as before — no CLI flags,
no config file. `squid_folder` is the acquisition folder itself, **not** the
`ome_tiff` (or old-layout `0`) subfolder — or a single `.ome.tiff`, as above.

## Automatic parameter calibration

`threshold` and `tolerance` can be estimated from the images instead of typed
in by hand. Either run it on its own and inspect the result:

```bash
python run_calibration.py     # writes channel_parameters_auto.yaml + calibration_report.csv
```

or set `auto_calibrate = True` in `run_spot_detection.py` to calibrate inline
at the start of a pipeline run.

**Why it's cheap.** The detector's peak mask is
`(maximum_filter(v) == v) & (raw >= threshold) & (raw - minimum_filter(v) >= tolerance)`.
The `maximum_filter(v) == v` term depends on neither parameter, so *one* filter
pass over a chunk yields every candidate maximum plus the two statistics the
thresholds act on. Any `(threshold, tolerance)` pair is then pure post-filtering
on two arrays — the whole parameter surface for the price of one pass, with no
re-running of detection. (Verified to reproduce the detector exactly.)

**Raw units in, raw units out.** The emitted `threshold`/`tolerance` are in raw
pixel values (see the section below), but they are *estimated* on the smoothed
statistics and mapped to raw by quantile matching. Estimating directly in raw
units is measurably worse: raw candidate intensities keep the pixel noise that
smoothing removes, broadening the noise mode until Otsu returns nearly the same
split for every channel — 219/220/225 on three gel channels whose hand-tuned
thresholds were 400/250/250 — losing the channel discrimination the
smoothed-domain estimate gets right.

**The two parameters are estimated differently**, because they behave
differently:

- **`threshold`** — Otsu's split on `log10(smoothed candidate intensity)`,
  mapped to raw. Smoothed candidate intensities are strongly bimodal: a tight
  noise mode plus a sparse bright tail. Against hand-tuned values on the 0615
  calibration gel (converted to raw), Otsu × 1.5 tracks the human picks with a
  1.42–1.60 spread across four channels (5%), and a 3.5% coefficient of
  variation across 12 chunks. Hence the `strictness = 1.5` default.
- **`tolerance`** — an explicit false-positive budget against a fitted null,
  *not* Otsu. On sparse samples real spots can be under 0.1% of candidates,
  leaving no second mode; Otsu then splits the noise itself. The null is
  modelled as a **Gumbel** distribution, because a background local maximum's
  prominence is `max − min` of noise over a window — an extreme-value
  statistic. A Gaussian null underestimates that tail. Set via
  `expected_fp_per_tile` (default 10).

**Guarding against empty chunks.** On sparse tissue many chunks contain no
signal, and there Otsu bisects the noise mode, producing a ~500× spot-density
blowout. Expressed as a z-score against a null fitted to the lower 95% of
candidates, failing chunks scored 0.0–0.4 while good chunks scored 17–123 — no
overlap — so chunks below `z_gate` (default 5) are dropped before aggregation.
A second pass then drops any surviving chunk whose density at the consensus
parameters is `outlier_factor` above the median and re-aggregates. Deciding
*which chunks to trust* is what the iteration is for; the parameter sweep
itself needs none.

**Three ways to set the operating point** — pick one:

| knob | meaning |
|---|---|
| `strictness` (default 1.5) | multiplier on the Otsu split. Higher = fewer, brighter spots. |
| `target_density` | solve for N spots per megavoxel. **Only valid when every channel carries the same spot density** — see below. |
| `objective='genes'` | tune all channels jointly to maximise gene calls and minimise unassigned groups. |

### `objective='genes'` — optimise the gene outcome directly

`target_density` asks every channel for the same spot density, which is wrong
as soon as genes use different colour combinations: a channel in three
barcodes should carry roughly three times the spots of one in a single
barcode. The channels are also *coupled* — whether a 488 spot is assigned
depends on whether its partner 561 and 638 spots were detected — so no
per-channel target can optimise the thing you care about.

Setting `calibration_kwargs={'objective': 'genes'}` instead optimises

```
score = assigned − chance_weight·chance − unassigned_weight·unassigned
```

by coordinate ascent over per-channel thresholds, scoring with the *same*
`group_points` routine the pipeline uses, so it optimises the real grouping
rather than an approximation. It needs `genes_file`, `radius` and `depth`,
which `run_pipeline` forwards automatically.

Both penalties are load-bearing, and they catch different failures:

- **`chance`** — expected assigned groups from random co-localization,
  estimated from *decoy* combinations (same size, not in genes.yaml). Without
  it, "maximise assigned" is degenerate: dropping every threshold manufactures
  barcodes out of accidents. On a synthetic check the degenerate solution had
  *more* assigned groups than the clean one (80 vs 70) but scored 0.0 vs 68.4.
- **`unassigned`** — the decoy term only ever looks at barcode-sized
  combinations, so single-channel junk is invisible to it. This term is what
  keeps thresholds above the junk. It also self-limits over-strictness: losing
  one colour of a three-colour gene costs an assigned group *and* creates an
  unassigned one.

Thresholds are searched over **absolute** percentiles of each channel's own
candidate distribution, not multiples of the Otsu seed. At the <1% signal
fractions typical here the seed can land inside the noise mode — on a fixture
with spots at 1500 over background 120 it came back at 133 — and a relative
search around such a seed cannot reach the right answer.

Validated on a fixture with `geneA={488,561,638}`, `geneB={488,730}` (so 488
carries 3× the spots of 730) plus per-channel junk:

| channel | junk at | real at | recovered |
|---|---|---|---|
| 488 | 700 | 1500 | 735 |
| 561 | 650 | 1400 | 675 |
| 638 | 260 | 420 | 281 |
| 730 | 600 | 1300 | 1229 |

Score −85 → 90, unassigned 214 → 35, assignment rate 38% → 78%. End-to-end
the pipeline then recovered 118/120 geneA and 55/60 geneB. Note the 2.6×
spread between 638 and 488 — no single density target can produce that.

Cost: chunks must be sampled **co-registered across channels** (all channels
at the same z/y/x) so they can be grouped, which is more I/O than the
per-channel sampler. The optimiser itself is cheap — candidates are extracted
once and every evaluation is post-filtering plus a KD-tree grouping. The
optimiser trace is written to `gene_optimization_trace.csv`.

If a `target_density` is unreachable, the threshold is floored at the
background level (never driven below it) and a warning names `tolerance` as
the binding constraint, with `expected_fp_per_tile` as the fix.

**Cost and sizing.** ~8 Mvoxel (`32×512×512`) per chunk is enough — threshold
is within 6% of a 51-Mvoxel reference, and even 1 Mvoxel is within ~6%. Each
z-slab is read once and several xy chunks are cut from it. A 4×3-chunk, 4-channel
calibration on a 145×2720×2720 tile takes ~100 s on CPU.

**Always check `calibration_report.csv`** — one row per chunk with its estimate,
gate verdict, resulting spot density and rejection reason. A bad calibration is
visible there rather than silent.

## What changed vs. the old package

**`io_utils.py`**

- `load_tile_volume(image_dir, fov, channel)` opens
  `ome_tiff/R0_<fov>.ome.tiff` and reads back only the requested channel's
  pages, so peak memory stays at one channel's z-stack (~2.1 GB for
  145 × 2720 × 2720 uint16) rather than the whole ~10 GB ZCYX file. The page
  index is computed from the file's own axes string, so `ZCYX`, `CZYX` and
  `TCZYX` files all load correctly.
- Channel names come from the OME-XML and are normalised from
  `Fluorescence 488 nm Ex` to `Fluorescence_488_nm_Ex`, so output folder
  names, `extract_wavelength_from_channel` and the gene-filter regex all keep
  working unchanged. If a file carries no channel names, the fallbacks are
  `acquisition_channels.yaml` → `acquisition.yaml` → the hardcoded
  `CHANNEL_INDEX_TO_WAVELENGTH` map.
- `get_squid_tile_channels` discovers tiles from the OME-TIFF filenames and
  channels from each file's metadata (instead of parsing per-slice filenames).
- New helpers: `get_image_dir`, `list_ome_tiles`, `tile_path`,
  `read_ome_tile_info`, `get_tile_shape`, `get_resolution_nm`,
  `load_tile_stage_coords`.
- Per-tile metadata is cached per `(path, mtime, size)`, so repeated tile
  reads don't re-parse the OME-XML.

**`stitching.py`**

- Tile size is read from the OME-TIFFs instead of being hardcoded to
  2720 × 2720 (pass `tile_shape=(h, w)` to override).
- Stage positions come from `load_tile_stage_coords`, which prefers
  `0/coordinates.csv` — the new layout labels each row with its `fov`, so
  tiles are matched by fov rather than by row order — and falls back to the
  positional top-level `coordinates.csv`.

**`corrections.py`**

- The pre-detection correction chain. Four independent steps, each off by
  default and each switched on by its own parameter; when more than one is on
  they are applied in this order:

  | # | Step | Parameter | Notes |
  |---|---|---|---|
  | 1 | Constant offset subtraction | `constant_offset` (counts) | the camera pedestal — taken out before anything that models the sample |
  | 2 | BaSiC flatfield/darkfield | `use_basic` | fit per tile+channel, see `basic_correction.py` |
  | 3 | Rolling-ball background subtraction | `rolling_ball_radius` (px) | 2-D, per z-slice; ImageJ's Subtract Background |
  | 4 | Gaussian blur | `gaussian_blur_sigma` (px) | scalar → per z-slice in xy; `(sz, sy, sx)` tuple → 3-D |

- **The offset is step 1 because steps 2 and 3 remove an additive background of
  their own.** Subtracting a constant *after* BaSiC's darkfield or a rolling ball
  takes the pedestal out twice; with `constant_offset=107` and
  `rolling_ball_radius=50` that floored an entire tile at zero, which made
  calibration return `threshold=0` and detection keep every voxel in the tile
  until it was OOM-killed. The corollary is that a rolling ball downstream
  re-absorbs the constant — it is equivariant under a shift — so
  `constant_offset` + `rolling_ball_radius` with BaSiC off equals the rolling ball
  alone to within float32 rounding. The offset only survives when a flatfield
  division sits between the two.
- The whole chain runs in float32 on one working copy of the tile and is
  rounded/clipped back into the image dtype only at the end, so a background
  subtraction that dips below zero is not truncated mid-chain.
- **Every step changes the intensity scale `threshold` and `tolerance` are
  expressed in.** Turning one on or off invalidates per-channel parameters
  calibrated under the old settings — recalibrate. `calibration.py` runs the
  same chain on its sampled chunks, so auto-calibrated values already refer to
  corrected pixels.
- Steps 3 and 4 are 2-D by default because z is sampled ~6× coarser than xy
  here (1500 nm vs 260 nm), so a z-isotropic rolling ball or blur would mix
  planes that are physically far apart.
- The rolling ball is rolled on a **shrunken** copy of each plane and the
  resulting background interpolated back up, exactly as ImageJ does — a direct
  roll of a radius-50 ball over one 2720 × 2720 plane takes ~27 s, i.e. over an
  hour per 145-slice tile. `rolling_ball_shrink=None` picks ImageJ's factor for
  the radius (≤10 → 1, ≤30 → 2, ≤100 → 4, else 8); set `1` for an exact
  (very slow) roll.
- `gaussian_blur_sigma` is **not** `smooth_sigma`. `smooth_sigma` is internal
  to the detector: it smooths a copy to *locate* peaks, then tests and reports
  raw values. `gaussian_blur_sigma` changes the image itself, so it changes the
  intensities threshold/tolerance are tested against and the values written to
  the output CSVs.
- With `output_corrected=True`, corrected copies are written as one OME-TIFF
  per tile under `<output_folder>/corrected/ome_tiff/`, preserving the original
  channel names and pixel sizes, with **every enabled correction baked in**.
  Channels selected for detection are corrected; the rest are copied through
  bit-identical. Detection then reads those copies, so in-memory and baked runs
  produce identical maxima.
- Those copies are written with axes **`CZYX`** rather than the acquisition's
  `ZCYX`, so each channel can be corrected, streamed to the writer and freed
  one at a time — writing `ZCYX` would require holding the whole ~10 GB tile
  in memory. `load_tile_volume` reads either order transparently. Files are
  written as BigTIFF.
- Note these copies are as large as the originals; leave
  `output_corrected=False` (correct in memory, discard after detection) unless
  you specifically want the corrected images on disk.

**`orchestrator.py`**

- Reads from `<squid_folder>/ome_tiff/` — or, for an old-layout acquisition,
  `<squid_folder>/0/` — and passes the discovered tile shape through to
  stitching. The layout it found is named in the line it prints before
  detection starts.
- `resolution` may now be `None`, in which case the voxel size is read from
  the OME-XML `PhysicalSizeX/Y/Z` (0.26 × 0.26 × 1.5 µm → `[260, 260, 1500]`
  nm on the 25× confocal data). Passing an explicit `[x, y, z]` in nm still
  works and takes precedence. Old-layout acquisitions have no such metadata,
  so `resolution` is required there.

## Module map

| Module | Responsibility |
|---|---|
| `torch_filters.py` | GPU (CUDA / MPS / CPU) separable filter primitives — Gaussian smoothing, max/min filters, Z-chunking for memory-bounded processing |
| `detection.py` | scipy CPU backend + the public `alg_smoothed_local_max_3d()` dispatcher (`backend='auto'/'torch'/'scipy'`) |
| `selftest.py` | scipy-vs-torch parity/speed check — run with `python -m spot_detection_new_squid.selftest` |
| `io_utils.py` | per-tile OME-TIFF loading, channel/wavelength bookkeeping, stage + virtual-stitch coordinate loaders; dispatches to `legacy_io.py` for an old per-slice folder |
| `legacy_io.py` | reader for the old per-(tile, z, channel) TIFF layout, presented to the rest of the package as per-tile OME-TIFFs |
| `basic_correction.py` | per-tile BaSiC flatfield/darkfield fit and its application |
| `corrections.py` | the pre-detection correction chain (constant offset → BaSiC → rolling ball → Gaussian blur), applied in memory or baked to `corrected/ome_tiff/` |
| `stitching.py` | combines per-tile maxima CSVs into per-channel `detected_points.csv` point clouds |
| `chromatic.py` | automatic chromatic z-shift computation between channels (co-localization based) |
| `gene_filter.py` | groups per-channel points into multi-color gene barcodes by cylindrical co-localization (`group_points`, shared with the gene objective); also writes the leftovers to `detected_points_unassigned.csv` |
| `gene_objective.py` | chance-corrected gene-assignment score and the coordinate-ascent optimiser behind `objective='genes'` |
| `calibration.py` | estimates per-channel `threshold`/`tolerance` from the images (see above) |
| `validation.py` | validates `genes.yaml`, `channel_parameters.yaml`, and the chromatic z-shift dict |
| `monitoring.py` | background RSS memory monitor for the whole process tree, written to `memory_log.csv` |
| `tasks.py` | per-(tile, channel) load + detect glue shared by the CPU multiprocessing pool and the GPU serial+prefetch dispatch |
| `orchestrator.py` | `run_pipeline(...)` — the full detect → stitch → chromatic z-shift → gene-filter flow |
| `run_spot_detection.py` | entry point script (see above) |

## Library use

```python
from spot_detection_new_squid.detection import alg_smoothed_local_max_3d

coords = alg_smoothed_local_max_3d(
    volume, footprint_size=10, threshold=800, smooth_sigma=1.0,
    tolerance=150, prominence_footprint_size=9,
    backend='auto', z_chunk=16,   # z_chunk caps GPU memory on large stacks
)
```

```python
from spot_detection_new_squid.orchestrator import run_pipeline

run_pipeline(
    squid_folder='...', channel_parameter_file='...', reference_channels=[405],
    overlap_percentage=10, resolution=None, genes_file='...', radius=750, depth=4500,
    chromatic_ref_channel=561, backend='auto', z_chunk=16,
)
```

Reading a single tile+channel volume:

```python
from spot_detection_new_squid.io_utils import get_image_dir, load_tile_volume

image_dir = get_image_dir('/path/to/0727A_in-tissue_2026-08-04_13-38-54.799045')
vol = load_tile_volume(image_dir, 0, 'Fluorescence_561_nm_Ex')   # (145, 2720, 2720)
```

## `threshold` and `tolerance` are in raw pixel units

Both parameters are compared against the **original image**, so they are
numbers you can find by opening a tile and hovering over a spot:

- **`threshold`** — the raw pixel value at the spot's peak.
- **`tolerance`** — the spot's height above its local background, in raw
  units: raw peak minus the local background level.

Smoothing is still used, but only to *locate* peaks — the accept/reject tests
themselves run on raw data:

```python
v        = gaussian(raw, smooth_sigma)
is_peak  = (maximum_filter(v, footprint_size) == v) & (raw >= threshold)
is_peak &= (raw - minimum_filter(v, prominence_footprint_size)) >= tolerance
```

The background estimate for `tolerance` deliberately comes from the *smoothed*
volume. A min-filter on raw data latches onto the darkest noise voxel in the
window and overstates prominence by ~8%; the smoothed min-filter tracks the
background you would actually read off the image. Measured on a 561 gel spot:
raw peak 405, background 123, so by eye 282 — `raw − minfilter(smoothed)` gives
277, `raw − minfilter(raw)` gives 301.

### This changes the meaning of existing parameter files

Previously both parameters were compared against the smoothed volume, where a
spot's peak sits about 1.36× lower than its raw value. **Old values are not
comparable and will be more permissive than intended if reused as-is.**
Recalibrate (`run_calibration.py`), or convert approximately:

| parameter | old (smoothed) → new (raw) |
|---|---|
| `threshold` | × 1.36 |
| `tolerance` | × 1.58 |

Those factors gave the best agreement with the old detector on the 0615 gel,
but only 73–90% overlap in the selected spot set — so treat them as a starting
point, not an exact conversion. The residual disagreement is inherent, not a
bug: thresholding raw values is a genuinely different test from thresholding
smoothed ones. A raw peak carries pixel noise that smoothing removed, so
borderline spots near the cut will flip in or out. That is the cost of having
parameters you can read off the image; spots comfortably above threshold are
unaffected.

## Unassigned points

Gene filtering writes one CSV per gene into `<output_folder>/genes/`. Every
co-localization group that matched no gene barcode now goes to
`genes/detected_points_unassigned.csv` in the same folder, so the gene files
plus that file account for every detected spot — the counts sum exactly.

Two things land there:

- **singletons** — a spot with no partner in any other channel inside the
  radius/depth cylinder (`n_channels` = 1);
- **channel combinations that are not a defined barcode** — a partial barcode
  missing a colour, or an over-complete one that picked up a spurious extra
  channel.

Columns are `x, y, z`, the same as the per-gene files. Which channels formed
each unassigned group is printed to stdout instead — the five most common
combinations (e.g. `561;638`), plus how many raw detections they account for.
A combination that keeps recurring usually means a barcode is missing from
`genes.yaml`, or that `radius`/`depth` is grouping too aggressively.

Unlike the per-gene files, which write a `-1,-1,-1` sentinel row when a gene
has no matches, this file is written with headers and no rows when everything
was assigned.

## On `footprint_size` and the voxel aspect ratio

Voxels are 260 × 260 × 1500 nm, so an isotropic-in-voxels `footprint_size=10`
is 2.6 × 2.6 × 15 µm — 5.8× more restrictive in z than in xy, in physical
units. That looks alarming, but measurements on the 0615 gel say it is fine,
and the reason is worth recording so it does not get "fixed" unnecessarily.

**Measured spot size** (median over ~160 detected spots, 561 and 488):

| | FWHM | in voxels |
|---|---|---|
| lateral | 1.56 µm | 6 px |
| axial | 6.0 µm | 4 slices |

Real spots are ~3.9× axially elongated in physical units — but z is sampled
5.8× more coarsely, so **in voxels a spot is wider laterally (6) than axially
(4)**. An isotropic voxel footprint is therefore already close to matched, and
10 voxels is ~1.7× the lateral and ~2.5× the axial FWHM, right in the usual
1.5–2× range for non-maximum suppression.

Diffraction theory is misleading here. The NA 1.1 axial PSF is ~1.1 µm FWHM,
under one z-step, which would suggest a z-footprint of ~3. Measured spots are
6 µm FWHM — 5× larger — so they are not diffraction-limited point sources.
Trust the measurement.

**Empirically it makes no difference at current densities.** Shrinking the
z-footprint from 10 → 7 → 5 → 3 changed the spot count by +1 out of 159, with
no z-duplicates. At ~2.3 spots/Mvoxel the expected number of competing spots
inside a 10×10×10 exclusion box is 0.002, so the exclusion almost never binds.
It would start to matter above roughly 50–100 spots/Mvoxel, ~40× the current
density.

**Recommendations:**

1. **Leave `footprint_size=10` alone.** It is well chosen and the anisotropic
   alternative is a no-op on this data.
2. **Leave `prominence_footprint_size=9` alone.** At 9 the measured prominence
   already recovers 96% of a spot's true excess over background, and enlarging
   the window raises the noise floor faster than the signal — the
   signal-to-noise ratio of the prominence statistic *fell* from 13.4 at 9 to
   11.9 at 21.
3. **If spot density rises above ~50/Mvoxel**, revisit with an anisotropic
   footprint of about `(7, 10, 10)` in (z, y, x), matching the measured 4-slice
   axial / 6-px lateral shape. scipy's filters already accept a per-axis tuple;
   `torch_filters.max_filter_torch` / `min_filter_torch` take a scalar `size`
   and would need a small change to loop per-axis. Do not go below 5 in z —
   spots stay above threshold for up to 8 slices, so a small z-footprint risks
   double-counting one spot, and the pipeline has no DBSCAN merge step to
   clean that up.
4. **`smooth_sigma=1.0` is smaller than the matched-filter optimum** (FWHM/2.355
   ≈ 2.5 lateral, ≈ 1.7 axial). Raising it would improve sensitivity to dim
   spots at some cost in separating close ones. This was historically awkward
   because it forces `threshold` to be re-tuned by hand — with `auto_calibrate`
   that re-tuning is automatic, so it is now cheap to try. Recalibrate whenever
   you change it, since thresholds are not comparable across different
   `smooth_sigma`.

## Notes

- Only region `R0` is scanned (`io_utils.REGION`), matching the old package's
  hardcoded `R0_` filename prefix.
- `z_chunk` (torch backend only) processes a volume in Z-slabs instead of all
  at once, capping peak GPU memory on large tiles. On a 150×2720×2720 tile
  with `footprint_size=10`, `z_chunk=16` peaks around 6.3 GB; `z_chunk=32` can
  exceed 10 GB and OOM on a card with ~11 GB.
- The CPU (`backend='scipy'`) path uses a `multiprocessing.Pool` with
  `processes` workers running concurrently, each holding a full copy of the
  volume plus intermediate filter buffers in RAM. On full-size real tiles this
  can be tens of GB with the default `processes=6`; lower it if you hit swap
  thrashing on CPU-only runs.
- `backend` must be the string `'auto'` (or `'torch'`) to use the GPU;
  `backend=None` falls through to the scipy CPU path.
- **BaSiC and the GPU detector do not share the device.** basicpy is
  torch-based, so a fit competes with detection for GPU memory. A fit moves the
  whole tile to the device — 145 × 2720 × 2720 float32 is **4.3 GB** (with the
  default `basic_max_fit_slices=150`, the entire 145-slice stack is used) — on
  top of detection's ~6.3 GB at `z_chunk=16`. That is already over an 11 GB
  card with a single fit, and the GPU path runs the tile loader in
  `io_workers` prefetch threads, so three fits could be in flight at once.
  Two guards: fits are **serialised per process** (`_FIT_LOCK` in
  `basic_correction.py`, verified to hold peak concurrency at 1), and
  `basic_device` falls back to **CPU** whenever the correction runs inline
  during detection. Fits use the GPU only with `output_corrected=True`, where
  they complete before detection starts.
  If CPU fitting becomes the bottleneck, either set `output_corrected=True`
  (GPU fits, no contention, but ~10 GB on disk per tile) or lower
  `basic_max_fit_slices` — at 40 slices a GPU fit needs ~1.2 GB and comfortably
  coexists with the detector.
- **Wrap calls to `run_pipeline` in `if __name__ == '__main__':`.** On the
  CPU/scipy path with `use_basic=True` and `output_corrected=False`, the worker
  pool is *spawned* rather than forked, and spawned workers re-import the
  calling module. Without the guard they re-run the pipeline recursively.
  Both `run_spot_detection.py` and `run_calibration.py` already have it.
  The reason for spawning: basicpy is torch-based, and torch's internal
  thread pools do not survive `fork()`. Once this process has initialised
  them — which `auto_calibrate=True` does — forked workers deadlock on first
  use and the run hangs indefinitely with no error. Confirmed by forcing the
  pool back to fork (hangs) versus spawn (completes in 33 s on the same
  input). Fork is still used when workers do not run BaSiC — the rolling ball,
  blur and offset are plain numpy/scipy and are fork-safe, so they do not force
  a spawn on their own.
- **The rolling ball is the expensive correction.** It runs on the CPU, per
  z-slice, in whichever process loaded the tile. At the default shrink factor a
  2720 × 2720 plane costs ~0.25 s, so ~36 s per 145-slice tile+channel — on the
  GPU path that lands in the `io_workers` prefetch threads and mostly overlaps
  with detection, but it can become the bottleneck. Raise `rolling_ball_shrink`
  if so.
- GPU and CPU backends are numerically identical (see `selftest.py`).
