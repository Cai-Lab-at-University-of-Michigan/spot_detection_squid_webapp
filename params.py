"""Parameter schema for the web form, plus parsing of submitted values.

Every entry mirrors one variable in the USER PARAMETERS block of the vendored
spot_detection_new_squid/run_spot_detection_localfdr.py (inside this folder). Defaults are read
straight out of that script at server start (see `script_defaults`), so editing
the script's values changes what the form starts with; the `default` below is
only the fallback for a value the script does not set or cannot be read.

Field kinds
-----------
dir       remote directory path; blank -> None
file      remote file path, or a small local file uploaded from the browser;
          blank -> None
bool      checkbox
number    int/float; blank -> None when `nullable`
choice    one of `choices`
literal   a Python literal (list, tuple, dict, number, None) parsed with
          ast.literal_eval; blank -> None
"""
from __future__ import annotations

import ast
import os

# The pipeline package is vendored inside this folder so the web app is
# self-contained; it is a copy of ../spot_detection_new_squid.
PACKAGE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'spot_detection_new_squid')
DEFAULTS_SCRIPT = os.path.join(PACKAGE_DIR, 'run_spot_detection_localfdr.py')

# (name, kind, group, default, help, extra)
_FIELDS = [
    # squid_folder is chosen in the "Input data" panel, not the generic form.
    ('output_folder', 'dir', 'Output', None,
     'Folder on the remote machine to write results to. Blank: inside the squid '
     'folder for remote data, or this job\'s folder for uploaded data.', {}),

    ('channel_parameter_file', 'file', 'Detection', None,
     'YAML with per-wavelength threshold/tolerance. Channel names must match the '
     'squid ones (488, not 1/2/3). Ignored when auto_calibrate is on.', {}),
    ('reference_channels', 'literal', 'Detection', [405],
     'Reference channels to skip, e.g. [405]. [] detects every channel.', {}),
    ('footprint_size', 'number', 'Detection', 10, 'Peak-search neighbourhood (px).', {}),
    ('smooth_sigma', 'number', 'Detection', 0,
     'Smoothing used only to locate peaks; does not change reported intensities.', {}),
    ('prominence_footprint_size', 'number', 'Detection', 9, '', {}),
    ('single_tiff_channels_by_index', 'bool', 'Detection', False,
     'Single .ome.tiff input only: name channels by C index (0 -> 405, 1 -> 488, ...) '
     'instead of trusting the names in the file.', {}),

    ('auto_calibrate', 'bool', 'Calibration', False,
     'Estimate threshold/tolerance per channel by local-FDR calibration instead of '
     'reading channel_parameter_file.', {}),
    ('calibration_kwargs', 'literal', 'Calibration', None,
     "Overrides for the calibrator, e.g. {'q': 0.05} or {'tiles': [3, 5]}.", {}),

    ('resolution', 'literal', 'Stitching & genes', [260, 260, 1500],
     'Voxel size in nm as [x, y, z]. Blank reads it from the OME-TIFF metadata.', {}),
    ('overlap_percentage', 'number', 'Stitching & genes', 0,
     'Tile overlap set in the squid software.', {}),
    ('virtual_stitch_coords_path', 'file', 'Stitching & genes', None,
     '.json from SAVAII virtual stitching. Blank: uniform overlap_percentage crop.', {}),
    ('genes_file', 'file', 'Stitching & genes', None,
     'YAML with the gene barcode specifications. Blank skips gene filtering.', {}),
    ('radius', 'number', 'Stitching & genes', 940,
     'Cylindrical radius (nm) for grouping a spot across channels.', {}),
    ('depth', 'number', 'Stitching & genes', 4500,
     'Depth (nm) for grouping a spot across channels.', {}),

    ('chromatic_ref_channel', 'number', 'Chromatic', None,
     'Wavelength to compute chromatic z-shifts against. Blank disables.', {'nullable': True}),
    ('chromatic_z_shift_px', 'literal', 'Chromatic', None,
     'Manual per-channel z-shifts, e.g. {488: 0, 561: 1}. Used only when '
     'chromatic_ref_channel is blank.', {}),

    ('backend', 'choice', 'Compute', 'auto',
     "'auto' uses the GPU when one is available; 'scipy' forces CPU.",
     {'choices': ['auto', 'torch', 'scipy']}),
    ('z_chunk', 'number', 'Compute', 16,
     'z-slices per GPU chunk. ~6.3 GB peak at 16 on a 150x2720x2720 tile.', {'nullable': True}),
    ('io_workers', 'number', 'Compute', 3, 'Volume-loading threads on the GPU path.', {}),
    ('processes', 'number', 'Compute', 6, 'Worker processes on the CPU path.', {}),

    ('use_basic', 'bool', 'Corrections', False,
     'Fit a BaSiC flatfield per tile and correct before detection.', {}),
    ('basic_get_darkfield', 'bool', 'Corrections', False, 'Also estimate a darkfield.', {}),
    ('basic_smoothness_flatfield', 'number', 'Corrections', None, 'Blank = auto.',
     {'nullable': True}),
    ('basic_smoothness_darkfield', 'number', 'Corrections', None, 'Blank = auto.',
     {'nullable': True}),
    ('basic_max_fit_slices', 'number', 'Corrections', 150,
     'Cap on z-slices used to fit each tile\'s BaSiC model.', {}),
    ('rolling_ball_radius', 'number', 'Corrections', 0,
     'Rolling-ball background radius (px). 0 disables.', {'nullable': True}),
    ('rolling_ball_shrink', 'number', 'Corrections', 0,
     'Rolling-ball downsample factor. 0/blank = auto.', {'nullable': True}),
    ('gaussian_blur_sigma', 'literal', 'Corrections', 0,
     'Blur applied to the image itself: a number (xy per slice) or (z, y, x). 0 disables.', {}),
    ('constant_offset', 'number', 'Corrections', None,
     'Counts subtracted from every voxel. Blank disables.', {'nullable': True}),
    ('output_corrected', 'bool', 'Corrections', False,
     'Also write full-size corrected OME-TIFFs. Not available for uploaded data '
     '(it would store a corrected copy of every tile on the remote machine).', {}),
]

# run_pipeline parameters the form does not expose: `device` is replaced by the
# GPU picker (which sets CUDA_VISIBLE_DEVICES for the job).
HIDDEN_PARAMS = {'device': None}


def script_defaults(path=DEFAULTS_SCRIPT):
    """{name: value} for every literal assignment in the script's
    `if __name__ == '__main__':` block. Non-literal values are skipped."""
    try:
        with open(path) as f:
            tree = ast.parse(f.read())
    except (OSError, SyntaxError):
        return {}
    out = {}
    for node in tree.body:
        if not (isinstance(node, ast.If) and '__main__' in ast.unparse(node.test)):
            continue
        for stmt in node.body:
            if (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1
                    and isinstance(stmt.targets[0], ast.Name)):
                try:
                    out[stmt.targets[0].id] = ast.literal_eval(stmt.value)
                except ValueError:
                    pass
    return out


def _to_text(kind, value):
    """Form representation of a default value."""
    if kind == 'bool':
        return bool(value)
    if value is None:
        return ''
    if kind == 'literal':
        return repr(value)
    return value


def schema():
    """List of field dicts for the frontend, defaults taken from the script."""
    from_script = script_defaults()
    fields = []
    for name, kind, group, default, help_text, extra in _FIELDS:
        value = from_script.get(name, default)
        # Paths in the script point at this machine only if they exist here;
        # stale paths from another computer would just fail validation.
        if kind in ('dir', 'file') and value and not os.path.exists(value):
            value = None
        fields.append({
            'name': name, 'kind': kind, 'group': group, 'help': help_text,
            'default': _to_text(kind, value), **extra,
        })
    return fields


def parse_values(raw):
    """Convert submitted form values to run_pipeline kwargs. Raises ValueError
    naming the offending field."""
    out = dict(HIDDEN_PARAMS)
    for name, kind, _group, _default, _help, extra in _FIELDS:
        value = raw.get(name)
        try:
            out[name] = _parse_one(kind, value, extra)
        except (ValueError, SyntaxError) as e:
            raise ValueError(f'{name}: {e}') from None
    return out


def _parse_one(kind, value, extra):
    if kind == 'bool':
        return bool(value)
    if isinstance(value, str):
        value = value.strip()
    if value is None or value == '' or value == 'None':
        if kind == 'number' and not extra.get('nullable'):
            raise ValueError('a number is required')
        return None
    if kind in ('dir', 'file'):
        path = os.path.expanduser(str(value))
        if not os.path.exists(path) and kind == 'file':
            raise ValueError(f'{path} does not exist on the remote machine')
        return path
    if kind == 'number':
        if isinstance(value, (int, float)):
            return value
        num = float(value)
        return int(num) if num.is_integer() and '.' not in str(value) else num
    if kind == 'choice':
        if value not in extra['choices']:
            raise ValueError(f'must be one of {extra["choices"]}')
        return value
    if kind == 'literal':
        return ast.literal_eval(value) if isinstance(value, str) else value
    raise ValueError(f'unknown field kind {kind}')
