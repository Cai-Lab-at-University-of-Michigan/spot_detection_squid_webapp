"""Automatic chromatic z-shift computation: finds per-channel z-offsets
relative to a reference channel by co-localizing spots, and saves the
result in the same JSON/CSV format used by the virtual-stitch coords."""
from __future__ import annotations

import json
import os
import re

import numpy as np
import pandas as pd
from scipy.spatial import KDTree

from .io_utils import WL_INT_TO_CH_IDX


def compute_chromatic_z_shift_auto(input_dir, ref_channel_wl, radius_nm, depth_nm, resolution):
    """
    Compute per-channel chromatic z-shifts relative to ref_channel_wl.

    Uses the same cylindrical search as gene_filter.filter_genes: for each
    reference channel spot, finds co-localized spots in every other channel
    within a cylinder (XY radius = radius_nm, Z half-extent = depth_nm) and
    records the z-offset (ref_z - ch_z). The median over all matched pairs
    gives the shift to apply to that channel to bring it into z-alignment
    with the reference.

    Returns
    -------
    z_shifts_px : dict {channel_wl_int: z_shift_px}
        Integer pixels to ADD to each channel's z to align with reference.
        Reference channel is always 0.
    stats : dict {channel_wl_int: dict}
        Per-channel diagnostic statistics (n, median_nm, std_nm, etc.).
    """
    pattern = re.compile(r'^Fluorescence_(.+)_nm_Ex$')

    channel_coords = {}
    for fname in sorted(os.listdir(input_dir)):
        m = pattern.match(fname)
        if m:
            wl = int(m.group(1))
            csv_path = os.path.join(input_dir, fname, 'detected_points.csv')
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                channel_coords[wl] = df[['x', 'y', 'z']].to_numpy(dtype=float)

    if ref_channel_wl not in channel_coords:
        raise ValueError(
            f'Reference channel {ref_channel_wl} not found in detected channels: '
            f'{sorted(channel_coords.keys())}. '
            f'Ensure it is not listed in reference_channels (which excludes it from detection).'
        )

    ref_coords = channel_coords[ref_channel_wl]
    print(f'Computing z-shifts relative to channel {ref_channel_wl} nm '
          f'({len(ref_coords)} reference spots across {len(channel_coords) - 1} other channel(s))')

    z_shifts_px = {ref_channel_wl: 0}
    stats = {}

    for wl, ch_coords in sorted(channel_coords.items()):
        if wl == ref_channel_wl:
            continue

        ch_tree = KDTree(ch_coords)
        z_diffs_nm = []

        for ref_pt in ref_coords:
            # Sphere query with radius=depth_nm bounds the Z extent of the cylinder.
            candidates = ch_tree.query_ball_point(ref_pt, r=depth_nm)
            for ci in candidates:
                ch_pt = ch_coords[ci]
                # Enforce cylindrical XY constraint.
                xy_dist2 = (ch_pt[0] - ref_pt[0]) ** 2 + (ch_pt[1] - ref_pt[1]) ** 2
                if xy_dist2 < radius_nm ** 2:
                    # Positive value: reference is higher in z than this channel spot.
                    z_diffs_nm.append(float(ref_pt[2] - ch_pt[2]))

        if len(z_diffs_nm) == 0:
            print(f'  Channel {wl}: WARNING — no co-localized spots found vs ref {ref_channel_wl}, '
                  f'z-shift set to 0')
            z_shifts_px[wl] = 0
            stats[wl] = {'n': 0, 'median_px': 0.0, 'median_nm': 0.0, 'std_px': 0.0, 'std_nm': 0.0}
        else:
            arr = np.array(z_diffs_nm)
            median_nm = float(np.median(arr))
            std_nm = float(np.std(arr))
            z_shift_px = int(round(median_nm / resolution[2]))
            z_shifts_px[wl] = z_shift_px
            stats[wl] = {
                'n': len(arr),
                'median_px': median_nm / resolution[2],
                'median_nm': median_nm,
                'std_px': std_nm / resolution[2],
                'std_nm': std_nm,
            }
            print(f'  Channel {wl}: {len(arr)} matched pairs  |  '
                  f'z-shift = {z_shift_px} px  '
                  f'(median {median_nm:.1f} nm, std {std_nm:.1f} nm / {std_nm / resolution[2]:.2f} px)')

    return z_shifts_px, stats


def save_chromatic_z_shift_file(z_shifts_px, stats, ref_channel_wl, output_dir, resolution):
    """
    Save computed chromatic z-shifts in two formats:

    1. chromatic_z_shift.json — mirrors the virtual stitch coords JSON structure.
       Each channel appears as one tile entry at grid position (0,0,0).
       crop_offset.z holds the integer pixel z-shift for that channel.
       crop_offset.x and crop_offset.y are always 0 (shifts are z-only).

    2. chromatic_z_shift.csv — human-readable summary table with per-channel
       shift and matching statistics.
    """
    tiles = []
    for wl in sorted(z_shifts_px.keys()):
        shift_px = z_shifts_px[wl]
        ch_idx = WL_INT_TO_CH_IDX.get(wl, -1)
        if ch_idx == -1:
            print(f'  Warning: wavelength {wl} not in CHANNEL_INDEX_TO_WAVELENGTH; '
                  f'channel index will be -1 in the z-shift JSON.')
        tiles.append({
            'grid_index': {'tx': 0, 'ty': 0, 'tz': 0, 'channel': ch_idx},
            'crop_offset': {'x': 0, 'y': 0, 'z': shift_px},
        })

    out_json = {
        'description': (
            f'Chromatic z-shifts computed automatically from co-localized spots. '
            f'Reference channel: {ref_channel_wl} nm (z-shift = 0). '
            f'crop_offset.z is the integer pixel shift to ADD to each channel\'s z '
            f'so it aligns with the reference channel. '
            f'crop_offset.x and crop_offset.y are 0 (no lateral chromatic shift corrected).'
        ),
        'reference_channel_wl': ref_channel_wl,
        'resolution_nm': resolution,
        'tiles': tiles,
    }

    json_path = os.path.join(output_dir, 'chromatic_z_shift.json')
    with open(json_path, 'w') as f:
        json.dump(out_json, f, indent=2)
    print(f'Chromatic z-shift JSON saved to {json_path}')

    # CSV summary
    rows = []
    for wl in sorted(z_shifts_px.keys()):
        shift_px = z_shifts_px[wl]
        s = stats.get(wl, {})
        rows.append({
            'channel_wl': wl,
            'is_reference': wl == ref_channel_wl,
            'z_shift_px': shift_px,
            'z_shift_nm': round(shift_px * resolution[2], 1),
            'median_nm': round(s.get('median_nm', 0.0), 1),
            'std_nm': round(s.get('std_nm', 0.0), 1),
            'std_px': round(s.get('std_px', 0.0), 3),
            'n_pairs': s.get('n', 0),
        })

    csv_path = os.path.join(output_dir, 'chromatic_z_shift.csv')
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f'Chromatic z-shift CSV saved to {csv_path}')
