"""Stitches per-tile maxima CSVs into per-channel `detected_points.csv` files,
using stage coordinates plus either a uniform overlap crop or explicit
virtual-stitch crop offsets.

New-layout notes
----------------
* Stage positions come from `io_utils.load_tile_stage_coords`, which prefers
  the new `0/coordinates.csv` (explicitly labelled with each row's ``fov``)
  over the top-level `coordinates.csv` (positional, one row per fov).
* The tile size is read from the acquisition's OME-TIFFs instead of being
  hardcoded to 2720x2720; pass `tile_shape=(height, width)` to override.
"""
from __future__ import annotations

import os
import re

import pandas as pd

from .io_utils import get_tile_shape, load_tile_stage_coords


def stitch_squid_tile_csvs(squid_folder, output_folder, overlap_percent, resolution,
                            virtual_stitch_coords=None, tile_shape=None):
    pattern = re.compile(r'^tile_(.+)$')
    folders, tile_nums = [], []

    for fname in os.listdir(output_folder):
        if pattern.match(fname):
            tile_nums.append(int(fname.split('_')[1]))
            folders.append(fname)

    coords = load_tile_stage_coords(squid_folder)
    coords['x (mm)'] -= coords.loc[0, 'x (mm)']
    coords['y (mm)'] -= coords.loc[0, 'y (mm)']
    coords = coords.set_index('fov')

    if tile_shape is None:
        tile_shape = get_tile_shape(squid_folder)
    tile_y, tile_x = int(tile_shape[0]), int(tile_shape[1])
    print(f'Tile size: {tile_x} x {tile_y} px')

    overlap_x_px = int(tile_x * overlap_percent / 100.0)
    overlap_y_px = int(tile_y * overlap_percent / 100.0)
    crop_left = overlap_x_px // 2
    crop_top = overlap_y_px // 2

    default_ex_min = crop_left
    default_ex_max = tile_x - (overlap_x_px - crop_left)
    default_ey_min = crop_top
    default_ey_max = tile_y - (overlap_y_px - crop_top)
    step_x = tile_x - overlap_x_px
    step_y = tile_y - overlap_y_px

    coords['x_px'] = coords['x (mm)'] * 1e6 / resolution[0]
    coords['y_px'] = coords['y (mm)'] * 1e6 / resolution[1]

    chunks = []
    bad_ones = []
    min_tile = min(tile_nums)

    print('Concatenating all points...')
    for folder, tile_num in zip(folders, tile_nums):
        print(f'Tile {tile_num}')
        tile_idx = tile_num - min_tile
        if tile_num in coords.index:
            coord_row = coords.loc[tile_num]
        else:
            # Coordinates were positional (no fov column); fall back to order.
            coord_row = coords.iloc[tile_idx]

        for channel_folder in os.listdir(os.path.join(output_folder, folder)):
            print(f'    Channel {channel_folder}')

            if virtual_stitch_coords is not None:
                tile_ch_map = virtual_stitch_coords[tile_idx]
                if channel_folder in tile_ch_map:
                    vc = tile_ch_map[channel_folder]
                    ex_min = vc[0]
                    ex_max = step_x + vc[0]
                    ey_min = vc[1]
                    ey_max = step_y + vc[1]
                    ez_min = vc[2]
                else:
                    print(f'      Warning: channel {channel_folder} not found in '
                          f'virtual stitch coords for tile {tile_num}, using default crop.')
                    ex_min, ex_max = default_ex_min, default_ex_max
                    ey_min, ey_max = default_ey_min, default_ey_max
                    ez_min = 0
            else:
                ex_min, ex_max = default_ex_min, default_ex_max
                ey_min, ey_max = default_ey_min, default_ey_max
                ez_min = 0

            try:
                cur_pts = pd.read_csv(
                    os.path.join(output_folder, folder, channel_folder, 'maxima_results.csv'),
                    index_col=0
                )
                passed = cur_pts[
                    (cur_pts['X'] >= ex_min) & (cur_pts['X'] < ex_max) &
                    (cur_pts['Y'] >= ey_min) & (cur_pts['Y'] < ey_max) &
                    (cur_pts['Slice'] >= ez_min)
                ]
                chunk = pd.DataFrame({
                    'x': ((passed['X'] - ex_min) + coord_row['x_px']) * resolution[0],
                    'y': ((passed['Y'] - ey_min) + coord_row['y_px']) * resolution[1],
                    'z': (passed['Slice'] - ez_min) * resolution[2],
                    'c': channel_folder,
                })
                chunks.append(chunk)
            except KeyError:
                bad_ones.append(tile_num)

    all_pts = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=['x', 'y', 'z', 'c'])

    print('Outputting to CSVs...')
    for c in all_pts['c'].unique():
        print(f'Channel {c}')
        out_dir = os.path.join(output_folder, c)
        os.makedirs(out_dir, exist_ok=True)
        all_pts[all_pts['c'] == c][['x', 'y', 'z']].to_csv(
            os.path.join(out_dir, 'detected_points.csv'), index=False
        )

    if bad_ones:
        print(f'Tiles needing attention: {bad_ones}')
