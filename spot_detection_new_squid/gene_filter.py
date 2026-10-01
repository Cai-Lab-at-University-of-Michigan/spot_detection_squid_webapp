"""Groups per-channel detected points into multi-color gene barcodes by
cylindrical co-localization, and writes one CSV per gene."""
from __future__ import annotations

import os
import re

import numpy as np
import pandas as pd
import yaml
from scipy.spatial import KDTree
from tqdm import tqdm


def group_points(coords_arr, channels_arr, radius, depth, progress=False):
    """Group detected points into cross-channel co-localization clusters.

    Walks the points in order; for each unvisited point, claims every unvisited
    point of a *different* channel inside a cylinder around it (XY radius
    `radius`, Z half-extent `depth`, both in nm) and emits one row for the
    group. `key` is the frozenset of channels it spans -- what gene assignment
    matches against.

    Returns a DataFrame with columns key, n_spots, x, y, z.

    Shared by `filter_genes` and by the gene-objective calibration in
    calibration.py, so calibration optimises against exactly the grouping the
    pipeline will later perform rather than an approximation of it.
    """
    n = len(coords_arr)
    if n == 0:
        return pd.DataFrame(columns=['key', 'n_spots', 'x', 'y', 'z'])

    visited = np.zeros(n, dtype=bool)
    rows = []
    tree = KDTree(coords_arr)

    it = range(n)
    if progress:
        it = tqdm(it, desc='Filtering points')

    for idx in it:
        if visited[idx]:
            continue
        visited[idx] = True

        pt_xyz = coords_arr[idx]
        pt_c = channels_arr[idx]

        candidate_idxs = tree.query_ball_point(pt_xyz, r=depth)

        neighbor_idxs = [
            i for i in candidate_idxs
            if not visited[i]
            and channels_arr[i] != pt_c
            and (coords_arr[i, 0] - pt_xyz[0]) ** 2 + (coords_arr[i, 1] - pt_xyz[1]) ** 2 < radius ** 2
        ]

        visited[neighbor_idxs] = True

        if len(neighbor_idxs) == 0:
            rows.append({'key': frozenset({int(pt_c)}), 'n_spots': 1,
                         'x': pt_xyz[0], 'y': pt_xyz[1], 'z': pt_xyz[2]})
        else:
            neighbor_arr = coords_arr[neighbor_idxs]
            key = frozenset(channels_arr[neighbor_idxs].tolist() + [int(pt_c)])
            all_coords = np.vstack([neighbor_arr, pt_xyz])
            rows.append({
                'key': key,
                'n_spots': len(all_coords),
                'x': np.mean(all_coords[:, 0]),
                'y': np.mean(all_coords[:, 1]),
                'z': np.mean(all_coords[:, 2]),
            })

    return pd.DataFrame(rows, columns=['key', 'n_spots', 'x', 'y', 'z'])


def write_unassigned_points(grouped, genes_data, gene_dir):
    """Write every co-localization group that matched no gene barcode to
    `detected_points_unassigned.csv`.

    Each grouped point goes to exactly one gene file or to this one, so the
    gene files plus this file account for every detected spot. Two things end
    up here:

      * singletons -- a spot with no partner in any other channel within the
        radius/depth cylinder (channel count 1), and
      * channel combinations that are not a defined barcode, e.g. a partial
        barcode missing one colour, or an over-complete one that picked up a
        spurious extra channel.

    The file holds x/y/z only, matching the per-gene files. Which channels
    formed each group is reported on stdout instead -- a combination that
    keeps recurring usually means a barcode is missing from genes.yaml, or
    that radius/depth is grouping too aggressively.
    """
    out_csv = os.path.join(gene_dir, 'detected_points_unassigned.csv')
    if os.path.exists(out_csv):
        os.remove(out_csv)

    gene_keys = {frozenset(channels) for channels in genes_data.values()}
    if len(grouped) == 0:
        unassigned = grouped
    else:
        unassigned = grouped[~grouped['key'].map(lambda k: k in gene_keys)]

    out = pd.DataFrame({
        'x': unassigned['x'].to_numpy(dtype=float),
        'y': unassigned['y'].to_numpy(dtype=float),
        'z': unassigned['z'].to_numpy(dtype=float),
    })
    out.to_csv(out_csv, index=False)

    n_total = len(grouped)
    n_un = len(out)
    pct = (100.0 * n_un / n_total) if n_total else 0.0
    print(f'Unassigned points: {n_un} of {n_total} groups ({pct:.1f}%) matched no gene '
          f'-> {out_csv}')
    if n_un:
        combos = pd.Series([';'.join(str(c) for c in sorted(k)) for k in unassigned['key']])
        n_spots = int(unassigned['n_spots'].sum())
        print(f'  ({n_un} group(s) covering {n_spots} detected spot(s))')
        print('  most common unassigned channel combinations:')
        for combo, n in combos.value_counts().head(5).items():
            print(f'    {combo:<24} {n}')
    return out


def filter_genes(input_dir, genes_path, radius, depth, resolution, chromatic_z_shift_px=None):
    pattern = re.compile(r'^Fluorescence_(.+)_nm_Ex$')
    folders = []
    channel_chunks = []

    for fname in os.listdir(input_dir):
        m = pattern.match(fname)
        if m:
            channel = m.group(1)
            folders.append(fname)
            if not os.path.exists(os.path.join(input_dir, fname, 'detected_points.csv')):
                raise Exception(
                    'Ensure all channel directories have been processed, '
                    'at least one is missing "detected_points.csv" file'
                )
            tmp_df = pd.read_csv(os.path.join(input_dir, fname, 'detected_points.csv'))
            tmp_df['c'] = int(channel)
            if chromatic_z_shift_px is not None:
                wl = int(channel)
                if wl in chromatic_z_shift_px:
                    shift_nm = chromatic_z_shift_px[wl] * resolution[2]
                    tmp_df['z'] = tmp_df['z'] + shift_nm
                    print(f'  Applying chromatic z-shift to channel {wl}: '
                          f'{chromatic_z_shift_px[wl]} px ({shift_nm} nm)')
                else:
                    print(f'  Warning: no chromatic_z_shift_px entry for channel {wl}, leaving z unshifted.')
            channel_chunks.append(tmp_df)

    print(f'Found directories for {len(folders)} channels')
    full_list = pd.concat(channel_chunks, ignore_index=True)

    with open(genes_path, 'r') as file:
        genes_data = yaml.safe_load(file)

    grouped = group_points(
        full_list[['x', 'y', 'z']].to_numpy(dtype=float),
        full_list['c'].to_numpy(),
        radius, depth, progress=True,
    )

    gene_dir = os.path.join(input_dir, 'genes')
    os.makedirs(gene_dir, exist_ok=True)

    for key in genes_data:
        print(f'Looking for points in gene: {key}')
        out_csv = os.path.join(gene_dir, 'detected_points_' + key + '.csv')
        if os.path.exists(out_csv):
            os.remove(out_csv)
        gene_key = frozenset(genes_data[key])
        gene_pts = grouped[grouped['key'] == gene_key]
        if len(gene_pts) == 0:
            pd.DataFrame({'x': [-1], 'y': [-1], 'z': [-1]}).to_csv(out_csv, index=False)
            continue
        gene_pts[['x', 'y', 'z']].to_csv(out_csv, index=False)

    write_unassigned_points(grouped, genes_data, gene_dir)
