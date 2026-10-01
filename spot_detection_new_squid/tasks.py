"""Per-(tile, channel) task glue: load a volume, run detection, write the
maxima CSV. Split into I/O and compute steps so the GPU path (serial compute
with threaded I/O prefetch) and the CPU path (multiprocessing pool, I/O +
compute per worker) can share the same detection code."""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from .corrections import apply_corrections
from .detection import alg_smoothed_local_max_3d, top_k_intensity_stats
from .io_utils import MAXIMA_COLUMNS, load_tile_volume


def load_tile_channel(task):
    """I/O step: load the full (Z, Y, X) volume for one tile+channel.

    If the task carries `correction_kwargs`, the enabled corrections (BaSiC,
    rolling ball, Gaussian blur, constant offset -- see corrections.py) are
    applied right here, right after the volume lands in memory. Only the
    corrected volume is kept, nothing is written to disk, and it is discarded
    once detection on this tile+channel finishes (used when
    output_corrected=False).
    """
    vol = load_tile_volume(task['image_dir'], task['n'], task['channel'], n_zslices=None)
    correction_kwargs = task.get('correction_kwargs')
    if correction_kwargs is not None:
        vol = apply_corrections(vol, **correction_kwargs)
    return task, vol


def detect_and_save(task, vol):
    """
    Compute step: run smoothed_local_max_3d directly in 3D on a preloaded
    volume -- no per-slice 2D detection and no DBSCAN merge step, since the 3D
    filter already finds each spot once -- and write the per-tile-channel CSV.

    A spot's intensity is measured over the 10 brightest voxels of its
    `footprint_size` neighbourhood -- the window it won its peak search in --
    rather than from the peak voxel alone. The CSV reports that set as:

        Median  median of the 10, the intensity to use
        Max     brightest of the 10, i.e. the brightest voxel in the box
        Min     dimmest of the 10, i.e. the 10th brightest in the box
        N       how many voxels those came from: 10, or fewer only for a
                 spot whose box is clipped by the volume edge

    Note Max is the box's brightest RAW voxel, which is not always the voxel at
    the located maximum: peaks are located on the smoothed volume, so the two
    differ for a few percent of spots. It is the located maximum's raw value
    that `threshold` and `tolerance` were tested against, and it is no longer
    recorded.
    """
    coords = alg_smoothed_local_max_3d(
        vol,
        footprint_size=task['footprint_size'],
        threshold=task['threshold'],
        smooth_sigma=task['smooth_sigma'],
        tolerance=task['tolerance'],
        prominence_footprint_size=task['prominence_footprint_size'],
        backend=task['backend'],
        device=task['device'],
        z_chunk=task['z_chunk'],
    )

    if len(coords) == 0:
        df_out = pd.DataFrame(columns=MAXIMA_COLUMNS)
    else:
        zz, yy, xx = coords[:, 0].astype(int), coords[:, 1].astype(int), coords[:, 2].astype(int)
        stats = top_k_intensity_stats(vol, coords, task['footprint_size'])
        df_out = pd.DataFrame({
            'X': xx, 'Y': yy, 'Slice': zz,
            'Median': stats.median, 'Max': stats.maximum, 'Min': stats.minimum,
            'N': stats.n_used,
        })

    df_out.to_csv(task['out_csv'])
    return task['n'], task['channel'], len(df_out)


def find_maxima_3d_tile_channel(task):
    """Load + detect for one tile+channel. Used by the CPU multiprocessing pool."""
    _, vol = load_tile_channel(task)
    return detect_and_save(task, vol)


def prefetch(tasks, loader, max_ahead):
    """Yield loader(task) results in order while keeping up to `max_ahead`
    loads in flight on a thread pool."""
    with ThreadPoolExecutor(max_workers=max_ahead) as pool:
        it = iter(tasks)
        inflight = []
        for _ in range(max_ahead):
            try:
                inflight.append(pool.submit(loader, next(it)))
            except StopIteration:
                break
        while inflight:
            result = inflight.pop(0).result()
            try:
                inflight.append(pool.submit(loader, next(it)))
            except StopIteration:
                pass
            yield result
