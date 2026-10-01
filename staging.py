"""Where uploaded tiles live inside a job's staging folder.

    staging/
      coordinates.csv, 0/coordinates.csv, acquisition parameters.json ...   metadata
      incoming/<fov>/...                 a tile still arriving (browser upload
                                         or Globus transfer)
      incoming/<fov>.json                browser uploads: which pieces arrived
      ome_tiff/R0_<fov>.ome.tiff         a complete new-layout tile
      tiles/<fov>/0/R0_<fov>_<z>_<ch>.tiff  a complete old-layout tile, one
                                         folder per tile so it can appear in a
                                         single rename
      <name>.ome.tiff                    a complete single-file upload

A tile is moved out of incoming/ (`finish_tile`) only once every byte of every
one of its files has arrived, so whatever the runner finds outside incoming/
is complete.
"""
from __future__ import annotations

import os
import shutil
import threading
import time

OME = 'ome'
LEGACY = 'legacy'
SINGLE = 'single'

# Written by the Globus transfer manager when it gives up; the pipeline's
# waits check for it so the job fails instead of waiting forever.
ABORT_FILE = '.transfer_failed'

# Sidecar the old-layout reader takes the z step from; it looks for it in the
# folder above the per-slice folder, so each tile folder gets a copy.
ACQ_PARAMS = 'acquisition parameters.json'


def ready_path(staging, layout, fov):
    """What exists once tile `fov` is fully uploaded (folder layouts only)."""
    if layout == LEGACY:
        return os.path.join(staging, 'tiles', str(fov))
    return os.path.join(staging, 'ome_tiff', f'R0_{fov}.ome.tiff')


def image_dir(staging, layout, fov):
    """The image folder to read tile `fov` from: the shared ome_tiff/ folder,
    or the tile's own per-slice folder."""
    if layout == LEGACY:
        return os.path.join(staging, 'tiles', str(fov), '0')
    return os.path.join(staging, 'ome_tiff')


def copy_acq_params(staging, tile_root):
    """Put a copy of the acquisition parameters in an old-layout tile folder
    (no-op if already there or not uploaded). Safe to call concurrently."""
    src = os.path.join(staging, ACQ_PARAMS)
    dst = os.path.join(tile_root, ACQ_PARAMS)
    if os.path.exists(dst) or not os.path.exists(src):
        return
    tmp = f'{dst}.{os.getpid()}.{threading.get_ident()}.tmp'
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst)


def remove_tile(staging, layout, fov):
    path = ready_path(staging, layout, fov)
    if layout == LEGACY:
        # Renamed away first so the folder disappears in one step, rather
        # than being seen half-deleted.
        trash = os.path.join(staging, 'tiles', f'.deleting-{fov}')
        try:
            os.rename(path, trash)
        except FileNotFoundError:
            return
        shutil.rmtree(trash, ignore_errors=True)
    else:
        try:
            os.remove(path)
        except FileNotFoundError:
            pass


def remove_staged_images(staging, manifest):
    """Delete every uploaded image of a job, complete or partial; the small
    metadata files are kept."""
    for sub in ('ome_tiff', 'tiles', 'incoming', 'calibration'):
        shutil.rmtree(os.path.join(staging, sub), ignore_errors=True)
    if manifest.get('layout') == SINGLE:
        for t in manifest['tiles']:
            try:
                os.remove(os.path.join(staging, t['ready']))
            except OSError:
                pass


def assembly_dir(staging, key):
    return os.path.join(staging, 'incoming', str(key))


def finish_tile(staging, layout, tile):
    """Move a fully arrived tile from incoming/ to where the runner looks."""
    assembly = assembly_dir(staging, tile['key'])
    ready = os.path.join(staging, tile['ready'])
    os.makedirs(os.path.dirname(ready), exist_ok=True)
    if layout == LEGACY:
        copy_acq_params(staging, assembly)
        os.rename(assembly, ready)        # the whole tile folder at once
    else:
        os.replace(os.path.join(assembly, tile['files'][0]['dest']), ready)
        shutil.rmtree(assembly, ignore_errors=True)


def upload_window(job, released):
    """Tiles that should be on their way now: the next (1 + prefetch) tiles in
    processing order that the runner has not finished reading (`released`).
    With auto_calibrate, at least all calibration tiles, which are needed at
    the same time."""
    from params import parse_values
    released = set(released)
    m = job['manifest']
    by_key = {t['key']: t for t in m['tiles']}
    pending = [k for k in m['order'] if k not in released]
    window = 1 + job['prefetch_tiles']
    if m['kind'] == 'folder' and job['params'].get('auto_calibrate'):
        cal = parse_values(job['params']).get('calibration_kwargs') or {}
        window = max(window, len(cal.get('tiles') or [None]))
    return [by_key[k] for k in pending[:window]]


def transfer_failed(staging):
    """The transfer manager's fatal error message, or None."""
    try:
        with open(os.path.join(staging, ABORT_FILE)) as f:
            return f.read()
    except OSError:
        return None


def wait_until_exists(path, staging, poll_s=0.5):
    """Block until `path` exists; raise if the transfer manager gave up."""
    while not os.path.exists(path):
        failed = transfer_failed(staging)
        if failed:
            raise RuntimeError(f'Transfer failed: {failed}')
        time.sleep(poll_s)
