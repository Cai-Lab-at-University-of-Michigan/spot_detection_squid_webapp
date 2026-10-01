"""Web front end for the SQUID spot-detection pipeline.

Run on the GPU machine, then open it in a browser on your own computer through
an SSH tunnel (VS Code Remote-SSH forwards the port for you):

    ./start.sh                      # or: python server.py --port 8765

It listens on 127.0.0.1 only, and every request needs the access token printed
at startup (kept in <jobs-dir>/.token so the URL survives restarts).

Each job runs in its own process (runner.py), so closing the browser does not
stop a job on remote data. Jobs on uploaded data need the browser tab open,
since the browser is what sends the tiles.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import secrets
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import zipfile

import psutil
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from starlette.background import BackgroundTask
from starlette.concurrency import run_in_threadpool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import params as params_mod  # noqa: E402
from jobstatus import read_status  # noqa: E402
from globus_link import GLOBUS_BIN, GlobusError, NotLoggedIn  # noqa: E402
from globus_link import client as globus_client  # noqa: E402
from staging import finish_tile, remove_staged_images, upload_window  # noqa: E402

# Size of one upload request; override with SPOTDET_CHUNK_MB.
CHUNK_SIZE = int(os.environ.get('SPOTDET_CHUNK_MB', '32')) * 1024 * 1024
MAX_SMALL_UPLOAD = 200 * 1024 * 1024
SIDECAR_NAMES = ['coordinates.csv', '0/coordinates.csv', 'acquisition parameters.json',
                 'acquisition.yaml', 'acquisition_channels.yaml']
# Tile files as the browser reports them, relative to the chosen folder.
TILE_RE = re.compile(r'^(?:ome_tiff/)?R0_(\d+)\.ome\.tiff?$', re.IGNORECASE)
LEGACY_SLICE_RE = re.compile(r'^0/R0_(\d+)_(\d+)_(.+)\.tiff?$', re.IGNORECASE)

app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
CONFIG = {'jobs_dir': None, 'token': None}
PROCS: dict[str, subprocess.Popen] = {}
UPLOAD_LOCK = threading.Lock()


# ── auth ───────────────────────────────────────────────────────────────────────
@app.middleware('http')
async def require_token(request: Request, call_next):
    token = CONFIG['token']
    given = request.query_params.get('token')
    if given is not None and secrets.compare_digest(given, token):
        # First visit via the printed URL: remember the token in a cookie and
        # drop it from the address bar.
        resp = RedirectResponse(request.url.path or '/')
        resp.set_cookie('sdtoken', token, httponly=True, samesite='strict', max_age=30 * 86400)
        return resp
    if not secrets.compare_digest(request.cookies.get('sdtoken', ''), token):
        return JSONResponse({'detail': 'Missing or wrong token. Open the URL printed by '
                                       'the server (it ends in ?token=...).'}, status_code=401)
    return await call_next(request)


# ── helpers ────────────────────────────────────────────────────────────────────
def jobs_dir():
    return CONFIG['jobs_dir']


def job_path(job_id, *parts):
    if not re.fullmatch(r'[0-9A-Za-z_-]+', job_id):
        raise HTTPException(404, 'no such job')
    return os.path.join(jobs_dir(), job_id, *parts)


def _upgrade(job):
    """Read a job saved by the first version, whose manifest had exactly one
    file per tile and no `layout`."""
    m = job.get('manifest')
    if m and 'layout' not in m:
        m['layout'] = 'single' if m['kind'] == 'single' else 'ome'
        for t in m['tiles']:
            if 'files' not in t:
                t['files'] = [{'relpath': t.pop('relpath'), 'size': t['size'],
                               'dest': os.path.basename(t['dest'])}]
                t['ready'] = t.pop('dest')
    return job


def load_job(job_id):
    try:
        with open(job_path(job_id, 'job.json')) as f:
            return _upgrade(json.load(f))
    except OSError:
        raise HTTPException(404, 'no such job') from None


def save_job(job):
    path = job_path(job['id'], 'job.json')
    with open(f'{path}.tmp', 'w') as f:
        json.dump(job, f, indent=1)
    os.replace(f'{path}.tmp', path)


def runner_alive(job):
    proc = PROCS.get(job['id'])
    if proc is not None:
        return proc.poll() is None
    # Started by an earlier server instance: check the pid still is our runner.
    pid = job.get('pid')
    if not pid:
        return False
    try:
        p = psutil.Process(pid)
        return (p.status() != psutil.STATUS_ZOMBIE
                and any(job['id'] in arg for arg in p.cmdline()))
    except psutil.Error:
        return False


def job_state(job, full=False):
    """Job plus its live state, as the frontend sees it. The per-file
    manifest (thousands of entries for an old-layout folder) is only included
    with full=True, which the browser asks for when it starts uploading."""
    status = read_status(job_path(job['id'], 'status.json')) or {}
    state = status.get('state', 'starting')
    if job.get('cancelled'):
        state = 'cancelled'
    elif state in ('starting', 'running') and not runner_alive(job):
        state = 'died'
    out = {k: v for k, v in job.items() if k != 'manifest'}
    out.update(state=state, phase=status.get('phase'), error=status.get('error'),
               waiting_for=status.get('waiting_for', []),
               released=len(status.get('released', [])),
               detected=status.get('detected', {}),
               started=status.get('started'), updated=status.get('updated'),
               transfers=status.get('transfers') or {}, globus_note=status.get('globus_note'),
               transferred=len(status.get('transferred') or []))
    if 'manifest' in job:
        m = job['manifest']
        out['manifest'] = {'kind': m['kind'], 'layout': m['layout'], 'root_name': m['root_name'],
                           'n_tiles': len(m['tiles']),
                           'n_files': sum(len(t['files']) for t in m['tiles']),
                           'total_bytes': sum(t['size'] for t in m['tiles']),
                           'sidecars': m['sidecars']}
        if full:
            out['manifest']['tiles'] = [
                {'key': t['key'], 'files': [{'relpath': f['relpath'], 'size': f['size']}
                                            for f in t['files']]}
                for t in m['tiles']]
        out['sidecars_done'] = os.path.exists(job_path(job['id'], 'staging', '.sidecars_done'))
    return out


def gpu_list():
    try:
        out = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,name,memory.used,memory.total,utilization.gpu',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=10, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    gpus = []
    for line in out.strip().splitlines():
        idx, name, used, total, util = [s.strip() for s in line.split(',')]
        gpus.append({'index': int(idx), 'name': name, 'mem_used_mb': int(used),
                     'mem_total_mb': int(total), 'util': int(util)})
    return gpus


# ── static page, schema, machine info ─────────────────────────────────────────
@app.get('/')
def index():
    return FileResponse(os.path.join(HERE, 'static', 'index.html'))


@app.get('/api/schema')
def get_schema():
    return {'fields': params_mod.schema(), 'chunk_size': CHUNK_SIZE,
            'sidecar_names': SIDECAR_NAMES, 'home': os.path.expanduser('~')}


@app.get('/api/gpus')
def get_gpus():
    gpus = gpu_list()
    busy = {}
    for job in list_jobs_raw():
        st = job_state(job)
        if st['state'] in ('starting', 'running') and job.get('gpu') not in (None, ''):
            busy.setdefault(str(job['gpu']), []).append(job['name'])
    for g in gpus:
        g['jobs'] = busy.get(str(g['index']), [])
    disk = shutil.disk_usage(jobs_dir())
    return {'gpus': gpus, 'disk_free': disk.free, 'disk_total': disk.total}


@app.get('/api/fs')
def browse(path: str = '~', show_files: bool = True):
    path = os.path.abspath(os.path.expanduser(path or '~'))
    if os.path.isfile(path):
        path = os.path.dirname(path)
    try:
        names = sorted(os.listdir(path), key=str.lower)
    except OSError as e:
        raise HTTPException(400, f'{path}: {e.strerror}') from None
    entries = []
    for name in names:
        if name.startswith('.'):
            continue
        full = os.path.join(path, name)
        is_dir = os.path.isdir(full)
        if not is_dir and not show_files:
            continue
        try:
            size = 0 if is_dir else os.path.getsize(full)
        except OSError:
            size = 0
        entries.append({'name': name, 'dir': is_dir, 'size': size})
    entries.sort(key=lambda e: (not e['dir'], e['name'].lower()))
    return {'path': path, 'parent': os.path.dirname(path), 'entries': entries}


@app.put('/api/files')
async def upload_small_file(request: Request, name: str):
    """A small local file (YAML/JSON) from the browser, stored under
    <jobs-dir>/_uploads/<id>/<name>. Returns its path on this machine."""
    name = os.path.basename(name)
    if not name:
        raise HTTPException(400, 'missing file name')
    body = await request.body()
    if len(body) > MAX_SMALL_UPLOAD:
        raise HTTPException(413, 'file too large for a parameter upload')
    folder = os.path.join(jobs_dir(), '_uploads', uuid.uuid4().hex[:12])
    os.makedirs(folder)
    dest = os.path.join(folder, name)
    with open(dest, 'wb') as f:
        f.write(body)
    return {'path': dest, 'name': name, 'size': len(body)}


# ── jobs ───────────────────────────────────────────────────────────────────────
def list_jobs_raw():
    jobs = []
    for name in os.listdir(jobs_dir()):
        p = os.path.join(jobs_dir(), name, 'job.json')
        if os.path.isfile(p):
            try:
                with open(p) as f:
                    jobs.append(_upgrade(json.load(f)))
            except (OSError, ValueError):
                pass
    return sorted(jobs, key=lambda j: j['created'], reverse=True)


@app.get('/api/jobs')
def list_jobs():
    return [job_state(j) for j in list_jobs_raw()]


@app.get('/api/jobs/{job_id}')
def get_job(job_id: str, full: bool = False):
    return job_state(load_job(job_id), full=full)


def _tile_order(tiles, raw_params):
    """Upload/processing order: fov order, with calibration tile(s) first."""
    fovs = sorted(t['key'] for t in tiles)
    if not raw_params.get('auto_calibrate'):
        return fovs
    kwargs = params_mod.parse_values(raw_params)
    cal = list((kwargs.get('calibration_kwargs') or {}).get('tiles') or [fovs[len(fovs) // 2]])
    missing = [t for t in cal if t not in fovs]
    if missing:
        raise ValueError(f'calibration tiles {missing} are not in the selected folder')
    return cal + [f for f in fovs if f not in cal]


def _build_tiles(layout, raw_tiles):
    """Validate the browser's tile list and decide where each file is staged.

    Every tile is assembled under staging/incoming/<key>/ and only moved to
    staging/<ready> once all of its files are complete, in one rename:

    ome     one file, ready = ome_tiff/R0_<key>.ome.tiff
    legacy  one file per (z, channel), assembled as <key>/0/<name>; the whole
            folder is renamed to ready = tiles/<key>, so the pipeline never
            sees a tile with some of its slices missing (it would silently
            read a shorter z-stack, not fail)
    single  one file, ready = <its name>
    """
    tiles = []
    for t in raw_tiles:
        files = t.get('files') or []
        if not files:
            raise ValueError('A tile has no files.')
        key = int(t.get('key', 0))
        staged = []
        for f in files:
            rel = f['relpath']
            if layout == 'single':
                if not rel.lower().endswith(('.tif', '.tiff')):
                    raise ValueError('Pick a .tif/.tiff file.')
                dest = os.path.basename(rel)
            elif layout == 'ome':
                mt = TILE_RE.match(rel)
                if not mt or int(mt.group(1)) != key:
                    raise ValueError(f'Not tile {key}: {rel}')
                dest = f'R0_{key}.ome.tiff'
            elif layout == 'legacy':
                mt = LEGACY_SLICE_RE.match(rel)
                if not mt or int(mt.group(1)) != key:
                    raise ValueError(f'Not a slice of tile {key}: {rel}')
                dest = f'0/{os.path.basename(rel)}'
            else:
                raise ValueError(f'unknown layout {layout!r}')
            staged.append({'relpath': rel, 'size': int(f['size']), 'dest': dest})
        if layout in ('ome', 'single') and len(staged) != 1:
            raise ValueError(f'Tile {key} should be one file, got {len(staged)}.')
        if len({f['dest'] for f in staged}) != len(staged):
            raise ValueError(f'Tile {key} lists the same file twice.')
        ready = {'ome': f'ome_tiff/R0_{key}.ome.tiff', 'legacy': f'tiles/{key}',
                 'single': staged[0]['dest']}[layout]
        tiles.append({'key': key, 'size': sum(f['size'] for f in staged),
                      'files': staged, 'ready': ready})
    if not tiles:
        raise ValueError('No tiles found in the selection.')
    if len({t['key'] for t in tiles}) != len(tiles):
        raise ValueError('The same tile appears twice.')
    if layout == 'single' and len(tiles) != 1:
        raise ValueError('Single-file mode takes exactly one file.')
    return tiles


@app.post('/api/jobs')
async def create_job(request: Request):
    body = await request.json()
    mode = body.get('mode')
    raw = body.get('params', {})
    try:
        kwargs = params_mod.parse_values(raw)
    except ValueError as e:
        raise HTTPException(400, str(e)) from None
    if not kwargs['auto_calibrate'] and kwargs['channel_parameter_file'] is None:
        raise HTTPException(400, 'Set channel_parameter_file, or turn on auto_calibrate.')

    job_id = time.strftime('%Y%m%d-%H%M%S-') + uuid.uuid4().hex[:4]
    job = {'id': job_id, 'created': time.time(), 'mode': mode, 'params': raw,
           'name': (body.get('name') or '').strip(), 'gpu': body.get('gpu'),
           'prefetch_tiles': max(0, int(body.get('prefetch_tiles', 2)))}
    out_dir = kwargs['output_folder']

    if mode == 'remote':
        squid = os.path.expanduser((body.get('squid_folder') or '').strip())
        if not squid or not os.path.exists(squid):
            raise HTTPException(400, f'Squid folder not found on the remote machine: {squid!r}')
        job['squid_folder'] = squid
        if out_dir is None:
            from spot_detection_new_squid.io_utils import default_output_folder
            out_dir = default_output_folder(squid)
        job['name'] = job['name'] or os.path.basename(squid.rstrip('/'))
    elif mode in ('upload_folder', 'upload_single', 'globus'):
        if kwargs['output_corrected']:
            raise HTTPException(400, 'output_corrected is not available for data from your '
                                     'computer: it would keep a full corrected copy of every '
                                     'tile here.')
        if mode == 'globus':
            # The server lists the folder itself rather than trusting the page.
            g = body.get('globus') or {}
            m, job['globus'] = await run_in_threadpool(
                globus_prepare, g.get('endpoint'), g.get('path'), bool(g.get('verify_checksum', True)))
            mode = job['mode'] = 'globus_single' if m['layout'] == 'single' else 'globus_folder'
        else:
            m = body.get('manifest') or {}
        single = mode.endswith('_single')
        layout = 'single' if single else m.get('layout', 'ome')
        try:
            tiles = _build_tiles(layout, m.get('tiles', []))
        except ValueError as e:
            raise HTTPException(400, str(e)) from None
        sidecars = [s for s in m.get('sidecars', []) if s in SIDECAR_NAMES]
        try:
            order = [0] if single else _tile_order(tiles, raw)
        except ValueError as e:
            raise HTTPException(400, str(e)) from None
        job['manifest'] = {'kind': 'single' if single else 'folder',
                           'layout': layout, 'root_name': m.get('root_name') or '', 'tiles': tiles,
                           'sidecars': sidecars, 'order': order, 'chunk_size': CHUNK_SIZE}
        window = len(_upload_window(job, released=()))
        need = max(t['size'] for t in tiles) * window * 1.05
        free = shutil.disk_usage(jobs_dir()).free
        if need > free:
            raise HTTPException(400, f'Not enough disk space here for {window} tiles '
                                     f'({need / 1e9:.1f} GB needed, {free / 1e9:.1f} GB free). '
                                     f'Lower "tiles to fetch ahead".')
        if out_dir is None:
            out_dir = job_path(job_id, 'output')
        job['name'] = job['name'] or m.get('root_name') or 'upload'
    else:
        raise HTTPException(400, f'unknown mode {mode!r}')

    job['output_folder'] = os.path.abspath(out_dir)
    os.makedirs(job_path(job_id, 'staging', 'incoming'))
    if mode in ('upload_folder', 'globus_folder'):
        if not job['manifest']['sidecars']:
            open(job_path(job_id, 'staging', '.sidecars_done'), 'w').close()
    save_job(job)
    start_runner(job)
    return job_state(job)


def start_runner(job):
    env = dict(os.environ, PYTHONUNBUFFERED='1', CUDA_DEVICE_ORDER='PCI_BUS_ID')
    if job.get('gpu') not in (None, ''):
        env['CUDA_VISIBLE_DEVICES'] = str(job['gpu'])
    log = open(job_path(job['id'], 'log.txt'), 'ab')
    proc = subprocess.Popen([sys.executable, os.path.join(HERE, 'runner.py'), job_path(job['id'])],
                            stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                            cwd=HERE, env=env, start_new_session=True)
    log.close()
    PROCS[job['id']] = proc
    job['pid'] = proc.pid
    save_job(job)


@app.post('/api/jobs/{job_id}/cancel')
def cancel_job(job_id: str):
    job = load_job(job_id)
    if runner_alive(job):
        try:
            os.killpg(job['pid'], signal.SIGTERM)
        except ProcessLookupError:
            pass
        proc = PROCS.get(job_id)
        if proc is not None:
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(job['pid'], signal.SIGKILL)
    job['cancelled'] = True
    save_job(job)
    if 'globus' in job:
        # Normally the runner cancels these on its way out; this covers a
        # runner that had to be killed.
        status = read_status(job_path(job_id, 'status.json')) or {}
        gl = globus_client()
        for t in (status.get('transfers') or {}).values():
            if t.get('state') == 'ACTIVE':
                gl.cancel(t['task_id'])
    # The runner's own cleanup may not have run if it was killed.
    if 'manifest' in job:
        remove_staged_images(job_path(job_id, 'staging'), job['manifest'])
    return job_state(job)


@app.delete('/api/jobs/{job_id}')
def delete_job(job_id: str):
    job = load_job(job_id)
    if runner_alive(job):
        raise HTTPException(400, 'Cancel the job before deleting it.')
    # The job folder holds results only when no output folder was chosen;
    # an output folder elsewhere is left alone.
    shutil.rmtree(job_path(job_id))
    PROCS.pop(job_id, None)
    return {'ok': True}


@app.get('/api/jobs/{job_id}/log')
def get_log(job_id: str, offset: int = 0):
    path = job_path(job_id, 'log.txt')
    try:
        size = os.path.getsize(path)
    except OSError:
        return {'offset': 0, 'text': ''}
    if offset > size:
        offset = 0
    # On first load just send the tail; long CPU runs can log a lot.
    if offset == 0 and size > 2_000_000:
        offset = size - 2_000_000
    with open(path, 'rb') as f:
        f.seek(offset)
        data = f.read(4_000_000)
    return {'offset': offset + len(data), 'text': data.decode('utf-8', 'replace')}


@app.get('/api/jobs/{job_id}/results.zip')
def download_results(job_id: str):
    """Zip the output folder, skipping image files (corrected TIFFs)."""
    job = load_job(job_id)
    out = job['output_folder']
    if not os.path.isdir(out):
        raise HTTPException(404, 'No output yet.')
    fd, tmp = tempfile.mkstemp(suffix='.zip', dir=job_path(job_id))
    os.close(fd)
    with zipfile.ZipFile(tmp, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, _dirs, files in os.walk(out):
            for name in files:
                if name.lower().endswith(('.tif', '.tiff')):
                    continue
                full = os.path.join(root, name)
                zf.write(full, os.path.relpath(full, out))
    fname = re.sub(r'[^\w.-]+', '_', job['name'] or job_id) + '_results.zip'
    return FileResponse(tmp, filename=fname, background=BackgroundTask(os.remove, tmp))


# ── Globus ────────────────────────────────────────────────────────────────────
_GLOBUS_CACHE = {'at': 0.0, 'data': None}
_GLOBUS_LOCK = threading.Lock()


def _jobs_dir_writable(gl):
    jd = jobs_dir()
    return any(jd == p or jd.startswith(p.rstrip('/') + '/') for p in gl.writable_paths())


@app.get('/api/globus/status')
def globus_status(refresh: bool = False):
    """Is everything in place for a Globus run? Checks: the CLI is logged in;
    Globus Connect Personal is running here (asked of the local install) and
    connected (asked of Globus); and which of your other GCP endpoints -- your
    computer -- are connected. Cached for 15 s; refresh=1 forces a new check."""
    with _GLOBUS_LOCK:
        if not refresh and _GLOBUS_CACHE['data'] and time.time() - _GLOBUS_CACHE['at'] < 15:
            return _GLOBUS_CACHE['data']
        gl = globus_client()
        out = {'logged_in': False, 'identity': None, 'error': None,
               'login_command': f'{GLOBUS_BIN} login --no-local-server',
               'remote': {'endpoint_id': gl.local_endpoint_id(),
                          'process': gl.gcp_process_status(),
                          'api': None,
                          'jobs_dir': jobs_dir(),
                          'jobs_dir_writable': _jobs_dir_writable(gl)},
               'local_endpoints': [], 'ready': False}
        try:
            out['identity'] = gl.whoami()['username']
            out['logged_in'] = True
            rid = out['remote']['endpoint_id']
            if rid:
                out['remote']['api'] = gl.endpoint(rid)
            out['local_endpoints'] = [e for e in gl.my_gcp_endpoints() if e['id'] != rid]
        except NotLoggedIn:
            pass
        except GlobusError as e:
            out['error'] = str(e)
        r = out['remote']
        out['ready'] = bool(out['logged_in'] and r['process']['connected'] and r['api']
                            and r['api']['gcp_connected'] and r['jobs_dir_writable']
                            and any(e['gcp_connected'] for e in out['local_endpoints']))
        _GLOBUS_CACHE.update(at=time.time(), data=out)
        return out


@app.post('/api/globus/start-remote')
def globus_start_remote():
    try:
        globus_client().start_local_gcp()
    except GlobusError as e:
        raise HTTPException(400, str(e)) from None
    time.sleep(5)
    return globus_status(refresh=True)


@app.get('/api/globus/ls')
def globus_ls(endpoint: str, path: str = '/~/'):
    try:
        out = globus_client().ls(endpoint, path or '/~/')
    except GlobusError as e:
        raise HTTPException(400, str(e)) from None
    p = out['path'].rstrip('/') or '/'
    out['path'] = p
    out['parent'] = os.path.dirname(p) if p not in ('/', '/~') else p
    out['entries'].sort(key=lambda e: (not e['dir'], e['name'].lower()))
    out['entries'] = [e for e in out['entries'] if not e['name'].startswith('.')]
    return out


def globus_inspect(gl, endpoint, path):
    """What is at `path` on your computer, as a manifest in the same form the
    browser upload sends: {layout, root_name, root, tiles, sidecars}."""
    path = (path or '').strip()
    if not path:
        raise ValueError('Choose a folder or file on your computer.')
    try:
        listing = gl.ls(endpoint, path)
    except GlobusError as e:
        # Not a folder: maybe a single file.
        parent, name = os.path.split(path.rstrip('/'))
        try:
            up = gl.ls(endpoint, parent or '/')
        except GlobusError:
            raise ValueError(str(e)) from None
        hit = next((x for x in up['entries'] if x['name'] == name and not x['dir']), None)
        if hit is None:
            raise ValueError(str(e)) from None
        if not name.lower().endswith(('.tif', '.tiff')):
            raise ValueError(f'{name} is not a .tif/.tiff file.')
        return {'layout': 'single', 'root_name': name, 'root': up['path'].rstrip('/'),
                'tiles': [{'key': 0, 'files': [{'relpath': name, 'size': hit['size']}]}],
                'sidecars': []}

    root = listing['path'].rstrip('/')
    names = {e['name']: e for e in listing['entries']}
    sidecars = [n for n in SIDECAR_NAMES if '/' not in n and n in names and not names[n]['dir']]
    zero = []
    if '0' in names and names['0']['dir']:
        zero = gl.ls(endpoint, f'{root}/0')['entries']
        if any(e['name'] == 'coordinates.csv' and not e['dir'] for e in zero):
            sidecars.append('0/coordinates.csv')

    def tiles_from(entries, prefix):
        out = {}
        for e in entries:
            m = TILE_RE.match(e['name'])
            if m and not e['dir'] and int(m.group(1)) not in out:
                out[int(m.group(1))] = {'key': int(m.group(1)),
                                        'files': [{'relpath': prefix + e['name'], 'size': e['size']}]}
        return [out[k] for k in sorted(out)]

    layout, tiles = 'ome', []
    if 'ome_tiff' in names and names['ome_tiff']['dir']:
        tiles = tiles_from(gl.ls(endpoint, f'{root}/ome_tiff')['entries'], 'ome_tiff/')
    if not tiles:
        tiles = tiles_from(listing['entries'], '')
    if not tiles and zero:
        layout, groups = 'legacy', {}
        for e in zero:
            m = LEGACY_SLICE_RE.match(f'0/{e["name"]}')
            if m and not e['dir']:
                groups.setdefault(int(m.group(1)), []).append({'relpath': f'0/{e["name"]}',
                                                               'size': e['size']})
        tiles = [{'key': k, 'files': sorted(groups[k], key=lambda f: f['relpath'])}
                 for k in sorted(groups)]
    if not tiles:
        raise ValueError(f'No tiles found in {root}: expected ome_tiff/R0_<fov>.ome.tiff '
                         f'(new layout) or 0/R0_<fov>_<z>_<channel>.tiff (old layout).')
    return {'layout': layout, 'root_name': os.path.basename(root) or root, 'root': root,
            'tiles': tiles, 'sidecars': sidecars}


@app.get('/api/globus/inspect')
def globus_inspect_route(endpoint: str, path: str):
    """Summary of a folder/file on your computer, for the page to show."""
    try:
        m = globus_inspect(globus_client(), endpoint, path)
    except (ValueError, GlobusError) as e:
        raise HTTPException(400, str(e)) from None
    out = {k: m[k] for k in ('layout', 'root_name', 'root', 'sidecars')}
    out['n_tiles'] = len(m['tiles'])
    out['n_files'] = sum(len(t['files']) for t in m['tiles'])
    out['total_bytes'] = sum(f['size'] for t in m['tiles'] for f in t['files'])
    out['largest_tile'] = max(sum(f['size'] for f in t['files']) for t in m['tiles'])
    if m['layout'] == 'legacy':
        depths = {}
        for t in m['tiles']:
            per = {}
            for f in t['files']:
                ch = LEGACY_SLICE_RE.match(f['relpath']).group(3)
                per[ch] = per.get(ch, 0) + 1
            for ch, n in per.items():
                depths.setdefault(ch, set()).add(n)
        out['uneven_z'] = {ch: sorted(ns) for ch, ns in depths.items() if len(ns) > 1}
    return out


def globus_prepare(endpoint, path, verify_checksum):
    """Checks and manifest for a new Globus job; raises HTTPException with a
    message the page can show as-is."""
    st = globus_status(refresh=True)
    if not st['logged_in']:
        raise HTTPException(400, f'Globus is not logged in on the remote machine. Run in a '
                                 f'terminal there: {st["login_command"]}')
    r = st['remote']
    if not (r['process']['connected'] and r['api'] and r['api']['gcp_connected']):
        raise HTTPException(400, 'Globus Connect Personal is not running/connected on the remote '
                                 'machine. Start it from the Globus panel.')
    if not r['jobs_dir_writable']:
        raise HTTPException(400, f'Globus Connect Personal on the remote machine may not write to '
                                 f'{r["jobs_dir"]}. Add it to ~/.globusonline/lta/config-paths '
                                 f'(writable), or start the app with --jobs-dir under a shared path.')
    src = next((e for e in st['local_endpoints'] if e['id'] == endpoint), None)
    if src is None:
        raise HTTPException(400, 'Choose your computer\'s Globus endpoint.')
    if not src['gcp_connected']:
        raise HTTPException(400, f'Globus Connect Personal on your computer ({src["display_name"]}) '
                                 f'is not connected. Start it there and check again.')
    try:
        m = globus_inspect(globus_client(), endpoint, path)
    except (ValueError, GlobusError) as e:
        raise HTTPException(400, str(e)) from None
    info = {'source_endpoint': endpoint, 'source_endpoint_name': src['display_name'],
            'source_root': m['root'], 'dest_endpoint': r['endpoint_id'],
            'verify_checksum': verify_checksum}
    return m, info


# ── uploads of streamed data ──────────────────────────────────────────────────
#
# A tile is one or more files (one OME-TIFF, or an old layout's per-slice
# TIFFs). Each file is sent in chunk_size pieces and assembled under
# staging/incoming/<key>/; incoming/<key>.json records which pieces of which
# file have arrived, so an interrupted upload resumes where it stopped.
def _state_path(job_id, key):
    return job_path(job_id, 'staging', 'incoming', f'{key}.json')


def _received(job_id, key):
    """{file index: set of chunk indices received} for one tile."""
    try:
        with open(_state_path(job_id, key)) as f:
            return {int(fi): set(chunks) for fi, chunks in json.load(f).items()}
    except (OSError, ValueError):
        return {}


def _n_chunks(size, chunk_size):
    return max(1, -(-size // chunk_size))


def _upload_window(job, released=None):
    if released is None:
        status = read_status(job_path(job['id'], 'status.json')) or {}
        released = status.get('released', [])
    return upload_window(job, released)


@app.get('/api/jobs/{job_id}/wanted')
def wanted(job_id: str):
    """What the browser should upload next: for each tile in the window that
    is not complete yet, its unfinished files and the chunks already here."""
    job = load_job(job_id)
    st = job_state(job)
    if st['state'] not in ('starting', 'running') or 'manifest' not in job:
        return {'stop': True, 'state': st['state'], 'tiles': []}
    cs = job['manifest']['chunk_size']
    tiles = []
    for t in _upload_window(job):
        if os.path.exists(job_path(job_id, 'staging', t['ready'])):
            continue
        got = _received(job_id, t['key'])
        files = []
        for fi, f in enumerate(t['files']):
            n = _n_chunks(f['size'], cs)
            have = got.get(fi, set())
            if len(have) < n:
                files.append({'fi': fi, 'size': f['size'], 'n_chunks': n, 'received': sorted(have)})
        tiles.append({'key': t['key'], 'n_files': len(t['files']),
                      'files_done': len(t['files']) - len(files), 'files': files})
    return {'stop': False, 'state': st['state'], 'chunk_size': cs, 'tiles': tiles}


@app.put('/api/jobs/{job_id}/sidecar')
async def upload_sidecar(job_id: str, path: str, request: Request):
    job = load_job(job_id)
    m = job.get('manifest') or {}
    if path not in m.get('sidecars', []):
        raise HTTPException(400, f'{path} is not an expected metadata file')
    body = await request.body()
    dest = job_path(job_id, 'staging', path)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, 'wb') as f:
        f.write(body)
    if all(os.path.exists(job_path(job_id, 'staging', s)) for s in m['sidecars']):
        open(job_path(job_id, 'staging', '.sidecars_done'), 'w').close()
    return {'ok': True}


def _finish_tile(job, tile):
    finish_tile(job_path(job['id'], 'staging'), job['manifest']['layout'], tile)
    os.remove(_state_path(job['id'], tile['key']))


def _write_chunk(job, tile, fi, ci, data):
    job_id, key = job['id'], tile['key']
    cs = job['manifest']['chunk_size']
    f = tile['files'][fi]
    target = job_path(job_id, 'staging', 'incoming', str(key), f['dest'])
    part = f'{target}.part'
    ready = job_path(job_id, 'staging', tile['ready'])
    with UPLOAD_LOCK:
        if os.path.exists(ready) or os.path.exists(target):
            return True  # a retried chunk of something already complete
        if not os.path.exists(part):
            os.makedirs(os.path.dirname(part), exist_ok=True)
            with open(part, 'wb') as fh:
                fh.truncate(f['size'])
    try:
        with open(part, 'r+b') as fh:
            fh.seek(ci * cs)
            fh.write(data)
    except FileNotFoundError:
        # Completed by a concurrent request between the check and the open.
        return os.path.exists(target) or os.path.exists(ready)
    with UPLOAD_LOCK:
        if os.path.exists(ready) or os.path.exists(target):
            return True
        got = _received(job_id, key)
        got.setdefault(fi, set()).add(ci)
        state = _state_path(job_id, key)
        with open(f'{state}.tmp', 'w') as fh:
            json.dump({str(k): sorted(v) for k, v in got.items()}, fh)
        os.replace(f'{state}.tmp', state)
        if len(got[fi]) == _n_chunks(f['size'], cs):
            os.replace(part, target)
        tile_done = all(len(got.get(i, ())) == _n_chunks(g['size'], cs)
                        for i, g in enumerate(tile['files']))
        if tile_done:
            _finish_tile(job, tile)
    return tile_done


@app.put('/api/jobs/{job_id}/tiles/{key}/files/{fi}/chunks/{ci}')
async def upload_chunk(job_id: str, key: int, fi: int, ci: int, request: Request):
    job = load_job(job_id)
    if 'manifest' not in job or job.get('cancelled'):
        raise HTTPException(409, 'job is not accepting uploads')
    window = {t['key']: t for t in _upload_window(job)}
    if key not in window:
        # Keeps disk use bounded even if a browser misbehaves.
        raise HTTPException(409, f'tile {key} is not in the current upload window')
    tile = window[key]
    if not 0 <= fi < len(tile['files']):
        raise HTTPException(400, 'file index out of range')
    size = tile['files'][fi]['size']
    cs = job['manifest']['chunk_size']
    if not 0 <= ci < _n_chunks(size, cs):
        raise HTTPException(400, 'chunk index out of range')
    expected = min(cs, size - ci * cs)
    data = await request.body()
    if len(data) != expected:
        raise HTTPException(400, f'chunk {ci} has {len(data)} bytes, expected {expected}')
    if os.path.exists(job_path(job_id, 'staging', tile['ready'])):
        return {'done': True}
    done = await run_in_threadpool(_write_chunk, job, tile, fi, ci, data)
    return {'done': done}


app.mount('/static', StaticFiles(directory=os.path.join(HERE, 'static')), name='static')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--port', type=int, default=8765)
    ap.add_argument('--host', default='127.0.0.1',
                    help='Keep 127.0.0.1 and use an SSH tunnel; anything else exposes '
                         'the app to the network.')
    ap.add_argument('--jobs-dir', default=os.path.join(HERE, 'jobs'),
                    help='Where job logs, uploaded tiles and (by default) results go.')
    args = ap.parse_args()

    CONFIG['jobs_dir'] = os.path.abspath(args.jobs_dir)
    os.makedirs(CONFIG['jobs_dir'], exist_ok=True)
    token_file = os.path.join(CONFIG['jobs_dir'], '.token')
    if not os.path.exists(token_file):
        with open(os.open(token_file, os.O_WRONLY | os.O_CREAT, 0o600), 'w') as f:
            f.write(secrets.token_urlsafe(24))
    with open(token_file) as f:
        CONFIG['token'] = f.read().strip()

    url = f'http://localhost:{args.port}/?token={CONFIG["token"]}'
    print('\n' + '=' * 72)
    print(f'  Spot detection web app on this machine, port {args.port}')
    print(f'  Open in your local browser (through VS Code port forwarding or')
    print(f'  `ssh -L {args.port}:localhost:{args.port} <this host>`):')
    print(f'\n    {url}\n')
    print(f'  Jobs folder: {CONFIG["jobs_dir"]}')
    print('=' * 72 + '\n', flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level='warning')


if __name__ == '__main__':
    main()
