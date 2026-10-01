"""A stand-in for globus_link.CliGlobus, for testing the app without Globus.

Enabled with SPOTDET_GLOBUS_FAKE=<state folder>. Both "endpoints" are this
machine's filesystem. A transfer is a detached process that copies the files
at a throttled rate (SPOTDET_GLOBUS_FAKE_MBPS, default 50) and records its
progress in <state folder>/tasks/<id>.json, so the server and the runner (two
processes) see the same task, as they would with real Globus.

Endpoint `fake-laptop` is "your computer", `fake-remote` is this machine.
Touch <state folder>/offline-<endpoint id> to make an endpoint disconnected:
running transfers then pause with nice_status GC_NOT_CONNECTED.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import uuid

from globus_link import GlobusError

LAPTOP, REMOTE = 'fake-laptop', 'fake-remote'


def _write_json(path, data):
    with open(f'{path}.tmp', 'w') as f:
        json.dump(data, f)
    os.replace(f'{path}.tmp', path)


class FakeGlobus:
    def __init__(self, state_dir):
        self.state = state_dir
        os.makedirs(os.path.join(state_dir, 'tasks'), exist_ok=True)

    def _online(self, ep):
        return not os.path.exists(os.path.join(self.state, f'offline-{ep}'))

    def whoami(self):
        return {'username': 'fake-user@example.org'}

    def local_endpoint_id(self):
        return REMOTE

    def gcp_process_status(self):
        on = self._online(REMOTE)
        return {'installed': True, 'connected': on,
                'text': f'Globus Online:   {"connected" if on else "disconnected"}'}

    def writable_paths(self):
        return ['/']

    def start_local_gcp(self):
        try:
            os.remove(os.path.join(self.state, f'offline-{REMOTE}'))
        except FileNotFoundError:
            pass

    def endpoint(self, endpoint_id):
        if endpoint_id not in (LAPTOP, REMOTE):
            raise GlobusError(f'No such endpoint {endpoint_id}')
        return {'id': endpoint_id, 'display_name': f'{endpoint_id} (fake)',
                'gcp_connected': self._online(endpoint_id), 'gcp_paused': False, 'is_gcp': True}

    def my_gcp_endpoints(self):
        return [self.endpoint(LAPTOP), self.endpoint(REMOTE)]

    def ls(self, endpoint_id, path):
        if not self._online(endpoint_id):
            raise GlobusError(f'Endpoint {endpoint_id} is not connected')
        path = os.path.expanduser(path.replace('/~/', '~/', 1)) if path.startswith('/~') else path
        try:
            names = sorted(os.listdir(path))
        except OSError as e:
            raise GlobusError(f'{path}: {e.strerror}') from None
        entries = []
        for n in names:
            full = os.path.join(path, n)
            entries.append({'name': n, 'dir': os.path.isdir(full),
                            'size': 0 if os.path.isdir(full) else os.path.getsize(full)})
        return {'path': path.rstrip('/') + '/', 'entries': entries}

    def submit(self, src_ep, dst_ep, pairs, label, verify_checksum=True):
        task_id = f'fake-{uuid.uuid4().hex[:12]}'
        path = os.path.join(self.state, 'tasks', f'{task_id}.json')
        total = sum(os.path.getsize(s) for s, _ in pairs)
        _write_json(path, {'status': 'ACTIVE', 'nice_status': 'Queued', 'bytes': 0, 'rate': 0,
                           'files': len(pairs), 'files_done': 0, 'faults': 0, 'error': None,
                           'total': total, 'label': label, 'src': src_ep, 'dst': dst_ep})
        spec = os.path.join(self.state, 'tasks', f'{task_id}.pairs.json')
        _write_json(spec, pairs)
        subprocess.Popen([sys.executable, os.path.abspath(__file__), self.state, task_id],
                         start_new_session=True, stdin=subprocess.DEVNULL,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return task_id

    def task(self, task_id):
        with open(os.path.join(self.state, 'tasks', f'{task_id}.json')) as f:
            return json.load(f)

    def cancel(self, task_id):
        open(os.path.join(self.state, 'tasks', f'{task_id}.cancel'), 'w').close()


def _copy_worker(state, task_id):
    """Body of the detached transfer process."""
    tdir = os.path.join(state, 'tasks')
    path = os.path.join(tdir, f'{task_id}.json')
    with open(path) as f:
        info = json.load(f)
    with open(os.path.join(tdir, f'{task_id}.pairs.json')) as f:
        pairs = json.load(f)
    mbps = float(os.environ.get('SPOTDET_GLOBUS_FAKE_MBPS', '50'))
    fail_marker = os.path.join(state, 'fail-next')   # touch to make the next task fail once
    if os.path.exists(fail_marker):
        os.remove(fail_marker)
        info.update(status='FAILED', nice_status='PERMISSION_DENIED', error='simulated failure')
        _write_json(path, info)
        return
    t0 = time.time()
    time.sleep(1.0)  # queueing latency
    for src, dst in pairs:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(src, 'rb') as fi, open(dst, 'wb') as fo:
            while True:
                if os.path.exists(os.path.join(tdir, f'{task_id}.cancel')):
                    info.update(status='FAILED', nice_status='CANCELED', error='canceled')
                    _write_json(path, info)
                    return
                while not all(not os.path.exists(os.path.join(state, f'offline-{ep}'))
                              for ep in (info['src'], info['dst'])):
                    info['nice_status'] = 'GC_NOT_CONNECTED'
                    _write_json(path, info)
                    time.sleep(0.3)
                buf = fi.read(1 << 20)
                if not buf:
                    break
                fo.write(buf)
                info['bytes'] += len(buf)
                info['nice_status'] = 'OK'
                info['rate'] = info['bytes'] / max(1e-3, time.time() - t0)
                _write_json(path, info)
                time.sleep(len(buf) / (mbps * 1e6))
        info['files_done'] += 1
        _write_json(path, info)
    info.update(status='SUCCEEDED', nice_status=None)
    _write_json(path, info)


if __name__ == '__main__':
    _copy_worker(sys.argv[1], sys.argv[2])
