"""Pulls tiles from your computer with Globus while a job runs.

Runs as a thread inside the job's runner process, so transfers carry on with
the browser closed. It keeps the next (1 + prefetch) tiles in flight -- at most
MAX_ACTIVE Globus tasks at a time, in processing order, so the tile needed
first is not slowed by the ones queued behind it -- and moves each tile into
place when its task succeeds (staging.finish_tile). The streaming pipeline
deletes tiles as it finishes reading them, which frees the window for more.

Each tile is one Globus task: one file for the new layout, all of the tile's
per-slice files for the old one. A failed task is resubmitted up to
MAX_ATTEMPTS times; after that the job fails with Globus's error message.
While an endpoint is disconnected (e.g. your laptop is asleep) Globus itself
pauses the task and resumes it on reconnect; the job just waits and says so.
"""
from __future__ import annotations

import os
import shutil
import threading
import time
import traceback

from globus_link import GlobusError, NotLoggedIn, client
from staging import ABORT_FILE, assembly_dir, finish_tile, upload_window

POLL_S = 3.0
ENDPOINT_CHECK_S = 30.0
MAX_ACTIVE = 2
MAX_ATTEMPTS = 3


class GlobusFetcher(threading.Thread):
    def __init__(self, job, staging, status):
        super().__init__(daemon=True, name='globus-fetcher')
        self.job = job
        self.m = job['manifest']
        self.g = job['globus']
        self.staging = staging
        self.status = status
        self.gl = client()
        self.stop_event = threading.Event()
        self.tasks = {}          # key ('sidecars' or tile fov) -> {'task_id', 'attempts', ...}
        self.done = set()
        self.last_endpoint_check = 0.0
        self.by_key = {t['key']: t for t in self.m['tiles']}

    # ── lifecycle ─────────────────────────────────────────────────────────────
    def run(self):
        while not self.stop_event.is_set():
            try:
                self.step()
            except NotLoggedIn as e:
                self.abort(f'Globus login is missing or expired on the remote machine: {e}')
                return
            except GlobusError as e:
                if 'consent' in str(e).lower():
                    # Needs a one-off `globus session consent ...`; the message says which.
                    self.abort(f'Globus needs additional consent: {e}')
                    return
                # Usually transient (network, Globus API hiccup): report and retry.
                self.status.update(globus_note=f'Globus error, retrying: {e}')
                print(f'Globus error, will retry: {e}', flush=True)
            except Exception:  # noqa: BLE001 -- a bug here must not hang the job silently
                self.abort(f'transfer manager crashed:\n{traceback.format_exc()}')
                return
            self.stop_event.wait(POLL_S)

    def stop(self):
        """Stop and cancel whatever is still in flight."""
        self.stop_event.set()
        if threading.current_thread() is not self:
            self.join(timeout=60)   # let a submit in progress finish, so it is cancelled too
        for info in list(self.tasks.values()):
            if info.get('state') == 'ACTIVE':
                self.gl.cancel(info['task_id'])

    def abort(self, message):
        print(f'\nTransfer failed: {message}', flush=True)
        with open(os.path.join(self.staging, ABORT_FILE), 'w') as f:
            f.write(message)
        self.stop()

    # ── one polling round ─────────────────────────────────────────────────────
    def step(self):
        self.check_endpoints()
        for key in list(self.tasks):
            if self.tasks[key].get('state') == 'ACTIVE':
                self.poll(key)

        # The small metadata task runs alongside the tiles and does not count
        # towards MAX_ACTIVE; the pipeline waits for it before reading tiles.
        if self.m['sidecars'] and 'sidecars' not in self.tasks:
            self.submit('sidecars')

        with self.status.lock:
            released = list(self.status.data['released'])
        active = sum(1 for k, i in self.tasks.items()
                     if k != 'sidecars' and i.get('state') == 'ACTIVE')
        for tile in upload_window(self.job, released):
            key = tile['key']
            if key in self.done or key in self.tasks and self.tasks[key].get('state') == 'ACTIVE':
                continue
            if active >= MAX_ACTIVE:
                break
            self.submit(key)
            active += 1
        self.publish()

    def check_endpoints(self):
        now = time.time()
        if now - self.last_endpoint_check < ENDPOINT_CHECK_S:
            return
        self.last_endpoint_check = now
        notes = []
        for which, ep in (('your computer', self.g['source_endpoint']),
                          ('the remote machine', self.g['dest_endpoint'])):
            info = self.gl.endpoint(ep)
            if not info['gcp_connected']:
                notes.append(f'Globus Connect Personal on {which} ({info["display_name"]}) is not '
                             f'connected; transfers are paused until it is.')
            elif info['gcp_paused']:
                notes.append(f'Globus Connect Personal on {which} is paused.')
        self.status.update(globus_note=' '.join(notes) or None)

    # ── tasks ─────────────────────────────────────────────────────────────────
    def pairs_for(self, key):
        src_root = self.g['source_root'].rstrip('/')
        if key == 'sidecars':
            return [(f'{src_root}/{rel}', os.path.join(self.staging, rel)) for rel in self.m['sidecars']]
        assembly = assembly_dir(self.staging, key)
        return [(f'{src_root}/{f["relpath"]}', os.path.join(assembly, f['dest']))
                for f in self.by_key[key]['files']]

    def submit(self, key):
        prev = self.tasks.get(key, {})
        attempts = prev.get('attempts', 0) + 1
        pairs = self.pairs_for(key)
        if key != 'sidecars':
            # Start every attempt from an empty folder.
            shutil.rmtree(assembly_dir(self.staging, key), ignore_errors=True)
        for _, dst in pairs:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
        label = f'spotdet {self.job["name"]} ' + ('metadata' if key == 'sidecars' else f'tile {key}')
        task_id = self.gl.submit(self.g['source_endpoint'], self.g['dest_endpoint'], pairs,
                                 label=label, verify_checksum=self.g.get('verify_checksum', True))
        what = 'metadata files' if key == 'sidecars' else f'tile {key} ({len(pairs)} file(s))'
        print(f'Globus: requested {what}, task {task_id}'
              f'{f" (attempt {attempts})" if attempts > 1 else ""}', flush=True)
        self.tasks[key] = {'task_id': task_id, 'attempts': attempts, 'state': 'ACTIVE',
                           'submitted': time.time(), 'bytes': 0, 'rate': 0, 'nice_status': 'Queued',
                           'total': sum(f['size'] for f in self.by_key[key]['files'])
                           if key != 'sidecars' else 0}
        self.publish()

    def poll(self, key):
        info = self.tasks[key]
        t = self.gl.task(info['task_id'])
        info.update(state=t['status'], bytes=t['bytes'], rate=t['rate'],
                    nice_status=t['nice_status'], files=t['files'], files_done=t['files_done'])
        if t['status'] == 'SUCCEEDED':
            if key == 'sidecars':
                open(os.path.join(self.staging, '.sidecars_done'), 'w').close()
            else:
                finish_tile(self.staging, self.m['layout'], self.by_key[key])
            info['elapsed'] = time.time() - info['submitted']
            self.done.add(key)
            what = 'metadata files' if key == 'sidecars' else f'tile {key}'
            print(f'Globus: {what} arrived ({info["bytes"] / 1e9:.2f} GB in '
                  f'{info["elapsed"]:.0f}s)', flush=True)
        elif t['status'] in ('FAILED', 'INACTIVE'):
            reason = t['error'] or t['nice_status'] or t['status']
            if t['status'] == 'INACTIVE':
                self.gl.cancel(info['task_id'])
            if info['attempts'] >= MAX_ATTEMPTS:
                self.abort(f'Globus task {info["task_id"]} for '
                           f'{"metadata" if key == "sidecars" else f"tile {key}"} failed '
                           f'{info["attempts"]} times; last error: {reason}')
                return
            print(f'Globus: task {info["task_id"]} failed ({reason}); retrying', flush=True)
            self.submit(key)

    def publish(self):
        """Transfer progress for the web page."""
        out = {}
        for key, i in self.tasks.items():
            if key in self.done and key != 'sidecars':
                continue
            out[str(key)] = {k: i.get(k) for k in ('task_id', 'state', 'bytes', 'rate', 'total',
                                                   'nice_status', 'files', 'files_done', 'attempts')}
        self.status.update(transfers=out, transferred=sorted(k for k in self.done if k != 'sidecars'))
