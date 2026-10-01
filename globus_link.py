"""Talking to Globus: login state, Globus Connect Personal (GCP) status,
listing files on your computer's endpoint, and submitting/watching transfers.

Everything goes through the `globus` command-line tool (globus-cli, installed in
the same environment as this app), so the app uses your own `globus login` on
this machine and needs no Globus app registration. Run once, in a terminal on
this machine:

    ~/miniconda3/envs/spotdet/bin/globus login --no-local-server

Set SPOTDET_GLOBUS_FAKE=<folder> to swap in globus_fake.FakeGlobus, which
"transfers" by copying local files; it exists only for testing the app.
"""
from __future__ import annotations

import glob
import json
import os
import shlex
import subprocess
import sys

GLOBUS_BIN = os.path.join(os.path.dirname(sys.executable), 'globus')
LTA_DIR = os.path.expanduser('~/.globusonline/lta')


class GlobusError(Exception):
    pass


class NotLoggedIn(GlobusError):
    pass


def client():
    fake = os.environ.get('SPOTDET_GLOBUS_FAKE')
    if fake:
        from globus_fake import FakeGlobus
        return FakeGlobus(fake)
    return CliGlobus()


def gcp_dir():
    """The Globus Connect Personal install on this machine, or None."""
    if os.environ.get('SPOTDET_GCP_DIR'):
        return os.environ['SPOTDET_GCP_DIR']
    found = sorted(glob.glob(os.path.expanduser('~/globusconnectpersonal-*/globusconnectpersonal')))
    return os.path.dirname(found[-1]) if found else None


def gcp_writable_paths():
    """Folders this machine's GCP lets Globus write to (its config-paths file:
    `path,sharing,writable` per line)."""
    out = []
    try:
        with open(os.path.join(LTA_DIR, 'config-paths')) as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 3 and parts[2].strip() == '1':
                    out.append(os.path.abspath(os.path.expanduser(parts[0])))
    except OSError:
        pass
    return out


class CliGlobus:
    """globus-cli wrapper. Every method raises GlobusError with the CLI's own
    message on failure, and NotLoggedIn when there is no login."""

    def _run(self, args, stdin=None, timeout=120):
        if not os.path.exists(GLOBUS_BIN):
            raise GlobusError(f'globus CLI not found at {GLOBUS_BIN}; pip install globus-cli')
        try:
            proc = subprocess.run([GLOBUS_BIN, *args, '-F', 'json'], input=stdin,
                                  capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            raise GlobusError(f'globus {args[0]} timed out') from None
        if proc.returncode != 0:
            msg = (proc.stderr or proc.stdout).strip()
            if any(s in msg for s in ('MissingLoginError', 'Missing login', 'invalid_grant')):
                raise NotLoggedIn(msg)
            raise GlobusError(msg or f'globus {args[0]} failed (exit {proc.returncode})')
        return json.loads(proc.stdout) if proc.stdout.strip() else {}

    # ── identity and endpoints ────────────────────────────────────────────────
    def whoami(self):
        data = self._run(['whoami'])
        return {'username': data.get('preferred_username') or data.get('username') or '?'}

    def local_endpoint_id(self):
        """This machine's GCP endpoint, from its local config (no login needed)."""
        try:
            with open(os.path.join(LTA_DIR, 'client-id.txt')) as f:
                return f.read().strip() or None
        except OSError:
            return None

    def gcp_process_status(self):
        """Whether GCP is running and connected on *this* machine, asked of
        the GCP install itself."""
        d = gcp_dir()
        if d is None:
            return {'installed': False, 'connected': False, 'text': 'Globus Connect Personal not found'}
        try:
            out = subprocess.run([os.path.join(d, 'globusconnectpersonal'), '-status'],
                                 capture_output=True, text=True, timeout=30, cwd=d)
            text = (out.stdout + out.stderr).strip()
        except (OSError, subprocess.SubprocessError) as e:
            return {'installed': True, 'connected': False, 'text': str(e)}
        connected = any(line.split(':', 1)[0].strip() == 'Globus Online'
                        and line.split(':', 1)[1].strip() == 'connected'
                        for line in text.splitlines() if ':' in line)
        return {'installed': True, 'connected': connected, 'text': text}

    def start_local_gcp(self):
        d = gcp_dir()
        if d is None:
            raise GlobusError('Globus Connect Personal is not installed on this machine.')
        log = open(os.path.join(d, 'webapp-start.log'), 'ab')
        subprocess.Popen([os.path.join(d, 'globusconnectpersonal'), '-start'], cwd=d,
                         stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                         start_new_session=True)
        log.close()

    def writable_paths(self):
        return gcp_writable_paths()

    @staticmethod
    def _endpoint_fields(e):
        return {'id': e['id'], 'display_name': e.get('display_name') or e.get('canonical_name') or e['id'],
                'gcp_connected': bool(e.get('gcp_connected')), 'gcp_paused': bool(e.get('gcp_paused')),
                'is_gcp': bool(e.get('is_globus_connect'))}

    def endpoint(self, endpoint_id):
        return self._endpoint_fields(self._run(['endpoint', 'show', endpoint_id]))

    def my_gcp_endpoints(self):
        data = self._run(['endpoint', 'search', '--filter-scope', 'my-gcp-endpoints', '--limit', '100'])
        return [self._endpoint_fields(e) for e in data.get('DATA', [])]

    # ── files ─────────────────────────────────────────────────────────────────
    def ls(self, endpoint_id, path):
        """{'path': absolute path as the endpoint reports it, 'entries': [...]}"""
        data = self._run(['ls', f'{endpoint_id}:{path}'], timeout=300)
        entries = [{'name': e['name'], 'dir': e['type'] == 'dir', 'size': int(e.get('size') or 0)}
                   for e in data.get('DATA', [])]
        return {'path': data.get('path') or path, 'entries': entries}

    # ── transfers ─────────────────────────────────────────────────────────────
    def submit(self, src_ep, dst_ep, pairs, label, verify_checksum=True):
        """One transfer task copying [(src_path, dst_path), ...]; returns its id."""
        batch = '\n'.join(f'{shlex.quote(s)} {shlex.quote(d)}' for s, d in pairs) + '\n'
        args = ['transfer', src_ep, dst_ep, '--batch', '-', '--label', label[:128],
                '--notify', 'off',
                '--verify-checksum' if verify_checksum else '--no-verify-checksum']
        data = self._run(args, stdin=batch)
        return data['task_id']

    def task(self, task_id):
        t = self._run(['task', 'show', task_id])
        fatal = t.get('fatal_error') or {}
        return {
            'status': t.get('status'),                      # ACTIVE, SUCCEEDED, FAILED, INACTIVE
            'nice_status': t.get('nice_status'),            # e.g. OK, Queued, GC_NOT_CONNECTED
            'nice_status_short': t.get('nice_status_short_description'),
            'bytes': int(t.get('bytes_transferred') or 0),
            'rate': float(t.get('effective_bytes_per_second') or 0),
            'files': int(t.get('files') or 0),
            'files_done': int(t.get('files_transferred') or 0) + int(t.get('files_skipped') or 0),
            'faults': int(t.get('faults') or 0),
            'error': fatal.get('description') or fatal.get('code'),
        }

    def cancel(self, task_id):
        try:
            self._run(['task', 'cancel', task_id])
        except GlobusError:
            pass  # already finished
