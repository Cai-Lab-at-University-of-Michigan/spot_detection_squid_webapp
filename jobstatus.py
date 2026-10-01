"""status.json: the runner process's progress, read by the web server.

The runner is the only writer; every write replaces the whole file atomically
(write to a temp file, then os.replace) so the server never reads a torn file.
"""
from __future__ import annotations

import json
import os
import threading
import time


class StatusFile:
    def __init__(self, path):
        self.path = path
        self.lock = threading.Lock()
        self.data = {
            'state': 'starting', 'phase': 'starting', 'error': None,
            'released': [], 'waiting_for': [], 'detected': {},
            'started': time.time(), 'updated': time.time(),
        }

    def _write(self):
        self.data['updated'] = time.time()
        tmp = f'{self.path}.tmp'
        with open(tmp, 'w') as f:
            json.dump(self.data, f)
        os.replace(tmp, self.path)

    def update(self, **fields):
        with self.lock:
            self.data.update(fields)
            self._write()

    def tile_released(self, fov):
        """Tile `fov` is fully read and deleted from staging; the server may
        ask the browser for the next one."""
        with self.lock:
            self.data['released'].append(fov)
            self._write()

    def waiting(self, fov, is_waiting):
        with self.lock:
            waiting = [f for f in self.data['waiting_for'] if f != fov]
            if is_waiting:
                waiting.append(fov)
            self.data['waiting_for'] = waiting
            self._write()

    def channel_detected(self, fov, channel, n_spots):
        with self.lock:
            self.data['detected'].setdefault(str(fov), {})[channel] = int(n_spots)
            self._write()


def read_status(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None
