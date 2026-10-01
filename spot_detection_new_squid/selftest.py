"""
Parity / speed self-test for the detector: confirms the torch backend (on
CPU and, if available, on the GPU) returns exactly the same peaks as the
scipy backend, then reports a rough GPU speedup.

Run with:
    python -m functionalized.selftest
"""
from __future__ import annotations

import time

import numpy as np

from .detection import alg_smoothed_local_max_3d
from .torch_filters import torch_available, gpu_available, select_device


def run():
    rng = np.random.default_rng(0)
    Z, Y, X = 32, 384, 384
    vol = rng.integers(0, 300, size=(Z, Y, X)).astype(np.uint16)
    for _ in range(60):                                   # plant bright blobs
        z = int(rng.integers(2, Z - 2))
        y = int(rng.integers(5, Y - 5))
        x = int(rng.integers(5, X - 5))
        vol[z, y - 2:y + 3, x - 2:x + 3] = 4000

    params = dict(footprint_size=5, threshold=1000, smooth_sigma=1.0,
                  tolerance=200, prominence_footprint_size=9)

    t0 = time.time()
    cpu = alg_smoothed_local_max_3d(vol, backend='scipy', **params)
    t_cpu = time.time() - t0
    print(f'[scipy]  {len(cpu):5d} peaks  in {t_cpu * 1e3:7.1f} ms')

    if not torch_available():
        print('[torch] not installed; scipy-only self-test complete')
        return

    import torch

    # Correctness: torch on CPU must match scipy exactly.
    t0 = time.time()
    tc = alg_smoothed_local_max_3d(vol, backend='torch', device='cpu', **params)
    t_tc = time.time() - t0
    s_cpu = set(map(tuple, cpu.astype(np.int64)))
    s_tc = set(map(tuple, tc.astype(np.int64)))
    print(f'[torch:cpu] {len(tc):5d} peaks in {t_tc * 1e3:7.1f} ms  '
          f'identical_to_scipy={s_cpu == s_tc}  '
          f'only_scipy={len(s_cpu - s_tc)}  only_torch={len(s_tc - s_cpu)}')

    # Speed: torch on the best available GPU.
    if not gpu_available():
        print('[torch] no GPU (cuda/mps) available; ran CPU parity only')
        return

    dev = select_device(None)
    _ = alg_smoothed_local_max_3d(vol, backend='torch', device=str(dev), **params)
    if dev.type == 'cuda':
        torch.cuda.synchronize()
    t0 = time.time()
    gpu = alg_smoothed_local_max_3d(vol, backend='torch', device=str(dev), **params)
    if dev.type == 'cuda':
        torch.cuda.synchronize()
    t_gpu = time.time() - t0
    s_gpu = set(map(tuple, gpu.astype(np.int64)))
    print(f'[torch:{dev.type}] {len(gpu):5d} peaks in {t_gpu * 1e3:7.1f} ms  '
          f'identical_to_scipy={s_cpu == s_gpu}  speedup={t_cpu / t_gpu:.1f}x')


if __name__ == '__main__':
    run()
