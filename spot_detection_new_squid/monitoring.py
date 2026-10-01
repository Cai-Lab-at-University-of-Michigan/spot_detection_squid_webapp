"""Background RSS memory monitor for the pipeline process tree."""
from __future__ import annotations

import csv
import os
import threading
import time

import psutil


def monitor_memory_tree(interval=0.5, output_csv="memory_log.csv"):
    """Start a background thread that samples RSS (parent + all children)
    every `interval` seconds. Returns a `stop_and_save()` callback that stops
    the thread, writes the log to `output_csv`, and returns the peak MB
    observed."""
    parent = psutil.Process(os.getpid())
    log = []
    peak = 0
    stop_event = threading.Event()

    def _monitor():
        nonlocal peak
        while not stop_event.is_set():
            try:
                mem = parent.memory_info().rss
                for child in parent.children(recursive=True):
                    try:
                        mem += child.memory_info().rss
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                mem_mb = mem / 1024 ** 2
                log.append((time.time(), mem_mb))
                if mem_mb > peak:
                    peak = mem_mb
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
            time.sleep(interval)

    threading.Thread(target=_monitor, daemon=True).start()

    def stop_and_save():
        stop_event.set()
        if log:
            t0 = log[0][0]
            with open(output_csv, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["time_s", "rss_mb"])
                writer.writerows([(t - t0, mem) for t, mem in log])
            print(f"Peak memory (parent + workers): {peak:.1f} MB")
            print(f"Memory log saved to {output_csv}")
        return peak

    return stop_and_save
