"""Runs one job in its own process: `python runner.py <job_dir>`.

The web server starts this with stdout/stderr going to <job_dir>/log.txt and
CUDA_VISIBLE_DEVICES set to the chosen GPU. Modes, from job.json:

remote         data already on this machine: run_pipeline, exactly as
               run_spot_detection_localfdr.py would call it.
globus_folder  a squid folder (either layout) on your computer, pulled tile
               by tile with Globus (globus_fetch.GlobusFetcher, a thread in
               this process) and processed as tiles arrive:
               streaming_pipeline.run_streaming_pipeline.
globus_single  one .ome.tiff on your computer: fetched whole, then run_pipeline.
upload_folder, upload_single
               the same, with the browser sending the data instead of Globus.

Transferred tiles are deleted when the job ends, whether it succeeded or not,
and Globus transfers still in flight are cancelled.
"""
import json
import os
import signal
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
# The vendored pipeline package lives next to this file.
sys.path.insert(0, HERE)

from jobstatus import StatusFile  # noqa: E402
from params import parse_values  # noqa: E402
from staging import remove_staged_images, wait_until_exists  # noqa: E402


def wait_for(path, staging, status, phase):
    if not os.path.exists(path):
        status.update(phase=phase)
        print(f'{phase}...', flush=True)
    wait_until_exists(path, staging)


def main(job_dir):
    with open(os.path.join(job_dir, 'job.json')) as f:
        job = json.load(f)
    status = StatusFile(os.path.join(job_dir, 'status.json'))
    status.update(state='running', phase='starting', pid=os.getpid())
    staging = os.path.join(job_dir, 'staging')
    exit_code = 0
    fetcher = None
    # Cancel from the web page sends SIGTERM; exit through `finally` so the
    # Globus transfers are cancelled and the tiles cleaned up.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    source, _, shape = job['mode'].partition('_')   # remote | upload | globus, folder | single
    try:
        kwargs = parse_values(job['params'])
        kwargs['output_folder'] = job['output_folder']
        print(f'Job {job["id"]} ({job["mode"]}) on GPU {os.environ.get("CUDA_VISIBLE_DEVICES", "default")}',
              flush=True)

        if job['mode'] == 'remote':
            from spot_detection_new_squid.orchestrator import run_pipeline
            status.update(phase='running pipeline')
            run_pipeline(squid_folder=job['squid_folder'], **kwargs)

        elif source not in ('upload', 'globus'):
            raise ValueError(f'unknown job mode {job["mode"]!r}')

        elif shape == 'single':
            from spot_detection_new_squid.orchestrator import run_pipeline
            if source == 'globus':
                fetcher = start_fetcher(job, staging, status)
            tile = job['manifest']['tiles'][0]
            path = os.path.join(staging, tile['ready'])
            wait_for(path, staging, status, 'waiting for the file to arrive')
            status.update(phase='running pipeline')
            run_pipeline(squid_folder=path, **kwargs)

        else:
            from streaming_pipeline import run_streaming_pipeline
            if source == 'globus':
                fetcher = start_fetcher(job, staging, status)
            wait_for(os.path.join(staging, '.sidecars_done'), staging, status,
                     'waiting for metadata files')
            run_streaming_pipeline(
                staging_dir=staging,
                layout=job['manifest']['layout'],
                tile_order=job['manifest']['order'],
                status=status,
                **kwargs,
            )

        status.update(state='finished', phase='done', waiting_for=[])
        print('\nJob finished.', flush=True)
    except BaseException as e:  # noqa: BLE001 -- record *why* the job died, then exit
        traceback.print_exc()
        sys.stdout.flush()
        status.update(state='failed', phase='failed', error=f'{type(e).__name__}: {e}')
        exit_code = 1
    finally:
        if fetcher is not None:
            fetcher.stop()
        if source in ('upload', 'globus'):
            # Image data only; the small sidecar files are kept for reference.
            remove_staged_images(staging, job['manifest'])
    return exit_code


def start_fetcher(job, staging, status):
    from globus_fetch import GlobusFetcher
    g = job['globus']
    print(f'Globus: {g["source_endpoint_name"]}:{g["source_root"]} -> this machine '
          f'({"with" if g.get("verify_checksum", True) else "without"} checksum verification)',
          flush=True)
    fetcher = GlobusFetcher(job, staging, status)
    fetcher.start()
    return fetcher


if __name__ == '__main__':
    sys.exit(main(sys.argv[1]))
