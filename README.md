# Spot detection web app

A browser front end for the SQUID spot-detection pipeline (the local-FDR
entry point, `run_spot_detection_localfdr.py`). It runs on the GPU machine and
you use it from a browser on your own computer.

This folder is self-contained: `spot_detection_new_squid/` inside it is a copy
of the pipeline package, and nothing here imports from
`../spot_detection_new_squid`. Changes to the original package do not reach the
web app until you copy them in again:

```bash
rsync -a --exclude __pycache__ ../spot_detection_new_squid/ spot_detection_new_squid/
```

## Starting it

```bash
cd ~/spot_detection_code/spot_detection_webapp
tmux new -s spotweb ./start.sh        # detach: Ctrl-b d · reattach: tmux attach -t spotweb
```

It prints a URL like `http://localhost:8765/?token=...`. Open it on your own
computer:

- **VS Code Remote-SSH:** VS Code forwards port 8765 automatically (check the
  *Ports* panel). Then open the printed URL.
- **Plain SSH:** `ssh -L 8765:localhost:8765 <user>@<gpu-machine>`, then open
  the URL.

The server listens on 127.0.0.1 only. Each request also needs the token, so
other users on the GPU machine can't reach it either. After the first visit the
token is stored in a browser cookie. It lives in `jobs/.token`; delete that
file and restart to issue a new one.

Options: `./start.sh --port 9000 --jobs-dir /big/disk/spotweb_jobs`.

## Input modes

**On the remote machine.** Pick a squid folder (or a single `.ome.tiff`, or an
old-layout folder) on the GPU machine with *Browse…*. Nothing is copied. The
job runs `run_pipeline` exactly as the script does, and keeps running if you
close the browser.

**From your computer (Globus).** The fast way to process data that lives on
your own computer. The remote machine pulls tiles through Globus, processes
each one as it arrives, and deletes it once it has been read.

- *Browse your computer…* lists your computer's files through Globus. Pick the
  acquisition folder (in either layout, see below) or a single `.ome.tiff`.
- At most the tile being processed plus *Tiles to fetch ahead* (default 2) are
  on the remote disk at once. With ~10 GB tiles that is ~30 GB however big the
  acquisition is. At most two Globus transfers run at a time, in processing
  order, so the next tile needed is never held up by the ones behind it.
- Each tile is one Globus transfer task. A tile becomes visible to the
  pipeline only once its whole task has succeeded. A failed task is retried
  twice before the run fails with Globus's error message.
- Transfers are run by the remote machine, so **you can close the browser
  tab**. Your computer has to stay awake, with Globus Connect Personal
  running, until the last tile has arrived. If it drops out, Globus pauses the
  transfer, the run waits and its card says why, and everything resumes when
  it reconnects.
- *Verify checksums* (on by default) has Globus check every file after it
  arrives. That's safer, but adds some time per tile on both machines.
- Cancelling a run cancels its Globus transfers.

The page checks four things before allowing a run, and the server checks them
again when the run starts:

1. **Globus is logged in on the remote machine.** Do this once, in a terminal
   on the remote machine:
   `~/miniconda3/envs/spotdet/bin/globus login --no-local-server`
2. **Globus Connect Personal is running and connected on the remote machine.**
   This is asked of the local install and of Globus. *Start it* starts it if
   it isn't running.
3. **The remote Globus Connect Personal may write into the jobs folder.** By
   default it shares your home folder (`~/.globusonline/lta/config-paths`),
   and the jobs folder is under it.
4. **Globus Connect Personal is connected on your computer.** The menu lists
   your account's personal endpoints and whether each is connected.

If Globus asks for an extra consent the first time it touches an endpoint,
the error shows the exact `globus session consent ...` command to run once on
the remote machine.

**Browser upload (fallback).** The same tile-by-tile processing, with the
browser sending the data over the SSH tunnel instead of Globus. It's slower,
and the tab must stay open. Use it when Globus isn't available.

- *Choose squid folder…* or *Choose single .ome.tiff…*. Choosing sends
  nothing yet.
- If the tab closes, the run waits. Reopen the page and click *Re-select
  folder to resume*; partly sent tiles continue where they stopped.
- Each upload is split into 32 MB pieces (change with `SPOTDET_CHUNK_MB`),
  and *Parallel upload streams* sends several at once.

**Rules for both ways of bringing data from your computer:**

- Either layout works:
  - **new layout:** `ome_tiff/R0_<fov>.ome.tiff`, one file per tile;
  - **old layout:** `0/R0_<fov>_<z>_<channel>.tiff`, one file per slice. A
    tile is then all of its slice files, and it only becomes visible to the
    pipeline once all of them have arrived. Set `resolution`: these TIFFs
    carry no pixel size.
- The small metadata files (`coordinates.csv`, `0/coordinates.csv`,
  `acquisition parameters.json`, acquisition YAMLs) come along first.
- With `auto_calibrate`, the calibration tile (the middle one, or
  `calibration_kwargs['tiles']`) is fetched first.
- `output_corrected` is not available, because it would keep a corrected copy
  of every tile.
- Results go to `jobs/<id>/output` unless you set `output_folder`. *Download
  results (.zip)* gets everything except image files.

## Parameter files

Every file field (`channel_parameter_file`, `genes_file`,
`virtual_stitch_coords_path`) takes one of:

- a path on the GPU machine (*Browse remote…*), or
- a file from your computer (*Upload local…*). It is copied to
  `jobs/_uploads/`.

The form's defaults are read from the `USER PARAMETERS` block of
`spot_detection_new_squid/run_spot_detection_localfdr.py` in this folder. Paths
there that don't exist on this machine are left blank. The form remembers your
last values in the browser; *Reset to script defaults* clears them.

## GPUs

The GPU menu sets `CUDA_VISIBLE_DEVICES` for the job (PCI order, same numbers
as `nvidia-smi`). It defaults to the GPU with the most free memory that isn't
running a job from this app. Several jobs can run at once on different GPUs.

## Files

| | |
|---|---|
| `server.py` | web server (FastAPI): jobs, Globus checks, uploads, file browsers |
| `runner.py` | runs one job in its own process; output goes to `jobs/<id>/log.txt` |
| `streaming_pipeline.py` | `run_pipeline` reworked to take tiles as they arrive, then delete them |
| `params.py` | form fields and parsing of their values |
| `jobstatus.py` | `status.json` shared between runner and server |
| `staging.py` | where arriving tiles are put while a job runs |
| `globus_link.py` | Globus checks, listing and transfers, through the `globus` CLI |
| `globus_fetch.py` | keeps the next tiles in flight with Globus during a run |
| `globus_fake.py` | stand-in for Globus used in testing (`SPOTDET_GLOBUS_FAKE=<dir>`) |
| `static/` | the page |
| `spot_detection_new_squid/` | copy of the pipeline package |
| `jobs/` | one folder per run (created on first start) |

## Environment

Uses the `spotdet` conda environment (`~/miniconda3/envs/spotdet`). To recreate
it:

```bash
conda create -n spotdet -c conda-forge --override-channels python=3.12
~/miniconda3/envs/spotdet/bin/pip install -r requirements.txt
```

`start.sh` uses that environment's Python; set `SPOTDET_PYTHON` to use another.
