#!/usr/bin/env bash
# Start the spot-detection web app on this machine.
#
#   ./start.sh                 # port 8765
#   ./start.sh --port 9000     # any server.py option
#
# To keep it running after you disconnect, start it inside tmux:
#   tmux new -s spotweb ./start.sh      (detach: Ctrl-b d, reattach: tmux attach -t spotweb)
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${SPOTDET_PYTHON:-$HOME/miniconda3/envs/spotdet/bin/python}"
if [ ! -x "$PY" ]; then
  echo "Python not found at $PY -- create the environment (see README.md) or set SPOTDET_PYTHON." >&2
  exit 1
fi
exec "$PY" "$HERE/server.py" "$@"
