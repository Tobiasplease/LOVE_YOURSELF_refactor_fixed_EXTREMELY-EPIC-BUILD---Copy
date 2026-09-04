#!/bin/bash
# End the dashboard sidecar session (the machine is untouched).
tmux kill-session -t impostor-dashboard 2>/dev/null \
  && echo "[dashboard] stopped" \
  || echo "[dashboard] no session found"
