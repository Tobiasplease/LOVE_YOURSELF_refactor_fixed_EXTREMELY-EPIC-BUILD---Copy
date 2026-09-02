#!/bin/bash
# Canonical supervised launcher (Sep 2 2026) — the 3.8 stack under tmux with
# auto-restart and a STOP-file kill, so the machine can run unattended.
#
#   start:   ./start_impostor.sh
#   watch:   tmux attach -t impostor-system   (detach: Ctrl+B, D)
#   stop:    ./stop_machine.sh               (works over any SSH session)
#
# ComfyUI is NOT launched here — utils/comfy_launcher.py auto-starts it at
# machine boot if port 8188 is silent (detached, survives restarts; log at
# event_log/comfyui.log). One launcher, one loop, one kill.
# History: this script used to pin the parked 3.6 arm (run_27b.sh) — the
# exhibition launchers booted the wrong model (trim-plan-aug30 §4).

SESSION_NAME="impostor-system"
REPO="$(cd "$(dirname "$0")" && pwd)"
STOP_FILE="$REPO/STOP"

rm -f "$STOP_FILE"
tmux kill-session -t "$SESSION_NAME" 2>/dev/null

tmux new-session -d -s "$SESSION_NAME" -n Machine
tmux send-keys -t "$SESSION_NAME:0" "cd $REPO" C-m
tmux send-keys -t "$SESSION_NAME:0" \
  'while [ ! -f STOP ]; do echo "=== starting machine (3.8 stack, code default) ==="; ./run_38.sh; echo "=== machine exited — restarting in 5s (./stop_machine.sh to end) ==="; sleep 5; done; echo "=== STOP present — supervisor loop ended ==="' C-m

echo "[start] session '$SESSION_NAME' running the 3.8 stack"
echo "[start] attach:  tmux attach -t $SESSION_NAME"
echo "[start] stop:    $REPO/stop_machine.sh"
