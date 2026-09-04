#!/bin/bash
# Dashboard sidecar launcher (Sep 2026) — its own tmux session so it
# OUTLIVES machine stops/crashes: the whole point is start/stop and the
# room view when the machine is down.
#
#   start:   dashboard/start_dashboard.sh
#   open:    http://<host-or-tailscale-ip>:8800   (bound 0.0.0.0; Tailscale is the access control)
#   watch:   tmux attach -t impostor-dashboard
#   stop:    dashboard/stop_dashboard.sh
#
# Optional boot persistence (not installed by default) — systemd user unit:
#   ~/.config/systemd/user/impostor-dashboard.service
#     [Unit]
#     Description=LOVE YOURSELF dashboard sidecar
#     [Service]
#     WorkingDirectory=%h/LOVE_YOURSELF_refactor_fixed_EXTREMELY-EPIC-BUILD---Copy
#     ExecStart=%h/LOVE_YOURSELF_refactor_fixed_EXTREMELY-EPIC-BUILD---Copy/.venv/bin/python dashboard/server.py
#     Restart=always
#     [Install]
#     WantedBy=default.target
#   then: systemctl --user enable --now impostor-dashboard && loginctl enable-linger $USER
#   (if using systemd, don't also run this script)

SESSION_NAME="impostor-dashboard"
REPO="$(cd "$(dirname "$0")/.." && pwd)"

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "[dashboard] session '$SESSION_NAME' already running — attach: tmux attach -t $SESSION_NAME"
  exit 0
fi

tmux new-session -d -s "$SESSION_NAME" -n Dashboard
tmux send-keys -t "$SESSION_NAME:0" "cd $REPO" C-m
tmux send-keys -t "$SESSION_NAME:0" \
  'while true; do echo "=== dashboard sidecar starting ==="; .venv/bin/python dashboard/server.py; echo "=== sidecar exited — restarting in 3s (dashboard/stop_dashboard.sh to end) ==="; sleep 3; done' C-m

echo "[dashboard] session '$SESSION_NAME' running on http://0.0.0.0:8800"
echo "[dashboard] attach:  tmux attach -t $SESSION_NAME"
echo "[dashboard] stop:    $REPO/dashboard/stop_dashboard.sh"
