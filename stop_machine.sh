#!/bin/bash
# Remote-safe kill (Sep 2 2026): end the supervisor loop, then stop the
# machine gracefully (SIGINT = the same clean shutdown as Ctrl+C — state
# saves, servos detach, GRBL pen-up). Safe to run twice; safe from any SSH.
REPO="$(cd "$(dirname "$0")" && pwd)"
touch "$REPO/STOP"
pkill -INT -f "machine.py" 2>/dev/null && echo "[stop] SIGINT sent — machine is shutting down (state saves on exit)" || echo "[stop] no machine process found"
echo "[stop] STOP file set — the supervisor loop will not restart it"
echo "[stop] to run again later: $REPO/start_impostor.sh"
