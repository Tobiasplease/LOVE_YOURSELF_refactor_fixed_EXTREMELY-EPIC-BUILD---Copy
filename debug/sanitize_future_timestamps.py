"""Clamp future timestamps out of persistent state files (the RTC skew cleanup).

The machine's RTC runs ~53 days fast; boots start in October until NTP steps
the clock back. State written during October phases carries future epochs,
which breaks every decay/aging computation (a "last seen in the future" entry
never decays). This walks the state JSONs and clamps any timestamp-shaped
value in the future back to now. Backs up each file first; refuses while
machine.py runs. Run logs (<run>-event-log.json) are left alone — history
stays history.

Usage:
    python debug/sanitize_future_timestamps.py          # dry run, report only
    python debug/sanitize_future_timestamps.py --apply
"""

import json
import os
import re
import shutil
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import MOOD_SNAPSHOT_FOLDER

STATE_FILES = [
    "spatial_registry.json",
    "vocab_promotion.json",
    "presence_arrivals.json",
    "body_schema.json",
    "machine_identity.json",
    "lifetime_state.json",
    "system_state.json",
    "drawing_memory.json",
    "episodic_events.json",
    "durable_ledger.json",
    "context_compression_cache.json",
    "activation_snapshot.json",
]

KEY_RE = re.compile(r"(^|_)(ts|time|timestamp|seen|since|at|formed|spent|updated|hit|start|end|promoted|audit|sample|harvest)($|_)", re.I)


def clamp(node, now, horizon, path=""):
    fixed = []
    if isinstance(node, dict):
        for k, v in node.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool) and KEY_RE.search(str(k)) and horizon < v < 2.0e9:
                fixed.append((f"{path}.{k}", v))
                node[k] = now
            else:
                fixed += clamp(v, now, horizon, f"{path}.{k}")
    elif isinstance(node, list):
        for i, v in enumerate(node):
            fixed += clamp(v, now, horizon, f"{path}[{i}]")
    return fixed


def main():
    apply = "--apply" in sys.argv
    check = subprocess.run(["pgrep", "-f", "venv/bin/python.*machine.py"], capture_output=True, text=True)
    if check.stdout.strip():
        print("REFUSED: machine.py is running — its next save would clobber this.")
        sys.exit(1)

    now = time.time()
    horizon = now + 3600.0  # anything more than an hour in the future is skew
    total = 0
    for name in STATE_FILES:
        path = os.path.join(MOOD_SNAPSHOT_FOLDER, name)
        if not os.path.exists(path):
            continue
        try:
            with open(path) as f:
                data = json.load(f)
        except Exception as e:
            print(f"{name}: unreadable ({e}) — skipped")
            continue
        fixed = clamp(data, now, horizon)
        if not fixed:
            print(f"{name}: clean")
            continue
        total += len(fixed)
        print(f"{name}: {len(fixed)} future timestamps" + ("" if apply else " (dry run)"))
        for p, v in fixed[:4]:
            drift_days = (v - now) / 86400.0
            print(f"    {p} = {v:.0f} (+{drift_days:.1f} days)")
        if len(fixed) > 4:
            print(f"    ... and {len(fixed) - 4} more")
        if apply:
            shutil.copy(path, path + ".pre-clock-fix-bak")
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
    print(f"\n{total} future timestamps {'clamped' if apply else 'found (re-run with --apply)'}")


if __name__ == "__main__":
    main()
