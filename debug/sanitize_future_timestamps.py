"""Clamp future timestamps out of persistent state files (the RTC skew cleanup).

The machine's RTC runs ~53 days fast; boots start in October until NTP steps
the clock back. State written during October phases carries future epochs,
which breaks every decay/aging computation (a "last seen in the future" entry
never decays). This walks the state JSONs and pulls timestamp-shaped future
values back into the PAST — never to now. Clamping to now() converts garbage
into false recency, which is worse than the skew itself: on Aug 31 a clamp of
three executed-drawing stamps to "now" had the machine told "your last drawing
reached the paper a few minutes ago" every caption, all day, and it reasonably
concluded it was mid-drawing. The repair is a uniform per-file shift: skew is
a constant offset per boot era, so subtracting one delta preserves both the
order and the relative spacing of every affected stamp, landing the newest a
full day in the past ("yesterday", never "just now"). Backs up each file
first; refuses while machine.py runs. Run logs (<run>-event-log.json) are
left alone — history stays history.

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


MARGIN_S = 86400.0  # the newest repaired stamp lands this far in the past


def collect(node, horizon, path=""):
    """Every timestamp-shaped value beyond the horizon, with its container."""
    found = []
    if isinstance(node, dict):
        for k, v in node.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool) and KEY_RE.search(str(k)) and horizon < v < 2.0e9:
                found.append((node, k, v, f"{path}.{k}"))
            else:
                found += collect(v, horizon, f"{path}.{k}")
    elif isinstance(node, list):
        for i, v in enumerate(node):
            found += collect(v, horizon, f"{path}[{i}]")
    return found


def repair(found, now):
    """Uniform shift into the past: order and spacing preserved, newest lands
    MARGIN_S ago. Returns (shift_seconds, [(path, old, new), ...])."""
    shift = max(v for _, _, v, _ in found) - (now - MARGIN_S)
    fixed = []
    for node, k, v, p in found:
        node[k] = v - shift
        fixed.append((p, v, node[k]))
    return shift, fixed


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
        found = collect(data, horizon)
        if not found:
            print(f"{name}: clean")
            continue
        total += len(found)
        shift, fixed = repair(found, now)
        print(f"{name}: {len(found)} future timestamps, shifted back {shift / 86400.0:.1f} days" + ("" if apply else " (dry run)"))
        for p, old, new in fixed[:4]:
            print(f"    {p}: {old:.0f} (+{(old - now) / 86400.0:.1f}d) -> {new:.0f} ({(now - new) / 86400.0:.1f}d ago)")
        if len(fixed) > 4:
            print(f"    ... and {len(fixed) - 4} more")
        if apply:
            shutil.copy(path, path + ".pre-clock-fix-bak")
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
    print(f"\n{total} future timestamps {'shifted into the past' if apply else 'found (re-run with --apply)'}")


if __name__ == "__main__":
    main()
