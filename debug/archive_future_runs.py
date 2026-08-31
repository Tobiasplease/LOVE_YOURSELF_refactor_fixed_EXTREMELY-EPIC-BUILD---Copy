"""Quarantine clock-skew ghost runs (the dying-RTC era, Oct-2026 timestamps).

Run logs whose start_time_iso is more than a day in the future are physically
impossible — they were written while the RTC was ~53 days fast. They sort as
"newest" forever and have poisoned every mtime/iso-based analysis since Aug 10.
Moves each ghost log + its -images dir into event_log/archive_clock_skew/.
History is kept, just out of the way. Refuses nothing while machine.py runs —
moving dead runs' files is safe.

Usage:
    python debug/archive_future_runs.py          # dry run
    python debug/archive_future_runs.py --apply
"""

import glob
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import MOOD_SNAPSHOT_FOLDER

ARCHIVE = os.path.join(MOOD_SNAPSHOT_FOLDER, "archive_clock_skew")
apply = "--apply" in sys.argv
horizon = time.time() + 86400

ghosts = []
for path in glob.glob(os.path.join(MOOD_SNAPSHOT_FOLDER, "*-event-log.json")):
    try:
        head = open(path, errors="replace").read(400)
        i = head.find('"start_time"')
        if i < 0:
            continue
        start = float(head[i:].split(":", 1)[1].split(",")[0].strip())
        if start > horizon:
            ghosts.append((path, head[head.find('"start_time_iso"') :].split('"')[3] if '"start_time_iso"' in head else "?"))
    except Exception:
        continue

print(f"{len(ghosts)} future-dated ghost runs found")
for path, iso in sorted(ghosts, key=lambda g: g[1]):
    run_id = os.path.basename(path).split("-")[0]
    images = os.path.join(MOOD_SNAPSHOT_FOLDER, f"{run_id}-images")
    tag = f"{iso}  {os.path.basename(path)}" + ("  (+images)" if os.path.isdir(images) else "")
    if apply:
        os.makedirs(ARCHIVE, exist_ok=True)
        shutil.move(path, os.path.join(ARCHIVE, os.path.basename(path)))
        if os.path.isdir(images):
            shutil.move(images, os.path.join(ARCHIVE, os.path.basename(images)))
        print("archived:", tag)
    else:
        print("would archive:", tag)

if not apply and ghosts:
    print("\n(re-run with --apply)")
