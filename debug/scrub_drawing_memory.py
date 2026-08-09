"""Remove error-sentinel entries from drawing memory.

The Aug 2 timeout bug stored a failed call's error text as a drawing: an entry
whose compressed_summary and comfy_prompt both begin "[WARNING] llama-server
API failed...". It has been sitting in the body of work ever since, fed into
every drawing-intent call as one of the machine's remembered drawings.

The reading side now skips such entries (get_executed_sequence), so this is
housekeeping rather than a fix — but a false drawing in the ledger will keep
surfacing anywhere else that reads history, so it should go.

Refuses to run while machine.py is alive: a live process holds drawing memory
in memory and its next save would overwrite the edit (learned the hard way when
a persona clear was silently clobbered).

    python debug/scrub_drawing_memory.py            # report only
    python debug/scrub_drawing_memory.py --apply    # back up and remove
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config

PATH = os.path.join(config.MOOD_SNAPSHOT_FOLDER, "drawing_memory.json")
SENTINELS = ("[WARNING]", "[ERROR]")


def is_poisoned(entry):
    for key in ("compressed_summary", "comfy_prompt", "narrative_thread", "emotional_tone"):
        v = entry.get(key)
        if isinstance(v, str) and v.lstrip().startswith(SENTINELS):
            return key
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="actually rewrite the file")
    args = ap.parse_args()

    if not os.path.exists(PATH):
        print(f"no drawing memory at {PATH}")
        return

    with open(PATH) as f:
        data = json.load(f)
    key = None
    if isinstance(data, list):
        history = data
    else:
        key = "drawings" if "drawings" in data else "history"
        history = data.get(key, [])

    bad = [(i, e, is_poisoned(e)) for i, e in enumerate(history)]
    bad = [(i, e, k) for i, e, k in bad if k]
    print(f"{len(history)} entries, {len(bad)} poisoned")
    for i, e, k in bad:
        print(f"  [{i}] {k}: {str(e.get(k))[:90]}")
    if not bad:
        return
    if not args.apply:
        print("\n(report only — re-run with --apply to remove)")
        return

    probe = subprocess.run(["pgrep", "-f", "machine.py"], capture_output=True, text=True)
    alive = [p for p in probe.stdout.split() if p.strip() and int(p) != os.getpid()]
    if alive:
        print(f"\nREFUSING: machine.py is running (pid {', '.join(alive)}) — its next save would undo this.")
        print("Stop the machine, then re-run.")
        sys.exit(1)

    backup = f"{PATH}.scrub-bak-{time.strftime('%Y%m%d_%H%M%S')}"
    shutil.copy2(PATH, backup)
    keep = [e for i, e in enumerate(history) if i not in {i for i, _, _ in bad}]
    if key:
        data[key] = keep
    else:
        data = keep
    with open(PATH, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nbackup: {backup}")
    print(f"removed {len(bad)} entries; {len(keep)} remain")


if __name__ == "__main__":
    main()
