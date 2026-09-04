"""One-shot: clear the stream tail + last caption + recent_memory from system_state.json so
the next boot starts the register flywheel CLEAN (Sep 4 — the old-register
tail otherwise splices into every restart and re-teaches itself).

Run BETWEEN machine-stop and machine-start. Refuses while machine.py runs.
Ledgers, identity, lore, registry are untouched — this clears only the
in-context seed.

Run: python debug/fresh_stream.py
"""

import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config.config import MOOD_SNAPSHOT_FOLDER  # noqa: E402

STATE = os.path.join(MOOD_SNAPSHOT_FOLDER, "system_state.json")

if subprocess.run(["pgrep", "-f", "python machine.py"], capture_output=True).stdout.strip():
    print("machine.py is running — stop it first (this edits its state file)")
    sys.exit(1)

if not os.path.exists(STATE):
    print("no system_state.json — nothing to clear")
    sys.exit(0)

state = json.load(open(STATE))
cap = state.get("captioner", {})
had = len(cap.get("stream_tail", []) or [])
cap["stream_tail"] = []
cap["last_caption"] = ""
had_mem = len(cap.get("recent_memory", []) or [])
cap["recent_memory"] = []  # memory mode's caption thread splices from it — same in-context seed (Sep 4 evening)
json.dump(state, open(STATE, "w"), indent=2, ensure_ascii=False)
print(f"cleared: {had} tail entries + last_caption + {had_mem} recent_memory entries. Next boot seeds the stream fresh.")
