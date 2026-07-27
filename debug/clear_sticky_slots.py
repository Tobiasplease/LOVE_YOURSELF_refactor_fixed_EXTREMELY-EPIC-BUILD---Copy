"""Clear the sticky identity slots (persona / desire / belief) SAFELY.

Why this exists: the July 27 persona clear was silently clobbered — the file
was edited while machine.py was running, and the live process's next
identity-save wrote its in-memory copy straight back. This script refuses to
run while machine.py is alive, backs up first, and clears only the three
re-injected slots. Ledgers (journal, events, self_notes, histories) are
untouched; the slots re-grow from the next reflections.

Usage: stop the machine, then  python debug/clear_sticky_slots.py
"""

import json
import os
import shutil
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config

IDENTITY_FILE = os.path.join(config.MOOD_SNAPSHOT_FOLDER, "machine_identity.json")

probe = subprocess.run(["pgrep", "-f", "machine.py"], capture_output=True, text=True)
alive = [p for p in probe.stdout.split() if p.strip() and int(p) != os.getpid()]
if alive:
    print(f"REFUSING: machine.py is running (pid {', '.join(alive)}) — its next identity save would clobber this clear.")
    print("Stop the machine first, then re-run.")
    sys.exit(1)

if not os.path.exists(IDENTITY_FILE):
    print(f"No identity file at {IDENTITY_FILE} — nothing to clear.")
    sys.exit(0)

stamp = time.strftime("%Y%m%d_%H%M%S")
backup = f"{IDENTITY_FILE}.slots-bak-{stamp}"
shutil.copy2(IDENTITY_FILE, backup)

with open(IDENTITY_FILE) as f:
    data = json.load(f)

cleared = {
    "persona": data.get("core_facts", {}).get("self", ""),
    "desire": data.get("current_desire", ""),
    "belief": data.get("current_belief", ""),
}
data.setdefault("core_facts", {})["self"] = ""
data["current_desire"] = ""
data["current_belief"] = ""
data["desire_since"] = 0.0

with open(IDENTITY_FILE, "w") as f:
    json.dump(data, f, indent=2)

print(f"Backup: {backup}")
for slot, val in cleared.items():
    print(f"cleared {slot}: {val!r}" if val else f"{slot}: was already empty")
print("Slots re-grow from the next reflections. Restore with: cp <backup> machine_identity.json")
