"""Re-seed the stores for mind mode (Sep 5 eve). The ledgers hold a month of
failure narrative (44 durable facts, 50 wants, 55 threads, all pen/paper/
hesitation) that would re-infect any new shape. ARCHIVE — never delete — into
event_log/archive-<stamp>/ and let the singletons start empty. Kept: the
events (episodic_events.json — the life), lifetime_state.json (first boot),
spatial_registry.json (the room as known), presence_arrivals.json, and the
mind thread itself.
Run with the machine STOPPED:  python debug/reseed_stores.py [--dry]
"""
import os
import shutil
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG = os.path.join(ROOT, "event_log")
ARCHIVE = ("durable_ledger.json", "want_ledger.json", "lore_ledger.json", "machine_identity.json", "effigy_memory.json", "recent_memory.json", "last_caption.txt")
dry = "--dry" in sys.argv
stamp = time.strftime("%Y%m%d-%H%M")
dest = os.path.join(LOG, f"archive-{stamp}")
moved = []
for name in ARCHIVE:
    src = os.path.join(LOG, name)
    if os.path.exists(src):
        if not dry:
            os.makedirs(dest, exist_ok=True)
            shutil.move(src, os.path.join(dest, name))
        moved.append(name)
print(("would archive" if dry else "archived") + f" → {dest}: {moved or 'nothing'}")
