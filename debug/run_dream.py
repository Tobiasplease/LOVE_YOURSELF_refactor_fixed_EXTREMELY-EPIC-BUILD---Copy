"""Run the dream pass now, on demand (Sep 6). Usage:
  python debug/run_dream.py [hours=24] [--dry]
--dry prints the records and the page without storing them."""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
hours = float(next((a for a in sys.argv[1:] if not a.startswith("--")), 24))
dry = "--dry" in sys.argv
from captioner.dream import run_dream  # noqa: E402
from captioner.mind import Mind  # noqa: E402

m = Mind(None, path=os.path.join("event_log", "mind_thread.json"), backfill=False)
if dry:
    m._index = False
t0 = time.time()
res = run_dream(m, now=t0, since=t0 - hours * 3600, dry=dry)
print(f"day: {res.get('day_tokens')} tokens | {len(res.get('records', []))} records | page {len((res.get('page') or '').split())} words | {time.time() - t0:.0f}s {'(dry)' if dry else '(stored)'}")
if res.get("skipped"):
    print("skipped:", res["skipped"])
print("\n— records —")
for r in res.get("records", []):
    print(" ·", r)
print("\n— the night's page —\n")
print(res.get("page", ""))
