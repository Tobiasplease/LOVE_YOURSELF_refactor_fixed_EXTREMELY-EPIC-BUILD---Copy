"""Remove person-claims-while-empty from event_log/mind_thread.json and the
thoughts index (Sep 7). The running text is the belief: a phantom left in it
is replayed for the next 30 turns. Stop the machine first.
python debug/scrub_mind_thread.py [hours=24] [--dry]"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
hours = float(next((a for a in sys.argv[1:] if not a.startswith("--")), 24))
dry = "--dry" in sys.argv
from captioner.mind import Mind  # noqa: E402
from utils.presence_text import is_phantom_presence  # noqa: E402

m = Mind(None, path=os.path.join("event_log", "mind_thread.json"), backfill=False)
if dry:
    m._index = False
now = time.time()
hits = [e for e in m.thread if e.get("text") and now - float(e.get("ts", 0)) <= hours * 3600 and is_phantom_presence(e["text"])]
print(f"{len(hits)} phantom entries in the last {hours:.0f} h (of {len(m.thread)} total)")
for e in hits[:5]:
    print("  ·", time.strftime("%H:%M", time.localtime(e["ts"])), e["text"][:90])
if dry:
    print("(dry — nothing removed)")
else:
    print("removed:", m.scrub_phantoms(now, window_s=hours * 3600))
