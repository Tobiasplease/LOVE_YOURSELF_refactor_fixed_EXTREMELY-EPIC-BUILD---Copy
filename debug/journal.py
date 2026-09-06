"""Render the mind thread as journal pages (Sep 6 morning). The artist's
target: 'put together it should form pages of what looks like an actual
journal.' Read-only. Usage: python debug/journal.py [hours=8] [--all]
Entries become running text; a new paragraph at a gap ≥ 3 min, at a look, or
at a reflection; an hour heading when the hour turns. Looks are marked with
a small glyph so the eye can find them, nothing else is annotated."""
import json
import os
import sys
import time

HOURS = float(sys.argv[1]) if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else 8
ALL = "--all" in sys.argv
d = json.load(open(os.path.join("event_log", "mind_thread.json")))
since = 0 if ALL else time.time() - HOURS * 3600
entries = [e for e in d["thread"] if e.get("text") and e.get("kind") in ("wake", "look", "think", "reflection", "memory") and e.get("ts", 0) >= since]
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from captioner.mind import Mind  # noqa: E402

by_hour = {}
for e in entries:
    by_hour.setdefault(time.strftime("%A %d %B, %H:00", time.localtime(e["ts"])), []).append(e)
for h, es in by_hour.items():
    marked = [dict(e, text=({"look": "◦ ", "reflection": "» ", "wake": "* ", "dream": "☾ "}.get(e["kind"], "") + e["text"].strip())) for e in es]
    print(f"— {h} —\n")
    print(Mind.running_text(marked))
    print()
print(f"[{len(entries)} entries]")
