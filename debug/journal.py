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
last_ts, last_hour, para = None, None, []


def flush():
    global para
    if para:
        print(" ".join(para).strip())
        print()
    para = []


for e in entries:
    ts = e["ts"]
    hour = time.strftime("%H", time.localtime(ts))
    if hour != last_hour:
        flush()
        print(f"— {time.strftime('%A %d %B, %H:00', time.localtime(ts))} —\n")
        last_hour = hour
    if last_ts is not None and (ts - last_ts >= 180 or e["kind"] in ("look", "reflection", "wake")):
        flush()
    mark = {"look": "◦ ", "reflection": "» ", "wake": "* "}.get(e["kind"], "")
    para.append(mark + e["text"].strip())
    last_ts = ts
flush()
print(f"[{len(entries)} entries]")
