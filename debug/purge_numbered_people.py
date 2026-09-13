#!/usr/bin/env python3
"""Sep 13 evening — take the invented numbered people out of the machine's memory.

The chain: the stretch line reported counts; at 14:27:52 it first said "a hundred
times or more"; at 14:28:42 the captions began opening "104 looks now…"; by 16:32
the numeral had become a person ("I feel 104's awareness as an indifference
rather than attention") and by 17:44 the reveries were full of 102 and 103
sitting in chairs in an empty room. The counts are gone from the prompt now, but
the machine reinfects itself every restart: system_state.json carries the stream
tail into the window, and the reveries and identity facts are read back by the
reflection.

A bare three-digit number has no honest use in this machine's prose — it is
never a time (those are HH:MM), never a measurement it makes. Entries carrying
one are removed whole, because each is a sentence about somebody who was not
there. Every file is backed up first.

    .venv/bin/python debug/purge_numbered_people.py           # dry run
    .venv/bin/python debug/purge_numbered_people.py --apply
"""
import json
import os
import re
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import MOOD_SNAPSHOT_FOLDER  # noqa: E402

APPLY = "--apply" in sys.argv
# A three-digit number used AS SOMEBODY: possessive, or subject of a verb, or
# paired with another. Matched against string values only — matching the JSON
# dump caught the \u201c escapes in ordinary prose and would have deleted
# twenty real thoughts. A bare "100 years" stays; "102 is hunched over that
# desk" and "103's legs" go.
NUM = re.compile(
    r"\b\d{3}'s\b"
    r"|\b\d{3}\s+and\s+\d{3}\b"
    r"|\b\d{3}\s+(?:is|was|are|were|has|have|had|left|stayed|sat|sits|sitting|stands|standing|moved|disappeared|vanished|keeps|kept|hunched|leaned|shifted|turned)\b",
    re.I,
)


def carries(node) -> bool:
    """Does any string anywhere in this entry use a number as a person?"""
    if isinstance(node, str):
        return bool(NUM.search(node))
    if isinstance(node, dict):
        return any(carries(v) for v in node.values())
    if isinstance(node, list):
        return any(carries(v) for v in node)
    return False
TARGETS = {
    "system_state.json": [("captioner", "stream_tail"), ("captioner", "recent_memory")],
    "lore_ledger.json": [("reveries",)],
    "machine_identity.json": [("journal",), ("self_notes",), ("events",)],
    "durable_ledger.json": [("facts",)],
    "mind_thread.json": [("thread",)],
}


def dig(obj, path):
    for k in path:
        if not isinstance(obj, dict) or k not in obj:
            return None, None
        parent, obj = obj, obj[k]
    return parent, path[-1]


total = 0
for name, paths in TARGETS.items():
    p = os.path.join(MOOD_SNAPSHOT_FOLDER, name)
    if not os.path.exists(p):
        print(f"{name}: not here")
        continue
    with open(p) as f:
        data = json.load(f)
    removed_here = 0
    for path in paths:
        parent, key = dig(data, path)
        if parent is None or not isinstance(parent.get(key), list):
            continue
        items = parent[key]
        keep = [it for it in items if not carries(it)]
        gone = [it for it in items if carries(it)]
        if gone:
            print(f"\n{name} · {'.'.join(path)}: {len(gone)} of {len(items)} entries carry an invented number")
            for it in gone[:3]:
                t = it if isinstance(it, str) else (it.get("text") or it.get("note") or it.get("fact") or it.get("summary") or it.get("event") or json.dumps(it))
                print(f"   - {str(t)[:105]}")
        removed_here += len(gone)
        parent[key] = keep
    # Nested: a thread's own text can be clean while a RETURN recorded one of
    # these captions as where the thought got to (the write-back built the same
    # day faithfully wrote down the hallucination). Drop the return, keep the
    # thread.
    if name == "lore_ledger.json":
        for t in data.get("threads") or []:
            rs = t.get("returns") or []
            keep = [r for r in rs if not carries(r)]
            if len(keep) != len(rs):
                print(f"\n{name} · a thread's returns: {len(rs) - len(keep)} dropped from {t.get('text','')[:60]!r}")
                t["returns"] = keep
                removed_here += len(rs) - len(keep)

    # the single-string slots
    if name == "system_state.json":
        c = data.get("captioner") or {}
        for slot in ("last_caption",):
            if isinstance(c.get(slot), str) and NUM.search(c[slot]):
                print(f"\n{name} · captioner.{slot}: cleared — {c[slot][:90]!r}")
                c[slot] = ""
                removed_here += 1
    total += removed_here
    if APPLY and removed_here:
        shutil.copy2(p, f"{p}.bak-numpeople-{int(time.time())}")
        with open(p, "w") as f:
            json.dump(data, f, indent=2)
        print(f"   → {name} rewritten ({removed_here} entries removed; backup kept)")

print(f"\n{total} memories of people who were not there." + ("" if APPLY else "  Dry run — pass --apply to remove them."))
