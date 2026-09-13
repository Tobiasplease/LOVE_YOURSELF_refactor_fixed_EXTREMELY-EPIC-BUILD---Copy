#!/usr/bin/env python3
"""Sep 13 — take the video clock out of the machine's stored words.

llama-server prepends a chunk marker ("[0m0.17s]") to every native video input.
On Sep 13 at 13:14:39 the model spoke one, wearing the stream's own separator
("0m0.17s — I'm looking at the black office chair now."), the window taught it,
and within two runs 72% of captions were chanting "1m40. 2m07." at each other.
captioner._strip_leaked_stamps now removes the marker before anything is stored,
but the tail already written to system_state.json reseeded the chant two minutes
into the next boot, so the stored copies have to be cleaned once.

This EDITS the machine's own sentences rather than deleting them: exactly the
marker is removed, the words stay. Run with the machine STOPPED — it rewrites
files the captioner owns. A .bak-<timestamp> copy is written first.

    .venv/bin/python debug/clean_video_stamps.py [--apply]
"""
import json
import os
import re
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import MOOD_SNAPSHOT_FOLDER  # noqa: E402

VIDEO_STAMP_RE = re.compile(r"(?:(?<=^)|(?<=[\s(]))\[?\s*\d{1,3}m\d{1,2}(?:\.\d+)?s?\s*\]?[.…,]*\s*[—–-]?\s*")
APPLY = "--apply" in sys.argv
TARGETS = ("system_state.json", "lore_ledger.json")


def clean(text):
    out = VIDEO_STAMP_RE.sub("", text)
    return " ".join(out.split())


def walk(node, hits):
    if isinstance(node, dict):
        return {k: walk(v, hits) for k, v in node.items()}
    if isinstance(node, list):
        return [walk(v, hits) for v in node]
    if isinstance(node, str) and VIDEO_STAMP_RE.search(node):
        new = clean(node)
        hits.append((node, new))
        return new
    return node


total = 0
for name in TARGETS:
    path = os.path.join(MOOD_SNAPSHOT_FOLDER, name)
    if not os.path.exists(path):
        print(f"{name}: not here")
        continue
    with open(path) as f:
        data = json.load(f)
    hits = []
    cleaned = walk(data, hits)
    total += len(hits)
    print(f"\n{name}: {len(hits)} stored strings carry the marker")
    for old, new in hits[:3]:
        print(f"   - {old[:88]!r}\n     → {new[:88]!r}")
    empties = [n for _, n in hits if not n.strip()]
    if empties:
        print(f"   ({len(empties)} would be left empty — those were nothing but markers)")
    if APPLY and hits:
        shutil.copy2(path, f"{path}.bak-{int(time.time())}")
        with open(path, "w") as f:
            json.dump(cleaned, f, indent=2)
        print(f"   written (backup: {os.path.basename(path)}.bak-*)")

print(f"\n{total} stored strings carried the video clock." + ("" if APPLY else "  Dry run — pass --apply to rewrite."))
