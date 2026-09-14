#!/usr/bin/env python3
"""Sep 14 — take the numeral outbreaks out of the long-term store.

The three outbreaks (a video chunk marker, counts that became numbered people,
and grounding coordinates) were cleaned from the window, the lore ledger,
identity and the durable ledger on Sep 13-14. They also reached ChromaDB: the
observations collection keeps the caption text per concept. Concepts themselves
are clean, because the label check rejects a bare numeral, and no reflection
carries one.

Deletes only the observation rows whose text carries one of those forms. Run
with the machine STOPPED — chroma is a local store and two writers can corrupt
it. Prints what it would delete unless given --apply.

    .venv/bin/python debug/purge_chroma_numerals.py [--apply]
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

APPLY = "--apply" in sys.argv
BAD = re.compile(
    r"\b\d{3,4}\s*[x×]\s*-?\d{3,4}\b"          # grounding box / size
    r"|\b\d{3}'s\b"                             # a number owning something
    r"|\b\d{3}\s+(?:is|was|are|were|has|have|had|left|sat|sits|sitting|stands|standing|moved|looking|looks|hunched|shifted|turned|disappeared|vanished)\b"
    r"|\b\d{1,3}m\d{1,2}(?:\.\d+)?s?\b"         # the server's video chunk marker
    r"|(?<![\d:.])\d{5,}(?:\s*,\s*-?\d+)+",     # the original coordinate storm
    re.I,
)


def main():
    from captioner.semantic_memory import get_semantic_memory

    sm = get_semantic_memory()
    col = sm._observations
    got = col.get(include=["documents", "metadatas"])
    ids, docs = got.get("ids") or [], got.get("documents") or []
    bad = [(i, d) for i, d in zip(ids, docs) if d and BAD.search(d)]
    print(f"observations: {len(docs)} rows, {len(bad)} carry a numeral from the outbreaks")
    for _, d in bad[:8]:
        print(f"   - {d[:100]}")
    if bad and APPLY:
        col.delete(ids=[i for i, _ in bad])
        print(f"\ndeleted {len(bad)}; observations now {col.count()}")
    elif bad:
        print("\nDry run — pass --apply to delete them.")

    for name, c in (("concepts", sm._concepts), ("reflections", sm._reflections)):
        g = c.get(include=["documents"])
        n = sum(1 for d in (g.get("documents") or []) if d and BAD.search(d))
        print(f"{name}: {c.count()} rows, {n} carrying one")


main()
