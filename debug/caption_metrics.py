#!/usr/bin/env python3
"""Continuity metrics for a run's caption output — the A/B instrument for
STREAM_MODE (docs/continuity-plan.md, Verification).

Reads an event log and reports, over the caption sequence:
- opening repetition: captions whose first N words match a recent caption
- near-duplicates: fuzzy-similar to one of the previous 10 captions
- anaphoric openings: first clause leans on the prior thought (that/it/still/...)
- memory surfaces: familiarity/echo/desire/felt lines that reached prompts
- anti-echo gate activity (retries / skipped cycles)

Usage:
  python debug/caption_metrics.py                      # newest run log
  python debug/caption_metrics.py event_log/<id>-event-log.json
  python debug/caption_metrics.py <baseline.json> <compare.json>
"""
import difflib
import glob
import json
import os
import re
import sys

OPEN_WORDS = 5
NEARDUP_LOOKBACK = 10
NEARDUP_RATIO = 0.75
ANAPHORA = re.compile(r"^(and |but |so |still[, ]|that |it |they |the same|again[, ]|maybe th)", re.I)
SURFACE_MARKERS = (
    "Something that was on your mind",  # reflection echo
    "again —",  # familiarity
    "Preoccupied with:",  # desire
    "A memory surfaces",  # memory mode
)


def load(path):
    txt = open(path).read()
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        return [json.loads(l) for l in txt.splitlines() if l.strip()]


def analyze(path):
    entries = load(path)
    caps = [e.get("caption", "").strip() for e in entries if e.get("type") == "caption" and e.get("caption")]
    caps = [c for c in caps if len(c) > 10 and not c.startswith("[")]
    calls = [e for e in entries if e.get("type") in ("llm_api_call", "ollama_api_call")]
    dbg = [e for e in entries if e.get("type") in ("debug", "info")]

    n = len(caps)
    if n < 5:
        return {"path": os.path.basename(path), "captions": n, "note": "too few captions"}

    openings = [" ".join(c.lower().split()[:OPEN_WORDS]) for c in caps]
    open_rep = sum(1 for i in range(1, n) if openings[i] in openings[max(0, i - NEARDUP_LOOKBACK) : i])

    neardup = 0
    for i in range(1, n):
        for j in range(max(0, i - NEARDUP_LOOKBACK), i):
            if difflib.SequenceMatcher(None, caps[i], caps[j]).ratio() > NEARDUP_RATIO:
                neardup += 1
                break

    anaphors = sum(1 for c in caps if ANAPHORA.match(c))

    # Frame-level templates (Sep 4) — the survival strategies that live above
    # the token layer, where DRY and the presence penalty can't see them.
    # The deflation frame ("is just a") collapsed under presence_penalty 0.6;
    # the register moved UP a level into reframing/redemption shapes. Track,
    # don't gate (P7 — no shape gates; content-diet is the durable fix).
    import re as _re

    deflation = sum(1 for c in caps if _re.search(r"\bis just\b|\bit'?s just\b|\bwas just\b|\bjust a \b", c, _re.I))
    pivot = sum(
        1
        for c in caps
        if _re.search(r"(isn'?t|wasn'?t|aren'?t|weren'?t|never)\s+[^.;]{0,45}[—;,-]\s*(it'?s|they'?re|more like|it was|but)", c, _re.I)
    )
    transform = sum(1 for c in caps if _re.search(r"stopped being|started becoming|no longer \w+ but", c, _re.I))
    proof_tic = sum(1 for c in caps if _re.search(r"proof that|evidence that|reminder that", c, _re.I))
    bangs = sum(c.count("!") for c in caps)
    questions = sum(c.count("?") for c in caps)

    surfaces = sum(1 for e in calls if any(m in str(e.get("prompt", "")) for m in SURFACE_MARKERS))
    modes = {}
    for e in calls:
        m = e.get("stream_mode")
        if m:
            modes[m] = modes.get(m, 0) + 1
    gate_retries = sum(1 for e in dbg if e.get("action") == "anti_echo_retry")
    gate_skips = sum(1 for e in dbg if e.get("action") == "anti_echo_skip")

    return {
        "path": os.path.basename(path),
        "captions": n,
        "opening_repetition_pct": round(100 * open_rep / (n - 1), 1),
        "near_duplicate_pct": round(100 * neardup / (n - 1), 1),
        "anaphoric_opening_pct": round(100 * anaphors / n, 1),
        "frame_templates_pct": {
            "deflation_just": round(100 * deflation / n, 1),
            "negation_pivot": round(100 * pivot / n, 1),
            "transformation": round(100 * transform / n, 1),
            "proof_tic": round(100 * proof_tic / n, 1),
        },
        "tone": {"exclamations": bangs, "questions": questions},
        "memory_surface_lines": surfaces,
        "stream_mode_calls": modes,
        "anti_echo": {"retries": gate_retries, "skips": gate_skips},
    }


def main():
    args = sys.argv[1:]
    if not args:
        logs = [p for p in glob.glob("event_log/*-event-log.json") if os.path.getmtime(p) <= __import__("time").time()]  # skip future-dated strays
        args = [max(logs, key=os.path.getmtime)]
    for path in args:
        r = analyze(path)
        print(json.dumps(r, indent=2))


if __name__ == "__main__":
    main()
