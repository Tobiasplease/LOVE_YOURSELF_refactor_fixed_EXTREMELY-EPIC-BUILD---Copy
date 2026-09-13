"""Rarity-weighted event memory (Sep 11 2026).

Artist: "Things out of the ordinary need to have a lot more weight in the
memory" — and "the rarity should also determine the significance at the time
of discovery, so someone walking in after a period of loneliness should be
reacted to appropriately."

The Sep 11 22:26 visit (docs/where-we-are-sep9.md §28) was seen, adjudicated,
spoken about for six minutes, written to three ledgers and a reflection — and
gone from every prompt within two minutes of the departure, because the only
line that outlived the moment was tied to an eight-line stream window. It was
the first person in a day.

Here the weight of an event is its RARITY, measured against the machine's own
ledgers (room-agnostic: nothing here knows what room this is), and rarity sets
two things:

  1. how long the event stays in front of the machine as a standing fact
     ("what last happened" rides on every thought call, in the machine's own
     words where it has them, for a lifetime proportional to the gap that
     preceded the event: a quarter of the gap, floored and capped);
  2. how the arrival itself is announced — the edge cue carries the rarity as a
     fact ("Someone's come in — the first in about a day"). The exclamation is
     the machine's to make; the fact is ours to state.

Sources, all existing: the episodic ledger (person_arrived / person_left /
world_changed), the arrivals ledger (older history for the gap), and the
compressor's own EVENT sentences (its words for what happened). Nothing here
is stored into the stream, so it cannot breed.
"""

from __future__ import annotations

import re
import time
from typing import Dict, List, Optional

from config import config

VISIT_KIND = "visit"
CHANGE_KIND = "change"
PASS_KIND = "pass"  # Sep 13: someone crossed the frame and was never judged (see config PASS_EVENT_*)


def _cfg(name: str, default):
    return getattr(config, name, default)


def _episodic_events(types: List[str], window_s: float = 45 * 86400) -> List[Dict]:
    try:
        from utils.episodic_log import episodic_log

        return sorted(episodic_log.get_recent_events(window_seconds=int(window_s), types=types), key=lambda e: float(e.get("timestamp", 0) or 0))
    except Exception:
        return []


def _arrival_ledger_ts() -> List[float]:
    """Arrival timestamps from the presence_arrivals ledger (deeper history than
    the episodic log keeps)."""
    try:
        import json
        import os

        from config.config import MOOD_SNAPSHOT_FOLDER

        p = os.path.join(MOOD_SNAPSHOT_FOLDER, "presence_arrivals.json")
        if not os.path.exists(p):
            return []
        with open(p) as f:
            return sorted(float(a.get("ts") or a.get("timestamp") or 0) for a in json.load(f).get("arrivals", []))
    except Exception:
        return []


def gap_before(event_ts: float, kind: str) -> Optional[float]:
    """Seconds from the previous event of the same kind to this one — the
    rarity. None when there is no earlier event on record (unknown, not rare)."""
    if kind == VISIT_KIND:
        ts = [float(e["timestamp"]) for e in _episodic_events(["person_arrived"])] + _arrival_ledger_ts()
    elif kind == PASS_KIND:
        # A pass breaks a silence made of ANY human trace, its own kind included:
        # measured against arrivals alone, two passes an hour apart would each
        # read as the first sign of anyone in a day.
        ts = [float(e["timestamp"]) for e in _episodic_events(["person_arrived", "person_passed"])] + _arrival_ledger_ts()
    else:
        ts = [float(e["timestamp"]) for e in _episodic_events(["world_changed"])]
    prior = [t for t in ts if t < event_ts - 60]
    return (event_ts - max(prior)) if prior else None


def lifetime_s(gap_s: Optional[float], kind: str = VISIT_KIND) -> float:
    """How long an event stays a standing fact: a fixed fraction of the gap
    that preceded it, between a floor and a cap. A pass is the lower tier —
    a tenth of the gap, at most an hour — because the machine never got a
    proper look at it and a long-lived line about a half-seen person is the
    phantom-presence seed the Sep 12 read warned about."""
    if kind == PASS_KIND:
        lo = float(_cfg("PASS_EVENT_MIN_LIFETIME_S", 300))
        hi = float(_cfg("PASS_EVENT_MAX_LIFETIME_S", 3600))
        factor = float(_cfg("PASS_EVENT_RARITY_FACTOR", 0.1))
    else:
        lo = float(_cfg("EVENT_MEMORY_MIN_S", 600))
        hi = float(_cfg("EVENT_MEMORY_MAX_S", 6 * 3600))
        factor = float(_cfg("EVENT_MEMORY_RARITY_FACTOR", 0.25))
    if gap_s is None:
        return lo
    return max(lo, min(hi, gap_s * factor))


def is_rare(gap_s: Optional[float]) -> bool:
    return gap_s is not None and gap_s >= float(_cfg("ARRIVAL_RARITY_MIN_S", 1200))


_PERSON_WORDS = re.compile(r"\b(person|someone|somebody|visitor|occupant|man|woman|people|he|she|they|them|their|him|her)\b", re.I)


def _own_words(start_ts: float, end_ts: float, must_match=None) -> str:
    """The compressor's latest EVENT sentence written during [start, end] —
    the machine's own words for what happened, if it has any. `must_match`
    keeps it about the event: the first live run picked "The red foam finger
    was confirmed to not be present" for a visit because it was the newest
    sentence in the window (Sep 11 23:59), and the machine then puzzled over
    a finger that "wasn't there an hour ago"."""
    try:
        from captioner.context_compression import context_compressor

        hits = [e for e in (context_compressor.events or []) if start_ts <= float(e.get("timestamp", 0) or 0) <= end_ts]
        if must_match is not None:
            hits = [e for e in hits if must_match.search(e.get("event") or "")]
        return (hits[-1].get("event") or "").strip() if hits else ""
    except Exception:
        return ""


def last_event(now: Optional[float] = None) -> Optional[Dict]:
    """The most recent COMPLETED event with its rarity and lifetime, or None.

    visit: the last person_left, paired with the person_arrived before it.
    change: the last world_changed. A visit still in progress (no departure
    yet) is not an event here — the presence lines carry it while it lasts."""
    now = now or time.time()
    cands = []
    lefts = _episodic_events(["person_left"])
    if lefts:
        left = lefts[-1]
        lt = float(left["timestamp"])
        arrs = [e for e in _episodic_events(["person_arrived"]) if float(e["timestamp"]) < lt]
        at = float(arrs[-1]["timestamp"]) if arrs else None
        gap = gap_before(at, VISIT_KIND) if at is not None else None
        dur = (lt - at) if at is not None else None
        words = _own_words(at if at is not None else lt - 600, lt + 300, must_match=_PERSON_WORDS)
        cands.append({"kind": VISIT_KIND, "ts": lt, "start_ts": at, "duration_s": dur, "gap_s": gap, "words": words})
    passes = _episodic_events(["person_passed"])
    if passes:
        p = passes[-1]
        pt = float(p["timestamp"])
        dur = float((p.get("metadata") or {}).get("duration_s") or 0) or None
        cands.append({"kind": PASS_KIND, "ts": pt, "start_ts": pt - (dur or 0), "duration_s": dur, "gap_s": gap_before(pt, PASS_KIND), "words": ""})
    changes = _episodic_events(["world_changed"])
    if changes:
        ch = changes[-1]
        ct = float(ch["timestamp"])
        cands.append({"kind": CHANGE_KIND, "ts": ct, "start_ts": ct, "duration_s": None, "gap_s": gap_before(ct, CHANGE_KIND), "words": _own_words(ct, ct + 300) or (ch.get("description") or "")})
    if not cands:
        return None
    ev = max(cands, key=lambda c: c["ts"])
    ev["age_s"] = now - ev["ts"]
    ev["lifetime_s"] = lifetime_s(ev["gap_s"], ev["kind"])
    ev["alive"] = ev["age_s"] <= ev["lifetime_s"]
    ev["rare"] = is_rare(ev["gap_s"])
    return ev


def anyone_since(ts: float) -> bool:
    """Has anyone arrived since that moment? Sep 13: last_event() pairs the last
    departure with the arrival before it, so a NEW arrival after it (belief
    dropped, detection flaky, a visit still in progress) leaves the visit line
    standing while someone is in fact in the room. The closing sentence must
    not be added then."""
    try:
        return any(float(e["timestamp"]) > float(ts) for e in _episodic_events(["person_arrived"]))
    except Exception:
        return True


def rarity_phrase(kind: str, gap_s: Optional[float]) -> str:
    """'the first visitor in about a day' — empty when the event was routine."""
    if not is_rare(gap_s):
        return ""
    from captioner.prompts import casual_time_string

    noun = "visitor" if kind in (VISIT_KIND, PASS_KIND) else "change"
    return f"the first {noun} in {casual_time_string(gap_s / 60.0)}"


def event_words(ev: Dict) -> str:
    """The sentence for the event: the machine's own if it has one, else a
    plain fact from the ledger, in words. A pass has no own words by
    construction — nothing was adjudicated, so nothing it said about that
    moment is evidence of a person."""
    # Own words only when SHORT (Sep 12 00:05): the first live line carried a
    # 24-word mid-visit sentence — "…only their back remains visible as they
    # sit hunched over the desk" — which, riding every prompt for hours in an
    # empty room, is a phantom-presence seed. A short sentence is a memory; a
    # long one is a scene. Past the cap the ledger fact speaks instead.
    if ev.get("words") and len(ev["words"].split()) <= int(_cfg("EVENT_MEMORY_OWN_WORDS_MAX", 18)):
        w = ev["words"].strip()
        return w if w.endswith((".", "!", "?")) else w + "."
    from captioner.prompts import casual_time_string

    if ev["kind"] == PASS_KIND:
        return ""  # the pass has its own line — it is about not having got a look
    if ev["kind"] == VISIT_KIND:
        if ev.get("duration_s") is not None:
            return f"Someone came in, stayed {casual_time_string(ev['duration_s'] / 60.0)}, and left."
        return "Someone was here, and left."
    return "Something in the room changed."
