#!/usr/bin/env python3
"""Sep 13 — the walk-past event (artist: "Someone walked past about an hour ago…
It left no trace in the current real-time captioning… There should be a way to
differentiate a consistent world model from a truly novel event").

Two halves: the detector state machine in captioner._track_pass (faked clock,
faked episodic log — no camera, no server), and the event-memory tier it feeds
(shorter lifetime than a visit, its own line, rarity counted against arrivals
AND earlier passes). Importing captioner.captioner mints a stub run log;
sweep event_log for a fresh small *-event-log.json after running."""
import os
import sys
import time
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils.episodic_log as _el  # noqa: E402
from captioner import event_memory as em  # noqa: E402
from captioner.prompts import build_last_event_line, pass_cue_text  # noqa: E402

fails = 0


def check(name, ok, got=""):
    global fails
    print(("PASS  " if ok else "FAIL  ") + name + (f"   → {got!r}" if not ok else ""))
    fails += 0 if ok else 1


T = time.time()
EVENTS = []
_el.episodic_log.get_recent_events = lambda window_seconds=0, types=None: [e for e in EVENTS if types is None or e["type"] in types]
_el.episodic_log.record = lambda etype, desc, metadata=None, timestamp=None: EVENTS.append(
    {"type": etype, "description": desc, "metadata": metadata or {}, "timestamp": timestamp or time.time()}
)
em._arrival_ledger_ts = lambda: []
sys.modules["captioner.context_compression"] = types.SimpleNamespace(context_compressor=types.SimpleNamespace(events=[]))

# --- the detector ---------------------------------------------------------
from captioner.captioner import Captioner  # noqa: E402


class Fake(Captioner):
    def __init__(self):  # no camera, no model, no ledgers
        self._presence_believed = False
        self._pass_run = None
        self._presence_edge = None
        self._attention = None


def run(events, believed=False, verdict=None, seed=None):
    """events: (t, raw_person) in order. Returns the pass events recorded."""
    EVENTS[:] = list(seed or [])
    a = Fake()
    a._presence_believed = believed
    for t, raw in events:
        a._track_pass(T + t, raw, {"presence_adjudication": verdict})
    return [e for e in EVENTS if e["type"] == "person_passed"], a


# someone crosses for 3 s, then the room is empty
seen = [(0, False), (1, True), (2, True), (4, True), (5, False), (8, False), (11, False)]
rec, a = run(seen)
check("a three-second crossing is a pass", len(rec) == 1 and 2.5 <= rec[0]["metadata"]["duration_s"] <= 3.5, rec)
check("it carries the cue in the sticky edge slot", a._presence_edge and "went past" in a._presence_edge["text"] and a._presence_edge["sent"] is False, a._presence_edge)

check("a flicker is not a pass", run([(0, False), (1, True), (2, False), (9, False)])[0] == [])
check("nothing at all is not a pass", run([(0, False), (5, False), (20, False)])[0] == [])
check("still in frame → not yet a pass", run([(0, True), (3, True), (6, True), (9, True)])[0] == [])
check("gone for less than the end window → not yet", run([(0, True), (3, True), (4, False), (7, False)])[0] == [])
check("belief on → a visit, not a pass", run(seen, believed=True)[0] == [])
check("adjudicated a person → a visit, not a pass", run(seen, verdict="person")[0] == [])
check("adjudicated a thing → the studio's own furniture", run(seen, verdict="thing")[0] == [])

# a second crossing after the first is its own pass
rec2, _ = run(seen + [(40, True), (43, True), (44, False), (52, False)])
# Sep 13, first live false positive: 24 s of a person-shape with the machine's own
# arm in frame, the adjudicator never having run. A crossing is short.
long_cross = [(0, False)] + [(t, True) for t in range(1, 25)] + [(30, False), (40, False)]
check("a shape that lingers unjudged is not a pass", run(long_cross)[0] == [], run(long_cross)[0])
edge = [(0, False), (1, True), (11, True), (12, False), (20, False)]
check("ten seconds still counts", len(run(edge)[0]) == 1, run(edge)[0])

check("two crossings are two passes", len(rec2) == 2, rec2)

# --- the event-memory tier -------------------------------------------------
EVENTS[:] = [{"type": "person_arrived", "timestamp": T - 86400}, {"type": "person_left", "timestamp": T - 86000}]
EVENTS.append({"type": "person_passed", "timestamp": T - 600, "metadata": {"duration_s": 3.0}})
ev = em.last_event(T)
check("the pass is the newest event", ev and ev["kind"] == em.PASS_KIND, ev)
check("rarity measured from the last person on record", ev and abs(ev["gap_s"] - 85800) < 60, ev and ev["gap_s"])
check("a pass lives an hour at most, not six", ev and ev["lifetime_s"] == 3600, ev and ev["lifetime_s"])
check("alive and rare", ev and ev["alive"] and ev["rare"])


class A:
    pass


def agent():
    a = A()
    a.true_session_start = T - 90000
    a._presence_believed = False
    return a


line = build_last_event_line(agent())
check("its own line says the look was never had", line == "Earlier: someone went past. You didn't get a proper look. That was about ten minutes ago, the first visitor in about twenty-three hours.", line)
check("no digits in it", not any(ch.isdigit() for ch in line))
check("no 'No one has been here since' on a pass", "No one has been here since." not in line)

EVENTS.append({"type": "person_passed", "timestamp": T - 60, "metadata": {"duration_s": 2.5}})
ev2 = em.last_event(T)
check("a second pass is not the first sign of anyone in a day", ev2 and ev2["gap_s"] < 700 and not ev2["rare"], ev2 and ev2["gap_s"])
check("a routine pass gets the plain line", "the first" not in build_last_event_line(agent()), build_last_event_line(agent()))
check("its lifetime falls to the floor", ev2["lifetime_s"] == 300, ev2["lifetime_s"])

# The cue is built at the pass's own moment, BEFORE the event is written —
# measured afterwards the pass is its own last sign of anyone and never rare.
EVENTS[:] = [{"type": "person_arrived", "timestamp": T - 86400}]
check("the cue states the rarity", pass_cue_text(T - 300).startswith("Someone just went past — the first sign of anyone in "), pass_cue_text(T - 300))
EVENTS.append({"type": "person_passed", "timestamp": T - 300})
check("a pass minutes after another is not announced as the first in a day", pass_cue_text(T) == "Someone just went past.", pass_cue_text(T))
rec_live, a_live = run(seen, seed=[{"type": "person_arrived", "timestamp": T - 86400}])
check("the live order gives the rare cue, not the plain one", "the first sign of anyone" in (a_live._presence_edge or {}).get("text", ""), a_live._presence_edge)
rec_live2, a_live2 = run(seen, seed=[{"type": "person_passed", "timestamp": T - 120}])
check("a pass two minutes after another gets the plain cue", (a_live2._presence_edge or {}).get("text") == "Someone just went past.", a_live2._presence_edge)

# a real visit outranks a pass that happened before it
EVENTS[:] = [
    {"type": "person_passed", "timestamp": T - 3000, "metadata": {"duration_s": 3.0}},
    {"type": "person_arrived", "timestamp": T - 2000},
    {"type": "person_left", "timestamp": T - 1000},
]
ev3 = em.last_event(T)
check("the later visit is what is remembered", ev3 and ev3["kind"] == em.VISIT_KIND, ev3 and ev3["kind"])

print("\n" + ("ALL PASS" if not fails else f"{fails} FAILED"))
sys.exit(1 if fails else 0)
