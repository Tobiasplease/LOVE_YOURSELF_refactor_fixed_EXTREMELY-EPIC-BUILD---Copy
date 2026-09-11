#!/usr/bin/env python3
"""Sep 11 — rarity-weighted event memory (captioner/event_memory.py), the
rarity-aware sticky arrival cue, and the "He only on re-ID" pronoun. Fakes the
ledgers and the compressor; needs no server. Importing captioner.prompts mints
no run log."""
import os
import sys
import time
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils.episodic_log as _el  # noqa: E402
from captioner import event_memory as em  # noqa: E402
from captioner.prompts import arrival_cue_text, build_last_event_line, build_situational_line, presence_who  # noqa: E402

fails = 0


def check(name, ok, got=""):
    global fails
    print(("PASS  " if ok else "FAIL  ") + name + (f"   → {got!r}" if not ok else ""))
    fails += 0 if ok else 1


T = time.time()
EVENTS = []
_el.episodic_log.get_recent_events = lambda window_seconds=0, types=None: [e for e in EVENTS if types is None or e["type"] in types]
em._arrival_ledger_ts = lambda: []
sys.modules["captioner.context_compression"] = types.SimpleNamespace(context_compressor=types.SimpleNamespace(events=[]))
CC = sys.modules["captioner.context_compression"].context_compressor


class A:
    pass


def agent(**kw):
    a = A()
    a.true_session_start = T - 3600
    a._presence_believed = False
    a._prev_presence_for_line = False
    for k, v in kw.items():
        setattr(a, k, v)
    return a


# --- a rare visit: the first in a day, an hour ago
EVENTS[:] = [
    {"type": "person_arrived", "timestamp": T - 4000 - 86400},
    {"type": "person_left", "timestamp": T - 3900 - 86400},
    {"type": "person_arrived", "timestamp": T - 4000},
    {"type": "person_left", "timestamp": T - 3800},
]
ev = em.last_event(T)
check("visit found", ev and ev["kind"] == "visit", ev)
check("gap ≈ a day", ev and abs(ev["gap_s"] - 86400) < 5, ev and ev["gap_s"])
check("lifetime capped at six hours", ev and ev["lifetime_s"] == 6 * 3600, ev and ev["lifetime_s"])
check("alive and rare", ev and ev["alive"] and ev["rare"])
line = build_last_event_line(agent())
check("plain-fact line, in words", line == "Earlier: Someone came in, stayed a few minutes, and left. That was about an hour ago, the first visitor in about a day.", line)
CC.events[:] = [{"event": "The person sitting in the chair left, leaving only the empty seat behind.", "timestamp": T - 3700}]
line = build_last_event_line(agent())
check("the compressor's own words are preferred", line.startswith("Earlier: The person sitting in the chair left, leaving only the empty seat behind. That was about an hour ago, the first visitor"), line)
CC.events.append({"event": "The red foam finger was confirmed to not be present in the room.", "timestamp": T - 3650})
line = build_last_event_line(agent())
check("a newer sentence about something else is not the visit's words", "The person sitting in the chair left" in line and "foam finger" not in line, line)
CC.events[:] = [{"event": "The occupant of the chair is no longer present; only their back remains visible as they sit hunched over the desk looking at a phone.", "timestamp": T - 3900}]
line = build_last_event_line(agent())
check("a long mid-event sentence yields to the ledger fact", line.startswith("Earlier: Someone came in, stayed a few minutes, and left."), line)
check("not while a visit is in progress", build_last_event_line(agent(_presence_believed=True)) == "")

# --- a routine visit: five minutes after the previous one, an hour ago → dead
EVENTS[:] = [
    {"type": "person_arrived", "timestamp": T - 4300},
    {"type": "person_left", "timestamp": T - 4200},
    {"type": "person_arrived", "timestamp": T - 4000},
    {"type": "person_left", "timestamp": T - 3800},
]
CC.events[:] = []
ev = em.last_event(T)
check("routine gap → ten-minute lifetime", ev and ev["lifetime_s"] == 600 and not ev["alive"] and not ev["rare"], ev and (ev["lifetime_s"], ev["alive"]))
check("dead event → no line", build_last_event_line(agent()) == "")
# --- the same routine visit, two minutes ago → alive, plain form
EVENTS[:] = [
    {"type": "person_arrived", "timestamp": T - 500},
    {"type": "person_left", "timestamp": T - 400},
    {"type": "person_arrived", "timestamp": T - 200},
    {"type": "person_left", "timestamp": T - 120},
]
line = build_last_event_line(agent())
check("routine but fresh → plain line", line == "Earlier: Someone came in, stayed just now, and left. That was just now ago." or line.startswith("Earlier: Someone came in"), line)

# --- the arrival cue with rarity
EVENTS[:] = [{"type": "person_arrived", "timestamp": T - 86400}, {"type": "person_arrived", "timestamp": T - 1}]
check("rare arrival, unknown person", arrival_cue_text(agent()) == "Someone's come in — the first in about a day.", arrival_cue_text(agent()))
check("rare arrival, familiar", arrival_cue_text(agent(_presence_arrival_familiar=True)) == "He's back — the first time in about a day.", arrival_cue_text(agent(_presence_arrival_familiar=True)))
check("rare arrival, several", arrival_cue_text(agent(_presence_arrival_count=3)) == "People have come in — the first in about a day.")
EVENTS[:] = [{"type": "person_arrived", "timestamp": T - 300}, {"type": "person_arrived", "timestamp": T - 1}]
check("routine arrival, unknown person", arrival_cue_text(agent()) == "Someone's come in.", arrival_cue_text(agent()))
check("presence_who: Someone unless re-ID", presence_who(agent()) == "Someone" and presence_who(agent(_presence_arrival_familiar=True)) == "He")

# --- the sticky edge
EVENTS[:] = [{"type": "person_arrived", "timestamp": T - 86400}, {"type": "person_arrived", "timestamp": T - 1}]
a = agent(_presence_believed=True, _prev_presence_for_line=False)
l1 = build_situational_line(a)
check("arrival edge rides with its rarity", "Someone's come in — the first in about a day." in l1, l1)
l2 = build_situational_line(a)
check("still rides on the next build (not yet sent)", "Someone's come in" in l2, l2)
a._presence_edge["sent"] = True
l3 = build_situational_line(a)
check("gone once a prompt carrying it was sent", "Someone's come in" not in l3, l3)
a._presence_believed = False
l4 = build_situational_line(a)
check("departure edge", "They've gone — the room's quiet again." in l4, l4)
check("departure still rides until sent", "They've gone" in build_situational_line(a))

print("\n" + ("ALL PASS" if not fails else f"{fails} FAILED"))
sys.exit(1 if fails else 0)
