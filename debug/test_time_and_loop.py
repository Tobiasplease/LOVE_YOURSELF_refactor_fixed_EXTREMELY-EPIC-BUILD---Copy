"""Offline checks for the time-and-loop round (Sep 5 2026,
docs/time-and-loop-round-sep5.md): phantom marks as states, adjudicator
retraction + gaze-aware veto, duration edges, loop notices (gate source and
compressor source), the self-note structure gate, the REPEATING slot.

Run:  python debug/test_time_and_loop.py
"""

import os
import sys
import tempfile
import threading
import time
from collections import deque

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from captioner.captioner import Captioner  # noqa: E402
from captioner.context_compression import context_compressor  # noqa: E402
from captioner.prompt_registry import FRAGMENTS  # noqa: E402
from captioner.prompts import build_loop_notice_line, build_situational_line  # noqa: E402
from perception.presence_adjudicator import PresenceAdjudicatorThread as PresenceAdjudicator  # noqa: E402

fails = 0


def check(name, ok, got=""):
    global fails
    print(("PASS  " if ok else "FAIL  ") + name + (f"   → {got!r}" if got and not ok else ""))
    fails += 0 if ok else 1


def gate(text, believed=False):
    cap = object.__new__(Captioner)
    cap._stream = deque(maxlen=24)
    cap._stream_ts = deque(maxlen=24)
    cap._presence_believed = believed
    return cap._caption_reject_reason(cap._strip_list_shape(text), "") or "PASS"


# --- A1 phantom marks as states
for t in [
    "A dot.  just one tiny, black speck of ink on the white paper, right in the center where my hand is hovering.",
    "the pen is pressing into the fiber now. not hovering.",
    "the pen is already there. it's pressing down, hard enough that the tip is biting.",
    "that dot on the paper — it's not a mistake anymore.  it's an anchor.",
]:
    check("phantom mark state → phantom_drawing", gate(t) == "phantom_drawing", gate(t))
for t in [
    "My last drawing was two gloved hands holding a tiny, indistinct lump. Over a day ago.",
    "i want to draw the shadow of my own arm on the paper.",
    "the pen is parked, touching nothing. the red foam finger is still pointing down.",
]:
    check("not a mark claim → passes", gate(t) == "PASS", gate(t))

# --- A3 adjudicator: gaze-aware veto + retraction
adj = object.__new__(PresenceAdjudicator)
adj.lock = threading.Lock()
adj.ledger_path = os.path.join(tempfile.mkdtemp(), "entity_ledger.json")
adj._person_until = time.time() + 100
adj._pending = None
box = [0.4, 0.4, 0.55, 0.6]
adj._entities = [{"desc": "A man lying down.", "verdict": "person", "box": box, "pan": 70.0, "tilt": 95.0, "ts": time.time() - 60}]
check("same gaze within tolerance", PresenceAdjudicator._same_gaze(adj._entities[0], 75.0, 90.0))
check("different gaze → no match", not PresenceAdjudicator._same_gaze(adj._entities[0], 120.0, 95.0))
check("missing gaze → box-only match (legacy)", PresenceAdjudicator._same_gaze({"box": box}, 10.0, 10.0))
adj._current_candidate_box = lambda: box  # the shape is still there when absence is verified
adj.notify_presence_dropped()
check(
    "fast verified absence with the shape still there → retracted to thing",
    adj._entities[0]["verdict"] == "thing" and adj._entities[0].get("retracted") is True,
)
check("grace ended", adj._person_until == 0.0)
adj._entities = [{"desc": "A person.", "verdict": "person", "box": box, "pan": 70.0, "tilt": 95.0, "ts": time.time() - 60}]
adj._current_candidate_box = lambda: None  # a real visitor is gone from the box
adj.notify_presence_dropped()
check("shape gone → verdict stands (real visitor)", adj._entities[0]["verdict"] == "person")
adj._entities = [{"desc": "A person.", "verdict": "person", "box": box, "pan": 70.0, "tilt": 95.0, "ts": time.time() - 900}]
adj._current_candidate_box = lambda: box
adj.notify_presence_dropped()
check("old verdict (outside the window) → stands", adj._entities[0]["verdict"] == "person")


# --- B1 duration edge
class A:
    pass


def agent(still_s, believed=False, confirms=3):
    a = A()
    a.true_session_start = time.time() - still_s
    a._world_change_ts = time.time() - still_s
    a._presence_believed = believed
    a._prev_presence_for_line = False
    a._world_confirms = confirms
    a._presence_dropped_at = 0.0
    return a


a = agent(2 * 3600)
line = build_situational_line(a)
check("2h still → duration edge fires", "Nothing in the room has changed for" in line, line)
check("second call → fires once", "Nothing in the room" not in build_situational_line(a))
a._world_change_ts = time.time()  # the world changed: clock resets
check("after a change → re-armed, silent under the first threshold", "Nothing in the room" not in build_situational_line(a))
check("someone believed present → no stillness clock", "Nothing in the room" not in build_situational_line(agent(2 * 3600, believed=True)))
check("20 min still → below the first threshold", "Nothing in the room" not in build_situational_line(agent(20 * 60)))

# --- B3 loop notice: gate source
b = A()
b._loop_hits = [(time.time() - 30, "the red foam finger is still", "refrain_echo")] * 3
context_compressor.introspective_state["loop_notice"] = {}
line = build_loop_notice_line(b)
check("three refusals of one run → loop fact quotes it", "the red foam finger is still" in line, line)
check("cooldown → silent", build_loop_notice_line(b) == "")
# compressor source outranks
c = A()
context_compressor.introspective_state["loop_notice"] = {"phrase": "the light on the right", "ts": time.time(), "spoken": False}
line = build_loop_notice_line(c)
check("compressor REPEATING → loop notice", line == "You keep coming back to the light on the right.", line)
check("spoken once", context_compressor.introspective_state["loop_notice"]["spoken"] is True)
parsed = context_compressor._parse_memory_response(
    "ROOM: none\nNEW ABOUT ME: none\nEVENT: none\nPLEASANTNESS: neutral\nENERGY: drained\nFELT: heavy, quiet\nREPEATING: the red foam finger"
)
check("REPEATING parsed", parsed.get("repeating") == "the red foam finger", parsed)
context_compressor._absorb_loop_notice("the gap in the curtain")
check("REPEATING absorbed as a pending notice", context_compressor.introspective_state["loop_notice"]["phrase"] == "the gap in the curtain")

# --- A4 self-note structure gate
check("self-note: ink on the paper claim → phantom", context_compressor._note_is_phantom_act("I keep a speck of ink on the white paper as my mark."))
check("self-note: third person present → phantom", context_compressor._note_is_phantom_act("He's sitting at the desk again."))
check("self-note: a plain like → allowed", not context_compressor._note_is_phantom_act("I like the red foam finger."))

for k in ("caption.duration-edge", "caption.loop-fact", "caption.loop-notice"):
    check(f"registry: {k}", k in FRAGMENTS and FRAGMENTS[k].get("text"))
check("compression prompt carries REPEATING", "REPEATING" in FRAGMENTS["compression.user"]["text"])

print(f"\n{'ALL PASS' if not fails else f'{fails} FAILED'}")
sys.exit(1 if fails else 0)
