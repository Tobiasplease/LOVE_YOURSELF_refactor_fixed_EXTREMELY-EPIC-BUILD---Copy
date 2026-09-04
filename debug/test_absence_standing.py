"""Offline checks for the standing absence fact (Sep 4 evening,
docs/presence-stickiness-sep4.md): rides only while the presence belief is OFF
with a known drop time AND the recent stored stream still mentions a person;
yields to the departure edge line; stops when the stream stops.

Run:  python debug/test_absence_standing.py
"""

import collections
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from captioner.prompt_registry import FRAGMENTS, P  # noqa: E402
from captioner.prompts import build_situational_line, build_standing_absence_line  # noqa: E402

HIM = ["the black curtain sways behind him like it's breathing.", "the red foam finger is still pointing down."]
NO_HIM = ["the red foam finger is still pointing down.", "the shelves are packed, they sag under all of it."]
fails = 0


def check(name, ok, got=""):
    global fails
    print(("PASS  " if ok else "FAIL  ") + name + (f"   → {got!r}" if got and not ok else ""))
    fails += 0 if ok else 1


class A:
    pass


def agent(believed, dropped_ago, stream, regime=True):
    a = A()
    a._presence_believed = believed
    a._presence_dropped_at = (time.time() - dropped_ago) if dropped_ago is not None else 0.0
    a._stream = collections.deque(stream)
    a._presence_singular_regime = regime
    return a


line = build_standing_absence_line(agent(False, 5 * 60, HIM))
check("rides: belief off, dropped 5 min, stream mentions him", line == "He left a few minutes ago; the room's been empty since.", line)
check(
    "silent: stream without a person mention ('they' about shelves does not count)", build_standing_absence_line(agent(False, 5 * 60, NO_HIM)) == ""
)
check("silent: belief still on", build_standing_absence_line(agent(True, 5 * 60, HIM)) == "")
check("silent: drop time unknown", build_standing_absence_line(agent(False, None, HIM)) == "")
check("silent: only mention is beyond the scanned tail", build_standing_absence_line(agent(False, 5 * 60, HIM[:1] + NO_HIM * 5)) == "")
check("rides: mention inside the tail", build_standing_absence_line(agent(False, 5 * 60, NO_HIM * 3 + HIM[:1])) != "")
line = build_standing_absence_line(agent(False, 30, HIM))
check("grammar: 'just now' has no 'ago'", line == "He left just now; the room's been empty since.", line)
line = build_standing_absence_line(agent(False, 25 * 60, HIM, regime=False))
check("plural regime says Someone", line.startswith("Someone left about 20 minutes ago"), line)

# the departure cycle: the edge line speaks, the standing fact yields, then rides next call
a = agent(False, 1, HIM)
a._prev_presence_for_line = True
sit = build_situational_line(a)
check("edge cycle: situational line says they've gone", "They've gone" in sit, sit)
check("edge cycle: standing fact yields", build_standing_absence_line(a) == "")
check("next cycle: standing fact rides", build_standing_absence_line(a) != "")

# ride bookkeeping: one onset, counts per call, stop when the stream clears
a = agent(False, 5 * 60, HIM)
for _ in range(3):
    build_standing_absence_line(a)
check("bookkeeping: riding with 3 calls", getattr(a, "_absence_standing_riding", False) and a._absence_standing_calls == 3)
a._stream = collections.deque(NO_HIM)
build_standing_absence_line(a)
check("bookkeeping: stopped when the stream stopped mentioning him", not a._absence_standing_riding)

frag = FRAGMENTS.get("caption.absence-standing") or {}
check("registry: placeholders who/when", sorted(frag.get("placeholders", [])) == ["when", "who"])
check("registry: used by caption, caption_blind, drift_turn", set(frag.get("used_by", [])) >= {"caption", "caption_blind", "drift_turn"})
check("registry: P() renders", "left" in P("caption.absence-standing").format(who="He", when="just now"))

print(f"\n{'ALL PASS' if not fails else f'{fails} FAILED'}")
sys.exit(1 if fails else 0)
