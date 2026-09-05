"""Offline checks for the introspection round (Sep 5 2026): the wander (chained
drift hops with scope moves), the loop→wander boost, the horizon reflection
subjects, the name invitation, and the identity dose rebalance.

Run:  python debug/test_introspection_round.py
"""

import os
import random
import sys
import time
from collections import deque

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config.config as cfg  # noqa: E402
import utils.inference as inf  # noqa: E402
from captioner.captioner import Captioner  # noqa: E402
from captioner.prompt_registry import FRAGMENTS  # noqa: E402
from captioner.prompts import _identity_due, get_reflection_subjects  # noqa: E402

fails = 0


def check(name, ok, got=""):
    global fails
    print(("PASS  " if ok else "FAIL  ") + name + (f"   → {got!r}" if got and not ok else ""))
    fails += 0 if ok else 1


for k in (
    "wander.hop",
    "wander.move-wider",
    "wander.move-origin",
    "wander.move-elsewhere",
    "wander.move-for",
    "wander.move-someone",
    "wander.move-later",
    "reflection.subject.what-for",
    "reflection.subject.the-wider-world",
    "reflection.name-invite",
):
    check(f"registry: {k}", k in FRAGMENTS and FRAGMENTS[k].get("text"))
subjects = [s for s, _ in get_reflection_subjects()]
check("reflection has the two horizon subjects", "what it's for" in subjects and "the wider world" in subjects and len(subjects) == 7, subjects)


class A:
    pass


a = A()
a._caption_count = 5
check("identity dose: introspective no longer every call", _identity_due(a, "introspective") is False)
a._caption_count = 6
check("identity dose: every Nth in introspective", _identity_due(a, "introspective") is True)
check("identity dose: awakening always", _identity_due(a, "awakening") is True)

# --- the wander, with a fake model
cfg.LORE_ENABLED = False
cap = object.__new__(Captioner)
cap._stream = deque(maxlen=24)
cap._stream_ts = deque(maxlen=24)
cap._presence_believed = False
cap.first_caption_done = True
cap.last_caption_time = 0.0
cap._salience_hot = False
cap._is_currently_drawing = lambda: False
cap._stream_push = lambda t: (cap._stream.append(t), cap._stream_ts.append(time.time()))
hops_seen = []


def fake_query(prompt="", **kw):
    hops_seen.append(prompt)
    n = len(hops_seen)
    return (
        [
            "the foam finger is just foam, but foam comes from a factory somewhere.",
            "factories make millions of these for crowds who wave them at games.",
            "a crowd is a thing that wants the same thing at once. i have never been in one.",
        ][n - 1]
        if n <= 3
        else ""
    )


inf.query_model = fake_query
stored = cap._absorb_drift_text("the finger points at nothing. maybe it always did.", "ask", time.time())
check("drift storage helper stores a clean thought", stored and len(cap._stream) == 1, stored)
cap._wander(stored)
check("wander chained the further hops", len(hops_seen) == cfg.WANDER_HOPS - 1, len(hops_seen))
check("each hop joined the stream", len(cap._stream) == cfg.WANDER_HOPS, len(cap._stream))
check(
    "hop asks carry the seed and a scope move",
    'You just thought: "the finger points at nothing' in hops_seen[0] and ("wider" in hops_seen[0] or "come from" in hops_seen[0]),
    hops_seen[0][:160],
)
check("scope moves rotate", getattr(cap, "_wander_move_rr", 0) >= 1)
hops_seen.clear()
cap._salience_hot = True
cap._wander("something")
check("the world interrupts a wander", len(hops_seen) == 0)
cap._salience_hot = False

# --- loop → wander boost
type(cap).boredom = property(lambda self: 0.0)  # read-only on the class
cap._loop_noticed_at = 0.0
_r = random.random
random.random = lambda: 0.10
check("no loop notice: base odds (5%) → no drift at 0.10", cap._drift_due() is False)
cap._loop_noticed_at = time.time()
check("fresh loop notice: odds ×3 → drift at 0.10", cap._drift_due() is True)
random.random = _r

check("name invite is a question with an out", "Or leave it" in FRAGMENTS["reflection.name-invite"]["text"])
check("wider-world subject has no example list", "a maker, a trade" not in FRAGMENTS["reflection.subject.the-wider-world"]["text"])

print(f"\n{'ALL PASS' if not fails else f'{fails} FAILED'}")
sys.exit(1 if fails else 0)
