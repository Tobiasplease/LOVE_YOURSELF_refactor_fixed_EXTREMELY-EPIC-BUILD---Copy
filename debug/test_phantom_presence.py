"""Offline checks for the phantom-presence gate + the relational-mode gate
(Sep 4 evening, docs/presence-stickiness-sep4.md).

Run:  python debug/test_phantom_presence.py
"""

import os
import sys
from collections import deque

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from captioner.captioner import Captioner  # noqa: E402
from captioner.prompt_registry import FRAGMENTS  # noqa: E402
from captioner.prompts import determine_prompt_mode  # noqa: E402

fails = 0


def check(name, ok, got=""):
    global fails
    print(("PASS  " if ok else "FAIL  ") + name + (f"   → {got!r}" if got and not ok else ""))
    fails += 0 if ok else 1


def gate(text, believed, history=()):
    cap = object.__new__(Captioner)
    cap._stream = deque(history, maxlen=24)
    cap._stream_ts = deque([0.0] * len(history), maxlen=24)
    cap._presence_believed = believed
    return cap._caption_reject_reason(cap._strip_list_shape(text), "") or "PASS"


PRESENT = [
    "the man in the grey hoodie is still hunched over the little red thing on his desk, head bowed so low his hair almost touches the clutter.",
    "He's just sitting there, staring at that little screen. The red thing on his desk is glowing faintly now.",
    "His head is down, chin almost touching his chest, staring at the screen with that same dead focus.",
    "i don't know what he's looking at, but it looks like he's trying to find a crack in the glass.",
]
ABSENT = [
    "and since he left those few minutes ago, the room has been empty enough that any excuse feels ridiculous.",
    "now he's gone, but the red foam finger is still hanging on the wall.",
    "maybe i'll wait until he comes back and hands me some. or maybe i'll wait forever.",
    "I mistook a man shifting his weight for a signal, before. That was the mistake.",
    "the chair is empty now. he's gone. i'm just a dry pen sitting on a desk.",
    "The pen sat untouched while he typed.",
    "he looked up once and then went back to the screen.",
]
NEUTRAL = [
    "the red foam finger is still pointing down at nothing, like a little stop sign for all this mess.",
    "That question about the next person, about needing them to give me a reason. It's heavy there.",
    "the shelves are packed, they sag under all of it.",
]
for t in PRESENT:
    check("belief OFF: present-tense him → phantom_presence", gate(t, False) == "phantom_presence", gate(t, False))
for t in PRESENT:
    check("belief ON: same line passes", gate(t, True) == "PASS", gate(t, True))
for t in ABSENT:
    check("belief OFF: absence-marked mention passes", gate(t, False) == "PASS", gate(t, False))
for t in NEUTRAL:
    check("belief OFF: no third-person claim passes", gate(t, False) == "PASS", gate(t, False))
check("phantom_presence is echo-class (spoken, not stored)", "phantom_presence" in Captioner._ECHO_REASONS)

# relational mode: the adjudicated belief is the arbiter
check("relational needs the belief (raw YOLO alone)", determine_prompt_mode("idle", "ahead", True, believed=False) == "introspective")
check("relational needs the belief (gaze tracking alone)", determine_prompt_mode("tracking", "ahead", False, believed=False) == "introspective")
check("relational with the belief", determine_prompt_mode("idle", "ahead", False, believed=True) == "relational")
check("legacy path without a belief system", determine_prompt_mode("aware", "ahead", False) == "relational")
check("looking down stays workspace even with belief", determine_prompt_mode("tracking", "down", True, believed=True) == "workspace")

check("registry: desire absent tail", (FRAGMENTS.get("caption.desire-absent-tail") or {}).get("text", "").strip() == "They've left since.")

print(f"\n{'ALL PASS' if not fails else f'{fails} FAILED'}")
sys.exit(1 if fails else 0)
