"""Offline checks for the paper glance (Sep 5 2026): gaze-only sheet check on a
clock, gated by quiet + alone + not drawing; the verdict persists.

Run:  python debug/test_paper_glance.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import safety.paper_detection as pd  # noqa: E402
from captioner.captioner import Captioner  # noqa: E402
from utils.state_manager import state_manager  # noqa: E402

fails = 0


def check(name, ok, got=""):
    global fails
    print(("PASS  " if ok else "FAIL  ") + name + (f"   → {got!r}" if got and not ok else ""))
    fails += 0 if ok else 1


calls = []


def fake_check(camera, servos, captioner=None):
    calls.append(time.time())
    state_manager.paper_state = "no_paper"
    state_manager.paper_present = False
    state_manager.last_paper_check_ts = time.time()
    state_manager.last_paper_check_reason = "fake"
    return False


pd.check_paper_before_drawing = fake_check
state_manager.camera = object()
state_manager.paper_state, state_manager.last_paper_check_ts = "", 0.0
cap = object.__new__(Captioner)
cap.true_session_start = time.time() - 600
cap._salience_hot = False
cap._presence_believed = False
now = time.time()
cap._maybe_paper_glance(now)
check("glance runs when due, quiet, alone", len(calls) == 1 and state_manager.paper_state == "no_paper")
cap._maybe_paper_glance(now + 60)
check("not again inside the interval", len(calls) == 1)
state_manager.last_paper_check_ts = now - 4000
cap._last_paper_glance_attempt = 0.0
cap._salience_hot = True
cap._maybe_paper_glance(now)
check("never on a live cycle", len(calls) == 1)
cap._salience_hot = False
cap._presence_believed = True
cap._maybe_paper_glance(now)
check("never while someone is here", len(calls) == 1)
cap._presence_believed = False
cap._maybe_paper_glance(now)
check("runs again once the interval has passed", len(calls) == 2)
young = object.__new__(Captioner)
young.true_session_start = time.time() - 10
young._salience_hot = False
young._presence_believed = False
state_manager.last_paper_check_ts = 0.0
young._maybe_paper_glance(time.time())
check("not in the first minutes after boot", len(calls) == 2)

print(f"\n{'ALL PASS' if not fails else f'{fails} FAILED'}")
sys.exit(1 if fails else 0)
