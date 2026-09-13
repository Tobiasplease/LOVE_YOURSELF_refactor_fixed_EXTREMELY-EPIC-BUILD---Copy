#!/usr/bin/env python3
"""Sep 13 — tedium as pressure with discharge (captioner/tedium.py).

Artist: "the continuous flow of time and interplay with boredom is a constant
shift within the persistence. Tedium is material and repetition is in and of
itself an event. But we are missing something in the architecture to truly
convey this framework."

Covers the dial (rise only while nothing is happening, faster on repetition,
never while engaged; discharge by a fraction; slow decay) and the three plugs
(the drift's odds, the decision ask, the reflection's interval). No server, no
camera. Run: .venv/bin/python debug/test_tedium.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from captioner.tedium import Tedium, pressing, unbearable  # noqa: E402
from config import config as cfg  # noqa: E402

fails = 0


def check(name, ok, got=""):
    global fails
    print(("PASS  " if ok else "FAIL  ") + name + (f"   → {got!r}" if not ok else ""))
    fails += 0 if ok else 1


def build(calls, **kw):
    t = Tedium(0.0)
    now = 0.0
    for i in range(calls):
        now += 8
        t.update(now, **({k: (v(i) if callable(v) else v) for k, v in kw.items()}))
    return t, now


# --- the dial -------------------------------------------------------------
t, now = build(300, unchanged=True)
check("nothing happening builds slowly", 0.2 < t.value < 0.6, round(t.value, 3))
t2, _ = build(300, unchanged=True, repeated=True)
check("naming the same thing again builds much faster", t2.value > 0.9 and t2.value > t.value * 2, (round(t.value, 2), round(t2.value, 2)))
check("sameness alone never reaches the reflection threshold", not unbearable(build(4000, unchanged=True)[0].value), build(4000, unchanged=True)[0].value)
check("repetition does", unbearable(t2.value))

check("something happening → no pressure", build(300, unchanged=False, repeated=True)[0].value == 0.0)
check("someone here → no pressure", build(300, unchanged=True, repeated=True, engaged=True)[0].value == 0.0)

# --- discharge ------------------------------------------------------------
t3, now3 = build(300, unchanged=True, repeated=True)
full = t3.value
after_drift = t3.discharge(cfg.TEDIUM_DISCHARGE_DRIFT, "drift", now3)
check("a drift spends half of it", abs(after_drift - full * 0.5) < 0.02, (round(full, 2), round(after_drift, 2)))
check("never to nothing", after_drift > 0)
t4, now4 = build(300, unchanged=True, repeated=True)
check("a rare event empties it", t4.discharge(cfg.TEDIUM_DISCHARGE_EVENT, "someone came", now4) == 0.0)
t5, now5 = build(300, unchanged=True, repeated=True)
check("a reflection takes most of it", t5.discharge(cfg.TEDIUM_DISCHARGE_REFLECTION, "reflection", now5) < full * 0.35)

t6, now6 = build(300, unchanged=True, repeated=True)
before = t6.value
t6.update(now6 + 8, unchanged=True, silent=True)
one = t6.value
t6.update(now6 + 16, unchanged=True, silent=True)
check("one chosen silence is not a discharge", abs(one - before) < 0.01, (round(before, 3), round(one, 3)))
check("a run of them is", t6.value < one * 0.9, (round(one, 3), round(t6.value, 3)))
t6.update(now6 + 24, unchanged=True, repeated=True)
check("speaking again resets the run", t6.value > 0)

t7, now7 = build(300, unchanged=True, repeated=True)
v = t7.value
t7._decay_to(now7 + 3600)
check("it decays on its own when left alone", t7.value < v * 0.2, (round(v, 2), round(t7.value, 2)))

# --- the plugs ------------------------------------------------------------
p_calm = cfg.DRIFT_BASE_P * (1.0 + cfg.TEDIUM_DRIFT_GAIN * 0.0)
p_full = cfg.DRIFT_BASE_P * (1.0 + cfg.TEDIUM_DRIFT_GAIN * 1.0)
check("the drift comes due sooner under pressure", p_full > p_calm * 2, (p_calm, p_full))

from captioner.prompt_registry import P  # noqa: E402
from captioner.prompts import build_decision_ask  # noqa: E402


class A:
    pass


def agent(ted):
    a = A()
    a._tedium_value = ted
    a._decide_counter = cfg.DECIDE_EVERY_N - 1
    a._room_attention = 1.0
    return a


import captioner.prompts as _pr  # noqa: E402

_pr.decide_every = lambda base, att: 1
calm = build_decision_ask(agent(0.0), live=False, is_awakening=False)
hot = build_decision_ask(agent(0.95), live=False, is_awakening=False)
check("the calm ask is unchanged", calm == P("caption.decide"), calm[:60])
check("the pressed ask names it and shows the ways out", hot == P("caption.decide-tedium") and "stay with this" in hot, hot[:80])
check("staying is one of them", "stay with" in P("caption.decide-tedium"))
check("no digits, no explanation of what tedium is", not any(c.isdigit() for c in P("caption.decide-tedium")) and "tedium" not in P("caption.decide-tedium").lower())

check("pressing/unbearable thresholds", pressing(cfg.TEDIUM_ASK_AT) and not pressing(cfg.TEDIUM_ASK_AT - 0.01) and unbearable(cfg.TEDIUM_REFLECT_AT))
check("the caption interval is untouched by any of it", cfg.CAPTION_INTERVAL_FIXED == 8)

print("\n" + ("ALL PASS" if not fails else f"{fails} FAILED"))
sys.exit(1 if fails else 0)
