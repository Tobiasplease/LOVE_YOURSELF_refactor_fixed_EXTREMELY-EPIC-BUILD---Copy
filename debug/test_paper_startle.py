"""Prove the paper-check interrupt and the reworked organic startle.

1. paper_clear plays the 'paper' take (servos AND gantry plan), holds the
   body in PAPER state, returns a truthful wait; paper_release blends the
   SAME dataset back (continuity).
2. startle plays the startle take RELATIVE to the live pose: the first
   flinch send equals the live pose exactly (zero-offset entry — the old
   pose-freeze snapped instead), the motion peaks at NUDGE x the take's
   amplitude, the gantry flinches too, and the temperament returns after.
3. A startle during a paper hold is suppressed (safety owns the body).
4. Without a startle dataset the built-in delta nudge still fires.

  python debug/test_paper_startle.py
"""

import math
import os
import shutil
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import motor_panel.session as session_mod
from motor_panel import kinetic_bus as kb
from motor_panel.session import Session

RATE = 20
failures = []


def build_library(tmp, with_startle=True):
    session_mod.SESSIONS_DIR = tmp

    s = Session("calm_observant_a", loop_len=2.0)
    arm = next(t for t in s.tracks if t.name == "left arm")
    gt = next(t for t in s.tracks if t.channels == ["x", "y"])
    n = 40
    arm.samples = [
        {
            "t": i / RATE,
            "dt": 1 / RATE,
            "elbow": 90 + 12 * math.sin(2 * math.pi * i / n) + (i % 3) - 1,
            "shoulder": 88 + 6 * math.cos(2 * math.pi * i / n),
        }
        for i in range(n)
    ]
    gt.samples = [
        {"t": i / RATE, "dt": 1 / RATE, "x": 30 + 10 * math.sin(2 * math.pi * i / n), "y": 20 + 5 * math.cos(2 * math.pi * i / n)} for i in range(n)
    ]
    s.save(export=True)

    p = Session("paper_a", loop_len=2.0)
    arm = next(t for t in p.tracks if t.name == "left arm")
    gt = next(t for t in p.tracks if t.channels == ["x", "y"])
    # movement for 1.5s to the clear pose, then still tail
    arm.samples = [{"t": i / RATE, "dt": 1 / RATE, "elbow": 90 + min(i, 30) * 1.0, "shoulder": 88 - min(i, 30) * 0.8} for i in range(n)]
    gt.samples = [{"t": i / RATE, "dt": 1 / RATE, "x": 30 + min(i, 30) * 2.0, "y": 20 - min(i, 30) * 0.5} for i in range(n)]
    p.save(export=True)

    if with_startle:
        st = Session("startle_a", loop_len=1.5)
        arm = next(t for t in st.tracks if t.name == "left arm")
        gt = next(t for t in st.tracks if t.channels == ["x", "y"])
        m = 30
        # sharp rise to +40 elbow by 0.5s, settle back to +30, still tail
        arm.samples = [{"t": i / RATE, "dt": 1 / RATE, "elbow": 90 + (40 * min(i, 10) / 10 if i < 12 else 30), "shoulder": 90.0} for i in range(m)]
        gt.samples = [{"t": i / RATE, "dt": 1 / RATE, "x": 30 + (8 * min(i, 10) / 10 if i < 12 else 6), "y": 20.0} for i in range(m)]
        st.save(export=True)


def make_bus(tmp):
    state = {"elbow": 90.0, "shoulder": 88.0, "wrist": 90.0, "finger0": 90.0, "finger1": 90.0, "finger2": 90.0, "finger3": 90.0, "x": 30.0, "y": 20.0}
    eases, plans = [], []
    bus = kb.KineticBus(
        library=kb.TemperamentLibrary(sessions_dir=tmp, owned=kb.OWNED_CHANNELS | {"x", "y"}),
        get_emotion=lambda: "calm_observant",
        is_drawing=lambda: False,
        get_gaze=lambda: (0.0, 0.0),
        get_person=lambda: "absent",
        on_log=lambda m: None,
        send_ease=lambda d: (state.update(d), eases.append((time.time(), dict(d)))),
        send_plan=lambda d, dt: plans.append((time.time(), dict(d))),
        get_state=lambda: dict(state),
    )
    bus._dir_flips = {c: False for c in kb.DIRECTION_CHANNELS}
    return bus, state, eases, plans


tmp = tempfile.mkdtemp(prefix="paper_startle_")
try:
    build_library(tmp)
    bus, state, eases, plans = make_bus(tmp)
    bus.enable()
    time.sleep(3.0)
    pre_bundle = bus.status()["bundle"]
    if not pre_bundle:
        failures.append("temperament never bloomed")

    # --- 1: paper check ---------------------------------------------------------
    n_plans = len(plans)
    wait = bus.paper_clear()
    if wait <= 0:
        failures.append(f"paper_clear returned {wait}")
    if bus.status()["state"] != kb.PAPER_STATE:
        failures.append(f"state after paper_clear: {bus.status()['state']}")
    time.sleep(min(wait, 5.0) + 0.5)
    if state["elbow"] < 110:  # the take ends at elbow 120 — playback must have moved us there
        failures.append(f"paper take did not play through servos (elbow={state['elbow']:.0f})")
    paper_plans = [d for _, d in plans[n_plans:]]
    if not paper_plans or max(d.get("x", 0) for d in paper_plans) < 80:  # take drives x to 90
        failures.append(f"gantry did not clear for the paper (plan sends: {len(paper_plans)})")
    if bus.status()["state"] != kb.PAPER_STATE:
        failures.append("hold released early")
    bus.paper_release()
    time.sleep(4.0)
    st_now = bus.status()
    if st_now["state"] != "calm_observant" or st_now["bundle"] != pre_bundle:
        failures.append(f"resume broke continuity: {st_now['state']}/{st_now['bundle']} (was {pre_bundle})")
    print(f"paper: wait {wait:.1f}s, servos+gantry cleared, hold held, same bundle resumed")

    # --- 2: organic startle -----------------------------------------------------
    time.sleep(1.0)
    snap = dict(state)
    i_ease, i_plan = len(eases), len(plans)
    bus.startle()
    time.sleep(0.4)
    post = [d for _, d in eases[i_ease:]]
    if not post:
        failures.append("startle produced no motion")
    else:
        first = post[0]
        jump = max(abs(first[c] - snap[c]) for c in first if c in snap)
        if jump > 2.5:
            failures.append(f"startle entry snapped {jump:.1f} degrees (must start from the live pose)")
    time.sleep(1.5)
    post = [d for _, d in eases[i_ease:]]
    peak = max((d.get("elbow", 0) for d in post), default=0)
    expect = snap["elbow"] + 40 * kb.KINETIC_STARTLE_NUDGE
    if not (expect - 8 <= peak <= expect + 8):
        failures.append(f"flinch peak {peak:.0f} != live {snap['elbow']:.0f} + 40 x {kb.KINETIC_STARTLE_NUDGE} (~{expect:.0f})")
    if len(plans) == i_plan:
        failures.append("gantry did not flinch")
    time.sleep(kb.KINETIC_STARTLE_HOLD_S + 3.0)
    if bus.status()["state"] != "calm_observant":
        failures.append(f"no return after startle: {bus.status()['state']}")
    print(f"startle: zero-offset entry, peak {peak:.0f} (~{expect:.0f} expected), gantry flinched, temperament returned")

    # --- 3: suppression during a safety hold ------------------------------------
    bus.paper_clear()
    time.sleep(0.5)
    bus.startle()
    if bus.status()["state"] != kb.PAPER_STATE:
        failures.append("startle stole the body from the paper clearing")
    bus.paper_release()
    print("suppression: startle refused while the paper clearing owns the body")
    bus.shutdown()

    # --- 4: fallback without a startle dataset ----------------------------------
    tmp2 = tempfile.mkdtemp(prefix="paper_startle_nb_")
    try:
        build_library(tmp2, with_startle=False)
        bus2, state2, eases2, _ = make_bus(tmp2)
        bus2.enable()
        time.sleep(2.5)
        i = len(eases2)
        bus2.startle()
        time.sleep(0.5)
        if len(eases2) == i:
            failures.append("built-in delta fallback sent nothing")
        bus2.shutdown()
        print("fallback: built-in delta nudge still fires without a dataset")
    finally:
        shutil.rmtree(tmp2, ignore_errors=True)
finally:
    shutil.rmtree(tmp, ignore_errors=True)

print("\n" + ("ALL OK" if not failures else "FAILURES:\n  " + "\n  ".join(failures)))
sys.exit(0 if not failures else 1)
