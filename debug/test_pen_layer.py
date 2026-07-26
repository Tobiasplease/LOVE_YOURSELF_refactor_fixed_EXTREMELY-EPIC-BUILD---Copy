"""Prove the pen layer's step-channel semantics, closed-loop and headless.

The pen is a "step channel": recorded continuously like every channel, but
emitted ONLY on value change — never interpolated (a half-lowered pen drags)
and never streamed (each M3 barriers the GRBL writer). This script verifies:

  1. Player emits pen changes exactly when the take changes, not per-sample
  2. Pen never leaks into the ease path (no 20Hz M3 storm)
  3. train() + Generator keep pen states inside the demonstrated set and
     emit only on change
  4. Transport routes send_step through record/play wiring

    python debug/test_pen_layer.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from motor_panel import arms_markov as engine
from motor_panel.session import Session, Transport

PEN_UP, PEN_DOWN = 34.0, 52.0
LOOP = 2.0
N = int(LOOP * engine.SAMPLE_RATE)


def make_take():
    """A circle with pen down for the middle half of the loop."""
    import math

    out = []
    for i in range(N):
        t = i / engine.SAMPLE_RATE
        a = 2 * math.pi * t / LOOP
        out.append(
            {
                "t": t,
                "dt": 1.0 / engine.SAMPLE_RATE,
                "x": 20 + 10 * math.cos(a),
                "y": 20 + 10 * math.sin(a),
                "pen": PEN_DOWN if N // 4 <= i < 3 * N // 4 else PEN_UP,
            }
        )
    return out


def main():
    failures = []
    take = make_take()

    # --- 1+2: Player step semantics + plan lookahead --------------------------
    steps, eases, plans = [], [], []
    t_start = time.time()
    p = engine.Player(
        take,
        ["x", "y", "pen"],
        send_ease=lambda d: eases.append(dict(d)),
        send_plan=lambda d, dt: plans.append((time.time() - t_start, dict(d))),
        send_step=lambda d: steps.append(dict(d)),
    )
    p.start()
    time.sleep(LOOP + 0.5)
    p.stop()
    # lookahead: the final gantry waypoint (take-time ~1.8s) must arrive EARLY
    # (planner needs queued segments to blend junctions — on-time = choppy)
    if plans and not plans[-1][0] < LOOP - 0.05:
        failures.append(f"no plan lookahead: last waypoint arrived at {plans[-1][0]:.2f}s of a {LOOP}s take")
    plans = [d for _, d in plans]
    pen_steps = [d["pen"] for d in steps if "pen" in d]
    if pen_steps != [PEN_UP, PEN_DOWN, PEN_UP]:
        failures.append(f"player pen steps {pen_steps}, expected [{PEN_UP}, {PEN_DOWN}, {PEN_UP}]")
    if any("pen" in d for d in eases):
        failures.append("pen leaked into the ease path")
    if not plans:
        failures.append("no plan (x/y) traffic during playback")
    print(f"player: {len(steps)} step sends over {N} samples, {len(plans)} plan sends — " f"pen sequence {pen_steps}")

    # --- 3: train + generate -------------------------------------------------
    chain = engine.train(take, ["x", "y", "pen"])
    print(f"train: {chain['unique_states']} states over channels {chain['channels']}")
    steps, eases = [], []
    gen = engine.Generator(chain, send_ease=lambda d: eases.append(dict(d)), send_plan=lambda d, dt: None, send_step=lambda d: steps.append(dict(d)))
    gen.start({"x": 30.0, "y": 20.0, "pen": PEN_UP})
    time.sleep(3.0)
    gen.stop()
    gen_pens = [d["pen"] for d in steps if "pen" in d]
    bad = [v for v in gen_pens if v not in (PEN_UP, PEN_DOWN)]
    if bad:
        failures.append(f"generator produced undemonstrated pen states: {bad}")
    repeats = [i for i in range(1, len(gen_pens)) if gen_pens[i] == gen_pens[i - 1]]
    if repeats:
        failures.append(f"generator re-sent unchanged pen values at {repeats}")
    if any("pen" in d for d in eases):
        failures.append("generator interpolated the pen (ease path)")
    print(f"generate: {len(gen_pens)} pen changes, all in demonstrated set: {not bad}")

    # --- 4: Transport routing -------------------------------------------------
    session = Session("pen_test", loop_len=LOOP)
    for t in session.tracks:
        if t.channels == ["pen"]:
            t.samples = [{"t": s["t"], "dt": s["dt"], "pen": s["pen"]} for s in take]
        if t.channels == ["x", "y"]:
            t.samples = [{"t": s["t"], "dt": s["dt"], "x": s["x"], "y": s["y"]} for s in take]
    steps = []
    tr = Transport(
        session,
        get_state=lambda: {"x": 0.0, "y": 0.0, "pen": PEN_UP},
        send_ease=lambda d: None,
        send_plan=lambda d, dt: None,
        on_status=lambda m: None,
        send_step=lambda d: steps.append(dict(d)),
    )
    tr.play()
    time.sleep(1.5)
    tr.stop()
    if not steps:
        failures.append("Transport.play never routed a pen step")
    print(f"transport: {len(steps)} pen sends through play()")

    print("\n" + ("ALL OK" if not failures else "FAILURES:\n  " + "\n  ".join(failures)))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
