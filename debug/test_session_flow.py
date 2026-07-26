"""Prove the looper's record → play round-trip is real, not a facade.

Drives a synthetic performance (sinusoid joints) through a Transport with
fake routing callbacks, then verifies:

  1. a full record pass commits a take that actually contains the motion
  2. playback re-emits the performance accurately (value error within the
     20Hz-sampling tolerance)
  3. Stop mid-pass COMMITS the partial take (the old behavior silently
     discarded it — the "recording is a facade" bug)
  4. rec-enable flags clear after commit

    python debug/test_session_flow.py
"""

import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from motor_panel.session import Session, Transport

LOOP = 2.0


def performance(t):
    return {
        "elbow": 90 + 25 * math.sin(2 * math.pi * t / LOOP),
        "shoulder": 90 + 20 * math.cos(2 * math.pi * t / LOOP),
    }


def make_transport(session, played):
    start = {"t": None}

    def get_state():
        t = 0.0 if start["t"] is None else time.time() - start["t"]
        s = {"x": 0.0, "y": 0.0, "pen": 34.0, "lung": 85.0, "finger0": 90.0, "finger1": 90.0, "finger2": 90.0, "finger3": 90.0}
        s.update(performance(t))
        return s

    tr = Transport(
        session,
        get_state=get_state,
        send_ease=lambda d: played.append((time.time(), dict(d))),
        send_plan=lambda d, dt: None,
        on_status=lambda m: None,
        send_step=lambda d: None,
    )
    return tr, start


def main():
    failures = []

    # --- 1: full pass commits real motion ------------------------------------
    session = Session("flow_test", loop_len=LOOP)
    arm = next(t for t in session.tracks if t.name == "left arm")
    arm.armed = True
    played = []
    tr, start = make_transport(session, played)
    start["t"] = time.time()
    tr.record()
    time.sleep(LOOP + 0.6)
    if not arm.has_take:
        failures.append("full pass committed no take")
    else:
        elbows = [s["elbow"] for s in arm.samples]
        span = max(elbows) - min(elbows)
        if span < 40:
            failures.append(f"take is flat (elbow span {span:.1f}°, expected ~50)")
        if arm.armed:
            failures.append("rec-enable did not clear after commit")
        print(f"record: {len(arm.samples)} samples, elbow span {span:.1f}°")

    # --- 2: playback reproduces the performance -------------------------------
    played.clear()
    tr.play(speed=1.0)
    t_play = time.time()
    time.sleep(LOOP)
    tr.stop()
    errs = []
    for ts, d in played:
        if "elbow" not in d:
            continue
        expect = performance((ts - t_play) % LOOP)["elbow"]
        errs.append(abs(d["elbow"] - expect))
    if not errs:
        failures.append("playback emitted nothing")
    else:
        mean_err = sum(errs) / len(errs)
        if mean_err > 12:
            failures.append(f"playback inaccurate: mean elbow error {mean_err:.1f}° (tolerance 12)")
        print(f"play: {len(errs)} ease sends, mean elbow error {mean_err:.2f}°")

    # --- 3: stop mid-pass keeps the partial -----------------------------------
    arm.samples = None
    arm.armed = True
    start["t"] = time.time()
    tr.record()
    time.sleep(0.8)
    tr.stop()
    if not arm.has_take:
        failures.append("early Stop discarded the take (the facade bug is back)")
    else:
        dur = arm.samples[-1]["t"]
        if not 0.5 < dur < 1.5:
            failures.append(f"partial take duration {dur:.2f}s, expected ~0.8")
        print(f"early stop: partial take kept, {len(arm.samples)} samples over {dur:.2f}s")
    if arm.armed:
        failures.append("rec-enable did not clear after early stop")

    print("\n" + ("ALL OK" if not failures else "FAILURES:\n  " + "\n  ".join(failures)))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
