"""Prove the kinetic bus end-to-end, headless (sim device, temp library).

1. TemperamentLibrary buckets sessions by name prefix; drawing overrides
   emotion; missing emotion falls back to any bundle, never paralysis
2. Chains are restricted to owned (lefthand) channels — gantry/pen tracks
   in a session are ignored at runtime
3. Temperament switches are SEAMLESS: no finger jump bigger than the
   crossfade's substep rate when the emotion changes mid-motion
4. Gaze nudge biases sends; startle freezes + snaps on person arrival,
   with cooldown

  python debug/test_kinetic_bus.py
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
from motor_panel.devices import build_devices
from motor_panel.session import Session

LOOP = 2.0
RATE = 20


def make_session(name, finger_center):
    s = Session(name, loop_len=LOOP)
    hand = next(t for t in s.tracks if t.name == "hand (fingers)")
    arm = next(t for t in s.tracks if t.name == "left arm")
    gantry = next(t for t in s.tracks if t.channels == ["x", "y"])
    n = int(LOOP * RATE)
    hand.samples = [
        {"t": i / RATE, "dt": 1 / RATE, **{f"finger{j}": finger_center + 15 * math.sin(2 * math.pi * i / n + j) for j in range(4)}} for i in range(n)
    ]
    arm.samples = [
        {"t": i / RATE, "dt": 1 / RATE, "elbow": 90 + 10 * math.sin(2 * math.pi * i / n), "shoulder": 90.0, "wrist": 90.0} for i in range(n)
    ]
    gantry.samples = [{"t": i / RATE, "dt": 1 / RATE, "x": 20.0, "y": 20.0} for i in range(n)]  # must be IGNORED by the bus
    return s


def main():
    failures = []
    tmp = tempfile.mkdtemp(prefix="kinetic_test_")
    session_mod.SESSIONS_DIR = tmp  # save/load route through the temp library
    try:
        for name, center in (("energized_engaged_a", 60), ("calm_observant_a", 140), ("drawing_a", 100)):
            make_session(name, center).save()

        lib = kb.TemperamentLibrary(sessions_dir=tmp)
        buckets = lib.scan()
        if sorted(buckets) != ["calm_observant", "drawing", "energized_engaged"]:
            failures.append(f"bucketing wrong: {sorted(buckets)}")
        if lib.bundle_for("energized_engaged", drawing=True) != "session_drawing_a.json":
            failures.append("drawing state did not override emotion")
        if lib.bundle_for("withdrawn_distant", drawing=False) is None:
            failures.append("missing emotion did not fall back to another bundle")
        chains = lib.chains("session_energized_engaged_a.json")
        chan = {c for ch in chains.values() for c in ch["channels"]}
        if "x" in chan or "y" in chan:
            failures.append(f"bus trained gantry channels: {chan}")
        print(f"library: buckets {sorted(buckets)}, owned channels only: {sorted(chan)}")

        # --- live bus with injected context ----------------------------------
        device = build_devices()[1]  # sim lefthand
        ctx = {"drawing": False, "gaze": (0.0, 0.0), "person": "absent"}
        bus = kb.KineticBus(
            device=device,
            library=lib,
            is_drawing=lambda: ctx["drawing"],
            get_gaze=lambda: ctx["gaze"],
            get_person=lambda: ctx["person"],
            on_log=lambda m: None,
        )
        bus.enable()
        bus.set_emotion("energized_engaged")
        time.sleep(3.0)  # supervisor picks the bundle, generators enter
        if bus._active_state != "energized_engaged":
            failures.append(f"bundle not selected (state={bus._active_state})")

        # --- 3: seamless switch ----------------------------------------------
        trace = []
        t0 = time.time()
        switched = False
        while time.time() - t0 < 6.0:
            trace.append(device.channels["finger0"].value)
            if not switched and time.time() - t0 > 1.0:
                bus.set_emotion("calm_observant")  # 60-centered -> 140-centered
                switched = True
            time.sleep(0.03)
        jumps = [abs(b - a) for a, b in zip(trace, trace[1:])]
        max_jump = max(jumps)
        reached = any(v > 115 for v in trace[-60:])
        if max_jump > 12:
            failures.append(f"switch snapped: max per-30ms finger jump {max_jump}° (crossfade should bound this)")
        if not reached:
            failures.append(f"never reached the new temperament (last values ~{trace[-5:]})")
        if bus._active_state != "calm_observant":
            failures.append(f"active state {bus._active_state} after switch")
        print(f"switch: max 30ms jump {max_jump}°, settled in new temperament: {reached}")

        # --- 4: gaze nudge + startle ------------------------------------------
        ctx["gaze"] = (1.0, 0.0)
        bus._update_gaze_offsets()
        bus._send_ease({"shoulder": 90.0})
        target = device.channels["shoulder"].target
        if not 98 <= target <= 102:
            failures.append(f"gaze nudge missing: shoulder target {target}, expected ~100")
        print(f"gaze nudge: shoulder 90 -> target {target}")

        before = device.channels["finger0"].value
        ctx["person"] = "visible"
        time.sleep(0.5)
        after = device.channels["finger0"].value
        snap = after - before
        if bus._last_startle == 0.0:
            failures.append("arrival did not trigger startle")
        if not 15 <= snap <= 50:
            failures.append(f"startle snap was {snap}° (expected ~+35 within clamp)")
        first_startle = bus._last_startle
        ctx["person"] = "absent"
        time.sleep(0.3)
        ctx["person"] = "visible"  # within cooldown — must NOT retrigger
        time.sleep(0.5)
        if bus._last_startle != first_startle:
            failures.append("startle retriggered inside cooldown")
        print(f"startle: snap +{snap}° with freeze, cooldown respected")

        # --- drawing override --------------------------------------------------
        ctx["drawing"] = True
        time.sleep(2.5)
        if bus._active_state != "drawing":
            failures.append(f"drawing state did not take over (state={bus._active_state})")
        print(f"drawing override: active bundle {bus._active_bundle}")

        bus.shutdown()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("\n" + ("ALL OK" if not failures else "FAILURES:\n  " + "\n  ".join(failures)))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
