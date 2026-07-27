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
    arm.samples = [{"t": i / RATE, "dt": 1 / RATE, "elbow": 90 + 10 * math.sin(2 * math.pi * i / n), "shoulder": 90.0} for i in range(n)]
    wrist = next(t for t in s.tracks if t.channels == ["wrist"])
    wrist.samples = [{"t": i / RATE, "dt": 1 / RATE, "wrist": 90 + 8 * math.cos(2 * math.pi * i / n)} for i in range(n)]
    gantry.samples = [{"t": i / RATE, "dt": 1 / RATE, "x": 20.0, "y": 20.0} for i in range(n)]  # must be IGNORED by the bus
    return s


def main():
    failures = []
    tmp = tempfile.mkdtemp(prefix="kinetic_test_")
    session_mod.SESSIONS_DIR = tmp  # save/load route through the temp library
    try:
        for name, center in (("energized_engaged_a", 60), ("calm_observant_a", 140), ("drawing_a", 100)):
            make_session(name, center).save(export=True)
        make_session("energized_engaged_wip", 90).save()  # PROJECT save — runtime must not see it

        lib = kb.TemperamentLibrary(sessions_dir=tmp)
        buckets = lib.scan()
        if sorted(buckets) != ["calm_observant", "drawing", "energized_engaged"]:
            failures.append(f"bucketing wrong: {sorted(buckets)}")
        if any("wip" in fn for fns in buckets.values() for fn in fns):
            failures.append("a projects/ save leaked into the runtime library scan")
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
        for _ in range(60):  # lean settles toward the per-channel map over ~tau
            bus._update_lean()
        lean = bus._offsets.get("shoulder", 0.0)
        if not 6.5 <= lean <= 8.5:
            failures.append(f"lean did not settle toward shoulder map (got {lean:.1f}, expected ~8)")
        bias = bus._gaze_bias()
        if bias.get("shoulder") != 1.0 or bias.get("x") != 1.0 or abs(bias.get("elbow", 0.0)) > 1e-9:
            failures.append(f"gaze bias map wrong for gaze (1,0): {bias}")
        ctx["gaze"] = (0.0, 0.0)
        for _ in range(60):
            bus._update_lean()
        if abs(bus._offsets.get("shoulder", 0.0)) > 1.0:
            failures.append(f"lean did not release after gaze recentered ({bus._offsets.get('shoulder'):.1f})")
        print(f"gaze current: lean settled to {lean:.1f}° then released; bias on {sorted(k for k, v in bias.items() if v)}")

        before = device.channels["finger0"].value
        ctx["person"] = "visible"
        peak = before
        t1 = time.time()
        while time.time() - t1 < 0.6:  # freeze duration is random — catch the PEAK of the snap
            peak = max(peak, device.channels["finger0"].value)
            time.sleep(0.02)
        snap = peak - before
        if bus._last_startle == 0.0:
            failures.append("arrival did not trigger startle")
        if not 12 <= snap <= 50:
            failures.append(f"startle snap peaked at +{snap}° (expected ~+35 within clamp)")
        first_startle = bus._last_startle
        ctx["person"] = "absent"
        time.sleep(0.3)
        ctx["person"] = "visible"  # within cooldown — must NOT retrigger
        time.sleep(0.5)
        if bus._last_startle != first_startle:
            failures.append("startle retriggered inside cooldown")
        print(f"startle: snap +{snap}° with freeze, cooldown respected")

        # --- drawing override (poll: the fallback startle above holds ~3s) -----
        ctx["drawing"] = True
        t0d = time.time()
        while time.time() - t0d < 8.0 and bus._active_state != "drawing":
            time.sleep(0.3)
        if bus._active_state != "drawing":
            failures.append(f"drawing state did not take over (state={bus._active_state})")
        print(f"drawing override: active bundle {bus._active_bundle}")

        # --- recorded startle: flinch toward pose, hold, slow blend back -------
        make_session("startle_a", 120).save(export=True)
        bus._last_startle = 0.0  # clear cooldown from the fallback test
        finger_before = device.channels["finger0"].value
        bus.startle()
        if bus._active_state != "startle":
            failures.append(f"startle did not claim the body (state={bus._active_state})")
        if bus.status()["chains"] != 0:
            failures.append("generators kept running through the held flinch")
        time.sleep(0.4)
        moved = abs(device.channels["finger0"].value - finger_before)
        if moved < 3:
            failures.append(f"flinch nudge did not move the fingers (moved {moved:.1f}°)")
        t0s = time.time()
        while time.time() - t0s < 8.0 and bus._active_state == "startle":
            time.sleep(0.3)
        if bus._active_state != "drawing":
            failures.append(f"startle did not blend back to the running dataset (state={bus._active_state})")
        if bus.status()["chains"] == 0:
            failures.append("no chains resumed after the startle release")
        print(f"recorded startle: flinch (+{moved:.0f}° fingers), held, blended back to {bus._active_state}")

        # --- homing: refuse w/o dataset, PLAY the choreography, hold, sentinel --
        wait = bus.home_clear()
        if wait != 0.0:
            failures.append(f"home_clear guessed a path without a dataset (wait={wait})")
        n = int(LOOP * RATE)
        s = Session("homing_a", loop_len=LOOP * 2)
        arm = next(t for t in s.tracks if t.name == "left arm")
        # the ESCAPE MOVEMENT (2s: elbow 90 -> 62) followed by a 2s STILL
        # TAIL (holding the tuck while the record pass runs out) — the wait
        # must cover the MOTION only, never the tail
        arm.samples = [{"t": i / RATE, "dt": 1 / RATE, "elbow": 90.0 - 28.0 * i / (n - 1), "shoulder": 64.0} for i in range(n)] + [
            {"t": (n + i) / RATE, "dt": 1 / RATE, "elbow": 62.0, "shoulder": 64.0} for i in range(n)
        ]
        wrist_t = next(t for t in s.tracks if t.channels == ["wrist"])
        wrist_t.samples = [{"t": i / RATE, "dt": 1 / RATE, "wrist": 66.0} for i in range(2 * n)]
        s.save(export=True)
        wait = bus.home_clear()
        if not 2.5 <= wait <= 4.6:  # entry ease + MOTION + margin — a 5.5s+ wait means the still tail leaked in
            failures.append(f"wait must cover the motion, not the still tail (wait={wait}, take 4s, motion 2s)")
        if bus._active_state != "homing":
            failures.append(f"homing did not claim the body (state={bus._active_state})")
        if bus.status()["chains"] != 0:
            failures.append("markov generators kept running during homing (must be straight playback)")
        # the take must be TRAVERSED (playback), gently, ending at the tuck
        trace_t = []
        t0r = time.time()
        while time.time() - t0r < wait + 0.3:
            trace_t.append(device.channels["elbow"].target)
            time.sleep(0.1)
        max_step = max(abs(b - a) for a, b in zip(trace_t, trace_t[1:]))
        mid = trace_t[len(trace_t) // 2]
        if max_step > 6:
            failures.append(f"homing SNAPPED: elbow target jumped {max_step:.1f}° in 100ms")
        if not 64 < mid < 90:
            failures.append(f"take not traversed — mid-playback elbow {mid} (expected between start and tuck)")
        if abs(trace_t[-1] - 62.0) > 2.0:
            failures.append(f"choreography did not end at the tuck (elbow target {trace_t[-1]})")
        # RE-TRIGGER mid-playback: must RESTART cleanly, never overlap —
        # two player sets fighting shows up as zigzagging targets
        bus.home_clear()  # run 2 starts from the held tuck
        time.sleep(1.8)  # deep into run 2's playback
        wait3 = bus.home_clear()  # run 3 lands MID-PLAYBACK — the fight case
        if not 2.5 <= wait3 <= 4.6:
            failures.append(f"mid-playback re-trigger did not restart (wait={wait3})")
        trace2 = []
        t0r2 = time.time()
        while time.time() - t0r2 < wait3 + 0.3:
            trace2.append(device.channels["elbow"].target)
            time.sleep(0.1)
        max_step2 = max(abs(b - a) for a, b in zip(trace2, trace2[1:]))
        if max_step2 > 6:
            failures.append(f"re-triggered homing OVERLAPPED the old run (elbow jumped {max_step2:.1f}°/100ms)")
        if abs(trace2[-1] - 62.0) > 2.0:
            failures.append(f"restarted choreography did not end at the tuck ({trace2[-1]})")
        print(f"re-trigger mid-playback: restarted without overlap (max {max_step2:.1f}°/100ms, end {trace2[-1]:.0f}°)")

        # cross-process release: the idle SUBPROCESS homes; ensure_homed touches
        # the sentinel and the bus must notice the fresh mtime
        from utils.hooks import HOMING_SENTINEL

        with open(HOMING_SENTINEL, "w") as hf:
            hf.write(str(time.time()))
        t0h = time.time()
        while time.time() - t0h < 8.0 and bus._active_state == "homing":
            time.sleep(0.3)
        if bus._active_state != "drawing":
            failures.append(f"sentinel did not release the homing hold (state={bus._active_state})")
        print(
            f"homing: refused w/o dataset; played the escape ({max_step:.1f}°/100ms max, mid {mid:.0f}°, end {trace_t[-1]:.0f}°), sentinel released"
        )

        bus.shutdown()

        # --- practice-room mode: injected callbacks, full-body ownership -------
        full = {"x", "y", "pen", "elbow", "shoulder", "wrist", "finger0", "finger1", "finger2", "finger3", "lung"}
        lib2 = kb.TemperamentLibrary(sessions_dir=tmp, owned=full)
        sends = {"ease": 0, "plan": 0}
        live = {c: 90.0 for c in full}
        live.update({"x": 20.0, "y": 20.0, "pen": 34.0})
        ctx2 = {"drawing": False}
        bus2 = kb.KineticBus(
            library=lib2,
            is_drawing=lambda: ctx2["drawing"],
            get_gaze=lambda: (0.0, 0.0),
            get_person=lambda: "absent",
            on_log=lambda m: None,
            send_ease=lambda d: sends.__setitem__("ease", sends["ease"] + 1),
            send_plan=lambda d, dt: sends.__setitem__("plan", sends["plan"] + 1),
            send_step=lambda d: None,
            get_state=lambda: dict(live),
            owned=full,
        )
        bus2.enable()
        bus2.set_emotion("energized_engaged")
        time.sleep(3.0)
        if bus2.device is not None:
            failures.append("practice-room bus built a device it must not own")
        if sends["ease"] == 0 or sends["plan"] == 0:
            failures.append(f"practice-room bus not driving injected callbacks: {sends}")

        # --- the drawing gate: while the machine draws, the gantry/pen are
        # UNTOUCHABLE no matter what the active chains contain ---------------
        ctx2["drawing"] = True
        time.sleep(3.0)  # supervisor switches to the drawing dataset; its group chain still holds x/y dims
        plan_frozen = sends["plan"]
        ease_before = sends["ease"]
        time.sleep(2.0)
        if sends["plan"] != plan_frozen:
            failures.append(f"bus contested the gantry during drawing ({sends['plan'] - plan_frozen} plan sends leaked)")
        if sends["ease"] <= ease_before:
            failures.append("left hand stopped acting during drawing (it should keep its temperament)")
        print(f"drawing gate: plan sends frozen at {plan_frozen}, ease kept flowing ({sends['ease'] - ease_before} in 2s)")
        bus2.shutdown()

        # --- retire: bundle disappears from the runtime scan --------------------
        lib2.retire("session_drawing_a.json")
        if "drawing" in lib2.scan():
            failures.append("retired bundle still visible to the scan")
        print(f"practice-room: {sends['ease']} ease + {sends['plan']} plan sends via callbacks; retire hides the bundle")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("\n" + ("ALL OK" if not failures else "FAILURES:\n  " + "\n  ".join(failures)))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
