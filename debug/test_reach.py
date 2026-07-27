"""Prove the reach current: while a person is tracked, the arm leans out
toward the gaze direction — measured-calibration IK when available,
proportional fallback until then — ramping in on presence and melting away
on departure, always inside the caps.

    python debug/test_reach.py
"""

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from motor_panel import kinetic_bus as kb


def make_bus(ctx):
    return kb.KineticBus(
        get_emotion=lambda: "calm_observant",
        is_drawing=lambda: False,
        get_gaze=lambda: (ctx["gx"], ctx["gy"]),
        get_person=lambda: ctx["person"],
        on_log=lambda m: None,
        send_ease=lambda d: None,
        send_plan=lambda d, dt: None,
        send_step=lambda d: None,
        get_state=lambda: {},
        owned=kb.OWNED_CHANNELS,
    )


def settle(bus, n=250):
    for _ in range(n):
        bus._update_lean()


def main():
    failures = []

    # --- fallback reach (no calibration yet): proportional joint-space ---------
    ctx = {"gx": 1.0, "gy": 0.0, "person": "visible"}
    bus = make_bus(ctx)
    bus.arm_calib_path = "/nonexistent/never.json"
    settle(bus)
    off = bus._offsets.get("shoulder", 0.0)
    # shoulder fallback pose at gx=1: neutral + range/2*0.8 = +24° -> *0.6 strength = 14.4
    if not 12 <= off <= 16:
        failures.append(f"fallback reach wrong (shoulder offset {off:.1f}, expected ~14.4)")
    if bus.status()["reach"] < 0.95:
        failures.append(f"reach did not ramp to full presence ({bus.status()['reach']})")
    print(f"fallback reach: shoulder leans +{off:.1f}° toward gaze at full presence")

    # --- departure: reach melts away, ambient lean remains ---------------------
    ctx["person"] = "absent"
    settle(bus)
    off_after = bus._offsets.get("shoulder", 0.0)
    # ambient lean at gx=1: 8° * strength 1.0
    if not 6.5 <= off_after <= 9.5:
        failures.append(f"reach did not decay to the ambient lean (shoulder {off_after:.1f}, expected ~8)")
    print(f"departure: reach decayed, ambient lean holds at +{off_after:.1f}°")

    # --- calibrated reach: the 9-point grid IS the IK table --------------------
    # grid[iy][ix] = (shoulder, elbow); shoulder spans 60..120 with u, elbow with v
    grid = [[(60 + 30 * ix, 60 + 30 * iy) for ix in range(3)] for iy in range(3)]
    tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w")
    json.dump({"grid": grid}, tmp)
    tmp.close()
    try:
        ctx2 = {"gx": 1.0, "gy": 0.0, "person": "visible"}
        bus2 = make_bus(ctx2)
        bus2.arm_calib_path = tmp.name
        settle(bus2)
        s_off = bus2._offsets.get("shoulder", 0.0)
        e_off = bus2._offsets.get("elbow", 0.0)
        # gaze (1,0) -> u=1 (shoulder pose 120 -> delta +30*0.6=18), v=0.5 (elbow pose 90 -> delta 0)
        if not 16 <= s_off <= 20:
            failures.append(f"calibrated reach shoulder {s_off:.1f}, expected ~18 (pose 120)")
        if abs(e_off) > 2.0:
            failures.append(f"calibrated reach elbow {e_off:.1f}, expected ~0 (pose = neutral)")
        # cap: a grid with extreme poses must clamp at KINETIC_REACH_MAX_DEG
        wild = [[(0, 0) for _ in range(3)] for _ in range(3)]
        with open(tmp.name, "w") as f:
            json.dump({"grid": wild}, f)
        os.utime(tmp.name)
        bus2._calib_cache = (0.0, None)
        settle(bus2)
        s_wild = bus2._offsets.get("shoulder", 0.0)
        if abs(s_wild) > kb.KINETIC_REACH_MAX_DEG + 1:
            failures.append(f"reach exceeded the cap ({s_wild:.1f} vs {kb.KINETIC_REACH_MAX_DEG})")
        print(f"calibrated reach: shoulder +{s_off:.1f}° elbow {e_off:+.1f}°; wild grid capped at {s_wild:.1f}°")
    finally:
        os.unlink(tmp.name)

    # --- direction flips: this-mode-only reversal, persisted -------------------
    ctx3 = {"gx": 1.0, "gy": 0.0, "person": "visible"}
    bus3 = make_bus(ctx3)
    bus3.arm_calib_path = "/nonexistent/never.json"
    dtmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
    dtmp.close()
    os.unlink(dtmp.name)
    bus3.directions_path = dtmp.name
    bus3._dir_flips = bus3._load_direction_flips()
    bus3.set_direction_flip("shoulder", True)
    settle(bus3)
    off_flipped = bus3._offsets.get("shoulder", 0.0)
    if not -16 <= off_flipped <= -12:
        failures.append(f"flipped shoulder should reach the OTHER way (got {off_flipped:.1f}, expected ~-14.4)")
    bias = bus3._gaze_bias()
    if bias.get("shoulder", 0) >= 0 or bias.get("wrist", 0) <= 0:
        failures.append(f"flip must reverse the choice/tempo bias for that channel only: {bias}")
    bus4 = make_bus(ctx3)
    bus4.directions_path = dtmp.name
    bus4._dir_flips = bus4._load_direction_flips()
    if not bus4.direction_flips().get("shoulder"):
        failures.append("direction flip did not persist across bus instances")
    os.unlink(dtmp.name)
    print(f"direction flip: shoulder reaches {off_flipped:+.1f}° (reversed), bias flipped for shoulder only, persisted")

    print("\n" + ("ALL OK" if not failures else "FAILURES:\n  " + "\n  ".join(failures)))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
