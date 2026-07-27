"""Verify the kinetic bus ↔ runtime seams without running machine.py.

The integration relies on three contracts, each checked here:
  1. EMOTION: the bus PULLS mood_engine.get_emotion_for_hand_controller()
     every supervisor tick — verify the mood engine's vector→state mapping
     produces exactly the five dataset states, and that the bus follows a
     changing provider (no dependence on old push plumbing).
  2. HOMING: utils.hooks exposes on_grbl_homing_start/_done and
     grbl_utils.ensure_homed actually calls them (static check — no serial).
  3. MONITOR: motor_panel.runtime_monitor imports clean (machine.py opens
     it in the old hand controller's slot).

    python debug/test_runtime_wiring.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from motor_panel import kinetic_bus as kb


def main():
    failures = []

    # --- 1a: mood engine vector -> emotion state mapping ----------------------
    try:
        from mood.mood import MoodEngine

        engine = MoodEngine()
        cases = [
            ((0.5, 0.8, 0.0), "energized_engaged"),
            ((-0.5, 0.2, 0.0), "withdrawn_distant"),
            ((0.1, 0.1, 0.0), "calm_observant"),
            ((-0.02, 0.1, 0.0), "quiet_detached"),
            ((-0.2, 0.6, 0.0), "alert_curious"),
        ]
        for vec, expected in cases:
            engine.mood_vector = vec
            got = engine.get_emotion_for_hand_controller()
            if got != expected:
                failures.append(f"mood vector {vec} -> {got}, expected {expected}")
            if got not in kb.EMOTIONS:
                failures.append(f"mood engine produced unknown state {got}")
        print(f"mood mapping: {len(cases)} vectors -> dataset states, all correct")
    except Exception as e:
        failures.append(f"MoodEngine unusable headless: {e}")

    # --- 1b: the bus FOLLOWS a pulled emotion ---------------------------------
    current = {"emotion": "calm_observant"}
    bus = kb.KineticBus(
        get_emotion=lambda: current["emotion"],
        is_drawing=lambda: False,
        get_gaze=lambda: (0.0, 0.0),
        get_person=lambda: "absent",
        on_log=lambda m: None,
        send_ease=lambda d: None,
        send_plan=lambda d, dt: None,
        send_step=lambda d: None,
        get_state=lambda: {},
        owned=kb.OWNED_CHANNELS,
    )
    if bus._desired_state() != "calm_observant":
        failures.append(f"bus did not pull the emotion (got {bus._desired_state()})")
    current["emotion"] = "energized_engaged"
    if bus._desired_state() != "energized_engaged":
        failures.append("bus did not follow the changed emotion (pull is broken)")
    print("emotion pull: bus follows the provider, no push plumbing required")

    # --- 2: homing hooks exist and grbl_utils calls them ----------------------
    from utils import hooks

    for name in ("on_grbl_homing_start", "on_grbl_homing_done"):
        if not hasattr(hooks, name):
            failures.append(f"utils.hooks missing {name}")
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "grbl", "grbl_utils.py")).read()
    if "on_grbl_homing_start" not in src or "on_grbl_homing_done" not in src:
        failures.append("grbl_utils.ensure_homed does not call the homing hooks")
    if src.index("on_grbl_homing_start") > src.index('send_cmd(ser, "$H"'):
        failures.append("tuck-wait must come BEFORE $H in ensure_homed")
    print("homing hooks: present and wired before/after $H in ensure_homed")

    # --- 3: monitor imports (machine.py opens it in the old controller slot) --
    try:
        import motor_panel.runtime_monitor  # noqa: F401

        print("runtime monitor: imports clean")
    except Exception as e:
        failures.append(f"runtime_monitor import failed: {e}")

    print("\n" + ("ALL OK" if not failures else "FAILURES:\n  " + "\n  ".join(failures)))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
