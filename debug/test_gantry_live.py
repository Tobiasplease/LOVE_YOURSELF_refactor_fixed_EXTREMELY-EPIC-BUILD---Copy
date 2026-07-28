"""Live hardware probe: does the gantry obey dataset coordinates?

The runtime path (GantryLink) was streaming G1s into a machine that never
moved — and the link was deaf to GRBL's replies. This script is the same
conversation with the volume up: home, then send three dataset-style
targets and echo EVERY line GRBL says back, then ask where it thinks it
is. Run with machine.py STOPPED (the port is exclusive):

    python debug/test_gantry_live.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from grbl.grbl_utils import ensure_homed, find_grbl_port
from grbl.warp_calibration import clamp_to_reach

TARGETS = [(30.0, 20.0), (45.0, 25.0), (35.0, 12.0)]  # dataset-flavored coords
FEED = 800


def say(ser, cmd, wait=1.5):
    ser.write((cmd + "\n").encode() if not cmd == "?" else b"?")
    ser.flush()
    t0 = time.time()
    lines = []
    while time.time() - t0 < wait:
        line = ser.readline().decode(errors="replace").strip()
        if line:
            lines.append(line)
            print(f"   grbl> {line}")
            if line == "ok" and cmd != "?":
                break
    if not lines:
        print("   grbl> (silence)")
    return lines


def status(ser):
    print("status:")
    return " ".join(say(ser, "?", wait=1.0))


def main():
    print("Opening GRBL (this resets it)…")
    ser = find_grbl_port(preferred_port=os.getenv("GRBL_PORT", "/dev/arduino_cnc"))
    if not ser:
        print("No GRBL found — is machine.py still running? (port is exclusive)")
        return 1
    try:
        before = status(ser)
        print("\nHoming (ensure_homed — same call the runtime uses)…")
        ensure_homed(ser, max_retries=2)
        print("\nAfter homing:")
        status(ser)
        for x, y in TARGETS:
            cx, cy = clamp_to_reach(x, y)
            note = "" if (cx, cy) == (x, y) else f"   (clamped from {x},{y})"
            print(f"\nG1 X{cx:.2f} Y{cy:.2f} F{FEED}{note}")
            say(ser, f"G1 X{cx:.2f} Y{cy:.2f} F{FEED}")
            time.sleep(2.5)  # let it physically travel
            status(ser)
        print("\nBack off + final position:")
        say(ser, "G1 X5 Y5 F800")
        time.sleep(2.5)
        final = status(ser)
        print("\nVERDICT:")
        if "Alarm" in final:
            print("  GRBL is ALARM-locked — this is why the runtime arm never moves.")
        elif "MPos:0.000,0.000" in final.replace(" ", "") or "WPos:0.000,0.000" in final.replace(" ", ""):
            print("  GRBL reports it never left home — motion is being rejected or scaled away; read the replies above.")
        else:
            print("  The machine MOVED on dataset coordinates. If the runtime arm still parks, the fault is upstream of GRBL —")
            print("  watch machine.py's terminal for 'gantry: status', 'gantry: streaming', 'gantry: GRBL said' lines.")
    finally:
        try:
            ser.close()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
