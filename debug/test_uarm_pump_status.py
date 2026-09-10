#!/usr/bin/env python3
"""Does the uArm's pump status actually tell paper from bare table?

The post-GRBL paper move plays a recording blind — nothing checks the sheet is
where the cup reaches, so a slipped or missing sheet leaves the arm pressing on
the table with the pump running. The SDK reports:

    get_pump_status() -> 0 stop, 1 working (nothing held), 2 pump thing (holding)

If 2 really shows up with a sheet held and 1 with the cup on bare table, that is
the abort signal a watchdog can use — direct, no camera calibration. This proves
it before anything is built on it.

READ-ONLY BY DEFAULT: connects and prints status. It sends no movement. Pass
--pump to let it switch the pump on and off, which is the only way to see 1 vs 2
(status is 0 while the pump is off); the arm still does not move, so put the cup
where you want it by hand first.

Usage: python debug/test_uarm_pump_status.py [--pump] [--seconds N]

What to do with it:
  1. run with --pump holding the cup against a sheet of paper -> expect 2
  2. run with --pump with the cup on the bare table          -> expect 1
  3. run with --pump with the cup in free air                -> expect 1
If paper and table both read the same, this signal cannot carry the watchdog and
the gate has to come from the finished-drawing photograph instead.
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

parser = argparse.ArgumentParser()
parser.add_argument("--pump", action="store_true", help="switch the pump on for the test (no movement)")
parser.add_argument("--seconds", type=float, default=20.0)
args = parser.parse_args()

MEANING = {0: "stop (pump off)", 1: "WORKING — nothing held", 2: "PUMP THING — holding something"}

swift = None
try:
    from uarm.wrapper import SwiftAPI

    print("Connecting to the uArm…")
    swift = SwiftAPI()
    swift.waiting_ready(timeout=10)
    print(f"Connected: {swift.get_device_info()}")

    if args.pump:
        print("\nPump ON. Put the cup against paper, then the bare table, and watch the reading.")
        swift.set_pump(True)
        time.sleep(1.5)  # let the vacuum build before believing anything
    else:
        print("\nRead-only: the pump stays off, so this will read 0 throughout.")
        print("Re-run with --pump to tell 1 from 2.")

    seen = {}
    end = time.time() + args.seconds
    while time.time() < end:
        status = swift.get_pump_status()
        seen[status] = seen.get(status, 0) + 1
        print(f"  {time.strftime('%H:%M:%S')}  status={status}  {MEANING.get(status, '?')}")
        time.sleep(0.5)

    print(f"\nCounts: " + ", ".join(f"{k}={v}" for k, v in sorted(seen.items(), key=lambda kv: str(kv[0]))))
    if 2 in seen and 1 in seen:
        print("Both 1 and 2 seen — the signal separates held from not-held. A watchdog can use it.")
    elif 2 in seen:
        print("Only 2 seen — try again with the cup on bare table to check it drops to 1.")
    elif 1 in seen and args.pump:
        print("Only 1 seen — either nothing was gripped, or this cup/firmware never reports 2.")

finally:
    if swift is not None:
        try:
            swift.set_pump(False)
            print("Pump off.")
        except Exception:
            pass
        try:
            swift.disconnect()
        except Exception:
            pass
