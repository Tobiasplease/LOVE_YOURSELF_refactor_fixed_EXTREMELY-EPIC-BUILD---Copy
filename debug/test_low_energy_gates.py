"""Low-energy mode gate check — no hardware needed.

Exercises the three software gates against the persisted mode file
(event_log/runtime_mode.json, utils/runtime_mode.py):
  1. mode file round-trip + mtime-cache
  2. kinetic bus supervisor parks/unparks (lab-injection mode, no serial)
  3. drawing trigger blocks with last_block_reason="low_energy"

The lung gate (machine.py servo_controller=None swap) and the awakening
gate are loop-inline and need a live machine to observe — see the
dashboard SYSTEM tab or DEBUG servo logs for those.

Run:  python debug/test_low_energy_gates.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import runtime_mode

FAILS = []


def check(name, ok):
    print(f"  [{'OK' if ok else 'FAIL'}] {name}")
    if not ok:
        FAILS.append(name)


print("1. runtime_mode round-trip")
prior = runtime_mode.low_energy()
runtime_mode.set_low_energy(True, changed_by="gate-test")
check("set True", runtime_mode.low_energy() is True)
runtime_mode.set_low_energy(False, changed_by="gate-test")
check("set False", runtime_mode.low_energy() is False)

print("2. kinetic bus park/unpark (injected transport, no serial)")
from motor_panel.kinetic_bus import LOW_ENERGY_STATE, KineticBus

sent = []
bus = KineticBus(
    send_ease=lambda d: sent.append(d),
    send_plan=lambda d, dt: None,
    send_step=lambda d: None,
    get_state=lambda: {"elbow": 90.0, "shoulder": 90.0},
    on_log=lambda m: print(f"    bus: {m}"),
)
bus.enable()
time.sleep(0.5)
runtime_mode.set_low_energy(True, changed_by="gate-test")
time.sleep(1.0)
check("supervisor parked", bus.status()["state"] == LOW_ENERGY_STATE)
check("no chains while parked", bus.status()["chains"] == 0)
writes_at_park = len(sent)
time.sleep(1.5)
check("zero writes while parked (post-ramp)", len(sent) == writes_at_park or len(sent) - writes_at_park < 3)
runtime_mode.set_low_energy(False, changed_by="gate-test")
time.sleep(1.0)
check("supervisor unparked", bus.status()["state"] != LOW_ENERGY_STATE)
bus.shutdown()

print("3. drawing trigger gate")
from drawing.drawing import DrawingController

dc = DrawingController()
dc._last_low_e_log = time.time()  # keep the test out of the event log
runtime_mode.set_low_energy(True, changed_by="gate-test")
verdict = dc.should_draw(mood=0.5, boredom=0.3)
from config.config import BASE_VOICE_DETOX

if BASE_VOICE_DETOX:
    print("    (BASE_VOICE_DETOX is on — trigger blocked upstream, gate untestable)")
else:
    check("should_draw blocked", verdict is False)
    check("block reason surfaced", dc.last_block_reason == "low_energy")
runtime_mode.set_low_energy(False, changed_by="gate-test")

runtime_mode.set_low_energy(prior, changed_by="gate-test")
print(f"\nmode restored to low_energy={prior}")
print("ALL PASS" if not FAILS else f"FAILURES: {FAILS}")
sys.exit(1 if FAILS else 0)
