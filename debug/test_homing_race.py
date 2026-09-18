"""Regression test for the phantom-homing race. No hardware, no waiting.

Grbl answers the first '?' after $H with the state from BEFORE the cycle
started. ensure_homed used to read that Idle and declare homing complete in
~0.7s, then send G54 into a controller that was actually mid-cycle; G54 timed
out at 5s, the attempt counted as failed, and the retry's soft reset killed the
live homing. Every acquire hit the limit switch twice with one approach cut off
part-way — which read as a finicky switch for weeks.

    python debug/test_homing_race.py

Fakes the serial link, the clock and the logger, and drives ensure_homed
through three controllers:

  1. phantom Idle, then the real cycle  -> completes on the REAL landing
  2. answers Idle forever (no silence)  -> completes on the min-cycle floor
  3. alarm at the switch                -> still fails, and fails fast
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import grbl.grbl_utils as gu

failures = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        failures.append(name)


class FakeClock:
    """Virtual time: sleeps advance it, so a 10s homing cycle costs no seconds."""

    def __init__(self):
        self.now = 1000.0

    def time(self):
        return self.now

    def sleep(self, s):
        self.now += float(s)

    def monotonic(self):
        return self.now


class FakeSerial:
    def __init__(self):
        self.drained = 0

    def reset_input_buffer(self):
        self.drained += 1

    def write(self, *_):
        pass

    def flush(self):
        pass

    def readline(self):
        return b""

    def close(self):
        pass


def run_case(status_for_elapsed, max_retries=1):
    """Drive ensure_homed against a fake controller. Returns (events, sent, ser, clock)."""
    clock = FakeClock()
    ser = FakeSerial()
    events, sent = [], []
    cycle_start = [None]

    def fake_get_status(_ser):
        clock.now += 0.05  # a real '?' round-trip is not free
        if cycle_start[0] is None:
            return "<Alarm,WPos:193.000,193.000,0.000>"
        return status_for_elapsed(clock.now - cycle_start[0])

    def fake_send_cmd(_ser, cmd, wait_ok=True, timeout=None):
        sent.append((round(clock.now - 1000.0, 2), cmd))
        if cmd == "$H":
            cycle_start[0] = clock.now
        return ["ok"]

    orig = (gu.time, gu.get_status, gu.send_cmd, gu.log_json_entry, gu.ensure_pen_up_critical_safety, gu.wait_until_idle)
    gu.time = clock
    gu.get_status = fake_get_status
    gu.send_cmd = fake_send_cmd
    gu.log_json_entry = lambda t, d, **k: events.append(d)
    gu.ensure_pen_up_critical_safety = lambda *a, **k: True
    gu.wait_until_idle = lambda *a, **k: True
    try:
        try:
            gu.ensure_homed(ser, home_timeout=120, max_retries=max_retries)
            raised = None
        except Exception as e:  # homing genuinely failed
            raised = e
    finally:
        gu.time, gu.get_status, gu.send_cmd, gu.log_json_entry, gu.ensure_pen_up_critical_safety, gu.wait_until_idle = orig
    return events, sent, ser, clock, raised


print("1. phantom Idle at t=0, real cycle lands at t=9")


def realistic(el):
    if el < 0.3:
        return "<Idle,WPos:193.000,193.000,0.000>"  # the pre-cycle state
    if el < 9.0:
        return ""  # Grbl is silent while homing
    return "<Home,WPos:0.000,0.000,0.000>"


events, sent, ser, clock, raised = run_case(realistic)
done = [e for e in events if e.get("action") == "homing_complete"]
check("homing completed", bool(done) and raised is None, str(raised or ""))
if done:
    check("waited for the real landing, not the phantom", done[0]["duration"] >= 9.0, f"duration {done[0]['duration']:.1f}s")
    check("landed on Home, not the stale Idle", "Home" in done[0]["final_status"], done[0]["final_status"])
check("no second attempt was needed", not [e for e in events if e.get("action") == "soft_reset"])
check("buffer drained before $H", ser.drained >= 1, f"{ser.drained} drains")
g54 = [t for t, c in sent if c == "G54"]
h = [t for t, c in sent if c == "$H"]
check("G54 sent after the cycle finished", bool(g54) and bool(h) and g54[0] - h[0] >= 9.0, f"$H at {h[0]:.1f}s, G54 at {g54[0]:.1f}s" if g54 and h else "missing")

print("2. a controller that answers Idle throughout (never goes silent)")
events, sent, ser, clock, raised = run_case(lambda el: "<Idle,WPos:0.000,0.000,0.000>")
done = [e for e in events if e.get("action") == "homing_complete"]
check("completes via the min-cycle floor", bool(done) and raised is None)
if done:
    check("not before the floor", done[0]["duration"] >= gu.GRBL_HOMING_MIN_CYCLE_S, f"duration {done[0]['duration']:.2f}s, floor {gu.GRBL_HOMING_MIN_CYCLE_S}s")

print("3. alarm during the cycle still fails")
events, sent, ser, clock, raised = run_case(lambda el: "" if el < 2 else "<Alarm,WPos:-107.000,-107.000,0.000>", max_retries=2)
check("reports the alarm", any(e.get("action") == "homing_alarm" for e in events))
check("gives up after the retries", raised is not None, type(raised).__name__ if raised else "no exception")

print()
print("ALL PASS" if not failures else f"{len(failures)} FAILED: {failures}")
sys.exit(1 if failures else 0)
