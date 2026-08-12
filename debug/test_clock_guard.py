"""Standalone proof of the clock guard (Aug 12 2026, the frozen-run fix).

Replays the real failure with injected clocks: boot on an RTC 53 days fast,
NTP steps the clock back mid-run, the watch must call it — and must stay
quiet through normal time and through NTP's gentle slewing.
Run: python debug/test_clock_guard.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.clock_guard import ClockJumpWatch, wait_for_clock_sync

failures = []


def check(name, ok):
    print(f"{'✓' if ok else '✗'} {name}")
    if not ok:
        failures.append(name)


class FakeClocks:
    def __init__(self, wall_start):
        self.wall = wall_start
        self.mono = 1000.0

    def tick(self, s):
        self.wall += s
        self.mono += s


OCT_2 = 1790956403.0  # the real run 980f6e82's skewed boot epoch
STEP_BACK = -51 * 86400.0

fc = FakeClocks(OCT_2)
w = ClockJumpWatch(threshold_s=30, wall=lambda: fc.wall, mono=lambda: fc.mono)

fc.tick(5)
check("normal tick is quiet", w.check() == 0.0)

fc.tick(5)
fc.wall += STEP_BACK  # NTP yanks the clock back 51 days
drift = w.check()
check("51-day backwards step detected", drift < -50 * 86400)
check("total drift accumulated", abs(w.total_drift_s - drift) < 1.0)

fc.tick(5)
check("quiet again after the step", w.check() == 0.0)
check("total drift persists for re-alerts", w.total_drift_s < -50 * 86400)

fc.tick(5)
fc.wall += 45  # a forward step (clock set ahead) must alert too
check("forward step detected", w.check() > 40)

w2 = ClockJumpWatch(threshold_s=30, wall=lambda: fc.wall, mono=lambda: fc.mono)
for _ in range(60):
    fc.tick(5)
    fc.wall += 0.4  # NTP slewing: ~80ms/s correction, never a step
    if w2.check() != 0.0:
        failures.append("slewing false-positive")
        break
check("gradual NTP slew stays quiet", "slewing false-positive" not in failures)

status = wait_for_clock_sync(max_wait_s=0.1)
check(f"live sync check returns instantly ({status!r})", isinstance(status, str))

if failures:
    print(f"\nFAILED: {failures}")
    sys.exit(1)
print("\nAll clock-guard checks passed.")
