"""Standalone proof of the drawing drive (docs/drawing-drive-plan.md).

Simulated hours with injected monotonic clocks: a flat wantless machine
takes days; arousal alone reaches the pen; want + arousal refills within
the hour; completion discharges fully. Run: python debug/test_drawing_drive.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import drawing.drive as drive_mod
from drawing.drive import DrawingDrive

drive_mod._STATE_PATH = "/tmp/test_drawing_drive_state.json"
failures = []


def check(name, ok, detail=""):
    print(f"{'✓' if ok else '✗'} {name} {detail}")
    if not ok:
        failures.append(name)


class Mono:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def fresh(arousal):
    m = Mono()
    d = DrawingDrive(mono=m)
    d.level = 0.0
    d.get_arousal = lambda: arousal
    return d, m


def hours_to_full(d, m, want, cap_h=200):
    h = 0.0
    while not d.full() and h < cap_h:
        m.t += 300
        h += 300 / 3600
        d.tick(want_active=want)
    return h


d, m = fresh(0.0)
h = hours_to_full(d, m, want=False)
check("flat + wantless takes days", h > 24, f"({h:.0f}h)")

d, m = fresh(0.9)
h = hours_to_full(d, m, want=False)
check("high arousal alone reaches the pen in ~2h", 1.4 < h < 3.0, f"({h:.1f}h)")

d, m = fresh(0.4)
h = hours_to_full(d, m, want=True)
check("want + moderate arousal ≈ under an hour", 0.6 < h < 1.2, f"({h:.1f}h)")

d.spend()
check("completion discharges fully", d.level == 0.0)
m.t += 300
d.tick(want_active=True)
check("charging resumes after spend", 0.0 < d.level < 0.2, f"({d.level:.3f})")

d2 = DrawingDrive(mono=Mono())
check("boot seeds near-full (testing era)", d2.level >= 0.9, f"({d2.level:.2f})")

d3, m3 = fresh(0.0)
d3.level = 0.5
before = d3.level
m3.t += 0  # no time passes
d3.tick(want_active=True)
check("zero elapsed time charges nothing", d3.level == before)

print()
if failures:
    print(f"FAILED: {failures}")
    sys.exit(1)
print("All drawing-drive checks passed.")
