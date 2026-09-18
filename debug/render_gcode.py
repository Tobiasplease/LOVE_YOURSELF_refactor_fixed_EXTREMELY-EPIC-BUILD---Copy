"""Render servo G-code the way the paper sees it: only pen-down travel draws.

The point is to separate COMMAND space from PAPER space (Aug 12 ruling: judge
fidelity on the sheet, not in the file). If this render already looks like the
photographed sheet, the loss happened in vectorisation; if this render is clean
and the sheet is broken, the loss is the pen, the surface or the servo.

    python debug/render_gcode.py <file.gcode> [-o out.png] [--px-per-mm 8]

Prints pen-down/up travel, segment count and the bounding box of what is
actually drawn.
"""

import os
import re
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

args = [a for a in sys.argv[1:] if not a.startswith("-")]
if not args:
    print(__doc__)
    sys.exit(1)
path = args[0]
out = sys.argv[sys.argv.index("-o") + 1] if "-o" in sys.argv else "/tmp/claude-1000/gcode_render.png"
ppm = float(sys.argv[sys.argv.index("--px-per-mm") + 1]) if "--px-per-mm" in sys.argv else 8.0

try:
    from config.config import GRBL_PEN_DOWN_S, GRBL_PEN_UP_S
except Exception:
    GRBL_PEN_UP_S, GRBL_PEN_DOWN_S = 34, 56
DOWN_THRESHOLD = (GRBL_PEN_UP_S + GRBL_PEN_DOWN_S) / 2.0

word = re.compile(r"([GMXYSF])(-?\d+\.?\d*)")
x = y = 0.0
pen_s = 0.0
segments = []  # (x0, y0, x1, y1, down)
for raw in open(path):
    line = raw.split(";")[0].strip()
    if not line:
        continue
    vals = dict((m.group(1), float(m.group(2))) for m in word.finditer(line.upper()))
    if "M" in vals and "S" in vals and int(vals["M"]) == 3:
        pen_s = vals["S"]
    if "G" in vals and int(vals["G"]) in (0, 1) and ("X" in vals or "Y" in vals):
        nx, ny = vals.get("X", x), vals.get("Y", y)
        segments.append((x, y, nx, ny, pen_s >= DOWN_THRESHOLD))
        x, y = nx, ny

down = [s for s in segments if s[4]]
if not down:
    print("nothing is drawn — no pen-down motion in this file")
    sys.exit(1)
dist = lambda s: ((s[2] - s[0]) ** 2 + (s[3] - s[1]) ** 2) ** 0.5
xs = [v for s in down for v in (s[0], s[2])]
ys = [v for s in down for v in (s[1], s[3])]
x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
pad = 3.0
w = int((x1 - x0 + 2 * pad) * ppm)
h = int((y1 - y0 + 2 * pad) * ppm)
img = np.full((h, w, 3), 255, np.uint8)
to_px = lambda px, py: (int((px - x0 + pad) * ppm), int(h - (py - y0 + pad) * ppm))  # y up in g-code
for s in segments:
    if s[4]:
        cv2.line(img, to_px(s[0], s[1]), to_px(s[2], s[3]), (0, 0, 0), 1, cv2.LINE_AA)
os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
cv2.imwrite(out, img)

print(f"file           : {os.path.basename(path)}")
print(f"pen threshold  : S>={DOWN_THRESHOLD:.0f} counts as down (up {GRBL_PEN_UP_S}, down {GRBL_PEN_DOWN_S})")
print(f"segments       : {len(segments)} total, {len(down)} drawn")
print(f"pen-down travel: {sum(dist(s) for s in down):.0f} mm")
print(f"pen-up travel  : {sum(dist(s) for s in segments if not s[4]):.0f} mm")
print(f"drawn extent   : X {x0:.1f}..{x1:.1f} mm  Y {y0:.1f}..{y1:.1f} mm  ({x1 - x0:.0f} x {y1 - y0:.0f} mm)")
print(f"lifts          : {sum(1 for i in range(1, len(segments)) if segments[i][4] != segments[i - 1][4])}")
print(f"-> {out}")
