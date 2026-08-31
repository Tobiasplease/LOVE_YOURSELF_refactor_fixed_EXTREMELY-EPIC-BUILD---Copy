"""Proof-of-method for the measured warp calibration.

Simulates the real failure mechanism (per WARP_TRANSFORM_README: GRBL 'X/Y'
drive JOINT ANGLES of a 2-link ~300mm arm through linear gearing), then
compares drawing a square through:

  A. the legacy quad transform (map_to_quad + rotation/stretch/nudge)
  B. a 5x5 measured calibration fitted with the inverse TPS

Squareness metrics: edge bow (max deviation of each drawn edge from the
straight chord between its corners) and side-length spread.

    python debug/test_warp_calibration.py
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from grbl import warp_calibration as wc
from grbl import warp_transform as legacy

L1, L2 = 295.0, 320.0


def machine(cmd_x, cmd_y):
    """Ground truth 'physical arm': command coords -> ink position (mm).
    Linear command->angle gearing, then 2-link forward kinematics — the
    exact mechanism the README describes. Constants chosen so the 70x40
    command box lands on a ~120x90mm paper patch."""
    t1 = 1.15 + 0.0075 * cmd_x
    t2 = -1.85 + 0.016 * cmd_y
    x = L1 * math.cos(t1) + L2 * math.cos(t1 + t2)
    y = L1 * math.sin(t1) + L2 * math.sin(t1 + t2)
    return x, y


def draw_square(transform_line, n_per_edge=20, ideal=40.0):
    """Ideal 30x30 square -> transform -> 'machine' -> ink points per edge."""
    corners = [(5, 5), (35, 5), (35, 35), (5, 35)]
    edges_ink = []
    for a, b in zip(corners, corners[1:] + corners[:1]):
        pts = []
        for i in range(n_per_edge + 1):
            f = i / n_per_edge
            x = a[0] + (b[0] - a[0]) * f
            y = a[1] + (b[1] - a[1]) * f
            line = transform_line(f"G1 X{x:.3f} Y{y:.3f} F1000", ideal, ideal)
            import re

            cx = float(re.search(r"X([-+]?\d*\.?\d+)", line).group(1))
            cy = float(re.search(r"Y([-+]?\d*\.?\d+)", line).group(1))
            pts.append(machine(cx, cy))
        edges_ink.append(np.array(pts))
    return edges_ink


def squareness(edges):
    bows, lengths = [], []
    for e in edges:
        chord = e[-1] - e[0]
        clen = np.linalg.norm(chord)
        lengths.append(clen)
        if clen < 1e-6:
            bows.append(float("inf"))
            continue
        n = np.array([-chord[1], chord[0]]) / clen
        bows.append(float(np.abs((e - e[0]) @ n).max()))
    lengths = np.array(lengths)
    spread = float((lengths.max() - lengths.min()) / lengths.mean() * 100)
    return max(bows), spread, lengths.mean()


def main():
    assert not os.path.exists(wc.CALIBRATION_PATH), "remove real calibration before running this test"

    # A. legacy pipeline
    edges = draw_square(legacy.warp_transform_line)
    bow, spread, size = squareness(edges)
    print(f"LEGACY quad+bandaids : edge bow {bow:6.2f}mm   side spread {spread:5.1f}%   mean side {size:.1f}mm")

    # B. measured calibration: dot the 5x5 grid through the 'machine', fit inverse
    cmd_pts = wc.grid_points(5)
    paper_pts = [machine(x, y) for x, y in cmd_pts]
    cal = wc.WarpCalibration.fit(cmd_pts, paper_pts)
    rms, mx = cal.residuals_mm()
    edges = draw_square(cal.apply_to_line)
    bow2, spread2, size2 = squareness(edges)
    print(f"MEASURED 5x5 TPS     : edge bow {bow2:6.2f}mm   side spread {spread2:5.1f}%   mean side {size2:.1f}mm   (fit residual rms {rms:.2f}mm)")

    print(f"\nedge bow improvement: {bow / max(0.01, bow2):.0f}x")
    ok = bow2 < 0.5 and spread2 < 2.0
    print("VERDICT:", "measured calibration draws a true square" if ok else "NOT GOOD ENOUGH — investigate")


if __name__ == "__main__":
    main()
