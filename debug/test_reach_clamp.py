"""Verify the shared reach clamp that now bounds ALL panel motion.

Every panel target (drag, jog, playback, generation) runs through
grbl.warp_calibration.clamp_to_reach — the same projection + hysteresis
shell the drawing pipeline uses. Checks:

  1. points well inside the envelope pass through untouched
  2. anything outside (including absurd ±200 discovery-era coords) lands
     inside the walked boundary
  3. hysteresis: near-boundary points project to the deeper margin inset
     (no raw/clamped zigzag)
  4. convexity in practice: midpoints of segments between clamped points
     are still inside (straight moves can't leave the envelope)

    python debug/test_reach_clamp.py
"""

import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from grbl.warp_calibration import MEASURED_BOUNDARY, _point_in_polygon, clamp_to_reach, reach_polygon


def main():
    failures = []
    boundary = reach_polygon()
    cx = sum(p[0] for p in boundary) / len(boundary)
    cy = sum(p[1] for p in boundary) / len(boundary)

    # 1. deep-inside points untouched
    for f in (0.0, 0.2, 0.4):
        for px, py in boundary:
            x, y = cx + (px - cx) * f, cy + (py - cy) * f
            if clamp_to_reach(x, y) != (x, y):
                failures.append(f"inside point ({x:.1f},{y:.1f}) was moved")

    # 2. wild outside points land inside the walked boundary
    rng = random.Random(7)
    outside = [(-200, -200), (200, 200), (-200, 200), (200, -200), (-102.9, 76.7)]
    outside += [(rng.uniform(-250, 250), rng.uniform(-250, 250)) for _ in range(500)]
    landed = 0
    for x, y in outside:
        qx, qy = clamp_to_reach(x, y)
        if not _point_in_polygon(qx, qy, boundary):
            failures.append(f"({x:.0f},{y:.0f}) clamped to ({qx:.1f},{qy:.1f}) — OUTSIDE the walked boundary")
        else:
            landed += 1
    print(f"outside points: {landed}/{len(outside)} projected inside the envelope")

    # 3. hysteresis: points ON the walked boundary are outside the shell,
    # so they must be pulled inward to the margin inset (never left raw)
    for bx, by in boundary:
        qx, qy = clamp_to_reach(bx, by)
        if (qx, qy) == (bx, by) or not _point_in_polygon(qx, qy, boundary):
            failures.append(f"boundary vertex ({bx},{by}) not pulled to the margin inset (got {qx:.2f},{qy:.2f})")
    print(f"hysteresis: all {len(boundary)} boundary vertices pulled inward")

    # 4. segment midpoints between clamped points stay inside (convexity)
    pts = [clamp_to_reach(rng.uniform(-250, 250), rng.uniform(-250, 250)) for _ in range(200)]
    for (ax, ay), (bx2, by2) in zip(pts, pts[1:]):
        mx, my = (ax + bx2) / 2, (ay + by2) / 2
        if not _point_in_polygon(mx, my, boundary):
            failures.append(f"segment midpoint ({mx:.1f},{my:.1f}) escaped the envelope")
    print(f"convexity: {len(pts) - 1} clamped-segment midpoints all inside")

    print("\n" + ("ALL OK" if not failures else "FAILURES:\n  " + "\n  ".join(failures[:10])))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
