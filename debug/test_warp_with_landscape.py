#!/usr/bin/env python3
"""
Test how the warp transform handles landscape (65x35mm) vs square (50x50mm) input.
Shows corner mapping and aspect ratio distortion for different PRE_ROTATION_DEG values.

Run: python debug/test_warp_with_landscape.py
"""

import os
import sys
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from grbl.warp_transform import map_to_quad, warp_transform_line, find_max_xy_from_lines, PRE_ROTATION_DEG


def test_corners(max_x, max_y, label=""):
    """Map the four corners and center of a rectangle through the warp."""
    print(f"\n{'='*60}")
    print(f"  Input: {max_x}x{max_y}mm ({label})")
    print(f"  PRE_ROTATION_DEG: {PRE_ROTATION_DEG}°")
    print(f"{'='*60}")

    points = [
        (0, 0, "bottom-left"),
        (max_x, 0, "bottom-right"),
        (max_x, max_y, "top-right"),
        (0, max_y, "top-left"),
        (max_x/2, max_y/2, "center"),
    ]

    mapped = []
    for x, y, name in points:
        # Build a fake G-code line and transform it
        gcode = f"G1 X{x:.4f} Y{y:.4f}"
        transformed = warp_transform_line(gcode, max_x, max_y)

        # Parse back
        import re
        tx = float(re.search(r"X([-+]?\d*\.?\d+)", transformed).group(1))
        ty = float(re.search(r"Y([-+]?\d*\.?\d+)", transformed).group(1))
        mapped.append((tx, ty, name))
        print(f"  ({x:6.1f}, {y:5.1f}) {name:14s} → ({tx:6.2f}, {ty:6.2f})")

    # Calculate output dimensions
    xs = [m[0] for m in mapped[:4]]
    ys = [m[1] for m in mapped[:4]]
    out_w = max(xs) - min(xs)
    out_h = max(ys) - min(ys)
    print(f"\n  Output bounding box: {out_w:.1f} x {out_h:.1f} mm")
    print(f"  Output aspect ratio: {out_w/out_h:.2f}:1")
    print(f"  Input aspect ratio:  {max_x/max_y:.2f}:1")

    # Check squareness (how parallelogram-shaped it is)
    # Top edge vector
    top_dx = mapped[2][0] - mapped[3][0]  # top-right - top-left
    top_dy = mapped[2][1] - mapped[3][1]
    # Bottom edge vector
    bot_dx = mapped[1][0] - mapped[0][0]  # bottom-right - bottom-left
    bot_dy = mapped[1][1] - mapped[0][1]
    # Left edge vector
    left_dx = mapped[3][0] - mapped[0][0]  # top-left - bottom-left
    left_dy = mapped[3][1] - mapped[0][1]

    top_angle = math.degrees(math.atan2(top_dy, top_dx))
    bot_angle = math.degrees(math.atan2(bot_dy, bot_dx))
    left_angle = math.degrees(math.atan2(left_dy, left_dx))

    print(f"\n  Top edge angle:    {top_angle:+.1f}°")
    print(f"  Bottom edge angle: {bot_angle:+.1f}°")
    print(f"  Left edge angle:   {left_angle:+.1f}°")
    print(f"  Skew (top-bottom): {abs(top_angle - bot_angle):.1f}°")

    return mapped


def test_with_rotation(max_x, max_y, rotation_deg, label=""):
    """Temporarily override rotation and test."""
    import grbl.warp_transform as wt
    original = wt.PRE_ROTATION_DEG
    wt.PRE_ROTATION_DEG = rotation_deg
    result = test_corners(max_x, max_y, label)
    wt.PRE_ROTATION_DEG = original
    return result


if __name__ == "__main__":
    print("Warp Transform Landscape Test")
    print(f"Current PRE_ROTATION_DEG = {PRE_ROTATION_DEG}°\n")

    # Test current settings
    test_corners(65, 35, "CURRENT: 65x35mm @ 20° rotation")

    # Test with reduced rotation for landscape
    for rot in [15, 12, 10]:
        test_with_rotation(65, 35, rot, f"65x35mm @ {rot}° rotation")

    # Test smaller targets that might fit the quad better
    for target in [(50, 28), (45, 25)]:
        test_with_rotation(target[0], target[1], 15, f"{target[0]}x{target[1]}mm @ 15° rotation")

    print("\n" + "="*60)
    print("GOAL: Output should fit within quad bounds:")
    print("  X: 1 to 70 mm")
    print("  Y: 2 to 40 mm")
    print("  No negative coordinates")
    print("="*60)
