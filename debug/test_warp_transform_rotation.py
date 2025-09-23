#!/usr/bin/env python3
"""
Test script to understand the warp transform rotation issue
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "grbl"))

from warp_transform import map_to_quad


def test_corner_mapping():
    """Test how the four corners of a 40x40 square map"""
    print("Testing corner mapping for 40x40 square:")
    print("Input -> Output")

    # Test corners of the input square
    corners = [(0, 0, "bottom-left"), (40, 0, "bottom-right"), (40, 40, "top-right"), (0, 40, "top-left")]

    for x, y, label in corners:
        tx, ty = map_to_quad(x, y, 40, 40)
        print(f"({x:2}, {y:2}) {label:12} -> ({tx:6.2f}, {ty:6.2f})")


def test_simple_shape():
    """Test a simple rectangle to see rotation"""
    print("\nTesting simple 10x10 rectangle at center:")

    # Rectangle from (15,15) to (25,25)
    rect_points = [(15, 15), (25, 15), (25, 25), (15, 25), (15, 15)]  # closed rectangle

    print("Original points:")
    for i, (x, y) in enumerate(rect_points):
        print(f"  Point {i}: ({x:2}, {y:2})")

    print("Transformed points:")
    for i, (x, y) in enumerate(rect_points):
        tx, ty = map_to_quad(x, y, 40, 40)
        print(f"  Point {i}: ({tx:6.2f}, {ty:6.2f})")


if __name__ == "__main__":
    test_corner_mapping()
    test_simple_shape()
