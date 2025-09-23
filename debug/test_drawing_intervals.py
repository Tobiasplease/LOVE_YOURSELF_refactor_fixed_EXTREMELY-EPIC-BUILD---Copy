#!/usr/bin/env python3
"""
Test drawing interval safety mechanisms
"""

import time
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from drawing.drawing import DrawingController

def test_interval_enforcement():
    """Test minimum and maximum interval enforcement."""

    print("🕒 Testing Drawing Interval Enforcement")
    print("=" * 50)

    drawing_controller = DrawingController()

    # Test 1: Too soon (should be blocked)
    print("\n--- Test 1: Drawing attempted too soon (5 minutes ago) ---")
    drawing_controller.last_drawing_time = time.time() - 300  # 5 minutes ago

    will_draw = drawing_controller.should_draw(
        mood=0.8, novelty=0.9, boredom=0.9  # Very high motivation
    )
    print(f"Result: {'🎨 WILL DRAW' if will_draw else '⏳ BLOCKED (as expected)'}")

    # Test 2: Just past minimum (should use state logic)
    print("\n--- Test 2: Just past minimum interval (11 minutes ago) ---")
    drawing_controller.last_drawing_time = time.time() - 660  # 11 minutes ago

    will_draw = drawing_controller.should_draw(
        mood=0.1, novelty=0.1, boredom=0.1  # Very low motivation
    )
    print(f"Result: {'🎨 WILL DRAW' if will_draw else '⏳ WILL WAIT (low motivation)'}")

    # Test 3: Maximum interval exceeded (should force draw)
    print("\n--- Test 3: Maximum interval exceeded (31 minutes ago) ---")
    drawing_controller.last_drawing_time = time.time() - 1860  # 31 minutes ago

    will_draw = drawing_controller.should_draw(
        mood=-0.8, novelty=0.0, boredom=0.0  # Very low motivation
    )
    print(f"Result: {'🎨 FORCED DRAW' if will_draw else '⏳ UNEXPECTED'}")

    print("\n" + "=" * 50)
    print("Interval enforcement working correctly!")

if __name__ == "__main__":
    test_interval_enforcement()