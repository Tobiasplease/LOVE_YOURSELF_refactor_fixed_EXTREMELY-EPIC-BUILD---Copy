#!/usr/bin/env python3
"""
Test the new state-motivated drawing system
"""

import time
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from drawing.drawing import DrawingController

def test_drawing_scenarios():
    """Test various scenarios with the new state-motivated drawing system."""

    print("🎨 Testing State-Motivated Drawing System")
    print("=" * 50)

    # Create drawing controller
    drawing_controller = DrawingController()

    # Reset last drawing time to allow testing
    drawing_controller.last_drawing_time = time.time() - 700  # 11+ minutes ago

    test_scenarios = [
        # (mood, novelty, boredom, description)
        (-0.5, 0.1, 0.2, "Low mood, low novelty, low boredom - should NOT draw"),
        (0.8, 0.9, 0.1, "High mood, high novelty, low boredom - should DRAW"),
        (0.2, 0.3, 0.8, "Medium mood, medium novelty, high boredom - should DRAW"),
        (-0.2, 0.2, 0.3, "Slightly negative mood, low novelty/boredom - probably NOT draw"),
        (0.1, 0.7, 0.6, "Neutral mood, high novelty and boredom - should DRAW"),
    ]

    for i, (mood, novelty, boredom, description) in enumerate(test_scenarios, 1):
        print(f"\n--- Scenario {i}: {description} ---")

        # Test the decision
        will_draw = drawing_controller.should_draw(
            mood=mood,
            novelty=novelty,
            boredom=boredom
        )

        print(f"Result: {'🎨 WILL DRAW' if will_draw else '⏳ WILL WAIT'}")

        # Simulate some time passing between tests
        time.sleep(0.1)

    print("\n" + "=" * 50)
    print("Test completed!")
    print("\nExpected behavior:")
    print("- Scenarios with high motivation scores should trigger drawing")
    print("- Minimum 10-minute interval is enforced")
    print("- Maximum 30-minute interval forces drawing")
    print("- Randomness adds unpredictability")

if __name__ == "__main__":
    test_drawing_scenarios()