#!/usr/bin/env python3

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Enable debug mode temporarily for testing
import config.config as config

config.DEBUG_FAST_DRAWING = True
config.DRAWING_INTERVAL = 60 if config.DEBUG_FAST_DRAWING else 600
config.DRAWING_COOLDOWN = 30 if config.DEBUG_FAST_DRAWING else 600

from drawing.drawing import DrawingController


def test_with_debug_enabled():
    """Test drawing system with debug mode temporarily enabled."""

    print("Debug Drawing System Test (DEBUG MODE ENABLED)")
    print("=" * 60)
    print(f"DEBUG_FAST_DRAWING: {config.DEBUG_FAST_DRAWING}")
    print(f"DRAWING_INTERVAL: {config.DRAWING_INTERVAL} seconds")
    print(f"DRAWING_COOLDOWN: {config.DRAWING_COOLDOWN} seconds")
    print()

    # Create drawing controller (will use debug intervals)
    drawing_controller = DrawingController()

    # Reset the last drawing time to simulate ready state
    drawing_controller.last_drawing_time = 0

    # Test scenarios
    test_scenarios = [
        {"mood": 0.2, "novelty": 0.3, "boredom": 0.1, "name": "Low mood trigger"},
        {"mood": 0.6, "novelty": 0.7, "boredom": 0.1, "name": "High novelty trigger"},
        {"mood": 0.5, "novelty": 0.3, "boredom": 0.8, "name": "High boredom trigger"},
        {"mood": 0.6, "novelty": 0.3, "boredom": 0.3, "name": "No triggers"},
        {"mood": 0.1, "novelty": 0.8, "boredom": 0.9, "name": "Multiple triggers"},
    ]

    for i, scenario in enumerate(test_scenarios):
        print(f"\n--- Scenario {i+1}: {scenario['name']} ---")
        result = drawing_controller.should_draw(mood=scenario["mood"], novelty=scenario["novelty"], boredom=scenario["boredom"])
        print(f"Result: {'WOULD DRAW ✨' if result else 'WOULD NOT DRAW'}")

    print(f"\n{'='*60}")
    print("✅ DEBUG MODE RESULTS:")
    print("- Scenarios 1,2,3,5 should trigger drawing")
    print("- Scenario 4 should not trigger (no conditions met)")
    print("- All show detailed trigger reasons")


if __name__ == "__main__":
    test_with_debug_enabled()
