#!/usr/bin/env python3

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import DEBUG_FAST_DRAWING, DRAWING_COOLDOWN, DRAWING_INTERVAL
from drawing.drawing import DrawingController


def test_debug_drawing_system():
    """Test the debug drawing system with detailed logging."""

    print("Debug Drawing System Test")
    print("=" * 50)
    print(f"DEBUG_FAST_DRAWING: {DEBUG_FAST_DRAWING}")
    print(f"DRAWING_INTERVAL: {DRAWING_INTERVAL} seconds")
    print(f"DRAWING_COOLDOWN: {DRAWING_COOLDOWN} seconds")
    print()

    # Create drawing controller
    drawing_controller = DrawingController()

    # Test scenarios with different values
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
        print(f"Result: {'WOULD DRAW' if result else 'WOULD NOT DRAW'}")

    print(f"\n{'='*50}")
    print("INSTRUCTIONS TO ENABLE DEBUG MODE:")
    print("1. Edit config/config.py")
    print("2. Set: DEBUG_FAST_DRAWING = True")
    print("3. Restart machine.py")
    print("4. Drawing will check every 60 seconds with 30-second cooldown")
    print("5. Watch for [🎨] messages in console output")


if __name__ == "__main__":
    test_debug_drawing_system()
