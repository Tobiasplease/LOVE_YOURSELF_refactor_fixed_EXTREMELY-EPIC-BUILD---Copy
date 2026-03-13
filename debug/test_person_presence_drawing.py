#!/usr/bin/env python3
"""
Test script to demonstrate person presence influence on drawing motivation.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from drawing.drawing import DrawingController

def test_person_presence_drawing():
    """Test how person presence affects drawing decisions."""
    drawing_controller = DrawingController()

    # Simulate a drawing 20 minutes ago to pass minimum interval
    import time
    drawing_controller.last_drawing_time = time.time() - 1200  # 20 minutes ago

    print("Person Presence Drawing Motivation Test")
    print("=" * 50)

    # Test scenario: moderate mood/novelty/boredom
    test_conditions = {
        "mood": 0.5,
        "novelty": 0.4,
        "boredom": 0.5
    }

    print(f"Test conditions: {test_conditions}")
    print()

    # Test without person present
    print("--- WITHOUT PERSON PRESENT ---")
    result_no_person = drawing_controller.should_draw(
        mood=test_conditions["mood"],
        novelty=test_conditions["novelty"],
        boredom=test_conditions["boredom"],
        person_present=False
    )
    print(f"Decision: {'WOULD DRAW' if result_no_person else 'WOULD NOT DRAW'}")

    print()

    # Test with person present
    print("--- WITH PERSON PRESENT ---")
    result_with_person = drawing_controller.should_draw(
        mood=test_conditions["mood"],
        novelty=test_conditions["novelty"],
        boredom=test_conditions["boredom"],
        person_present=True
    )
    print(f"Decision: {'WOULD DRAW' if result_with_person else 'WOULD NOT DRAW'}")

    print()
    print("=" * 50)

    if result_with_person and not result_no_person:
        print("✅ SUCCESS: Person presence made the difference!")
        print("   System is more likely to draw when people are present.")
    elif result_with_person and result_no_person:
        print("ℹ️  Both scenarios triggered drawing.")
        print("   Person presence provides additional motivation.")
    elif not result_with_person and not result_no_person:
        print("ℹ️  Neither scenario triggered drawing.")
        print("   Conditions may need higher motivation or less time since last drawing.")
    else:
        print("⚠️  Unexpected result: person present blocked drawing.")

    print()
    print("Configuration values:")
    from config.config import DRAWING_PERSON_WEIGHT, DRAWING_PERSON_BONUS, DRAWING_BASE_THRESHOLD
    print(f"  DRAWING_PERSON_WEIGHT: {DRAWING_PERSON_WEIGHT}")
    print(f"  DRAWING_PERSON_BONUS: {DRAWING_PERSON_BONUS}")
    print(f"  DRAWING_BASE_THRESHOLD: {DRAWING_BASE_THRESHOLD}")
    print(f"  Maximum person influence: {DRAWING_PERSON_WEIGHT + DRAWING_PERSON_BONUS}")

if __name__ == "__main__":
    test_person_presence_drawing()