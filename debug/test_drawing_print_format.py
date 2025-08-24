#!/usr/bin/env python3
"""
Test the new drawing decision print format.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from drawing.drawing import DrawingController

def test_drawing_print_format():
    """Test the new print format for drawing decisions."""
    controller = DrawingController()
    
    print("Testing Drawing Decision Print Format")
    print("=" * 50)
    
    # Test scenario 1: Should draw (high novelty)
    print("\nScenario 1: High novelty")
    controller.should_draw(
        mood=0.6,
        novelty=0.8,
        boredom=0.3,
        reflection="I notice something unusual in the light patterns"
    )
    
    # Test scenario 2: Should not draw (if thresholds were normal)
    print("\nScenario 2: Low values")
    controller.should_draw(
        mood=0.9,
        novelty=0.02,
        boredom=0.1,
        reflection=None
    )
    
    # Test scenario 3: Cooldown (simulate recent drawing)
    print("\nScenario 3: Cooldown active")
    controller.last_drawing_time = controller.last_drawing_time or 0
    import time
    controller.last_drawing_time = time.time() - 10  # Drew 10 seconds ago
    controller.should_draw(
        mood=0.5,
        novelty=0.7,
        boredom=0.8,
        reflection="I want to express this feeling"
    )

if __name__ == "__main__":
    test_drawing_print_format()