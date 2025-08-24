#!/usr/bin/env python3
"""
Test drawing trigger system to verify it's working properly.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from drawing.drawing import DrawingController

def test_drawing_triggers():
    """Test the drawing trigger logic with various scenarios."""
    print("Testing Drawing Trigger System")
    print("=" * 50)
    
    controller = DrawingController()
    
    # Test scenarios (mood, novelty, boredom, expected_result)
    test_scenarios = [
        # High novelty scenarios
        (0.5, 0.5, 0.3, True, "Medium novelty should trigger"),
        (0.6, 0.45, 0.2, True, "High novelty (>0.4) should trigger"),
        (0.7, 0.7, 0.1, True, "Very high novelty should trigger"),
        
        # High boredom scenarios  
        (0.6, 0.3, 0.6, True, "High boredom (>0.5) should trigger"),
        (0.5, 0.2, 0.8, True, "Very high boredom should trigger"),
        
        # Low mood scenarios
        (0.35, 0.3, 0.4, True, "Low mood (<0.4) should trigger"),
        (0.2, 0.2, 0.3, True, "Very low mood should trigger"),
        
        # No trigger scenarios
        (0.6, 0.3, 0.4, False, "Medium values should not trigger"),
        (0.8, 0.2, 0.2, False, "High mood, low novelty/boredom should not trigger"),
        (0.5, 0.35, 0.45, False, "All values below thresholds should not trigger"),
    ]
    
    for i, (mood, novelty, boredom, expected, description) in enumerate(test_scenarios):
        print(f"\nTest {i+1}: {description}")
        print(f"  Values: mood={mood}, novelty={novelty}, boredom={boredom}")
        
        # Test the should_draw method
        should_draw = controller.should_draw(
            mood=mood,
            novelty=novelty, 
            boredom=boredom,
            reflection=None
        )
        
        result = "[PASS]" if should_draw == expected else "[FAIL]"
        action = "DRAW" if should_draw else "SKIP"
        print(f"  Result: {action} ({result})")
        
        if should_draw != expected:
            print(f"  Expected: {'DRAW' if expected else 'SKIP'}, Got: {action}")
    
    # Test reflection triggers
    print(f"\n{'='*50}")
    print("Testing Reflection Triggers:")
    
    reflection_tests = [
        ("I feel stuck in this routine", True, "Should trigger on 'feel stuck'"),
        ("I need to express something", True, "Should trigger on 'need to express'"),
        ("Nothing is changing here", True, "Should trigger on 'nothing is changing'"),
        ("I want to create something", True, "Should trigger on 'want to'"),
        ("My desire is to make art", True, "Should trigger on 'desire'"),
        ("Just observing the room", False, "Should not trigger on neutral reflection"),
        ("", False, "Should not trigger on empty reflection"),
    ]
    
    for reflection, expected, description in reflection_tests:
        print(f"\nReflection test: {description}")
        print(f"  Reflection: '{reflection}'")
        
        should_draw = controller.should_draw(
            mood=0.5,  # Neutral values
            novelty=0.3,
            boredom=0.4,
            reflection=reflection
        )
        
        result = "[PASS]" if should_draw == expected else "[FAIL]"
        action = "DRAW" if should_draw else "SKIP"
        print(f"  Result: {action} ({result})")

if __name__ == "__main__":
    test_drawing_triggers()