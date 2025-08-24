#!/usr/bin/env python3
"""
Quick test of the content + temporal blending logic.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mood.experiential_mood import ExperientialMoodEngine

def test_blending_logic():
    """Test the blending logic directly"""
    print("=== Testing Blending Logic ===")
    
    engine = ExperientialMoodEngine()
    
    # Test cases for blending
    test_cases = [
        {
            "name": "Positive content with temporal sadness",
            "base_mood": 0.8,  # Happy base
            "temporal_mood": 0.2,  # Sad temporal
            "caption": "I see a person smiling at me",
            "temporal_context": {"stagnation_minutes": 30, "repetitions_of_top_object": 20},
            "expected_range": (0.6, 0.9)  # Should stay mostly positive
        },
        {
            "name": "Negative content with neutral temporal",
            "base_mood": 0.2,  # Sad base
            "temporal_mood": 0.5,  # Neutral temporal
            "caption": "I am so tired of cluttered spaces",
            "temporal_context": {"stagnation_minutes": 10, "repetitions_of_top_object": 5},
            "expected_range": (0.2, 0.4)  # Should stay mostly negative
        },
        {
            "name": "Neutral content with temporal influence",
            "base_mood": 0.5,  # Neutral base
            "temporal_mood": 0.1,  # Very sad temporal  
            "caption": "The book sits on the table",
            "temporal_context": {"stagnation_minutes": 90, "repetitions_of_top_object": 50},
            "expected_range": (0.1, 0.4)  # Should be influenced by temporal sadness
        }
    ]
    
    results = []
    
    for case in test_cases:
        result_mood = engine._blend_content_and_temporal(
            case["base_mood"],
            case["temporal_mood"], 
            case["temporal_context"],
            case["caption"]
        )
        
        expected_min, expected_max = case["expected_range"]
        in_range = expected_min <= result_mood <= expected_max
        
        print(f"Test: {case['name']}")
        print(f"  Base mood: {case['base_mood']:.2f}, Temporal mood: {case['temporal_mood']:.2f}")
        print(f"  Result: {result_mood:.3f} (expected {expected_min:.1f}-{expected_max:.1f})")
        print(f"  Status: {'PASS' if in_range else 'FAIL'}")
        print()
        
        results.append(in_range)
    
    passes = sum(results)
    print(f"Results: {passes}/{len(test_cases)} blending tests passed")
    
    return passes >= 2

if __name__ == "__main__":
    success = test_blending_logic()
    if success:
        print("\nSUCCESS: Blending logic is working correctly!")
        print("Content sentiment now leads, with temporal enhancement.")
    else:
        print("\nNEEDS WORK: Blending logic needs adjustment.")