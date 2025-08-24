#!/usr/bin/env python3
"""
Test that the system now responds appropriately to different content types.
Should show varied emotional responses rather than predictable sadness.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
from mood.mood_factory import create_mood_engine

def test_emotional_variety():
    """Test that the system shows appropriate emotional variety"""
    print("=== Testing Emotional Responsiveness ===")
    
    engine = create_mood_engine()
    
    # Build some temporal context first (mild stagnation)
    if hasattr(engine, 'temporal_engine'):
        engine.temporal_engine.memory_bank.stagnation_start = time.time() - 600  # 10 min ago
        engine.temporal_engine.memory_bank.last_discovery = time.time() - 600
    
    # Add some stagnation
    for _ in range(5):
        engine.analyze_mood("The room remains the same", saw_person=False)
    
    # Test different content types
    test_scenarios = [
        {
            "caption": "I notice a person smiling warmly at me",
            "saw_person": True,
            "expected": "positive",
            "description": "Positive social interaction"
        },
        {
            "caption": "The beautiful morning sunlight creates peaceful shadows",
            "saw_person": False, 
            "expected": "positive",
            "description": "Beautiful environmental observation"
        },
        {
            "caption": "I am so tired of this cluttered, messy space",
            "saw_person": False,
            "expected": "negative", 
            "description": "Authentic negative sentiment"
        },
        {
            "caption": "The same book sits on the table as always",
            "saw_person": False,
            "expected": "temporal",
            "description": "Neutral content with temporal influence"
        },
        {
            "caption": "I find contentment in this familiar, comfortable routine",
            "saw_person": False,
            "expected": "positive",
            "description": "Finding joy in repetition"
        }
    ]
    
    results = []
    moods = []
    
    print("Testing emotional responses to different content types...\n")
    
    for i, scenario in enumerate(test_scenarios):
        mood = engine.analyze_mood(scenario["caption"], saw_person=scenario["saw_person"])
        moods.append(mood)
        
        # Evaluate result
        if scenario["expected"] == "positive" and mood > 0.55:
            result = "PASS"
        elif scenario["expected"] == "negative" and mood < 0.45:
            result = "PASS"
        elif scenario["expected"] == "temporal":  # Allow temporal influence for neutral content
            result = "PASS"  # Any response is valid for temporal
        else:
            result = "MIXED"
        
        print(f"Scenario {i+1}: {result} - {scenario['description']}")
        print(f"  '{scenario['caption']}'")
        print(f"  Mood: {mood:.3f} (expected {scenario['expected']})")
        print()
        
        results.append(result)
    
    # Check for emotional variety
    mood_range = max(moods) - min(moods)
    variety_score = len(set(round(m, 1) for m in moods)) / len(moods)
    
    print(f"Emotional Analysis:")
    print(f"  Mood range: {mood_range:.3f} (higher = more variety)")
    print(f"  Variety score: {variety_score:.2f} (higher = more diverse)")
    print(f"  Moods: {[round(m, 3) for m in moods]}")
    
    passes = results.count("PASS")
    print(f"\nResults: {passes}/{len(test_scenarios)} scenarios showed appropriate responses")
    
    # Success if we have good variety AND appropriate responses
    has_variety = mood_range > 0.2 and variety_score > 0.6
    has_responses = passes >= len(test_scenarios) * 0.6
    
    return has_variety and has_responses, has_variety, has_responses

if __name__ == "__main__":
    success, variety, responses = test_emotional_variety()
    
    print(f"\nEmotional Variety: {'GOOD' if variety else 'LIMITED'}")
    print(f"Appropriate Responses: {'GOOD' if responses else 'LIMITED'}")
    
    if success:
        print("\nSUCCESS: System now shows emotional variety and responds to content!")
        print("No longer predictably sad - emotions vary based on what it sees.")
    else:
        print("\nNEEDS WORK: System may still lack emotional variety or content responsiveness.")
        print("Continue refining the blending approach.")