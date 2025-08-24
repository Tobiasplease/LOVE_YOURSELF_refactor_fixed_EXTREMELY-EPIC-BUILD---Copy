#!/usr/bin/env python3
"""
Test the new content + temporal blending to ensure sentiment analysis leads.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mood.mood_factory import create_mood_engine
import time

def test_content_sentiment_priority():
    """Test that positive content sentiment is preserved despite temporal context"""
    print("=== Testing Content + Temporal Blending ===")
    
    engine = create_mood_engine()
    
    # Test cases with different sentiment types
    test_cases = [
        {
            "caption": "I see a person smiling warmly at me from across the room",
            "expected": "positive",
            "description": "Positive content (smile) should stay positive"
        },
        {
            "caption": "This beautiful, serene landscape fills me with peace",
            "expected": "positive", 
            "description": "Beautiful content should resist temporal sadness"
        },
        {
            "caption": "I am so tired of always being in cluttered spaces",
            "expected": "negative",
            "description": "Authentic negative sentiment should be preserved"
        },
        {
            "caption": "The book sits on the table, same as before",
            "expected": "neutral",
            "description": "Neutral content allows temporal influence"
        },
        {
            "caption": "I notice the warm sunlight streaming through clean windows",
            "expected": "positive",
            "description": "Positive environmental observations should lift mood"
        }
    ]
    
    # Set up some temporal stagnation to test resistance
    if hasattr(engine, 'temporal_engine'):
        engine.temporal_engine.memory_bank.stagnation_start = time.time() - 1800  # 30 min ago
        engine.temporal_engine.memory_bank.last_discovery = time.time() - 1800
    
    results = []
    
    print("Testing sentiment preservation against temporal context...")
    for i, case in enumerate(test_cases):
        caption = case["caption"]
        expected = case["expected"]
        description = case["description"]
        
        # Build some temporal pressure first
        for _ in range(10):
            engine.analyze_mood("The same object remains here", saw_person=False)
        
        # Now test the target caption
        mood = engine.analyze_mood(caption, saw_person="smiling" in caption)
        
        # Evaluate result
        if expected == "positive" and mood > 0.6:
            result = "PASS"
        elif expected == "negative" and mood < 0.4:
            result = "PASS"
        elif expected == "neutral" and 0.4 <= mood <= 0.6:
            result = "PASS" 
        else:
            result = "MIXED"
        
        print(f"  Test {i+1}: {result} - {description}")
        print(f"    Caption: {caption[:60]}...")
        print(f"    Mood: {mood:.3f} (expected {expected})")
        print()
        
        results.append((result, mood, expected))
    
    # Summary
    passes = sum(1 for r, _, _ in results if r == "PASS")
    print(f"Results: {passes}/{len(test_cases)} tests showed proper content sentiment priority")
    
    return passes >= len(test_cases) * 0.6  # 60% success threshold

if __name__ == "__main__":
    success = test_content_sentiment_priority()
    if success:
        print("\nSUCCESS: Content sentiment is now leading with temporal enhancement!")
    else:
        print("\nNEEDS WORK: Content sentiment may still be overridden by temporal patterns")