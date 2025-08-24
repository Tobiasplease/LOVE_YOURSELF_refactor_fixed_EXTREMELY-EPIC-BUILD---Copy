#!/usr/bin/env python3
"""
Quick test to check if the emotional memory system is actually being used in live captions.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from captioner.memory import MemoryMixin
from mood.experiential_mood import ExperientialMoodEngine
import time


def test_memory_integration():
    """Test if emotional memory is working with live caption system"""
    print("Testing Emotional Memory Integration")
    print("=" * 50)
    
    # Create memory system
    memory = MemoryMixin()
    
    # Create mood engine
    mood_engine = ExperientialMoodEngine()
    
    # Simulate some positive observations about "person"
    print("\n1. Storing positive memories about person...")
    for i in range(3):
        time.sleep(0.1)
        caption = f"The person I know well is here, feeling connected and peaceful #{i}"
        
        # Store in memory with positive mood
        memory.observe(
            text=caption,
            mood=(0.7, 0.2, 0.8),  # positive mood
            emotion_state="calm_observant",
            mood_vector=(0.7, 0.2, 0.8)
        )
        
        print(f"   Stored: {caption[:40]}...")
    
    # Check if emotional memory bank has positive associations
    print("\n2. Checking emotional associations...")
    if hasattr(memory, 'emotional_memory_bank'):
        person_motif = memory.emotional_memory_bank.motif_emotions.get("person")
        if person_motif:
            print(f"   Person valence: {person_motif.cumulative_valence:.2f}")
            print(f"   Person comfort: {person_motif.comfort_level:.2f}")
        else:
            print("   No person motif found in emotional memory")
            
        # Test memory influence calculation
        influence = memory.emotional_memory_bank.calculate_memory_mood_influence(
            ["person"], "calm_observant"
        )
        print(f"   Memory influence: {influence}")
    else:
        print("   No emotional memory bank found!")
    
    # Test mood analysis with memory context
    print("\n3. Testing mood analysis with memory...")
    test_caption = "The person I know well is still here with me"
    
    mood_result = mood_engine.analyze_mood(
        caption=test_caption,
        saw_person=True,
        memory_context=memory
    )
    
    print(f"   Test caption: {test_caption}")
    print(f"   Mood result: {mood_result:.2f}")
    
    # Expected: Should be positive due to emotional memory of "person"
    if mood_result > 0.6:
        print("   ✓ GOOD: Mood is positive (memory may be helping)")
    elif mood_result > 0.4:
        print("   ? NEUTRAL: Mood is neutral (memory effect unclear)")
    else:
        print("   ✗ NEGATIVE: Mood is negative (memory not helping)")
    
    print("\n" + "=" * 50)


if __name__ == "__main__":
    test_memory_integration()