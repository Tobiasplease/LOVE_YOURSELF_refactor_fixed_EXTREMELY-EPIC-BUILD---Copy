#!/usr/bin/env python3
"""
Test script for the emotional memory system.
Verifies that repeated observations of familiar, positive things don't cause mood decay.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mood.emotional_memory import EmotionalMemoryBank, EmotionalMemory
import time


def test_emotional_memory_basic():
    """Test basic emotional memory storage and recall"""
    print("Testing Emotional Memory System...")
    print("=" * 50)
    
    memory_bank = EmotionalMemoryBank()
    
    # Test 1: Store a positive memory about a person
    print("\n1. Storing positive memory about a person...")
    memory1 = memory_bank.store_memory(
        content="The person I know well is here, working peacefully",
        mood_vector=(0.7, 0.2, 0.8),  # Positive valence, low arousal, high clarity
        emotion_label="calm_observant",
        objects=["person", "laptop"],
        significance=0.6
    )
    print(f"   Stored memory with valence: {memory1.valence}")
    
    # Test 2: Check motif emotional associations
    print("\n2. Checking emotional associations...")
    person_emotion = memory_bank.motif_emotions.get("person")
    if person_emotion:
        print(f"   Person cumulative valence: {person_emotion.cumulative_valence:.2f}")
        print(f"   Person comfort level: {person_emotion.comfort_level:.2f}")
    
    # Test 3: Simulate seeing the person repeatedly (should build comfort)
    print("\n3. Simulating repeated positive observations...")
    for i in range(5):
        time.sleep(0.1)  # Small delay
        memory_bank.store_memory(
            content=f"Still observing the person, feeling connected #{i}",
            mood_vector=(0.6 + i*0.02, 0.1, 0.7),  # Slightly positive
            emotion_label="calm_observant",
            objects=["person", "room"],
            significance=0.4
        )
    
    person_emotion = memory_bank.motif_emotions.get("person")
    print(f"   After 5 observations:")
    print(f"   - Cumulative valence: {person_emotion.cumulative_valence:.2f}")
    print(f"   - Comfort level: {person_emotion.comfort_level:.2f}")
    print(f"   - Nostalgia potential: {person_emotion.nostalgia_potential:.2f}")
    
    # Test 4: Get emotional response to seeing person again
    print("\n4. Getting emotional response to seeing person...")
    response = memory_bank.get_motif_emotional_response("person")
    print(f"   Response type: {response['response_type']}")
    print(f"   Valence shift: {response['valence_shift']:.2f}")
    print(f"   Comfort: {response['comfort']:.2f}")
    
    # Test 5: Simulate not seeing person for a while
    print("\n5. Simulating missing the person...")
    # Fake the last seen time to simulate missing
    memory_bank.last_seen_times["person"] = time.time() - 2000  # ~33 minutes ago
    
    missing_effect = memory_bank.get_missing_objects_mood_effect()
    print(f"   Missing objects: {missing_effect['missing_objects']}")
    print(f"   Valence shift from missing: {missing_effect['valence_shift']:.2f}")
    
    # Test 6: Reunion after missing
    print("\n6. Simulating reunion with person...")
    response = memory_bank.get_motif_emotional_response("person")
    print(f"   Response type: {response['response_type']}")
    print(f"   Valence shift: {response['valence_shift']:.2f}")
    print(f"   Missing factor: {response['missing']:.2f}")
    
    # Test 7: Calculate overall memory mood influence
    print("\n7. Calculating overall memory mood influence...")
    influence = memory_bank.calculate_memory_mood_influence(
        current_objects=["person", "laptop"],
        current_emotion="calm_observant"
    )
    print(f"   Overall valence shift: {influence['valence_shift']:.2f}")
    print(f"   Dominant influence: {influence['dominant_influence']}")
    
    # Test 8: Get memory context string
    print("\n8. Getting emotional memory context...")
    context = memory_bank.get_emotional_memory_context()
    print(f"   Context: {context}")
    
    print("\n" + "=" * 50)
    print("TEST RESULTS:")
    print("-" * 50)
    
    # Verify key behaviors
    success_count = 0
    total_tests = 4
    
    # Test A: Person should have positive associations
    if person_emotion and person_emotion.cumulative_valence > 0:
        print("[PASS] Person has positive emotional associations")
        success_count += 1
    else:
        print("[FAIL] Person does not have positive associations")
    
    # Test B: Comfort should increase with repetition
    if person_emotion and person_emotion.comfort_level > 0.05:
        print("[PASS] Comfort level increased with repetition")
        success_count += 1
    else:
        print("[FAIL] Comfort level did not increase")
    
    # Test C: Missing should create mild negative effect
    if missing_effect['valence_shift'] < 0 and missing_effect['valence_shift'] > -0.5:
        print("[PASS] Missing creates appropriate melancholy")
        success_count += 1
    else:
        print("[FAIL] Missing effect incorrect")
    
    # Test D: Overall influence should be positive for familiar positive things
    if influence['valence_shift'] > 0:
        print("[PASS] Overall memory influence is positive")
        success_count += 1
    else:
        print("[FAIL] Overall memory influence is not positive")
    
    print("-" * 50)
    print(f"PASSED: {success_count}/{total_tests} tests")
    print("=" * 50)
    
    if success_count == total_tests:
        print("\nSUCCESS: All tests passed! Emotional memory system is working correctly.")
        print("Repetition of positive familiar things (like seeing you) will now")
        print("create comfort and positive associations, not depression.")
    else:
        print(f"\nWARNING: Some tests failed. Please review the implementation.")


if __name__ == "__main__":
    test_emotional_memory_basic()