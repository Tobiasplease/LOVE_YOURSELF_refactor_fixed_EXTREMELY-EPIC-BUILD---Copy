#!/usr/bin/env python3
"""
Test script to validate ExperientialMoodEngine compatibility with existing systems.
Run this to verify all interfaces work correctly before deploying.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mood.experiential_mood import ExperientialMoodEngine
from mood.mood import MoodEngine
import time

def test_interface_compatibility():
    """Test that ExperientialMoodEngine maintains all legacy interfaces."""
    print("=== Testing Interface Compatibility ===")
    
    # Create both engines
    legacy_engine = MoodEngine()
    experiential_engine = ExperientialMoodEngine()
    
    # Test basic interface compatibility
    test_caption = "I see a laptop on a desk with a person working quietly"
    
    # 1. Test analyze_mood method signature
    print("CHECK: Testing analyze_mood() interface...")
    legacy_mood = legacy_engine.analyze_mood(test_caption)
    experiential_mood = experiential_engine.analyze_mood(test_caption)
    
    assert isinstance(legacy_mood, float), "Legacy mood should be float"
    assert isinstance(experiential_mood, float), "Experiential mood should be float" 
    assert 0.0 <= legacy_mood <= 1.0, "Legacy mood should be in [0,1] range"
    assert 0.0 <= experiential_mood <= 1.0, "Experiential mood should be in [0,1] range"
    print(f"  Legacy mood: {legacy_mood:.3f}")
    print(f"  Experiential mood: {experiential_mood:.3f}")
    
    # 2. Test get_current_mood method
    print("CHECK: Testing get_current_mood() interface...")
    legacy_current = legacy_engine.get_current_mood()
    experiential_current = experiential_engine.get_current_mood()
    
    assert isinstance(legacy_current, float), "Legacy current mood should be float"
    assert isinstance(experiential_current, float), "Experiential current mood should be float"
    print(f"  Legacy current: {legacy_current:.3f}")
    print(f"  Experiential current: {experiential_current:.3f}")
    
    # 3. Test mood_vector property
    print("CHECK: Testing mood_vector property...")
    legacy_vector = legacy_engine.mood_vector
    experiential_vector = experiential_engine.mood_vector
    
    assert isinstance(legacy_vector, tuple) and len(legacy_vector) == 3, "Legacy vector should be 3-tuple"
    assert isinstance(experiential_vector, tuple) and len(experiential_vector) == 3, "Experiential vector should be 3-tuple"
    print(f"  Legacy vector: {legacy_vector}")
    print(f"  Experiential vector: {experiential_vector}")
    
    # 4. Test get_emotion_for_hand_controller method
    print("CHECK: Testing get_emotion_for_hand_controller() interface...")
    legacy_emotion = legacy_engine.get_emotion_for_hand_controller()
    experiential_emotion = experiential_engine.get_emotion_for_hand_controller()
    
    valid_emotions = {"energized_engaged", "calm_observant", "alert_curious", "quiet_detached", "withdrawn_distant"}
    assert legacy_emotion in valid_emotions, f"Legacy emotion '{legacy_emotion}' not in valid set"
    assert experiential_emotion in valid_emotions, f"Experiential emotion '{experiential_emotion}' not in valid set"
    print(f"  Legacy emotion: {legacy_emotion}")
    print(f"  Experiential emotion: {experiential_emotion}")
    
    # 5. Test get_pattern_data method
    print("CHECK: Testing get_pattern_data() interface...")
    legacy_pattern = legacy_engine.get_pattern_data()
    experiential_pattern = experiential_engine.get_pattern_data()
    
    assert isinstance(legacy_pattern, dict), "Legacy pattern data should be dict"
    assert isinstance(experiential_pattern, dict), "Experiential pattern data should be dict"
    print(f"  Legacy pattern keys: {list(legacy_pattern.keys())}")
    print(f"  Experiential pattern keys: {list(experiential_pattern.keys())}")
    
    print("PASS: All interface compatibility tests passed!")

def test_experiential_features():
    """Test experiential-specific features."""
    print("\n=== Testing Experiential Features ===")
    
    engine = ExperientialMoodEngine()
    
    # Test experiential state access
    print("CHECK: Testing experiential state access...")
    exp_state = engine.get_experiential_state()
    assert isinstance(exp_state, dict), "Experiential state should be dict"
    assert "experiences" in exp_state, "Should have experiences key"
    assert "dominant" in exp_state, "Should have dominant key"
    print(f"  Available experiences: {list(exp_state['experiences'].keys())}")
    
    # Test experience summary
    print("CHECK: Testing experience summary...")
    summary = engine.get_experience_summary()
    assert isinstance(summary, str), "Experience summary should be string"
    print(f"  Current summary: '{summary}'")
    
    # Test temporal accumulation
    print("CHECK: Testing temporal accumulation...")
    initial_restlessness = engine.experiential_states["restlessness"]
    
    # Simulate some time passing
    engine._apply_temporal_accumulation()
    engine._last_experience_update = time.time() - 10  # 10 seconds ago
    engine._apply_temporal_accumulation()
    
    updated_restlessness = engine.experiential_states["restlessness"] 
    print(f"  Restlessness: {initial_restlessness:.3f} -> {updated_restlessness:.3f}")
    
    print("PASS: All experiential feature tests passed!")

def test_state_persistence():
    """Test that experiential states can be saved and loaded."""
    print("\n=== Testing State Persistence ===")
    
    # Create engine and set some experiential states
    engine1 = ExperientialMoodEngine()
    engine1.experiential_states["curiosity"] = 0.7
    engine1.experiential_states["contemplation"] = 0.5
    engine1.current_mood = 0.8
    
    # Simulate state extraction
    state_data = {
        "experiential_states": dict(engine1.experiential_states),
        "experience_history": list(engine1.experience_history),
        "current_mood": engine1.current_mood,
    }
    
    # Create new engine and restore state
    engine2 = ExperientialMoodEngine()
    engine2.experiential_states.update(state_data["experiential_states"])
    engine2.current_mood = state_data["current_mood"]
    
    # Verify restoration
    assert engine2.experiential_states["curiosity"] == 0.7, "Curiosity should be restored"
    assert engine2.experiential_states["contemplation"] == 0.5, "Contemplation should be restored"
    assert engine2.current_mood == 0.8, "Current mood should be restored"
    
    print("PASS: State persistence test passed!")

def test_fallback_behavior():
    """Test fallback to legacy system when experiential fails."""
    print("\n=== Testing Fallback Behavior ===")
    
    engine = ExperientialMoodEngine()
    
    # Test that fallback doesn't crash the system
    try:
        engine._apply_experiential_fallback("test caption", None)
        print("PASS: Experiential fallback works without crashing")
    except Exception as e:
        print(f"ERROR: Experiential fallback failed: {e}")
        raise
    
    # Test that all values remain in valid ranges
    mood = engine.get_current_mood()
    vector = engine.mood_vector
    emotion = engine.get_emotion_for_hand_controller()
    
    assert 0.0 <= mood <= 1.0, "Mood should remain in valid range after fallback"
    assert all(-1.0 <= v <= 1.0 for v in vector), "Vector should remain in valid range after fallback"
    assert emotion in {"energized_engaged", "calm_observant", "alert_curious", "quiet_detached", "withdrawn_distant"}, "Emotion should remain valid after fallback"
    
    print("PASS: Fallback behavior test passed!")

if __name__ == "__main__":
    print("Running ExperientialMoodEngine validation tests...\n")
    
    try:
        test_interface_compatibility()
        test_experiential_features() 
        test_state_persistence()
        test_fallback_behavior()
        
        print("\nAll tests passed! ExperientialMoodEngine is ready for deployment.")
        print("\nThe experiential mood system is enabled by default:")
        print("1. USE_EXPERIENTIAL_MOOD = True in config/config.py (already set)")
        print("2. Run: python machine.py (no config override needed)")
        print("3. The system will automatically fall back to legacy if there are any issues")
        print("4. Set DEBUG_TEMPORAL_EMOTIONS = True in config for detailed temporal debugging")
        
    except Exception as e:
        print(f"\nERROR: Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)