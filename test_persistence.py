#!/usr/bin/env python3
"""
Test the persistence system without camera dependencies
"""

import sys
import os
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Test imports
try:
    from utils.state_manager import state_manager
    from captioner.captioner import Captioner
    from mood.mood import MoodEngine
    from utils.continuity import describe_duration
    print("✅ All imports successful")
except Exception as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

def test_persistence():
    """Test the persistence system"""
    print("\n=== Testing Persistence System ===")
    
    # Initialize components
    print("1. Initializing components...")
    mood_engine = MoodEngine()
    captioner = Captioner()
    
    # Simulate some state
    print("2. Setting up test state...")
    captioner.current_mood = 0.7
    captioner.last_caption = "Testing persistence system"
    captioner.boredom = 0.3
    captioner.awakening_done = True
    
    # Add some memory and motifs
    captioner.observe("test observation")
    captioner.motif_counter["creativity"] = 8  # Above threshold (7) to form belief
    captioner.motif_counter["personal_space"] = 9  # Above threshold
    captioner.motif_first_seen["creativity"] = time.time() - 86400 * 2  # 2 days ago to meet minimum
    captioner.motif_first_seen["personal_space"] = time.time() - 86400 * 3  # 3 days ago
    captioner.motif_last_seen["creativity"] = time.time()
    captioner.motif_last_seen["personal_space"] = time.time()
    captioner.update_beliefs()  # Form beliefs from motifs
    
    mood_engine.current_mood = 0.7
    
    print(f"   - Captioner mood: {captioner.current_mood}")
    print(f"   - Captioner beliefs: {len(captioner.beliefs)}")
    print(f"   - Mood engine mood: {mood_engine.current_mood}")
    
    # Save state
    print("3. Saving state...")
    success = state_manager.save_session_state(captioner, mood_engine)
    if success:
        print("   ✅ State saved successfully")
    else:
        print("   ❌ Failed to save state")
        return False
    
    # Simulate restart - create new instances
    print("4. Simulating restart...")
    new_mood_engine = MoodEngine()
    new_captioner = Captioner()
    
    print(f"   - New captioner mood (before load): {new_captioner.current_mood}")
    print(f"   - New captioner beliefs (before load): {len(new_captioner.beliefs)}")
    
    # Load state
    print("5. Loading previous state...")
    previous_state = state_manager.load_session_state()
    
    if previous_state:
        print("   ✅ State loaded successfully")
        
        # Apply state
        state_manager.apply_state_to_captioner(previous_state, new_captioner)
        state_manager.apply_state_to_mood_engine(previous_state, new_mood_engine)
        
        print(f"   - Restored captioner mood: {new_captioner.current_mood}")
        print(f"   - Restored captioner beliefs: {len(new_captioner.beliefs)}")
        print(f"   - Restored mood engine mood: {new_mood_engine.current_mood}")
        
        # Test awakening message
        save_time = previous_state["metadata"]["save_time"]
        time_since_last = describe_duration(save_time)
        previous_beliefs = previous_state["captioner"].get("beliefs", {})
        
        new_captioner.memory_loaded_from_previous = True
        awakening_msg = new_captioner.generate_awakening_message(time_since_last, previous_beliefs)
        print(f"   - Awakening message: {awakening_msg}")
        
        return True
    else:
        print("   ❌ No state found")
        return False

if __name__ == "__main__":
    success = test_persistence()
    if success:
        print("\n🎉 Persistence system test PASSED!")
    else:
        print("\n💥 Persistence system test FAILED!")
        sys.exit(1)
