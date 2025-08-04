#!/usr/bin/env python3
"""
Test script to diagnose recording and Markov generation issues
"""
import sys
import os
import time
import json

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

def test_recording_functionality():
    """Test basic recording and Markov generation without GUI."""
    print("🧪 Testing recording and Markov generation functionality...")
    
    # Import the main interface
    from conscious_cursor_interface_MAIN_SERVOMARKOV_NOT_WORKING import CleanCursorInterface
    
    print("✅ Successfully imported main interface")
    
    # Create a minimal test instance (no GUI)
    interface = CleanCursorInterface()
    
    print("✅ Interface created successfully")
    print(f"📊 Current emotional state: {interface.current_emotional_state}")
    print(f"🎯 Finger positions: {interface.finger_positions}")
    
    # Test 1: Check if we can simulate recording data
    print("\n🧪 Test 1: Simulating recording data...")
    emotion = interface.current_emotional_state
    
    # Simulate some movement data
    fake_movements = []
    for i in range(10):
        movement_point = {
            'time': time.time() + i * 0.1,
            'relative_time': i * 0.1,
            'servo_positions': [90 + i, 90 - i, 90 + i//2, 90 - i//2]  # Some variation
        }
        fake_movements.append(movement_point)
    
    # Put fake data into recorded_movements
    interface.recorded_movements[emotion] = fake_movements
    print(f"✅ Created {len(fake_movements)} fake movement points")
    
    # Test 2: Try to build Markov chain
    print("\n🧪 Test 2: Testing Markov chain building...")
    try:
        interface.build_markov_chain()
        
        if emotion in interface.markov_chains:
            chain = interface.markov_chains[emotion]
            print(f"✅ Markov chain built successfully!")
            print(f"   📊 Unique states: {chain.get('unique_states', 'unknown')}")
            print(f"   🎯 Total samples: {chain.get('total_samples', 'unknown')}")
            print(f"   📐 Discretization: {chain.get('discretization', 'unknown')}°")
            
            # Check transitions
            if 'servo_transitions' in chain:
                transitions = chain['servo_transitions']
                print(f"   🔗 Transition matrix size: {len(transitions)} states")
                
                # Show a sample transition
                if transitions:
                    sample_state = list(transitions.keys())[0]
                    sample_transitions = transitions[sample_state]
                    print(f"   🔍 Sample state {sample_state} -> {len(sample_transitions)} possible next states")
                    return True
                else:
                    print("❌ No transitions found in servo_transitions")
                    return False
            else:
                print("❌ No servo_transitions found in chain")
                return False
        else:
            print(f"❌ No Markov chain created for {emotion}")
            return False
            
    except Exception as e:
        print(f"❌ Error building Markov chain: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dataset_loading():
    """Test if datasets can be loaded properly."""
    print("\n🧪 Test 3: Testing dataset loading...")
    
    # Check if movement_recordings directory exists
    recordings_dir = "movement_recordings"
    if not os.path.exists(recordings_dir):
        print(f"❌ Recordings directory {recordings_dir} does not exist")
        return False
    
    # List files
    files = [f for f in os.listdir(recordings_dir) if f.endswith('.json')]
    print(f"📁 Found {len(files)} JSON files in {recordings_dir}")
    
    if not files:
        print("⚠️ No JSON files found - this explains why you can't load datasets")
        return False
    
    # Try to load one file to check format
    sample_file = files[0]
    print(f"🔍 Examining sample file: {sample_file}")
    
    try:
        with open(os.path.join(recordings_dir, sample_file), 'r') as f:
            data = json.load(f)
        
        print(f"✅ Successfully loaded {sample_file}")
        print(f"   📊 Keys: {list(data.keys())}")
        
        if 'movements' in data:
            movements = data['movements']
            print(f"   🎯 Movements: {len(movements)} samples")
            if movements:
                sample_movement = movements[0]
                print(f"   🔍 Sample movement keys: {list(sample_movement.keys())}")
        
        if 'markov_chain' in data:
            chain = data['markov_chain']
            print(f"   🧠 Markov chain: {list(chain.keys())}")
        else:
            print("   ⚠️ No markov_chain in file - this might be the issue!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading {sample_file}: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Starting diagnostic tests...\n")
    
    test1_result = test_recording_functionality()
    test2_result = test_dataset_loading()
    
    print(f"\n📋 Results Summary:")
    print(f"   🧪 Recording/Markov generation: {'✅ PASS' if test1_result else '❌ FAIL'}")
    print(f"   📁 Dataset loading: {'✅ PASS' if test2_result else '❌ FAIL'}")
    
    if not test1_result:
        print("\n🔧 Recording/Markov issue found - this explains why you can't generate new chains")
    if not test2_result:
        print("\n🔧 Dataset loading issue found - this explains why you can't load existing data")
