#!/usr/bin/env python3
"""Test to reproduce machine.py hand controller crash"""
import sys
import os
import time
import traceback
import threading

def test_machine_hand_crash():
    """Test machine.py integration to reproduce crash"""
    print("Starting machine.py hand controller crash test...")
    
    try:
        # Add project root to path
        project_root = os.path.dirname(os.path.dirname(__file__))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
            
        # Import machine.py components to simulate real environment
        print("Importing machine.py components...")
        
        from hand_control.direct_hand_control import (
            start_hand_controller,
            change_to_emotion,
            send_reactivity_data,
            get_status
        )
        
        print("SUCCESS: Imported hand control modules")
        
        # Test both headless and non-headless modes
        print("Testing headless mode first...")
        result = start_hand_controller(headless=True)
        
        if not result:
            print("ERROR: Even headless mode failed")
            return
            
        print("SUCCESS: Headless mode works, now testing GUI mode...")
        # Stop the headless controller first
        from hand_control.direct_hand_control import stop_hand_controller
        stop_hand_controller()
        
        time.sleep(1)
        
        print("Starting hand controller (non-headless like machine.py)...")
        result = start_hand_controller(headless=False)
        
        if not result:
            print("ERROR: Failed to start hand controller")
            return
            
        print("SUCCESS: Hand controller started")
        
        # Simulate machine.py behavior - rapid emotion changes and data sending
        emotions = ['calm_observant', 'alert_curious', 'energized_engaged', 'quiet_detached', 'withdrawn_distant']
        
        print("Starting intensive testing to reproduce crash...")
        
        for cycle in range(5):  # Run multiple cycles
            print(f"\n=== CYCLE {cycle + 1} ===")
            
            for i, emotion in enumerate(emotions):
                print(f"Step {i+1}: Changing to {emotion}")
                
                # Change emotion like machine.py does
                change_result = change_to_emotion(emotion)
                print(f"  Change result: {change_result}")
                
                # Send reactivity data like machine.py does
                reactivity_data = {
                    'action': 'resume' if i % 2 == 0 else 'pause',
                    'activity_level': 0.3 + (i * 0.1),
                    'person_present': True,
                    'mood_vector': [0.1, 0.2, 0.3]
                }
                
                try:
                    send_reactivity_data(reactivity_data)
                    print(f"  Sent reactivity data: {reactivity_data}")
                except Exception as e:
                    print(f"  ERROR sending reactivity data: {e}")
                    traceback.print_exc()
                
                # Check status frequently like machine.py would
                try:
                    status = get_status()
                    print(f"  Status check: {status.get('available', 'unknown')}")
                except Exception as e:
                    print(f"  ERROR getting status: {e}")
                    traceback.print_exc()
                    print("  CRASH DETECTED!")
                    return
                
                # Short wait between changes
                time.sleep(0.5)
                
            # Longer wait between cycles
            print(f"Cycle {cycle + 1} completed, waiting 2 seconds...")
            time.sleep(2)
            
        print("SUCCESS: Completed all cycles without crash")
        
    except Exception as e:
        print(f"ERROR: Test failed: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    test_machine_hand_crash()