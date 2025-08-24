#!/usr/bin/env python3
"""Test machine.py integration crash"""
import sys
import os
import time
import traceback
import threading

def test_machine_integration():
    """Test machine.py integration with hand controller"""
    print("Testing machine.py integration...")
    
    try:
        # Add the project root to Python path
        project_root = os.path.dirname(os.path.dirname(__file__))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        
        # Import the hand controller modules  
        from hand_control.direct_hand_control import (
            start_hand_controller,
            change_to_emotion,
            get_status
        )
        
        print("SUCCESS: Imported hand controller modules")
        
        # Test starting the hand controller like machine.py does
        print("Starting hand controller (headless=False)...")
        result = start_hand_controller(headless=False)
        print(f"Hand controller start result: {result}")
        
        if result:
            print("SUCCESS: Hand controller started")
            
            # Test changing emotions like machine.py does
            emotions_to_test = ['calm_observant', 'alert_curious', 'energized_engaged']
            
            for i, emotion in enumerate(emotions_to_test):
                print(f"\nTest {i+1}: Changing to emotion: {emotion}")
                change_result = change_to_emotion(emotion)
                print(f"Change result: {change_result}")
                
                # Check status
                status = get_status()
                print(f"Status: {status}")
                
                # Wait a bit to see if it crashes
                print(f"Waiting 5 seconds...")
                time.sleep(5)
                
                # Check if still alive
                try:
                    status_check = get_status()
                    print(f"Status check after 5s: {status_check}")
                except Exception as e:
                    print(f"ERROR: Status check failed: {e}")
                    traceback.print_exc()
                    break
                    
            print("SUCCESS: All emotion tests completed")
            
        else:
            print("ERROR: Failed to start hand controller")
            
    except Exception as e:
        print(f"ERROR: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    test_machine_integration()