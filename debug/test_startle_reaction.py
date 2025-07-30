#!/usr/bin/env python3
"""
Test the hand startle reaction when faces are detected.

This script simulates face detection events to test the startle response.
"""

import time
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from servo_control.hand_expression import HandExpressionController

def test_startle_reaction():
    """Test the hand startle reaction system."""
    print("=" * 60)
    print("HAND STARTLE REACTION TEST")
    print("=" * 60)
    print("This test simulates face detection events to trigger startle reactions.")
    print("You should see:")
    print("- Quick finger snap when 'face detected'")
    print("- Smooth return to normal behavior after 0.8 seconds")
    print("- Arduino should use 3ms speed for startle (very fast)")
    
    # Initialize hand controller
    try:
        hand_controller = HandExpressionController()
        print(f"[HAND] Connected successfully")
    except Exception as e:
        print(f"[ERROR] Failed to connect to hand controller: {e}")
        return

    try:
        print("\n" + "=" * 60)
        print("TEST 1: Normal behavior (no face)")
        print("=" * 60)
        print("Running normal behavior for 5 seconds...")
        
        start_time = time.time()
        while time.time() - start_time < 5:
            positions = hand_controller.update_from_consciousness(
                mood=0.5,
                novelty=0.3, 
                boredom=0.2,
                person_present=False
            )
            
            # Print status every second
            if int(time.time() - start_time) != int(time.time() - start_time - 0.1):
                elapsed = int(time.time() - start_time)
                print(f"  {elapsed}s | Normal behavior | F0:{positions['finger0']:3d} F1:{positions['finger1']:3d} F2:{positions['finger2']:3d} F3:{positions['finger3']:3d}")
            
            time.sleep(0.1)

        print("\n" + "=" * 60)
        print("TEST 2: Startle reaction when face detected")
        print("=" * 60)
        
        for test_num in range(3):
            print(f"\nStartle test #{test_num + 1}:")
            print("- Starting with no face present...")
            
            # Run without face for 2 seconds
            start_time = time.time()
            while time.time() - start_time < 2:
                positions = hand_controller.update_from_consciousness(
                    mood=0.5,
                    novelty=0.3,
                    boredom=0.2,
                    person_present=False
                )
                time.sleep(0.1)
            
            print("- 😲 FACE DETECTED! Triggering startle...")
            
            # Trigger startle by switching to person_present=True
            startle_start = time.time()
            while time.time() - startle_start < 3:  # 3 seconds to see full startle + recovery
                elapsed = time.time() - startle_start
                positions = hand_controller.update_from_consciousness(
                    mood=0.5,
                    novelty=0.3,
                    boredom=0.2,
                    person_present=True  # This triggers the startle on first call
                )
                
                status = "STARTLE!" if elapsed < 0.8 else "recovering"
                print(f"  {elapsed:.1f}s | {status:9s} | F0:{positions['finger0']:3d} F1:{positions['finger1']:3d} F2:{positions['finger2']:3d} F3:{positions['finger3']:3d}")
                time.sleep(0.2)
            
            # Reset for next test - need to clear person_present
            hand_controller.person_was_present = False
            time.sleep(1)

        print("\n" + "=" * 60)
        print("TEST 3: Multiple rapid face detections")
        print("=" * 60)
        print("Testing that startle doesn't re-trigger if face stays present...")
        
        # First detection triggers startle
        print("- First face detection (should startle)")
        positions = hand_controller.update_from_consciousness(
            mood=0.5, novelty=0.3, boredom=0.2, person_present=True
        )
        time.sleep(0.1)
        
        # Subsequent calls with face still present should NOT retrigger startle
        for i in range(10):
            positions = hand_controller.update_from_consciousness(
                mood=0.5, novelty=0.3, boredom=0.2, person_present=True
            )
            print(f"  Call {i+2}: F0:{positions['finger0']:3d} F1:{positions['finger1']:3d} F2:{positions['finger2']:3d} F3:{positions['finger3']:3d} (should NOT startle)")
            time.sleep(0.1)

        print("\n" + "=" * 60)
        print("STARTLE REACTION TEST COMPLETE!")
        print("=" * 60)
        print("Key observations:")
        print("- Startle should only trigger on NEW face detection (not continuous)")
        print("- Arduino should show 3ms speeds for startle movements")
        print("- Fingers should snap quickly then settle into normal behavior")
        print("- No 'kikiki' sounds - should maintain smooth 'rrrrr' servo operation")

    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"Test error: {e}")
    finally:
        hand_controller.close()
        print("Hand controller connection closed")

if __name__ == "__main__":
    test_startle_reaction()
