#!/usr/bin/env python3
"""
Test script to observe and verify gaze tracking behavior.
Run this to see servo output without the full system running.
"""
import os
import sys
import time

import numpy as np

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vision.gaze import update_gaze


def simulate_no_face_scenario(duration=30):
    """Test idle wandering behavior with no face detected"""
    print("=== TESTING: No Face Detected (Idle Wandering) ===")
    print("Duration: {} seconds".format(duration))
    print("Format: [Time] X: servo_x, Y: servo_y")
    print("-" * 50)

    # Create dummy frame
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    start_time = time.time()
    while time.time() - start_time < duration:
        # No face detected - pass None for face_box
        person_present, servo_x, servo_y = update_gaze(dummy_frame, None)

        elapsed = time.time() - start_time
        print("[{:05.1f}s] X: {:3d}, Y: {:3d}".format(elapsed, servo_x, servo_y))

        time.sleep(0.5)  # Check every 500ms


def simulate_face_tracking_scenario(duration=15):
    """Test face tracking behavior with simulated face movement"""
    print("\n=== TESTING: Face Tracking ===")
    print("Duration: {} seconds".format(duration))
    print("Simulating face moving left to right across frame")
    print("-" * 50)

    # Create dummy frame (640x480)
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    start_time = time.time()
    while time.time() - start_time < duration:
        elapsed = time.time() - start_time

        # Simulate face moving left to right
        progress = (elapsed / duration) % 1.0  # 0.0 to 1.0
        face_x = int(100 + progress * 440)  # Move from x=100 to x=540
        face_y = 240  # Center Y

        # Create face bounding box
        face_box = (face_x - 50, face_y - 50, face_x + 50, face_y + 50)

        person_present, servo_x, servo_y = update_gaze(dummy_frame, face_box)

        print("[{:05.1f}s] Face at ({:3d},{:3d}) -> Servos X:{:3d}, Y:{:3d}".format(elapsed, face_x, face_y, servo_x, servo_y))

        time.sleep(0.2)  # Check every 200ms


def main():
    print("Gaze Tracking Behavior Test")
    print("=" * 50)

    # Test 1: Face tracking behavior
    simulate_face_tracking_scenario(15)

    # Test 2: Idle wandering behavior
    simulate_no_face_scenario(20)

    print("\n=== TEST COMPLETED ===")
    print("Review the output to identify:")
    print("1. Smooth vs erratic movement patterns")
    print("2. Appropriate response to face tracking")
    print("3. Reasonable idle wandering behavior")
    print("4. Any sudden jumps or oscillations")


if __name__ == "__main__":
    main()
