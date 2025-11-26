#!/usr/bin/env python3
"""
Test smooth servo movement - shows step-by-step easing to verify smoothness.
This should show gradual "shhhhh" movement instead of "kakakaka" jumps.
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vision.gaze import update_gaze


def test_smooth_transition():
    """Test gradual movement from one position to another"""
    print("=== SMOOTH MOVEMENT TEST ===")
    print("Watch for gradual steps, not jumps")
    print("Movement should be: shhhhhhh (smooth) not kakakaka (choppy)")
    print("-" * 60)

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Start with face on left
    face_box = (100 - 25, 240 - 25, 100 + 25, 240 + 25)
    person_present, servo_x, servo_y = update_gaze(dummy_frame, face_box)
    print("Starting position - Face LEFT -> PAN:{:6.2f}, TILT:{:6.2f}".format(servo_x, servo_y))

    # Move face to right gradually and show each step
    for i in range(20):  # 20 steps
        progress = i / 19.0  # 0.0 to 1.0
        face_x = int(100 + progress * 440)  # Move from 100 to 540
        face_y = 240  # Keep Y centered

        face_box = (face_x - 25, face_y - 25, face_x + 25, face_y + 25)
        person_present, servo_x, servo_y = update_gaze(dummy_frame, face_box)

        print("Step {:2d}: Face X:{:3d} -> PAN:{:6.2f}, TILT:{:6.2f}".format(i, face_x, servo_x, servo_y))

        time.sleep(0.1)  # Small delay to simulate real frame rate

    print("\nShould see smooth gradual changes in PAN values!")
    print("If PAN jumps in large steps, it's still choppy")
    print("If PAN changes gradually (small increments), it's smooth!")


def test_approach_target():
    """Test how servo approaches a target position step by step"""
    print("\n=== TARGET APPROACH TEST ===")
    print("Servo should gradually approach target, not jump there")
    print("-" * 50)

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Set target face position (right side)
    target_face_box = (500 - 25, 240 - 25, 500 + 25, 240 + 25)

    # Show how servo approaches this target over multiple updates
    for i in range(15):
        person_present, servo_x, servo_y = update_gaze(dummy_frame, target_face_box)

        print("Update {:2d}: PAN:{:6.2f}, TILT:{:6.2f}".format(i, servo_x, servo_y))

        time.sleep(0.05)  # Fast updates to show easing

    print("\nPAN should gradually approach final value, not jump there immediately!")


if __name__ == "__main__":
    test_smooth_transition()
    test_approach_target()
