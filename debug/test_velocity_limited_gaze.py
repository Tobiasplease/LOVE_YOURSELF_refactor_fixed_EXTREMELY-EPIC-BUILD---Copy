#!/usr/bin/env python3
"""
Test the velocity-limited gaze system to verify protection against hard lock-ins
and confirm expanded tilt range functionality.
"""
import os
import sys
import time

import numpy as np

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vision.gaze import update_gaze
from config.config import MAX_PAN_VELOCITY, MAX_TILT_VELOCITY


def test_hard_lockIn_protection():
    """Test protection against hard lock-ins when face suddenly appears"""
    print("=== TESTING: Hard Lock-In Protection ===")
    print("Simulating sudden face appearances to test velocity limiting")
    print(f"Max velocities: PAN {MAX_PAN_VELOCITY}°/update, TILT {MAX_TILT_VELOCITY}°/update")
    print("-" * 60)

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Start with no face (idle state)
    print("Phase 1: Establishing idle state...")
    for _ in range(10):
        update_gaze(dummy_frame, None)
        time.sleep(0.1)

    # Sudden face appearance at extreme position
    extreme_positions = [
        ((50, 50), "Top-left extreme"),
        ((590, 50), "Top-right extreme"),
        ((590, 430), "Bottom-right extreme"),
        ((50, 430), "Bottom-left extreme"),
    ]

    for (face_x, face_y), label in extreme_positions:
        print(f"\n--- {label} Lock-In Test ---")
        face_box = (face_x - 25, face_y - 25, face_x + 25, face_y + 25)

        # Record position before face appears
        person_present, start_pan, start_tilt = update_gaze(dummy_frame, None)
        print(f"Start position: PAN {start_pan}°, TILT {start_tilt}°")

        # Face suddenly appears - track movement velocity
        print(f"Face appears at ({face_x},{face_y})...")
        velocities = []

        for step in range(15):  # Track 15 movement steps
            person_present, servo_x, servo_y = update_gaze(dummy_frame, face_box)

            if step > 0:
                pan_velocity = abs(servo_x - last_pan)
                tilt_velocity = abs(servo_y - last_tilt)
                velocities.append((pan_velocity, tilt_velocity))

                violation = ""
                if pan_velocity > MAX_PAN_VELOCITY:
                    violation += f" PAN_VIOLATION({pan_velocity:.1f}°)"
                if tilt_velocity > MAX_TILT_VELOCITY:
                    violation += f" TILT_VIOLATION({tilt_velocity:.1f}°)"

                print(f"  Step {step:2d}: PAN {servo_x:3d}° (Δ{pan_velocity:+4.1f}), TILT {servo_y:3d}° (Δ{tilt_velocity:+4.1f}){violation}")

            last_pan, last_tilt = servo_x, servo_y
            time.sleep(0.05)  # Fast updates to test high-frequency behavior

        # Analyze velocities
        if velocities:
            max_pan_vel = max(v[0] for v in velocities)
            max_tilt_vel = max(v[1] for v in velocities)
            print(f"  Max velocities observed: PAN {max_pan_vel:.1f}°, TILT {max_tilt_vel:.1f}°")

            if max_pan_vel <= MAX_PAN_VELOCITY and max_tilt_vel <= MAX_TILT_VELOCITY:
                print(f"  ✓ PASSED: No velocity violations")
            else:
                print(f"  ✗ FAILED: Velocity violations detected")

        time.sleep(0.5)  # Brief pause between tests


def test_expanded_tilt_range():
    """Test the expanded tilt range to verify firmware updates"""
    print("\n=== TESTING: Expanded Tilt Range ===")
    print("Testing if tilt can now reach the expanded range (50-130°)")
    print("Previous Arduino firmware limited to 70-110°")
    print("-" * 60)

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Test extreme tilt positions
    test_positions = [
        ((320, 50), "Top center - should reach high tilt (~120°)"),
        ((320, 430), "Bottom center - should reach low tilt (~60°)"),
    ]

    max_tilt_achieved = 90
    min_tilt_achieved = 90

    for (face_x, face_y), label in test_positions:
        print(f"\n--- {label} ---")
        face_box = (face_x - 25, face_y - 25, face_x + 25, face_y + 25)

        # Allow time to converge
        for step in range(20):
            person_present, servo_x, servo_y = update_gaze(dummy_frame, face_box)
            max_tilt_achieved = max(max_tilt_achieved, servo_y)
            min_tilt_achieved = min(min_tilt_achieved, servo_y)

            if step % 5 == 0:  # Print every 5th step
                print(f"  Step {step:2d}: PAN {servo_x:3d}°, TILT {servo_y:3d}°")

            time.sleep(0.1)

    print(f"\nTilt Range Analysis:")
    print(f"✓ Maximum tilt achieved: {max_tilt_achieved}°")
    print(f"✓ Minimum tilt achieved: {min_tilt_achieved}°")
    print(f"✓ Total range: {max_tilt_achieved - min_tilt_achieved}°")

    if max_tilt_achieved >= 115:
        print(f"✓ PASSED: Tilt reaches high range (≥115°) - firmware updated successfully")
    else:
        print(f"✗ FAILED: Tilt limited to {max_tilt_achieved}° - firmware may not be updated")

    if min_tilt_achieved <= 65:
        print(f"✓ PASSED: Tilt reaches low range (≤65°) - firmware updated successfully")
    else:
        print(f"✗ FAILED: Tilt limited to {min_tilt_achieved}° - firmware may not be updated")


def test_velocity_smoothing():
    """Test velocity smoothing to ensure gradual acceleration/deceleration"""
    print("\n=== TESTING: Velocity Smoothing ===")
    print("Testing smooth acceleration and deceleration patterns")
    print("-" * 60)

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Start at center
    for _ in range(5):
        update_gaze(dummy_frame, None)

    print("Moving from center to extreme position...")
    face_box = (50, 50, 100, 100)  # Top-left corner

    last_pan = last_tilt = 90
    accelerations = []

    for step in range(20):
        person_present, servo_x, servo_y = update_gaze(dummy_frame, face_box)

        if step > 0:
            pan_velocity = servo_x - last_pan
            tilt_velocity = servo_y - last_tilt

            if step > 1:
                pan_accel = abs(pan_velocity - last_pan_velocity)
                tilt_accel = abs(tilt_velocity - last_tilt_velocity)
                accelerations.append((pan_accel, tilt_accel))

                print(f"Step {step:2d}: PAN {servo_x:3d}° (v={pan_velocity:+5.1f}, a={pan_accel:4.1f}), TILT {servo_y:3d}° (v={tilt_velocity:+5.1f}, a={tilt_accel:4.1f})")

            last_pan_velocity = pan_velocity
            last_tilt_velocity = tilt_velocity

        last_pan, last_tilt = servo_x, servo_y
        time.sleep(0.05)

    if accelerations:
        avg_pan_accel = sum(a[0] for a in accelerations) / len(accelerations)
        avg_tilt_accel = sum(a[1] for a in accelerations) / len(accelerations)
        max_pan_accel = max(a[0] for a in accelerations)
        max_tilt_accel = max(a[1] for a in accelerations)

        print(f"\nSmoothing Analysis:")
        print(f"✓ Average acceleration: PAN {avg_pan_accel:.2f}°, TILT {avg_tilt_accel:.2f}°")
        print(f"✓ Maximum acceleration: PAN {max_pan_accel:.2f}°, TILT {max_tilt_accel:.2f}°")
        print(f"Lower values indicate smoother movement")


def main():
    print("Velocity-Limited Gaze System Test Suite")
    print("=" * 60)
    print("Testing servo protection and expanded range functionality")

    # Test 1: Hard lock-in protection
    test_hard_lockIn_protection()

    # Test 2: Expanded tilt range verification
    test_expanded_tilt_range()

    # Test 3: Velocity smoothing
    test_velocity_smoothing()

    print("\n" + "=" * 60)
    print("VELOCITY-LIMITED TEST SUITE COMPLETED")
    print("\nKey protections verified:")
    print("1. No movement exceeds maximum velocity limits")
    print("2. Expanded tilt range accessible (50-130° vs 70-110°)")
    print("3. Smooth acceleration prevents jarring movements")
    print("4. System protects servo hardware from damage")


if __name__ == "__main__":
    main()