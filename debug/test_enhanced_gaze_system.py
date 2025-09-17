#!/usr/bin/env python3
"""
Test the enhanced gaze system with wider range and faster, more dynamic movement.
"""
import os
import sys
import time

import numpy as np

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vision.gaze import update_gaze


def test_expanded_range_tracking(duration=15):
    """Test face tracking across the expanded range"""
    print("=== TESTING: Expanded Range Face Tracking ===")
    print(f"Duration: {duration} seconds")
    print("Testing new expanded range: PAN 45-135° (±45°), TILT 50-130° (±40°)")
    print("-" * 60)

    # Create dummy frame (640x480)
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    start_time = time.time()
    max_pan = min_pan = 90
    max_tilt = min_tilt = 90

    while time.time() - start_time < duration:
        elapsed = time.time() - start_time

        # Test extreme positions to verify expanded range
        phase = (elapsed / duration) * 2 * np.pi

        # Move face to extreme positions - corners and edges
        if elapsed < duration * 0.25:
            # Top-left corner
            face_x, face_y = 50, 50
        elif elapsed < duration * 0.5:
            # Top-right corner
            face_x, face_y = 590, 50
        elif elapsed < duration * 0.75:
            # Bottom-right corner
            face_x, face_y = 590, 430
        else:
            # Bottom-left corner
            face_x, face_y = 50, 430

        # Create face bounding box
        face_box = (face_x - 25, face_y - 25, face_x + 25, face_y + 25)

        person_present, servo_x, servo_y = update_gaze(dummy_frame, face_box)

        # Track extremes
        max_pan = max(max_pan, servo_x)
        min_pan = min(min_pan, servo_x)
        max_tilt = max(max_tilt, servo_y)
        min_tilt = min(min_tilt, servo_y)

        print(f"[{elapsed:05.1f}s] Face: ({face_x:3d},{face_y:3d}) -> PAN {servo_x:3d}°, TILT {servo_y:3d}°")

        time.sleep(0.2)

    print(f"\nRange achieved:")
    print(f"✓ PAN: {min_pan}° to {max_pan}° (range: {max_pan - min_pan}°)")
    print(f"✓ TILT: {min_tilt}° to {max_tilt}° (range: {max_tilt - min_tilt}°)")
    print(f"Expected: PAN 45-135° (90°), TILT 50-130° (80°)")


def test_dynamic_idle_movement(duration=20):
    """Test the faster, more dynamic idle movement"""
    print("\n=== TESTING: Dynamic Idle Movement ===")
    print(f"Duration: {duration} seconds")
    print("Watch for:")
    print("- Faster movement patterns")
    print("- Shorter base pauses with occasional longer ones")
    print("- More dynamic frequency changes")
    print("-" * 60)

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    start_time = time.time()
    last_pan, last_tilt = 90, 90
    movement_count = 0
    pause_durations = []
    last_movement_time = start_time

    while time.time() - start_time < duration:
        person_present, servo_x, servo_y = update_gaze(dummy_frame, None)

        elapsed = time.time() - start_time
        pan_delta = servo_x - last_pan
        tilt_delta = servo_y - last_tilt

        # Detect movement
        if abs(pan_delta) > 0.5 or abs(tilt_delta) > 0.5:
            if movement_count > 0:  # Skip first movement
                pause_duration = elapsed - (last_movement_time - start_time)
                pause_durations.append(pause_duration)

            movement_count += 1
            last_movement_time = time.time()
            print(f"[{elapsed:05.1f}s] MOVE #{movement_count}: PAN {servo_x:6.1f} (Δ{pan_delta:+4.1f}), TILT {servo_y:6.1f} (Δ{tilt_delta:+4.1f})")
        else:
            print(f"[{elapsed:05.1f}s] IDLE: PAN {servo_x:6.1f}, TILT {servo_y:6.1f}")

        last_pan, last_tilt = servo_x, servo_y
        time.sleep(0.2)

    if pause_durations:
        avg_pause = sum(pause_durations) / len(pause_durations)
        min_pause = min(pause_durations)
        max_pause = max(pause_durations)
        print(f"\nPause Analysis:")
        print(f"✓ Average pause: {avg_pause:.1f}s")
        print(f"✓ Range: {min_pause:.1f}s to {max_pause:.1f}s")
        print(f"✓ Total movements: {movement_count}")
        print(f"Expected: Mostly 1-4s pauses with occasional 6-10s pauses")


def test_movement_responsiveness():
    """Test the responsiveness of movement with new easing factors"""
    print("\n=== TESTING: Movement Responsiveness ===")
    print("Testing faster easing factors for more dynamic response")
    print("-" * 60)

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Test sudden face position changes
    positions = [
        (100, 100, "Top-left"),
        (540, 100, "Top-right"),
        (540, 380, "Bottom-right"),
        (100, 380, "Bottom-left"),
        (320, 240, "Center"),
    ]

    for i, (face_x, face_y, label) in enumerate(positions):
        print(f"\n--- {label} Position Test ---")
        face_box = (face_x - 25, face_y - 25, face_x + 25, face_y + 25)

        # Show convergence to target
        for step in range(10):
            person_present, servo_x, servo_y = update_gaze(dummy_frame, face_box)
            print(f"Step {step:2d}: PAN {servo_x:3d}°, TILT {servo_y:3d}°")
            time.sleep(0.05)  # Fast updates to show responsiveness

    print("\nLook for:")
    print("✓ Quick convergence to target positions")
    print("✓ Smooth approach without oscillation")
    print("✓ Reaching extreme angles (near 45°/135° pan, 50°/130° tilt)")


def main():
    print("Enhanced Gaze System Test Suite")
    print("=" * 60)
    print("Testing wider range and faster, more dynamic movement")

    # Test 1: Expanded range tracking
    test_expanded_range_tracking(12)

    # Test 2: Dynamic idle movement
    test_dynamic_idle_movement(15)

    # Test 3: Movement responsiveness
    test_movement_responsiveness()

    print("\n" + "=" * 60)
    print("ENHANCED TEST SUITE COMPLETED")
    print("\nKey enhancements to verify:")
    print("1. Much wider tilt range (±40° vs ±30°)")
    print("2. Faster, more dynamic movement patterns")
    print("3. Shorter base pauses (1-4s) with occasional longer ones")
    print("4. More responsive face tracking")


if __name__ == "__main__":
    main()