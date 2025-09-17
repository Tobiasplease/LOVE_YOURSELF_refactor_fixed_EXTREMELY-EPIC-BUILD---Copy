#!/usr/bin/env python3
"""
Test script for the improved gaze system with organic movement and responsive tracking.
This tests the new Perlin noise-based idle movement and clean state transitions.
"""
import os
import sys
import time

import numpy as np

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vision.gaze import update_gaze


def test_organic_idle_movement(duration=30):
    """Test the new organic idle movement using Perlin noise"""
    print("=== TESTING: Organic Idle Movement (Perlin Noise) ===")
    print(f"Duration: {duration} seconds")
    print("Watch for:")
    print("- Smooth, organic curves instead of jerky movements")
    print("- Independent pan/tilt timing (not synchronized)")
    print("- Natural contemplative pauses")
    print("-" * 60)

    # Create dummy frame
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    start_time = time.time()
    last_pan, last_tilt = 90, 90

    while time.time() - start_time < duration:
        # No face detected - should use organic idle movement
        person_present, servo_x, servo_y = update_gaze(dummy_frame, None)

        elapsed = time.time() - start_time
        pan_delta = servo_x - last_pan
        tilt_delta = servo_y - last_tilt

        print(f"[{elapsed:05.1f}s] PAN: {servo_x:6.1f} (Δ{pan_delta:+4.1f}), TILT: {servo_y:6.1f} (Δ{tilt_delta:+4.1f})")

        last_pan, last_tilt = servo_x, servo_y
        time.sleep(0.2)  # Check every 200ms

    print("\nLook for:")
    print("✓ Smooth curves instead of sudden jumps")
    print("✓ Pan and tilt moving independently (different deltas)")
    print("✓ Natural, organic feeling movement patterns")


def test_responsive_face_tracking(duration=20):
    """Test responsive face tracking with expanded range"""
    print("\n=== TESTING: Responsive Face Tracking (Expanded Range) ===")
    print(f"Duration: {duration} seconds")
    print("Testing face movement across expanded range (55-125° pan, 60-120° tilt)")
    print("-" * 60)

    # Create dummy frame (640x480)
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    start_time = time.time()
    while time.time() - start_time < duration:
        elapsed = time.time() - start_time

        # Simulate face moving in figure-8 pattern across full frame
        t = (elapsed / duration) * 4 * np.pi  # Two complete cycles

        # Face moves across full frame width and height
        face_x = int(320 + 280 * np.sin(t))  # 40 to 600 (near edges)
        face_y = int(240 + 200 * np.sin(2 * t))  # 40 to 440 (near edges)

        # Create face bounding box
        face_box = (face_x - 50, face_y - 50, face_x + 50, face_y + 50)

        person_present, servo_x, servo_y = update_gaze(dummy_frame, face_box)

        print(f"[{elapsed:05.1f}s] Face: ({face_x:3d},{face_y:3d}) -> Servos: PAN {servo_x:3d}°, TILT {servo_y:3d}°")

        time.sleep(0.1)  # Fast updates for tracking responsiveness

    print("\nCheck that:")
    print("✓ PAN range reaches near 55° and 125° (expanded range)")
    print("✓ TILT range reaches near 60° and 120° (expanded range)")
    print("✓ Movement is responsive (follows face quickly)")


def test_state_transitions(cycles=3):
    """Test clean transitions between idle and tracking states"""
    print("\n=== TESTING: Clean State Transitions ===")
    print(f"Testing {cycles} idle->tracking->idle cycles")
    print("Watch for smooth transitions without jerky movements")
    print("-" * 60)

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    for cycle in range(cycles):
        print(f"\n--- Cycle {cycle + 1} ---")

        # Start with idle for 5 seconds
        print("Phase: IDLE (5s)")
        start_time = time.time()
        while time.time() - start_time < 5:
            person_present, servo_x, servo_y = update_gaze(dummy_frame, None)
            elapsed = time.time() - start_time
            print(f"  [{elapsed:4.1f}s] IDLE: PAN {servo_x:3d}°, TILT {servo_y:3d}°")
            time.sleep(0.5)

        # Sudden face appearance - test transition
        print("Phase: FACE APPEARS -> TRACKING (3s)")
        face_box = (300, 200, 400, 300)  # Center-right face
        start_time = time.time()
        while time.time() - start_time < 3:
            person_present, servo_x, servo_y = update_gaze(dummy_frame, face_box)
            elapsed = time.time() - start_time
            print(f"  [{elapsed:4.1f}s] TRACK: PAN {servo_x:3d}°, TILT {servo_y:3d}°")
            time.sleep(0.3)

        # Face disappears - test return to idle
        print("Phase: FACE DISAPPEARS -> IDLE (3s)")
        start_time = time.time()
        while time.time() - start_time < 3:
            person_present, servo_x, servo_y = update_gaze(dummy_frame, None)
            elapsed = time.time() - start_time
            print(f"  [{elapsed:4.1f}s] IDLE: PAN {servo_x:3d}°, TILT {servo_y:3d}°")
            time.sleep(0.3)

    print("\nTransitions should be:")
    print("✓ Smooth without sudden position jumps")
    print("✓ Quick response when face appears")
    print("✓ Gradual return to organic movement when face disappears")


def main():
    print("Improved Gaze System Test Suite")
    print("=" * 60)
    print("Testing the new organic movement patterns and responsive tracking")

    # Test 1: Organic idle movement
    test_organic_idle_movement(20)

    # Test 2: Responsive face tracking with expanded range
    test_responsive_face_tracking(15)

    # Test 3: Clean state transitions
    test_state_transitions(2)

    print("\n" + "=" * 60)
    print("TEST SUITE COMPLETED")
    print("\nKey improvements to verify:")
    print("1. Idle movement feels organic and contemplative (not jerky)")
    print("2. Pan and tilt move independently during idle")
    print("3. Face tracking is responsive across wider range")
    print("4. State transitions are smooth without artifacts")
    print("5. Overall movement feels natural and lifelike")


if __name__ == "__main__":
    main()