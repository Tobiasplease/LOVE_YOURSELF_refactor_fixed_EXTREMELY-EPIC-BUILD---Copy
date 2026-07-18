#!/usr/bin/env python3
"""
Test the refined face tracking with optimized velocity limits and reduced dead zone.
Focus on responsiveness while maintaining smooth, natural movement.
"""
import os
import sys
import time

import numpy as np

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# NOTE (July 10): superseded by test_face_tracking_stability.py — the tiny-dead-zone
# tuning this script was written to verify caused close-range bobbing and was replaced
# by a face-size-scaled dead zone (config FACE_TRACK_DEAD_ZONE). Safe to delete.
from vision.gaze import FACE_PAN_VELOCITY, FACE_TILT_VELOCITY, FACE_VELOCITY_SMOOTHING, update_gaze


def test_precision_tracking():
    """Test precise face tracking with very small dead zone"""
    print("=== TESTING: Precision Face Tracking ===")
    print(f"Face tracking velocities: PAN {FACE_PAN_VELOCITY}°, TILT {FACE_TILT_VELOCITY}°")
    print(f"Face smoothing: {FACE_VELOCITY_SMOOTHING} (less smoothing = more responsive)")
    print("-" * 60)

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Test small face movements to verify dead zone effectiveness
    base_x, base_y = 320, 240  # Center position
    small_movements = [
        (0, 0, "Center baseline"),
        (2, 0, "Tiny right movement (2px)"),
        (5, 0, "Small right movement (5px)"),
        (10, 3, "Medium movement (10,3px)"),
        (0, 0, "Return to center"),
        (-8, -5, "Small left-up movement"),
        (0, 0, "Return to center again"),
    ]

    print("Testing small movements for dead zone effectiveness...")

    for dx, dy, label in small_movements:
        face_x, face_y = base_x + dx, base_y + dy
        face_box = (face_x - 25, face_y - 25, face_x + 25, face_y + 25)

        print(f"\n--- {label} ---")
        print(f"Face at ({face_x},{face_y})")

        # Track convergence
        for step in range(8):
            person_present, servo_x, servo_y = update_gaze(dummy_frame, face_box)

            if step == 0:
                start_pan, start_tilt = servo_x, servo_y
            elif step > 0:
                pan_vel = servo_x - last_pan
                tilt_vel = servo_y - last_tilt
                print(f"  Step {step}: PAN {servo_x:6.1f}° (Δ{pan_vel:+5.1f}), TILT {servo_y:6.1f}° (Δ{tilt_vel:+5.1f})")

            last_pan, last_tilt = servo_x, servo_y
            time.sleep(0.05)

        total_pan_move = abs(servo_x - start_pan)
        total_tilt_move = abs(servo_y - start_tilt)
        print(f"  Total movement: PAN {total_pan_move:.1f}°, TILT {total_tilt_move:.1f}°")


def test_responsive_tracking():
    """Test responsiveness of face tracking across different movement speeds"""
    print("\n=== TESTING: Responsive Face Tracking ===")
    print("Testing quick face movements to verify responsiveness vs smoothness balance")
    print("-" * 60)

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Test different face movement patterns
    test_patterns = [
        ("Quick horizontal sweep", [(100, 240), (200, 240), (300, 240), (400, 240), (500, 240)]),
        ("Quick vertical sweep", [(320, 100), (320, 180), (320, 260), (320, 340), (320, 420)]),
        ("Diagonal movement", [(150, 150), (250, 200), (350, 250), (450, 300), (550, 350)]),
        ("Sudden position jump", [(100, 100), (500, 400)]),  # Test extreme jump
    ]

    for pattern_name, positions in test_patterns:
        print(f"\n--- {pattern_name} ---")

        max_velocities = {"pan": 0, "tilt": 0}

        for i, (face_x, face_y) in enumerate(positions):
            face_box = (face_x - 25, face_y - 25, face_x + 25, face_y + 25)

            print(f"Position {i+1}: Face at ({face_x},{face_y})")

            # Allow 5 steps to converge
            for step in range(5):
                person_present, servo_x, servo_y = update_gaze(dummy_frame, face_box)

                if step > 0:
                    pan_vel = abs(servo_x - last_pan)
                    tilt_vel = abs(servo_y - last_tilt)
                    max_velocities["pan"] = max(max_velocities["pan"], pan_vel)
                    max_velocities["tilt"] = max(max_velocities["tilt"], tilt_vel)

                    violation = ""
                    if pan_vel > FACE_PAN_VELOCITY:
                        violation += f" PAN_VIOLATION({pan_vel:.1f}°)"
                    if tilt_vel > FACE_TILT_VELOCITY:
                        violation += f" TILT_VIOLATION({tilt_vel:.1f}°)"

                    print(f"    Step {step}: PAN {servo_x:3d}° (v={pan_vel:4.1f}), TILT {servo_y:3d}° (v={tilt_vel:4.1f}){violation}")

                last_pan, last_tilt = servo_x, servo_y
                time.sleep(0.05)

            time.sleep(0.1)  # Brief pause between positions

        print(f"  Pattern summary:")
        print(f"    Max PAN velocity: {max_velocities['pan']:.1f}° (limit: {FACE_PAN_VELOCITY}°)")
        print(f"    Max TILT velocity: {max_velocities['tilt']:.1f}° (limit: {FACE_TILT_VELOCITY}°)")

        if max_velocities["pan"] <= FACE_PAN_VELOCITY and max_velocities["tilt"] <= FACE_TILT_VELOCITY:
            print(f"    ✓ PASSED: No velocity violations")
        else:
            print(f"    ✗ FAILED: Velocity violations detected")


def test_tracking_vs_idle_behavior():
    """Compare face tracking velocity vs idle movement velocity"""
    print("\n=== TESTING: Face Tracking vs Idle Movement ===")
    print("Verifying different velocity limits for face tracking vs idle movement")
    print("-" * 60)

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Test face tracking movement
    print("Phase 1: Face tracking movement")
    face_box = (500, 150, 550, 200)  # Top-right position
    face_velocities = []

    for step in range(10):
        person_present, servo_x, servo_y = update_gaze(dummy_frame, face_box)

        if step > 0:
            pan_vel = abs(servo_x - last_pan)
            tilt_vel = abs(servo_y - last_tilt)
            face_velocities.append((pan_vel, tilt_vel))

            if step <= 5:  # Show first few steps
                print(f"  Face tracking step {step}: PAN v={pan_vel:4.1f}°, TILT v={tilt_vel:4.1f}°")

        last_pan, last_tilt = servo_x, servo_y
        time.sleep(0.05)

    # Test idle movement
    print("\nPhase 2: Idle movement (no face)")
    idle_velocities = []

    for step in range(15):
        person_present, servo_x, servo_y = update_gaze(dummy_frame, None)

        if step > 0:
            pan_vel = abs(servo_x - last_pan)
            tilt_vel = abs(servo_y - last_tilt)
            if pan_vel > 0.1 or tilt_vel > 0.1:  # Only count actual movement
                idle_velocities.append((pan_vel, tilt_vel))

                if len(idle_velocities) <= 5:  # Show first few movements
                    print(f"  Idle movement {len(idle_velocities)}: PAN v={pan_vel:4.1f}°, TILT v={tilt_vel:4.1f}°")

        last_pan, last_tilt = servo_x, servo_y
        time.sleep(0.1)

    # Analysis
    if face_velocities:
        max_face_pan = max(v[0] for v in face_velocities)
        max_face_tilt = max(v[1] for v in face_velocities)
        print(f"\nFace tracking analysis:")
        print(f"  Max PAN velocity: {max_face_pan:.1f}° (limit: {FACE_PAN_VELOCITY}°)")
        print(f"  Max TILT velocity: {max_face_tilt:.1f}° (limit: {FACE_TILT_VELOCITY}°)")

    if idle_velocities:
        max_idle_pan = max(v[0] for v in idle_velocities)
        max_idle_tilt = max(v[1] for v in idle_velocities)
        print(f"\nIdle movement analysis:")
        print(f"  Max PAN velocity: {max_idle_pan:.1f}°")
        print(f"  Max TILT velocity: {max_idle_tilt:.1f}°")

        print(f"\nComparison:")
        print(f"  Face tracking is {'more' if max_face_pan > max_idle_pan else 'less'} aggressive in PAN")
        print(f"  Face tracking is {'more' if max_face_tilt > max_idle_tilt else 'less'} aggressive in TILT")


def main():
    print("Refined Face Tracking Test Suite")
    print("=" * 60)
    print("Testing optimized face tracking with reduced dead zone and balanced velocity")

    # Test 1: Precision tracking with small dead zone
    test_precision_tracking()

    # Test 2: Responsive tracking behavior
    test_responsive_tracking()

    # Test 3: Face tracking vs idle movement comparison
    test_tracking_vs_idle_behavior()

    print("\n" + "=" * 60)
    print("REFINED FACE TRACKING TEST COMPLETED")
    print("\nKey refinements verified:")
    print("1. Dead zone now face-size-scaled (config FACE_TRACK_DEAD_ZONE)")
    print(f"2. Optimized face tracking velocities (PAN {FACE_PAN_VELOCITY}°, TILT {FACE_TILT_VELOCITY}°)")
    print(f"3. Reduced smoothing ({FACE_VELOCITY_SMOOTHING}) for more responsiveness")
    print("4. Maintained velocity limits to prevent hard lock-ins")
    print("5. Balanced responsiveness vs smoothness for natural movement")


if __name__ == "__main__":
    main()
