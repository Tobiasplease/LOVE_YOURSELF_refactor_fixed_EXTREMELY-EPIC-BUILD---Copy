#!/usr/bin/env python3
"""
Test encoder reading when motors are detached vs engaged
"""

import time
from uarm_controller import UarmController


def test_encoder_reading_with_motor_states():
    """Test if encoders can be read when motors are detached"""
    print("Testing Encoder Reading with Different Motor States")
    print("==================================================")

    controller = UarmController(auto_home=False)

    if not controller.is_connected():
        print("ERROR: Failed to connect to uArm")
        return

    print("✅ Connected to uArm")

    try:
        # Test 1: Read position with motors engaged
        print("\n=== Test 1: Motors ENGAGED ===")
        controller.enable_motors()
        time.sleep(1)

        print("Reading position with motors engaged...")
        engaged_positions = []
        for i in range(10):
            pos = controller.robot.get_position()
            if pos:
                engaged_positions.append(pos)
                print(f"  Reading {i+1}: X={pos[0]:.2f}, Y={pos[1]:.2f}, Z={pos[2]:.2f}")
            time.sleep(0.1)

        # Test 2: Read position with motors detached
        print("\n=== Test 2: Motors DETACHED ===")
        controller.release_motors()
        time.sleep(1)

        print("Reading position with motors detached...")
        print("🔧 Please move the arm manually during this test...")

        detached_positions = []
        for i in range(20):  # Longer test to allow manual movement
            pos = controller.robot.get_position()
            if pos:
                detached_positions.append(pos)
                print(f"  Reading {i+1}: X={pos[0]:.2f}, Y={pos[1]:.2f}, Z={pos[2]:.2f}")
            time.sleep(0.2)  # Slower to give time to move

        # Analysis
        print("\n=== ANALYSIS ===")

        if engaged_positions:
            print(f"Motors engaged - {len(engaged_positions)} readings:")
            eng_x = [p[0] for p in engaged_positions]
            eng_y = [p[1] for p in engaged_positions]
            eng_z = [p[2] for p in engaged_positions]

            print(f"  X range: {min(eng_x):.2f} to {max(eng_x):.2f} (variation: {max(eng_x)-min(eng_x):.2f})")
            print(f"  Y range: {min(eng_y):.2f} to {max(eng_y):.2f} (variation: {max(eng_y)-min(eng_y):.2f})")
            print(f"  Z range: {min(eng_z):.2f} to {max(eng_z):.2f} (variation: {max(eng_z)-min(eng_z):.2f})")

        if detached_positions:
            print(f"\nMotors detached - {len(detached_positions)} readings:")
            det_x = [p[0] for p in detached_positions]
            det_y = [p[1] for p in detached_positions]
            det_z = [p[2] for p in detached_positions]

            print(f"  X range: {min(det_x):.2f} to {max(det_x):.2f} (variation: {max(det_x)-min(det_x):.2f})")
            print(f"  Y range: {min(det_y):.2f} to {max(det_y):.2f} (variation: {max(det_y)-min(det_y):.2f})")
            print(f"  Z range: {min(det_z):.2f} to {max(det_z):.2f} (variation: {max(det_z)-min(det_z):.2f})")

            total_variation = (max(det_x)-min(det_x)) + (max(det_y)-min(det_y)) + (max(det_z)-min(det_z))

            if total_variation < 1.0:
                print("  ❌ PROBLEM: No significant position changes detected!")
                print("     Encoders may not be updating when motors are detached")
            else:
                print("  ✅ Good: Position changes detected during manual movement")

        # Test 3: Alternative position reading methods
        print("\n=== Test 3: Alternative Position Reading Methods ===")

        # Check for alternative encoder reading methods
        robot = controller.robot
        all_methods = [method for method in dir(robot) if not method.startswith('_')]

        encoder_methods = [m for m in all_methods if any(keyword in m.lower() for keyword in
                          ['encoder', 'angle', 'servo', 'joint'])]

        print("Available encoder/angle methods:")
        for method in sorted(encoder_methods):
            print(f"  {method}")

        # Test servo angle reading
        servo_methods = ['get_servo_angle', 'get_servo_angles', 'get_joint_angles']

        for method_name in servo_methods:
            if hasattr(robot, method_name):
                try:
                    method = getattr(robot, method_name)
                    print(f"\nTesting {method_name}:")

                    # Try different parameter combinations
                    for param in [None, 0, 1, 2, 3, [0, 1, 2, 3]]:
                        try:
                            if param is None:
                                result = method()
                            else:
                                result = method(param)
                            print(f"  {method_name}({param}): {result}")
                        except Exception as e:
                            if "required" not in str(e).lower():
                                print(f"  {method_name}({param}): Error - {e}")

                except Exception as e:
                    print(f"  {method_name}: Failed - {e}")

    except Exception as e:
        print(f"Test error: {e}")

    finally:
        # Re-enable motors before disconnecting
        try:
            controller.enable_motors()
            time.sleep(0.5)
        except:
            pass

        controller.disconnect()


if __name__ == "__main__":
    test_encoder_reading_with_motor_states()