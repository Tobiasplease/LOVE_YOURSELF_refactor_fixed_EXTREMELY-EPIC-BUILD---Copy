#!/usr/bin/env python3
"""
Investigate firmware button override methods for uArm Swift Pro

The built-in firmware is intercepting button presses for its own play functions.
We need to find ways to:
1. Disable firmware button handlers
2. Access raw button states
3. Override button behavior during recording
"""

import time
from uarm_controller import UarmController


def investigate_firmware_methods(controller):
    """Investigate available methods to override firmware button behavior"""
    print("=== Investigating Firmware Override Methods ===")

    robot = controller.robot

    # Get all available methods
    methods = [method for method in dir(robot) if not method.startswith('_')]

    # Look for button, mode, or control related methods
    button_methods = [m for m in methods if any(keyword in m.lower() for keyword in
                     ['button', 'key', 'mode', 'control', 'disable', 'enable', 'override', 'raw'])]

    print("Potential button/control methods:")
    for method in sorted(button_methods):
        print(f"  {method}")

    # Test mode-related methods
    mode_methods = ['set_mode', 'get_mode', 'set_buzzer', 'flush_cmd']

    print("\n=== Testing Mode/Control Methods ===")
    for method_name in mode_methods:
        if hasattr(robot, method_name):
            try:
                method = getattr(robot, method_name)
                print(f"{method_name}: Available")

                # Try calling with different parameters
                if method_name == 'get_mode':
                    try:
                        result = method()
                        print(f"  Current mode: {result}")
                    except Exception as e:
                        print(f"  Error getting mode: {e}")

                elif method_name == 'set_mode':
                    try:
                        # Try different modes
                        for mode in [0, 1, 2, 3, 4, 5]:
                            try:
                                result = method(mode)
                                print(f"  set_mode({mode}): {result}")
                                time.sleep(0.5)
                            except Exception as e:
                                print(f"  set_mode({mode}): Error - {e}")
                    except Exception as e:
                        print(f"  set_mode general error: {e}")

                elif method_name == 'flush_cmd':
                    try:
                        result = method()
                        print(f"  flush_cmd(): {result}")
                    except Exception as e:
                        print(f"  flush_cmd error: {e}")

            except Exception as e:
                print(f"{method_name}: Error accessing - {e}")
        else:
            print(f"{method_name}: Not available")


def test_raw_commands(controller):
    """Try sending raw commands to disable button handlers"""
    print("\n=== Testing Raw Commands ===")

    robot = controller.robot

    # Common G-code commands to disable button functions
    raw_commands = [
        "M2019",  # Disable button
        "M2020",  # Enable button
        "M70",    # Disable button functions
        "M71",    # Enable button functions
        "M2231 V0",  # Set button mode to manual
        "M2231 V1",  # Set button mode to normal
        "M2240 V0",  # Disable key detection
        "M2240 V1",  # Enable key detection
    ]

    if hasattr(robot, 'send_cmd_sync'):
        for cmd in raw_commands:
            try:
                result = robot.send_cmd_sync(cmd)
                print(f"Command '{cmd}': {result}")
                time.sleep(0.5)
            except Exception as e:
                print(f"Command '{cmd}': Error - {e}")
    else:
        print("send_cmd_sync not available")


def test_button_override_modes(controller):
    """Test different approaches to override button behavior"""
    print("\n=== Testing Button Override Approaches ===")

    robot = controller.robot

    # Approach 1: Try to set robot to manual mode
    print("Approach 1: Manual mode")
    try:
        if hasattr(robot, 'set_mode'):
            robot.set_mode(0)  # Mode 0 might be manual
            print("  Set to manual mode")

            # Test button reading in manual mode
            for pin in [2, 3, 12, 13]:
                try:
                    state = robot.get_digital(pin)
                    print(f"  Pin {pin} in manual mode: {state}")
                except Exception as e:
                    print(f"  Pin {pin} error: {e}")

    except Exception as e:
        print(f"  Manual mode failed: {e}")

    # Approach 2: Try to disable servo mode
    print("\nApproach 2: Servo detach mode")
    try:
        robot.set_servo_detach()
        print("  Servos detached")

        # Test button reading with servos detached
        for pin in [2, 3, 12, 13]:
            try:
                state = robot.get_digital(pin)
                print(f"  Pin {pin} with servos detached: {state}")
            except Exception as e:
                print(f"  Pin {pin} error: {e}")

        # Re-attach servos
        robot.set_servo_attach()
        print("  Servos re-attached")

    except Exception as e:
        print(f"  Servo detach failed: {e}")

    # Approach 3: Try firmware commands
    print("\nApproach 3: Firmware override commands")
    if hasattr(robot, 'send_cmd_sync'):
        # Try to disable built-in button functions
        override_commands = [
            "M2019 S1",  # Disable button with parameter
            "M2240 V0",  # Disable key detection
        ]

        for cmd in override_commands:
            try:
                result = robot.send_cmd_sync(cmd)
                print(f"  Command '{cmd}': {result}")

                # Test reading after command
                for pin in [2, 3]:
                    try:
                        state = robot.get_digital(pin)
                        print(f"    Pin {pin} after '{cmd}': {state}")
                    except Exception as e:
                        print(f"    Pin {pin} error: {e}")

            except Exception as e:
                print(f"  Command '{cmd}' failed: {e}")


def main():
    print("uArm Firmware Button Override Investigation")
    print("==========================================")

    controller = UarmController()

    if not controller.is_connected():
        print("ERROR: Failed to connect to uArm")
        return

    print("Connected successfully!\n")

    try:
        # Run investigations
        investigate_firmware_methods(controller)
        test_raw_commands(controller)
        test_button_override_modes(controller)

        print("\n" + "="*60)
        print("RECOMMENDATIONS:")
        print("="*60)
        print("1. Try setting robot to manual mode during recording")
        print("2. Use servo detach mode to override firmware functions")
        print("3. Send raw M-codes to disable built-in button handlers")
        print("4. Consider intercepting button events at a higher level")

    except Exception as e:
        print(f"Investigation failed: {e}")
    finally:
        controller.disconnect()


if __name__ == "__main__":
    main()