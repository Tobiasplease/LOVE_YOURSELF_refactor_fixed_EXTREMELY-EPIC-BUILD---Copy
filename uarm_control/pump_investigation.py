#!/usr/bin/env python3
"""
Investigate suction pump control methods for uArm Swift Pro
"""

import time
from uarm_controller import UarmController


def investigate_pump_methods(controller):
    """Investigate all available pump control methods"""
    print("=== Investigating Pump Control Methods ===")

    robot = controller.robot

    # Get all available methods
    methods = [method for method in dir(robot) if not method.startswith('_')]

    # Look for pump, suction, vacuum related methods
    pump_methods = [m for m in methods if any(keyword in m.lower() for keyword in
                   ['pump', 'suction', 'vacuum', 'pneumatic', 'air'])]

    print("Potential pump control methods:")
    for method in sorted(pump_methods):
        print(f"  {method}")

    # Test pump-related methods
    print("\n=== Testing Pump Methods ===")

    # Method 1: set_pump (what we've been using)
    print("Method 1: set_pump")
    try:
        print("  Activating pump with set_pump(True)...")
        result = robot.set_pump(True)
        print(f"  Result: {result}")
        time.sleep(2)

        print("  Deactivating pump with set_pump(False)...")
        result = robot.set_pump(False)
        print(f"  Result: {result}")
    except Exception as e:
        print(f"  set_pump error: {e}")

    # Method 2: Try different pump control methods
    pump_test_methods = [
        ('set_vacuum', [True]),
        ('set_vacuum', [False]),
        ('set_suction', [True]),
        ('set_suction', [False]),
        ('set_gripper', [True]),   # Sometimes the suction is controlled via gripper
        ('set_gripper', [False]),
    ]

    for method_name, args in pump_test_methods:
        if hasattr(robot, method_name):
            try:
                print(f"\nTesting {method_name}({args})...")
                method = getattr(robot, method_name)
                result = method(*args)
                print(f"  Result: {result}")
                time.sleep(1)
            except Exception as e:
                print(f"  {method_name} error: {e}")
        else:
            print(f"\n{method_name}: Not available")


def test_raw_pump_commands(controller):
    """Test raw G-code commands for pump control"""
    print("\n=== Testing Raw Pump Commands ===")

    robot = controller.robot

    # Common G-code commands for pump/suction control
    pump_commands = [
        "M2231 V1",    # Enable pump
        "M2231 V0",    # Disable pump
        "M2232 V1000", # Set pump with value
        "M2232 V0",    # Set pump to 0
        "M106 S255",   # Fan/pump on (3D printer G-code)
        "M106 S0",     # Fan/pump off
        "M107",        # Fan/pump off
        "M3 S1000",    # Spindle on (sometimes used for pumps)
        "M5",          # Spindle off
    ]

    if hasattr(robot, 'send_cmd_sync'):
        for cmd in pump_commands:
            try:
                print(f"Sending command: {cmd}")
                result = robot.send_cmd_sync(cmd)
                print(f"  Result: {result}")
                time.sleep(2)  # Give time to hear the pump
            except Exception as e:
                print(f"  Command '{cmd}' error: {e}")
    else:
        print("send_cmd_sync not available")


def test_digital_output_pump(controller):
    """Test if pump is controlled via digital output pins"""
    print("\n=== Testing Digital Output Pump Control ===")

    robot = controller.robot

    # Try controlling pump via digital output pins
    pump_pins = [0, 1, 2, 3, 6, 7, 8, 9, 13]  # Common pump control pins

    for pin in pump_pins:
        try:
            print(f"Testing digital output pin {pin}...")

            # Set pin HIGH
            if hasattr(robot, 'set_digital'):
                result = robot.set_digital(pin, 1)
                print(f"  Pin {pin} HIGH: {result}")
                time.sleep(2)

                # Set pin LOW
                result = robot.set_digital(pin, 0)
                print(f"  Pin {pin} LOW: {result}")
                time.sleep(1)

        except Exception as e:
            print(f"  Pin {pin} error: {e}")


def test_alternative_pump_methods(controller):
    """Test alternative pump activation methods"""
    print("\n=== Testing Alternative Pump Methods ===")

    robot = controller.robot

    # Alternative methods that might control the pump
    alt_methods = [
        ('set_servo_angle', [3, 90]),   # Servo 3 might control pump
        ('set_servo_angle', [3, 0]),
        ('set_wrist', [90]),            # Wrist control might include pump
        ('set_wrist', [0]),
    ]

    for method_name, args in alt_methods:
        if hasattr(robot, method_name):
            try:
                print(f"Testing {method_name}({args})...")
                method = getattr(robot, method_name)
                result = method(*args)
                print(f"  Result: {result}")
                time.sleep(2)
            except Exception as e:
                print(f"  {method_name} error: {e}")
        else:
            print(f"{method_name}: Not available")


def main():
    print("uArm Swift Pro Pump Investigation")
    print("=================================")

    controller = UarmController()

    if not controller.is_connected():
        print("ERROR: Failed to connect to uArm")
        return

    print("Connected successfully!\n")

    try:
        # Run all investigations
        investigate_pump_methods(controller)
        test_raw_pump_commands(controller)
        test_digital_output_pump(controller)
        test_alternative_pump_methods(controller)

        print("\n" + "="*60)
        print("PUMP CONTROL INVESTIGATION COMPLETE")
        print("="*60)
        print("Listen for pump activation sounds during the tests above.")
        print("If you heard the pump activate, note which method worked.")

    except Exception as e:
        print(f"Investigation failed: {e}")
    finally:
        controller.disconnect()


if __name__ == "__main__":
    main()