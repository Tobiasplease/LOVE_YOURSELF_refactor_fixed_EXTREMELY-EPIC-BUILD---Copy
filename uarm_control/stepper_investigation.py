#!/usr/bin/env python3
"""
Investigate stepper motor and encoder specific commands
"""

from uarm_controller import UarmController


def investigate_stepper_commands():
    """Look for stepper and encoder specific methods"""
    print("Investigating Stepper Motor and Encoder Commands")
    print("===============================================")

    controller = UarmController(auto_home=False)

    if not controller.is_connected():
        print("ERROR: Failed to connect to uArm")
        return

    robot = controller.robot
    print("✅ Connected to uArm")

    # Get all available methods
    all_methods = [method for method in dir(robot) if not method.startswith('_')]

    # Look for stepper, encoder, step, motor related methods
    stepper_methods = [m for m in all_methods if any(keyword in m.lower() for keyword in
                      ['step', 'motor', 'encoder', 'position', 'angle', 'count'])]

    print(f"\n📋 Found {len(stepper_methods)} stepper/encoder related methods:")
    for method in sorted(stepper_methods):
        print(f"  {method}")

    # Test raw G-code commands for stepper/encoder access
    print(f"\n🔍 Testing Raw G-code Commands for Stepper Access:")

    stepper_commands = [
        "M2013",      # Get stepper positions
        "M2014",      # Get encoder positions
        "M2015",      # Read encoder counts
        "M92",        # Get steps per unit
        "M114",       # Get current position (standard G-code)
        "M119",       # Get endstop states
        "G28",        # Home (to test if this updates positions)
        "M17",        # Enable steppers
        "M18",        # Disable steppers
        "M84",        # Disable steppers (alternative)
        "M3006 N0",   # Read encoder 0
        "M3006 N1",   # Read encoder 1
        "M3006 N2",   # Read encoder 2
        "M2231 V0",   # Query encoder mode
    ]

    if hasattr(robot, 'send_cmd_sync'):
        for cmd in stepper_commands:
            try:
                print(f"\nSending: {cmd}")
                result = robot.send_cmd_sync(cmd)
                print(f"  Result: {result}")
                if result and result != ['OK'] and result != 'OK':
                    print(f"  ✅ Got data: {result}")
            except Exception as e:
                print(f"  Error: {e}")

    # Test with motors released vs engaged
    print(f"\n🔧 Testing Encoder Reading with Different Motor States:")

    print("\n--- Motors ENGAGED ---")
    controller.enable_motors()

    # Try different position reading methods
    position_methods = [
        ('get_position', []),
        ('get_servo_angle', []),
        ('get_servo_angle', [0]),
        ('get_servo_angle', [1]),
        ('get_servo_angle', [2]),
    ]

    for method_name, args in position_methods:
        if hasattr(robot, method_name):
            try:
                method = getattr(robot, method_name)
                if args:
                    result = method(*args)
                else:
                    result = method()
                print(f"  {method_name}({args}): {result}")
            except Exception as e:
                print(f"  {method_name}({args}): Error - {e}")

    print("\n--- Motors RELEASED ---")
    controller.release_motors()

    for method_name, args in position_methods:
        if hasattr(robot, method_name):
            try:
                method = getattr(robot, method_name)
                if args:
                    result = method(*args)
                else:
                    result = method()
                print(f"  {method_name}({args}): {result}")
            except Exception as e:
                print(f"  {method_name}({args}): Error - {e}")

    # Test raw encoder reading commands when motors are released
    print(f"\n--- Raw Encoder Commands (Motors Released) ---")
    encoder_commands = [
        "M114",       # Current position
        "M2013",      # Stepper positions
        "M2014",      # Encoder positions
        "M3006 N0",   # Encoder 0
        "M3006 N1",   # Encoder 1
        "M3006 N2",   # Encoder 2
    ]

    for cmd in encoder_commands:
        try:
            result = robot.send_cmd_sync(cmd)
            if result and result != ['OK']:
                print(f"  {cmd}: {result}")
        except Exception as e:
            print(f"  {cmd}: Error - {e}")

    controller.disconnect()


if __name__ == "__main__":
    investigate_stepper_commands()