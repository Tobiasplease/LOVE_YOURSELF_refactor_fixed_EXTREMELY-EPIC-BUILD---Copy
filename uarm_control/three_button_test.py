#!/usr/bin/env python3
"""
Test for 3-button system: on/off, menu, play
"""

from uarm_controller import UarmController


def test_three_buttons():
    print("Testing 3-Button System (On/Off, Menu, Play)")
    print("===========================================")

    controller = UarmController()

    if not controller.is_connected():
        print("ERROR: Failed to connect to uArm")
        return False

    robot = controller.robot

    # Check what key-related methods are available
    methods = [method for method in dir(robot) if 'key' in method.lower()]
    print("Available key methods:")
    for method in sorted(methods):
        print(f"  {method}")

    # Check for key2 callback (third button)
    if hasattr(robot, 'register_key2_callback'):
        print("\n✅ Key2 callback available (third button detected)")

        def key2_callback(value):
            pressed = bool(value)
            print(f"🔘 ON/OFF button {'PRESSED' if pressed else 'RELEASED'}")

        try:
            robot.register_key2_callback(key2_callback)
            print("✅ Key2 (on/off) callback registered")
        except Exception as e:
            print(f"❌ Key2 callback registration failed: {e}")

    else:
        print("❌ Key2 callback not available")

    # Show current callback status
    print(f"\nCallback status:")
    print(f"  Callbacks registered: {controller.callbacks_registered}")

    controller.disconnect()
    return True


if __name__ == "__main__":
    test_three_buttons()