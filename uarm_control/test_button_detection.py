#!/usr/bin/env python3
"""
Test the updated button detection for menu and play buttons
"""

import time
from uarm_controller import UarmController


def test_button_detection():
    print("Testing uArm Button Detection")
    print("============================")

    controller = UarmController()

    if not controller.is_connected():
        print("ERROR: Failed to connect to uArm")
        return

    print("Connected! Testing button detection...")
    print("Press the menu and play buttons to test detection.")
    print("Press Ctrl+C to stop.\n")

    try:
        last_states = {"menu": False, "play": False}

        while True:
            current_states = controller.get_button_state()

            # Check for state changes
            for button, pressed in current_states.items():
                if pressed != last_states[button]:
                    action = "PRESSED" if pressed else "RELEASED"
                    print(f"{button.upper()} button {action}")

            last_states = current_states.copy()

            # Show current status every 5 seconds
            if int(time.time()) % 5 == 0:
                menu_status = "PRESSED" if current_states["menu"] else "released"
                play_status = "PRESSED" if current_states["play"] else "released"
                print(f"Status: Menu={menu_status}, Play={play_status}")

            time.sleep(0.1)  # 10Hz polling

    except KeyboardInterrupt:
        print("\nButton test stopped.")

    controller.disconnect()
    print("Test complete!")


if __name__ == "__main__":
    test_button_detection()