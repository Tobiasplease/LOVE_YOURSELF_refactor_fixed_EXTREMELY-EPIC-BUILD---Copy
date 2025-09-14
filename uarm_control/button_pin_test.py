#!/usr/bin/env python3
"""
Focused Button Pin Test for uArm Swift Pro

Tests specific pins and methods that are most likely to be the menu and play buttons.
"""

import time
import sys
from uarm_controller import UarmController


def test_specific_button_methods(controller):
    """Test uArm-specific button reading methods"""
    print("=== Testing uArm-Specific Button Methods ===")

    # Try various button-related methods that might exist
    button_methods = [
        'get_key',
        'get_button',
        'get_buttons',
        'read_button',
        'read_buttons',
        'get_digital_input',
        'get_io',
        'get_is_moving',
        'get_device_info'
    ]

    for method_name in button_methods:
        try:
            method = getattr(controller.robot, method_name, None)
            if method:
                try:
                    result = method()
                    print(f"{method_name}(): {result}")
                except Exception as e:
                    print(f"{method_name}(): Error - {e}")

                # Try with different arguments
                for arg in [0, 1, 2]:
                    try:
                        result = method(arg)
                        print(f"{method_name}({arg}): {result}")
                    except:
                        pass
            else:
                print(f"{method_name}: Method not available")
        except Exception as e:
            print(f"{method_name}: Error accessing - {e}")


def monitor_pin_changes_with_prompts(controller):
    """Monitor pins with user prompts for button presses"""
    print("\n=== Monitoring with Button Press Prompts ===")
    print("This test will check pins before and after button presses")

    # Get baseline
    print("Getting baseline readings (no buttons pressed)...")
    baseline = {}
    for pin in range(16):
        try:
            baseline[pin] = bool(controller.robot.get_digital(pin))
        except:
            baseline[pin] = None

    print("Baseline captured. Now testing each button...")

    # Test menu button
    print("\n--- MENU BUTTON TEST ---")
    print("Please press and hold the MENU button now...")
    time.sleep(3)  # Give time to press

    menu_states = {}
    for pin in range(16):
        try:
            menu_states[pin] = bool(controller.robot.get_digital(pin))
        except:
            menu_states[pin] = None

    print("Menu button readings captured. Release the button.")
    time.sleep(2)

    # Test play button
    print("\n--- PLAY BUTTON TEST ---")
    print("Please press and hold the PLAY button now...")
    time.sleep(3)  # Give time to press

    play_states = {}
    for pin in range(16):
        try:
            play_states[pin] = bool(controller.robot.get_digital(pin))
        except:
            play_states[pin] = None

    print("Play button readings captured. Release the button.")
    time.sleep(2)

    # Analyze results
    print("\n--- ANALYSIS ---")

    menu_pins = []
    play_pins = []

    for pin in range(16):
        if baseline[pin] is not None:
            baseline_state = baseline[pin]
            menu_state = menu_states.get(pin)
            play_state = play_states.get(pin)

            if menu_state is not None and menu_state != baseline_state:
                menu_pins.append(pin)
                print(f"Pin {pin}: MENU button detected (baseline: {baseline_state}, menu: {menu_state})")

            if play_state is not None and play_state != baseline_state and pin not in menu_pins:
                play_pins.append(pin)
                print(f"Pin {pin}: PLAY button detected (baseline: {baseline_state}, play: {play_state})")

    if not menu_pins:
        print("No MENU button pins detected")
    if not play_pins:
        print("No PLAY button pins detected")

    return menu_pins, play_pins


def test_inverted_logic(controller):
    """Test if buttons use inverted logic (active LOW)"""
    print("\n=== Testing Inverted Logic (Active LOW) ===")

    # Look for pins that are normally HIGH and go LOW when pressed
    print("Pins currently HIGH (could be active LOW buttons):")
    high_pins = []

    for pin in range(16):
        try:
            state = controller.robot.get_digital(pin)
            if state:
                high_pins.append(pin)
                print(f"Pin {pin}: HIGH (could be released button)")
        except:
            pass

    if high_pins:
        print(f"Active LOW candidate pins: {high_pins}")
        print("These pins might go LOW when buttons are pressed")
    else:
        print("No HIGH pins found")


def test_common_button_pins(controller):
    """Test pins commonly used for buttons on Arduino-based devices"""
    print("\n=== Testing Common Button Pins ===")

    # Common button pins on Arduino/uArm devices
    common_pins = [2, 3, 4, 7, 8, 12, 13]

    print("Checking common button pins:")
    for pin in common_pins:
        try:
            state = controller.robot.get_digital(pin)
            print(f"Pin {pin}: {state} {'(HIGH - possibly active LOW button)' if state else '(LOW - possibly active HIGH button)'}")
        except Exception as e:
            print(f"Pin {pin}: Error - {e}")


def main():
    print("uArm Swift Pro Focused Button Pin Test")
    print("=====================================")

    controller = UarmController()

    if not controller.is_connected():
        print("ERROR: Failed to connect to uArm")
        return

    print("Connected to uArm successfully!\n")

    # Test 1: uArm-specific button methods
    test_specific_button_methods(controller)

    # Test 2: Common button pins
    test_common_button_pins(controller)

    # Test 3: Inverted logic check
    test_inverted_logic(controller)

    # Test 4: Monitor with prompts (but skip interactive part for automation)
    print("\n=== Automated Pin State Analysis ===")
    print("Current pin states that could be buttons:")

    current_states = {}
    for pin in range(16):
        try:
            state = controller.robot.get_digital(pin)
            current_states[pin] = state
            if state is not None:
                logic = "Active LOW (released)" if state else "Active HIGH (released)"
                print(f"Pin {pin}: {state} - {logic}")
        except Exception as e:
            print(f"Pin {pin}: Error - {e}")

    print("\n" + "="*50)
    print("RECOMMENDATIONS FOR BUTTON PINS:")
    print("="*50)

    # High pins (likely active LOW buttons when released)
    high_pins = [pin for pin, state in current_states.items() if state is True]
    if high_pins:
        print(f"Most likely MENU/PLAY button pins (active LOW): {high_pins}")
        print("These pins should go LOW (False) when buttons are pressed")

    # Low pins (likely active HIGH buttons when released)
    low_pins = [pin for pin, state in current_states.items() if state is False]
    if low_pins:
        print(f"Alternative button pins (active HIGH): {low_pins}")
        print("These pins should go HIGH (True) when buttons are pressed")

    # Suggest most likely candidates based on uArm documentation
    print(f"\nBased on uArm Swift Pro design:")
    print("- Menu button is likely on pins 2, 3, 12, or 13")
    print("- Play button is likely on pins 2, 3, 12, or 13")
    print("- Try pins with current HIGH states first (active LOW logic)")

    controller.disconnect()
    print("\nTest complete!")


if __name__ == "__main__":
    main()