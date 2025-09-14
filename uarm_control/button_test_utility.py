#!/usr/bin/env python3
"""
uArm Swift Pro Button Testing Utility

This utility helps identify which digital pins correspond to the two base buttons:
- Menu button
- Play button

Run this script and press buttons to see which pins respond.
"""

import time
import threading
from uarm_controller import UarmController


class ButtonTester:
    def __init__(self):
        self.controller = UarmController()
        self.running = False
        self.button_states = {}

    def test_all_pins(self):
        """Test digital pins 0-15 to find active buttons"""
        if not self.controller.is_connected():
            print("ERROR: uArm not connected!")
            return False

        print("=== uArm Button Detection Test ===")
        print("Press the MENU button and PLAY button on the base while this runs...")
        print("Monitoring digital pins 0-15 for button activity...")
        print("Press Ctrl+C to stop\n")

        self.running = True

        try:
            while self.running:
                current_states = {}

                # Test digital pins 0-15
                for pin in range(16):
                    try:
                        state = self.controller.robot.get_digital(pin)
                        current_states[pin] = bool(state) if state is not None else False
                    except:
                        current_states[pin] = False

                # Check for state changes
                for pin, state in current_states.items():
                    if pin not in self.button_states:
                        self.button_states[pin] = state
                    elif self.button_states[pin] != state:
                        action = "PRESSED" if state else "RELEASED"
                        print(f"Pin {pin}: {action} (state: {state})")
                        self.button_states[pin] = state

                time.sleep(0.1)  # 10Hz polling

        except KeyboardInterrupt:
            print("\n=== Button Test Complete ===")
            self.running = False

        return True

    def test_keys_interface(self):
        """Test the keys interface method"""
        if not self.controller.is_connected():
            print("ERROR: uArm not connected!")
            return False

        print("\n=== Testing Keys Interface ===")
        print("Press buttons while this runs...")

        try:
            for i in range(50):  # 5 second test
                try:
                    keys = self.controller.robot.get_keys()
                    if keys:
                        print(f"Keys detected: {keys}")
                except Exception as e:
                    if i == 0:  # Only print error once
                        print(f"Keys interface failed: {e}")
                        break
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass

        print("Keys interface test complete\n")

    def test_specific_combinations(self):
        """Test specific pin combinations that might work"""
        if not self.controller.is_connected():
            print("ERROR: uArm not connected!")
            return False

        print("=== Testing Common Pin Combinations ===")

        # Common pin assignments for buttons
        test_pins = [
            (0, 1, "Standard assignment"),
            (2, 3, "Alternative assignment"),
            (7, 8, "Common Arduino pins"),
            (11, 12, "High pins"),
            (13, 14, "End pins")
        ]

        for pin1, pin2, desc in test_pins:
            try:
                state1 = self.controller.robot.get_digital(pin1)
                state2 = self.controller.robot.get_digital(pin2)
                print(f"{desc}: Pin {pin1}={state1}, Pin {pin2}={state2}")
            except Exception as e:
                print(f"{desc}: Failed to read pins {pin1}/{pin2}: {e}")

        print()

    def interactive_test(self):
        """Interactive test - user presses buttons and we detect"""
        if not self.controller.is_connected():
            print("ERROR: uArm not connected!")
            return False

        print("=== Interactive Button Test ===")
        input("Press ENTER, then immediately press and hold the MENU button...")

        # Baseline reading
        baseline = {}
        for pin in range(16):
            try:
                baseline[pin] = bool(self.controller.robot.get_digital(pin))
            except:
                baseline[pin] = False

        print("Baseline captured. Now press and hold the MENU button for 3 seconds...")
        time.sleep(1)

        # Test during button press
        menu_detected = []
        for i in range(30):  # 3 seconds
            for pin in range(16):
                try:
                    current = bool(self.controller.robot.get_digital(pin))
                    if current != baseline[pin]:
                        if pin not in menu_detected:
                            menu_detected.append(pin)
                            print(f"MENU button detected on pin {pin}!")
                except:
                    pass
            time.sleep(0.1)

        if not menu_detected:
            print("No MENU button activity detected")

        print("\nNow test the PLAY button...")
        input("Press ENTER, then immediately press and hold the PLAY button...")

        time.sleep(1)
        play_detected = []
        for i in range(30):  # 3 seconds
            for pin in range(16):
                try:
                    current = bool(self.controller.robot.get_digital(pin))
                    if current != baseline[pin] and pin not in menu_detected:
                        if pin not in play_detected:
                            play_detected.append(pin)
                            print(f"PLAY button detected on pin {pin}!")
                except:
                    pass
            time.sleep(0.1)

        if not play_detected:
            print("No PLAY button activity detected")

        return menu_detected, play_detected


def main():
    print("uArm Swift Pro Button Testing Utility")
    print("====================================\n")

    tester = ButtonTester()

    if not tester.controller.is_connected():
        print("Failed to connect to uArm. Please check:")
        print("1. uArm is powered on")
        print("2. USB cable is connected")
        print("3. Correct port in config")
        return

    print("Connected to uArm successfully!\n")

    while True:
        print("Choose a test:")
        print("1. Monitor all pins (live)")
        print("2. Test keys interface")
        print("3. Test common pin combinations")
        print("4. Interactive button identification")
        print("5. Exit")

        choice = input("\nEnter choice (1-5): ").strip()

        if choice == "1":
            tester.test_all_pins()
        elif choice == "2":
            tester.test_keys_interface()
        elif choice == "3":
            tester.test_specific_combinations()
        elif choice == "4":
            menu_pins, play_pins = tester.interactive_test()
            if menu_pins or play_pins:
                print(f"\nSUMMARY:")
                print(f"Menu button pins: {menu_pins}")
                print(f"Play button pins: {play_pins}")
        elif choice == "5":
            break
        else:
            print("Invalid choice")

        print("\n" + "="*50 + "\n")

    tester.controller.disconnect()
    print("Test complete!")


if __name__ == "__main__":
    main()