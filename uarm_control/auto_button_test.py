#!/usr/bin/env python3
"""
Automatic uArm Button Detection Test

This script automatically tests all possible button detection methods
and prints results for menu and play buttons on the base.
"""

import time
from uarm_controller import UarmController


def test_digital_pins(controller):
    """Test all digital pins 0-15 for button states"""
    print("=== Testing Digital Pins 0-15 ===")

    results = {}
    for pin in range(16):
        try:
            state = controller.robot.get_digital(pin)
            results[pin] = bool(state) if state is not None else None
            print(f"Pin {pin:2d}: {results[pin]}")
        except Exception as e:
            results[pin] = f"Error: {e}"
            print(f"Pin {pin:2d}: {results[pin]}")

    return results


def test_keys_interface(controller):
    """Test the keys interface"""
    print("\n=== Testing Keys Interface ===")

    try:
        keys = controller.robot.get_keys()
        print(f"get_keys() returned: {keys}")
        if keys:
            print(f"Type: {type(keys)}, Length: {len(keys) if hasattr(keys, '__len__') else 'N/A'}")
            if hasattr(keys, '__iter__'):
                for i, key in enumerate(keys):
                    print(f"  Key {i}: {key}")
        return keys
    except Exception as e:
        print(f"get_keys() failed: {e}")
        return None


def test_analog_pins(controller):
    """Test analog pins that might be connected to buttons"""
    print("\n=== Testing Analog Pins 0-7 ===")

    results = {}
    for pin in range(8):
        try:
            value = controller.robot.get_analog(pin)
            results[pin] = value
            print(f"Analog {pin}: {value}")
        except Exception as e:
            results[pin] = f"Error: {e}"
            print(f"Analog {pin}: {results[pin]}")

    return results


def test_alternative_methods(controller):
    """Test alternative button reading methods"""
    print("\n=== Testing Alternative Methods ===")

    # Test various API methods that might read buttons
    methods = [
        ('get_servo_angle', [0]),
        ('get_servo_angle', [1]),
        ('get_servo_angle', [2]),
        ('get_servo_angle', [3]),
    ]

    for method_name, args in methods:
        try:
            method = getattr(controller.robot, method_name, None)
            if method:
                result = method(*args)
                print(f"{method_name}({args}): {result}")
            else:
                print(f"{method_name}: Method not available")
        except Exception as e:
            print(f"{method_name}({args}): Error - {e}")


def continuous_monitoring(controller, duration=10):
    """Monitor all pins for changes over time"""
    print(f"\n=== Continuous Monitoring for {duration} seconds ===")
    print("Press menu and play buttons now...")

    # Get baseline readings
    baseline_digital = {}
    for pin in range(16):
        try:
            baseline_digital[pin] = bool(controller.robot.get_digital(pin))
        except:
            baseline_digital[pin] = None

    print("Baseline captured, monitoring for changes...")

    changes_detected = []
    start_time = time.time()

    while time.time() - start_time < duration:
        # Check digital pins for changes
        for pin in range(16):
            try:
                current = bool(controller.robot.get_digital(pin))
                if baseline_digital[pin] is not None and current != baseline_digital[pin]:
                    change_info = f"Pin {pin}: {baseline_digital[pin]} -> {current}"
                    if change_info not in changes_detected:
                        changes_detected.append(change_info)
                        print(f"CHANGE DETECTED: {change_info}")
                    baseline_digital[pin] = current
            except:
                pass

        time.sleep(0.1)  # 10Hz polling

    print(f"Monitoring complete. Total changes detected: {len(changes_detected)}")
    return changes_detected


def main():
    print("uArm Swift Pro Automatic Button Detection")
    print("========================================")

    controller = UarmController()

    if not controller.is_connected():
        print("ERROR: Failed to connect to uArm")
        return

    print("Connected to uArm successfully!\n")

    # Test 1: Digital pins
    digital_results = test_digital_pins(controller)

    # Test 2: Keys interface
    keys_result = test_keys_interface(controller)

    # Test 3: Analog pins
    analog_results = test_analog_pins(controller)

    # Test 4: Alternative methods
    test_alternative_methods(controller)

    # Test 5: Continuous monitoring
    changes = continuous_monitoring(controller, duration=10)

    # Summary
    print("\n" + "="*50)
    print("DETECTION SUMMARY")
    print("="*50)

    # Show pins that returned True (likely button pressed states)
    active_pins = [pin for pin, state in digital_results.items() if state is True]
    if active_pins:
        print(f"Digital pins currently HIGH (possibly pressed): {active_pins}")
    else:
        print("No digital pins currently HIGH")

    # Show pins that returned False (likely button released states)
    inactive_pins = [pin for pin, state in digital_results.items() if state is False]
    if inactive_pins:
        print(f"Digital pins currently LOW: {inactive_pins}")

    # Show any detected changes
    if changes:
        print(f"Pins that changed during monitoring: {len(changes)} changes")
        for change in changes:
            print(f"  {change}")
    else:
        print("No pin changes detected during monitoring")

    print("\nRecommendations:")
    print("1. Try pressing buttons and note which pins are HIGH when pressed")
    print("2. Menu and Play buttons likely correspond to pins that change state")
    print("3. If no changes detected, buttons might use I2C or other interface")

    controller.disconnect()


if __name__ == "__main__":
    main()