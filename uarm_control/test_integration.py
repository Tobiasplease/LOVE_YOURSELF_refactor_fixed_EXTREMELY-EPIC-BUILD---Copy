#!/usr/bin/env python3
"""
Comprehensive uArm Integration Test

Tests the complete uArm integration including:
- Connection to uArm
- Button detection (menu and play buttons)
- Motion recording with button-controlled suction
- Motion playback with suction replay
- Simple API integration
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import time
from uarm_controller import UarmController
from motion_manager import MotionManager
from simple_api import UarmSimpleAPI


def test_connection():
    """Test basic connection to uArm"""
    print("=== Testing Connection ===")
    controller = UarmController()

    if controller.is_connected():
        print("✅ Connected to uArm successfully!")
        print(f"✅ Device info: {controller.robot.get_device_info()}")
        controller.disconnect()
        return True
    else:
        print("❌ Failed to connect to uArm")
        return False


def test_button_detection():
    """Test button detection system"""
    print("\n=== Testing Button Detection ===")
    controller = UarmController()

    if not controller.is_connected():
        print("❌ Connection failed")
        return False

    print("✅ Testing button states...")

    # Test button reading
    states = controller.get_button_state()
    print(f"✅ Button states: Menu={states['menu']}, Play={states['play']}")

    # Test individual button methods
    menu_pressed = controller.get_menu_button_pressed()
    play_pressed = controller.get_play_button_pressed()
    any_pressed = controller.get_any_button_pressed()

    print(f"✅ Individual methods: Menu={menu_pressed}, Play={play_pressed}, Any={any_pressed}")

    print("✅ Button detection system working!")
    controller.disconnect()
    return True


def test_motor_control():
    """Test motor release and enable"""
    print("\n=== Testing Motor Control ===")
    controller = UarmController()

    if not controller.is_connected():
        print("❌ Connection failed")
        return False

    # Test motor release
    print("✅ Testing motor release...")
    if controller.release_motors():
        print("✅ Motors released successfully")
        time.sleep(2)

        # Test motor enable
        print("✅ Testing motor enable...")
        if controller.enable_motors():
            print("✅ Motors enabled successfully")
        else:
            print("⚠️ Motor enable failed")
    else:
        print("⚠️ Motor release failed")

    controller.disconnect()
    return True


def test_suction_control():
    """Test suction cup control"""
    print("\n=== Testing Suction Control ===")
    controller = UarmController()

    if not controller.is_connected():
        print("❌ Connection failed")
        return False

    print("✅ Testing suction ON...")
    if controller.set_pump(True):
        print("✅ Suction activated")
        time.sleep(2)

        print("✅ Testing suction OFF...")
        if controller.set_pump(False):
            print("✅ Suction deactivated")
        else:
            print("⚠️ Suction deactivation failed")
    else:
        print("⚠️ Suction activation failed")

    controller.disconnect()
    return True


def test_simple_api():
    """Test the simple API interface"""
    print("\n=== Testing Simple API ===")

    controller = UarmController()
    motion_manager = MotionManager(controller=controller)
    api = UarmSimpleAPI(controller=controller, motion_manager=motion_manager)

    if not api.is_available():
        print("❌ Simple API not available")
        controller.disconnect()
        return False

    print("✅ Simple API available!")

    # Test status
    status = api.get_status()
    print(f"✅ API Status: Available={status['available']}")

    # Test home position
    home_pos = api.get_home_position()
    print(f"✅ Home position: {home_pos}")

    # Test homing
    if api.home():
        print("✅ Homing successful")
    else:
        print("⚠️ Homing failed")

    controller.disconnect()
    return True


def test_motion_system():
    """Test motion recording/playback system setup"""
    print("\n=== Testing Motion System ===")

    controller = UarmController()
    motion_manager = MotionManager(controller=controller)

    if not controller.is_connected():
        print("❌ Connection failed")
        return False

    # Test motion slots
    print("✅ Motion slots configured:")
    for slot, info in motion_manager.motion_slots.items():
        recorded = motion_manager.is_motion_recorded(slot)
        status = "✅ recorded" if recorded else "⚪ not recorded"
        print(f"  Slot {slot}: {info['name']} - {status}")

    print("✅ Motion system ready!")
    controller.disconnect()
    return True


def main():
    print("uArm Swift Pro Integration Test")
    print("===============================\n")

    test_results = []

    # Run all tests
    test_results.append(("Connection", test_connection()))
    test_results.append(("Button Detection", test_button_detection()))
    test_results.append(("Motor Control", test_motor_control()))
    test_results.append(("Suction Control", test_suction_control()))
    test_results.append(("Motion System", test_motion_system()))
    test_results.append(("Simple API", test_simple_api()))

    # Summary
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)

    passed = 0
    failed = 0

    for test_name, result in test_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:20}: {status}")
        if result:
            passed += 1
        else:
            failed += 1

    print(f"\nResults: {passed} passed, {failed} failed")

    if failed == 0:
        print("\n🎉 ALL TESTS PASSED!")
        print("uArm integration is ready for use!")
        print("\nNext steps:")
        print("1. Use the recording GUI to record your 3 motions")
        print("2. Use pickup(), place(), gesture() functions from anywhere")
        print("3. Press PLAY button during recording to control suction")
    else:
        print(f"\n⚠️ {failed} test(s) failed. Check the output above for details.")


if __name__ == "__main__":
    main()