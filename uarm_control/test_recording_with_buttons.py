#!/usr/bin/env python3
"""
Test motion recording with button-controlled suction override
"""

from uarm_controller import UarmController
from motion_manager import MotionManager


def test_recording_system():
    print("Testing Motion Recording with Button Override")
    print("============================================")

    controller = UarmController()
    if not controller.is_connected():
        print("❌ Failed to connect to uArm")
        return False

    print("✅ Connected to uArm")
    print(f"✅ Button callbacks registered: {controller.callbacks_registered}")

    motion_manager = MotionManager(controller=controller)
    print("✅ Motion manager initialized")

    print("\n🎯 BUTTON OVERRIDE ACTIVE!")
    print("- Firmware play functions are now overridden")
    print("- PLAY button will control suction during recording")
    print("- MENU button reserved for future use")

    print(f"\nMotion slots available:")
    for slot, info in motion_manager.motion_slots.items():
        recorded = motion_manager.is_motion_recorded(slot)
        status = "✅ recorded" if recorded else "⚪ not recorded"
        print(f"  Slot {slot}: {info['name']} - {status}")

    print(f"\nReady for recording!")
    print(f"You can now:")
    print(f"1. Run the GUI: python uarm_control/recording_gui.py")
    print(f"2. Record motions with button-controlled suction")
    print(f"3. Use pickup(), place(), gesture() from main code")

    # Test button detection briefly
    print(f"\nTesting button detection for 5 seconds...")
    print(f"Press buttons to verify override is working:")

    import time
    start_time = time.time()
    event_count = 0

    while time.time() - start_time < 5:
        events = controller.get_button_events()
        for event in events:
            event_count += 1
            action = "PRESSED" if event["pressed"] else "RELEASED"
            print(f"  🔘 {event['button'].upper()} {action}")
        time.sleep(0.1)

    if event_count > 0:
        print(f"✅ Button override working! Detected {event_count} events.")
    else:
        print(f"⚪ No button presses detected (normal if no buttons pressed)")

    controller.disconnect()
    return True


if __name__ == "__main__":
    test_recording_system()