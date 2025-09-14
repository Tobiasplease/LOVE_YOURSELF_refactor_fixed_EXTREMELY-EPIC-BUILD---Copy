#!/usr/bin/env python3
"""
Test the new callback-based button system that overrides firmware functions
"""

import time
from uarm_controller import UarmController


def test_callback_button_system():
    print("Testing uArm Callback-Based Button System")
    print("=========================================")

    controller = UarmController()

    if not controller.is_connected():
        print("ERROR: Failed to connect to uArm")
        return False

    print("✅ Connected successfully!")
    print(f"✅ Callbacks registered: {controller.callbacks_registered}")

    print("\nTesting button detection...")
    print("Press the MENU and PLAY buttons on the base to test.")
    print("The callbacks should override the firmware functions.")
    print("Press Ctrl+C to stop.\n")

    try:
        last_states = {"menu": False, "play": False}
        event_count = 0

        while True:
            # Check current button states
            current_states = controller.get_button_state()

            # Check for state changes
            for button, pressed in current_states.items():
                if pressed != last_states[button]:
                    action = "PRESSED" if pressed else "RELEASED"
                    print(f"🔘 {button.upper()} button {action}")

            last_states = current_states.copy()

            # Check for new events
            events = controller.get_button_events()
            if events:
                for event in events:
                    event_count += 1
                    action = "PRESSED" if event["pressed"] else "RELEASED"
                    timestamp = time.strftime("%H:%M:%S", time.localtime(event["time"]))
                    print(f"📝 Event #{event_count}: {event['button'].upper()} {action} at {timestamp}")

            # Show periodic status
            if int(time.time()) % 5 == 0:
                menu_status = "PRESSED" if current_states["menu"] else "released"
                play_status = "PRESSED" if current_states["play"] else "released"
                print(f"📊 Status: Menu={menu_status}, Play={play_status}, Events={event_count}")

            time.sleep(0.1)

    except KeyboardInterrupt:
        print(f"\n✅ Button test completed. Total events captured: {event_count}")

    print("\nTesting wait_for_button_press method...")
    print("Press any button within 5 seconds:")

    button_pressed = controller.wait_for_button_press(timeout=5.0)
    if button_pressed:
        print(f"✅ Button press detected: {button_pressed.upper()}")
    else:
        print("⚠️ No button press detected within timeout")

    controller.disconnect()
    return True


def test_button_during_motion():
    """Test button detection while arm is moving"""
    print("\nTesting buttons during motion...")

    controller = UarmController()

    if not controller.is_connected():
        print("ERROR: Failed to connect to uArm")
        return False

    print("✅ Moving arm while monitoring buttons...")
    print("Press buttons during the movement to test callback system.")

    try:
        # Start a slow movement
        start_pos = controller.get_position()
        if start_pos:
            print(f"Starting position: {start_pos}")

            # Move in a pattern while monitoring buttons
            for i in range(5):
                # Move to different positions
                x_offset = 20 * (1 if i % 2 == 0 else -1)
                target_x = start_pos[0] + x_offset

                print(f"Moving to position {i+1}/5...")
                controller.move_to(target_x, start_pos[1], start_pos[2], speed=50)

                # Monitor buttons during movement
                start_time = time.time()
                while time.time() - start_time < 2.0:
                    events = controller.get_button_events()
                    for event in events:
                        if event["pressed"]:
                            print(f"🔘 {event['button'].upper()} pressed during movement!")
                    time.sleep(0.1)

            # Return to start position
            print("Returning to start position...")
            controller.move_to(start_pos[0], start_pos[1], start_pos[2], speed=100)

    except Exception as e:
        print(f"Motion test error: {e}")

    controller.disconnect()
    return True


def main():
    print("uArm Callback Button System Test")
    print("=================================\n")

    # Test 1: Basic button detection
    if not test_callback_button_system():
        print("❌ Basic button test failed")
        return

    # Test 2: Buttons during motion
    if not test_button_during_motion():
        print("❌ Motion button test failed")
        return

    print("\n🎉 ALL CALLBACK TESTS COMPLETED!")
    print("The button override system should now work during recording.")
    print("\nNext: Test the recording system with button-controlled suction.")


if __name__ == "__main__":
    main()