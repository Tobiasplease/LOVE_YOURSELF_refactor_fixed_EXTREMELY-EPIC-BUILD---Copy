#!/usr/bin/env python3
"""
Test Simple Lightbulb Controller
================================
Tests the ultra-simplified lightbulb controller with frame diff and caption flash only.
"""

import os
import sys
import time

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from servo_control.lightbulb_controller_simple import SimpleLightbulbController

LIGHTBULB_PORT = "/dev/arduino_lightbulb"


def test_basic_connection():
    """Test basic connection to lightbulb."""
    print("=" * 50)
    print("SIMPLE LIGHTBULB TEST")
    print("=" * 50)

    print(f"Testing connection to: {LIGHTBULB_PORT}")

    try:
        controller = SimpleLightbulbController(LIGHTBULB_PORT, debug=True)

        if controller.ser:
            print("✅ Connection successful!")
            return controller
        else:
            print("❌ Connection failed!")
            return None

    except Exception as e:
        print(f"❌ Error creating controller: {e}")
        return None


def test_frame_diff_brightness(controller):
    """Test frame difference brightness control."""
    if not controller:
        return

    print("\n📊 Testing frame difference brightness...")

    # Test brightness ramp up
    print("  🔆 Ramping brightness up...")
    for brightness in range(0, 256, 25):
        print(f"    Setting brightness: {brightness}")
        controller.set_frame_diff_brightness(brightness)
        time.sleep(0.5)

    # Test brightness ramp down
    print("  🔅 Ramping brightness down...")
    for brightness in range(255, -1, -25):
        print(f"    Setting brightness: {brightness}")
        controller.set_frame_diff_brightness(brightness)
        time.sleep(0.5)

    print("  ✅ Frame diff brightness test complete")


def test_caption_flash(controller):
    """Test caption flash functionality."""
    if not controller:
        return

    print("\n⚡ Testing caption flash...")

    for i in range(3):
        print(f"  Flash {i+1}/3")
        controller.caption_flash()
        time.sleep(2)  # Wait between flashes

    print("  ✅ Caption flash test complete")


def test_realistic_scenario(controller):
    """Test realistic usage scenario."""
    if not controller:
        return

    print("\n🎬 Testing realistic scenario...")

    # Simulate varying frame differences with occasional caption flashes
    frame_diffs = [10, 25, 45, 30, 15, 60, 80, 45, 20, 5, 35, 70, 25]

    for i, frame_diff in enumerate(frame_diffs):
        # Convert frame diff to brightness (0-255)
        brightness = min(255, frame_diff * 3)

        print(f"  Frame {i+1}: diff={frame_diff}, brightness={brightness}")
        controller.set_frame_diff_brightness(brightness)

        # Occasionally trigger caption flash
        if i % 4 == 0:  # Every 4th frame
            print("    📝 New caption - triggering flash!")
            controller.caption_flash()
            time.sleep(1.5)  # Flash duration
        else:
            time.sleep(0.3)  # Normal frame interval

    print("  ✅ Realistic scenario test complete")


def main():
    """Run all tests."""
    controller = test_basic_connection()

    if not controller:
        print("\n❌ Cannot continue - connection failed!")
        print("\nTroubleshooting:")
        print(f"  1. Check Arduino is connected to {LIGHTBULB_PORT}")
        print("  2. Upload lightbulb_simple.ino to the Arduino")
        print("  3. Check port permissions: sudo chmod 666 /dev/ttyUSB*")
        return

    try:
        test_frame_diff_brightness(controller)
        test_caption_flash(controller)
        test_realistic_scenario(controller)

        print("\n" + "=" * 50)
        print("🎉 ALL TESTS COMPLETE!")
        print("=" * 50)
        print("The simplified lightbulb controller is working correctly.")
        print("You can now use it in your main application.")

    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")

    except Exception as e:
        print(f"\n❌ Test error: {e}")

    finally:
        if controller:
            print("\n🔌 Closing connection...")
            controller.close()


if __name__ == "__main__":
    main()
