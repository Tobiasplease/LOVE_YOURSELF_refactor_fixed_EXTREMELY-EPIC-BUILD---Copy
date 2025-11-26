#!/usr/bin/env python3
"""
Left Arm Servo Test Script
=========================
Tests the new left arm servo functionality by sending mood commands
to the hand controller and observing the autonomous movement.
"""

import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hand_control.hand_expression import HandExpressionController


def test_left_arm_servos():
    """Test left arm servo functionality with different moods."""
    print("🧪 Left Arm Servo Test Starting...")
    print("=" * 50)

    # Initialize hand controller (this should connect to the Arduino)
    print("🔌 Connecting to hand controller...")
    try:
        controller = HandExpressionController(port="/dev/arduino_lefthand", clean_output=False)
        if not controller.serial_connection:
            print("❌ Failed to connect to Arduino. Check:")
            print("   - Arduino is connected to /dev/arduino_lefthand")
            print("   - hand_controller.ino is uploaded")
            print("   - Serial port permissions")
            return False
        print("✅ Connected successfully!")

    except Exception as e:
        print(f"❌ Connection error: {e}")
        return False

    # Test different moods with explanations
    mood_tests = [
        ("calm_observant", "Very subtle movement, occasional pauses (1% pause chance, 20-45s)"),
        ("energized_engaged", "Mostly continuous gentle movement (0.2% pause chance, 5-15s)"),
        ("withdrawn_distant", "Minimal movement, frequent pauses (4% pause chance, 45-90s)"),
        ("quiet_detached", "Gentle movement with pauses (2% pause chance, 30-60s)"),
        ("alert_curious", "Steady gentle movement (0.5% pause chance, 10-25s)"),
    ]

    print("\n🎭 Testing different emotional states...")
    print("Watch the left arm servos (pins 4 & 5) for autonomous movement")
    print("Movement range: 87°-93° (6 degrees total - VERY subtle)")
    print("Speed: Consistent 400ms (very slow & gentle for mechanics)")
    print("\nBehavior: Very gentle breathing-like movement with random pauses")
    print("Each servo moves independently - should be barely noticeable")
    print("Initial startup: 5-10 second pause to prevent jolting")

    for i, (mood, description) in enumerate(mood_tests):
        print(f"\n--- Test {i+1}/5: {mood.upper()} ---")
        print(f"Expected: {description}")

        # Send mood command using existing set_mood method if available, or direct serial
        try:
            if hasattr(controller, "set_mood"):
                controller.set_mood(mood)
            else:
                # Send mood command directly via serial
                mood_command = f"MOOD,{mood}\n"
                controller.serial_connection.write(mood_command.encode())
                print(f"📤 Sent: {mood_command.strip()}")

        except Exception as e:
            print(f"❌ Failed to send mood command: {e}")
            continue

        # Wait and observe
        print(f"⏳ Observing for 15 seconds...")
        print("   (Left arm servos should adjust their timing)")

        for second in range(15):
            print(f"   {15-second}s remaining...", end="\r")
            time.sleep(1)
        print("   Complete!                    ")

    print(f"\n🎯 Test Summary:")
    print("- Left arm servos should be moving very subtly and autonomously")
    print("- Each mood produces different pause frequencies and durations")
    print("- Movement speed is always consistent (400ms) for mechanical safety")
    print("- Current mood: withdrawn_distant (4% pause chance, 45-90s pauses)")
    print("- Movement range strictly limited to 6 degrees (87°-93°)")
    print("- Movement should be barely noticeable - like gentle breathing")
    print("- No startup jolt - begins with 5-10s pause for settling")

    # Send a status command if the Arduino supports it
    try:
        controller.serial_connection.write(b"STATUS\n")
        print("\n📊 Sent STATUS command - check Arduino Serial Monitor for details")
    except:
        pass

    # Cleanup
    try:
        controller.cleanup()
        print("\n✅ Test completed successfully!")
        return True
    except Exception as e:
        print(f"⚠️ Cleanup warning: {e}")
        return True


def quick_mood_test(mood):
    """Quick test of a specific mood."""
    print(f"🚀 Quick test: {mood}")

    try:
        controller = HandExpressionController(port="/dev/arduino_lefthand", clean_output=False)
        if controller.serial_connection:
            mood_command = f"MOOD,{mood}\n"
            controller.serial_connection.write(mood_command.encode())
            print(f"✅ Sent mood: {mood}")
            time.sleep(1)
            controller.cleanup()
        else:
            print("❌ No connection")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Quick mood test mode
        mood = sys.argv[1]
        quick_mood_test(mood)
    else:
        # Full test mode
        print("Left Arm Servo Test")
        print("Usage:")
        print("  python test_left_arm_servos.py              # Full test")
        print("  python test_left_arm_servos.py calm_observant  # Quick mood test")
        print()

        response = input("Run full test? (y/N): ").strip().lower()
        if response in ["y", "yes"]:
            test_left_arm_servos()
        else:
            print("Test cancelled. Use quick mode:")
            print("  python test_left_arm_servos.py calm_observant")
