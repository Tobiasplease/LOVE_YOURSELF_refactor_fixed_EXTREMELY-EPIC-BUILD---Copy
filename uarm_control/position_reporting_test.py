#!/usr/bin/env python3
"""
Test the position reporting system - this is likely the key to live position tracking
"""

import time
from uarm_controller import UarmController


def test_position_reporting_system():
    """Test the position reporting callback system"""
    print("Testing Position Reporting System")
    print("================================")

    controller = UarmController(auto_home=False)

    if not controller.is_connected():
        print("ERROR: Failed to connect to uArm")
        return

    robot = controller.robot
    print("✅ Connected to uArm")

    # Storage for received positions
    received_positions = []
    position_count = 0

    def position_callback(*args):
        """Callback function to receive position updates"""
        nonlocal position_count
        position_count += 1
        timestamp = time.time()

        print(f"[{position_count:3d}] Position callback: {args}")
        received_positions.append({
            "time": timestamp,
            "data": args,
            "count": position_count
        })

    try:
        print("\n🔧 Setting up position reporting...")

        # Enable position reporting
        if hasattr(robot, 'set_report_position'):
            try:
                result = robot.set_report_position(True)
                print(f"set_report_position(True): {result}")
            except Exception as e:
                print(f"set_report_position error: {e}")

        # Register position callback
        if hasattr(robot, 'register_report_position_callback'):
            try:
                result = robot.register_report_position_callback(position_callback)
                print(f"register_report_position_callback: {result}")
            except Exception as e:
                print(f"register_report_position_callback error: {e}")

        print("\n✅ Position reporting setup complete")

        # Test with motors engaged first
        print("\n=== TEST 1: Motors ENGAGED ===")
        controller.enable_motors()
        time.sleep(1)

        print("Monitoring for 5 seconds with motors engaged...")
        start_positions = len(received_positions)
        time.sleep(5)
        engaged_positions = len(received_positions) - start_positions
        print(f"Received {engaged_positions} position updates with motors engaged")

        # Test with motors released
        print("\n=== TEST 2: Motors RELEASED ===")
        controller.release_motors()
        time.sleep(1)

        print("Monitoring for 10 seconds with motors released...")
        print("🔧 MOVE THE ARM MANUALLY NOW!")

        start_positions = len(received_positions)
        for i in range(10):
            print(f"  {i+1}/10 seconds - positions received: {len(received_positions) - start_positions}")
            time.sleep(1)

        released_positions = len(received_positions) - start_positions
        print(f"\nReceived {released_positions} position updates with motors released")

        # Analysis
        print("\n📊 ANALYSIS:")
        if released_positions > 0:
            print(f"✅ SUCCESS: Position reporting works with motors released!")
            print(f"   Engaged: {engaged_positions} updates")
            print(f"   Released: {released_positions} updates")

            if len(received_positions) >= 2:
                first_pos = received_positions[0]
                last_pos = received_positions[-1]
                print(f"   First: {first_pos['data']}")
                print(f"   Last:  {last_pos['data']}")

        else:
            print(f"❌ Position reporting not working with motors released")

    except Exception as e:
        print(f"Test error: {e}")

    finally:
        # Cleanup
        print("\n🧹 Cleaning up position reporting...")

        if hasattr(robot, 'release_report_position_callback'):
            try:
                robot.release_report_position_callback()
                print("Position callback released")
            except Exception as e:
                print(f"Callback release error: {e}")

        if hasattr(robot, 'set_report_position'):
            try:
                robot.set_report_position(False)
                print("Position reporting disabled")
            except Exception as e:
                print(f"Position reporting disable error: {e}")

        controller.disconnect()

    return len(received_positions) > 0


if __name__ == "__main__":
    success = test_position_reporting_system()

    if success:
        print("\n🎉 POSITION REPORTING SYSTEM FOUND!")
        print("This is likely the key to implementing proper recording!")
    else:
        print("\n❌ Position reporting system not working")