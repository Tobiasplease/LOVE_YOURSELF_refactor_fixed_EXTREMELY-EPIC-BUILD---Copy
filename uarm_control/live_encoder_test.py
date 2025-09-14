#!/usr/bin/env python3
"""
Test live encoder reading using servo angles during manual movement
"""

import time
from uarm_controller import UarmController


def test_live_encoder_reading():
    """Test reading live encoder positions during manual movement"""
    print("Testing Live Encoder Reading During Manual Movement")
    print("==================================================")

    controller = UarmController(auto_home=False)

    if not controller.is_connected():
        print("ERROR: Failed to connect to uArm")
        return

    print("✅ Connected to uArm")

    try:
        # Release motors for manual movement
        print("\n🔧 Releasing motors for manual movement...")
        controller.release_motors()
        time.sleep(1)

        print("\n🤖 Testing LIVE encoder reading method:")
        print("Move the arm manually and watch the readings change!")
        print("Press Ctrl+C to stop\n")

        last_angles = None
        last_position = None
        reading_count = 0

        start_time = time.time()

        while True:
            try:
                # Method 1: Read live servo angles
                angles = controller.robot.get_servo_angle()

                # Method 2: Convert angles to Cartesian coordinates
                if angles and len(angles) >= 3:
                    try:
                        # Convert angles to position using built-in method
                        position = controller.robot.angles_to_coordinate(*angles)

                        reading_count += 1
                        current_time = time.time() - start_time

                        # Check if values changed
                        angles_changed = last_angles is None or angles != last_angles
                        position_changed = last_position is None or (
                            position and last_position and
                            (abs(position[0] - last_position[0]) > 0.1 or
                             abs(position[1] - last_position[1]) > 0.1 or
                             abs(position[2] - last_position[2]) > 0.1)
                        )

                        if angles_changed or position_changed or reading_count % 20 == 0:
                            print(f"[{current_time:6.1f}s] Angles: [{angles[0]:6.1f}°, {angles[1]:6.1f}°, {angles[2]:6.1f}°] → Position: [{position[0]:6.1f}, {position[1]:6.1f}, {position[2]:6.1f}]")

                            if position_changed:
                                print("  ✅ MOVEMENT DETECTED!")

                        last_angles = angles.copy() if hasattr(angles, 'copy') else list(angles)
                        last_position = position.copy() if hasattr(position, 'copy') else list(position) if position else None

                    except Exception as e:
                        print(f"  Coordinate conversion error: {e}")

                time.sleep(0.1)  # 10Hz

            except KeyboardInterrupt:
                print(f"\n⏹️ Stopped after {reading_count} readings")
                break
            except Exception as e:
                print(f"Reading error: {e}")
                time.sleep(0.1)

        # Summary
        elapsed_time = time.time() - start_time
        avg_hz = reading_count / elapsed_time if elapsed_time > 0 else 0

        print(f"\n📊 RESULTS:")
        print(f"  Duration: {elapsed_time:.1f}s")
        print(f"  Readings: {reading_count}")
        print(f"  Average rate: {avg_hz:.1f} Hz")
        print(f"  Method: get_servo_angle() + angles_to_coordinate()")

        if avg_hz >= 50:
            print("  ✅ Excellent sampling rate for high-quality recording!")
        elif avg_hz >= 20:
            print("  ✅ Good sampling rate for recording")
        else:
            print("  ⚠️ Low sampling rate - may affect recording quality")

    except Exception as e:
        print(f"Test error: {e}")

    finally:
        # Re-enable motors
        try:
            controller.enable_motors()
            time.sleep(0.5)
        except:
            pass

        controller.disconnect()


if __name__ == "__main__":
    test_live_encoder_reading()