#!/usr/bin/env python3
"""
Test manual position recording to see what the arm actually reports
when physically moved to different heights
"""

import sys
import time
try:
    from uarm.wrapper.swift_api import SwiftAPI
except Exception as e:
    print(f"Error: uArm SDK not available: {e}")
    sys.exit(1)

def main():
    swift = SwiftAPI(port="/dev/arduino_uarm")

    try:
        print("Connecting to uArm...")
        swift.connect()

        if not swift.connected:
            print("Failed to connect to uArm")
            return

        print("✅ Connected to uArm")

        # Set height offset again
        print("Setting height offset to +50mm...")
        result = swift.set_height_offset(50)
        print(f"Height offset result: {result}")

        # Detach servos for manual movement
        print("Detaching servos for manual movement...")
        swift.set_servo_detach()

        print("\n=== Manual Height Test ===")
        print("Please manually move the uArm to different heights.")
        print("Press Enter after positioning at each height level...")
        print("Type 'quit' to exit")

        positions = []

        while True:
            input_val = input(f"\nPosition {len(positions)+1} (or 'quit'): ").strip()
            if input_val.lower() == 'quit':
                break

            try:
                # Get current position
                pos = swift.get_position()
                if pos and len(pos) >= 3:
                    positions.append(pos)
                    print(f"Recorded position: X={pos[0]:.2f}, Y={pos[1]:.2f}, Z={pos[2]:.2f}")

                    # Also try to check if this position is valid
                    is_limit = swift.check_pos_is_limit(pos[:3])
                    limit_status = "❌ LIMIT" if is_limit else "✅ VALID"
                    print(f"Position validation: {limit_status}")
                else:
                    print("Failed to get position")
            except Exception as e:
                print(f"Error getting position: {e}")

        print(f"\n=== Summary of {len(positions)} recorded positions ===")
        if positions:
            max_z = max(pos[2] for pos in positions)
            min_z = min(pos[2] for pos in positions)
            print(f"Z-range: {min_z:.2f}mm to {max_z:.2f}mm")

            for i, pos in enumerate(positions, 1):
                print(f"{i}: X={pos[0]:.2f}, Y={pos[1]:.2f}, Z={pos[2]:.2f}")

    except Exception as e:
        print(f"Error during testing: {e}")

    finally:
        try:
            swift.set_servo_attach()
        except:
            pass
        swift.disconnect()
        print("Disconnected from uArm")

if __name__ == "__main__":
    main()