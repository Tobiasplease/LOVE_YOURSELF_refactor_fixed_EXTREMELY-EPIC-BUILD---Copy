#!/usr/bin/env python3
"""
Test uArm height limits and attempt to increase workspace
"""

import sys

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

        # Test current position limits
        test_positions = [
            [200, 0, 150],  # Default safe position
            [200, 0, 174],  # Current observed limit
            [200, 0, 180],  # Try slightly higher
            [200, 0, 200],  # Try much higher
        ]

        print("\n=== Testing Position Limits ===")
        for pos in test_positions:
            try:
                is_valid = swift.check_pos_is_limit(pos)
                status = "✅ VALID" if not is_valid else "❌ LIMIT"
                print(f"Position {pos}: {status}")
            except Exception as e:
                print(f"Position {pos}: ❌ ERROR - {e}")

        # Try to increase height offset
        print(f"\n=== Attempting to Increase Height Workspace ===")
        try:
            # Try setting a positive height offset to allow higher Z positions
            print("Setting height offset to +30mm...")
            result = swift.set_height_offset(30)
            print(f"Height offset result: {result}")

            # Test positions again with new offset
            print("\n=== Re-testing Position Limits After Offset ===")
            for pos in test_positions:
                try:
                    is_valid = swift.check_pos_is_limit(pos)
                    status = "✅ VALID" if not is_valid else "❌ LIMIT"
                    print(f"Position {pos}: {status}")
                except Exception as e:
                    print(f"Position {pos}: ❌ ERROR - {e}")

        except Exception as e:
            print(f"Failed to set height offset: {e}")

        # Get current position for reference
        try:
            current_pos = swift.get_position()
            print(f"\nCurrent position: {current_pos}")
        except Exception as e:
            print(f"Failed to get current position: {e}")

    except Exception as e:
        print(f"Error during testing: {e}")

    finally:
        swift.disconnect()
        print("Disconnected from uArm")


if __name__ == "__main__":
    main()
