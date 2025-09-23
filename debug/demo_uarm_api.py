#!/usr/bin/env python3
"""
Demo script showing how to use the uArm Simple API

This demonstrates the 3 utilitarian motions without any emotion logic.
Just simple function calls to trigger pre-recorded movements.
"""

import os
import sys
import time

# Add parent directory for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the simple API functions
from uarm_control.simple_api import gesture, home, is_available, pickup, place, status


def demo_api_usage():
    print("=== uArm Simple API Demo ===")
    print("This shows how to call the 3 utilitarian motions from anywhere in your code.\n")

    # Check if uArm is available
    print("1. Checking uArm availability...")
    if is_available():
        print("✅ uArm is connected and ready!")
    else:
        print("❌ uArm not available (expected without hardware)")
        print("   Connect hardware and run machine.py first to initialize\n")

    # Show status
    print("2. Getting uArm status...")
    uarm_status = status()
    print(f"   Status: {uarm_status}\n")

    # Demo the API calls (will fail without hardware, which is expected)
    print("3. API Function Examples:")

    print("\n   # Move to home position")
    print("   from uarm_control.simple_api import home")
    print("   home()")
    result = home()
    print(f"   → Result: {result}\n")

    print("   # Execute pickup motion")
    print("   from uarm_control.simple_api import pickup")
    print("   pickup()")
    result = pickup()
    print(f"   → Result: {result}\n")

    print("   # Execute place motion")
    print("   from uarm_control.simple_api import place")
    print("   place()")
    result = place()
    print(f"   → Result: {result}\n")

    print("   # Execute gesture motion")
    print("   from uarm_control.simple_api import gesture")
    print("   gesture()")
    result = gesture()
    print(f"   → Result: {result}\n")


def show_integration_examples():
    print("=== Integration Examples ===")
    print(
        """
How to use the uArm API in your main script:

# Example 1: Simple function calls
from uarm_control.simple_api import pickup, place, gesture

# Trigger motions from anywhere in your code
def handle_object_detected():
    if pickup():  # Returns True if successful
        print("Object picked up successfully!")
        time.sleep(2)
        place()

# Example 2: With error handling
def safe_pickup():
    if not is_available():
        print("uArm not ready")
        return False

    if pickup():
        print("Pickup completed")
        return True
    else:
        print("Pickup failed")
        return False

# Example 3: Status checking
def check_motions_ready():
    uarm_status = status()
    motions = uarm_status.get('motions_ready', {})

    if all(motions.values()):
        print("All 3 motions are recorded and ready!")
    else:
        print("Some motions need to be recorded first")

# Example 4: Integration in main processing loop
def process_frame(frame):
    # Your existing frame processing...

    # Trigger uArm based on conditions
    if some_condition:
        pickup()
    elif some_other_condition:
        place()
    elif gesture_needed:
        gesture()
"""
    )


def main():
    demo_api_usage()
    show_integration_examples()

    print("\n=== Next Steps ===")
    print("1. Connect your uArm Swift Pro via USB")
    print("2. Run: python machine.py --debug")
    print("3. Use the recording UI: python uarm_control/recording_ui.py")
    print("4. Record your 3 utilitarian motions")
    print("5. Call pickup(), place(), gesture() from anywhere in your code!")


if __name__ == "__main__":
    main()
