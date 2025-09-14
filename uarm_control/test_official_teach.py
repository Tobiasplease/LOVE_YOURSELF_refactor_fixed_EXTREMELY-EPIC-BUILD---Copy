#!/usr/bin/env python3
"""
Test Official Teach System
==========================

Direct test of the SDK's official teach functionality.
"""

import time
import os
from uarm_controller import UarmController
from uarm.swift.teach import Teach


def test_official_teach():
    """Test the official teach system"""
    print("=== TESTING OFFICIAL TEACH SYSTEM ===")

    # Connect to uArm
    controller = UarmController(auto_home=False)
    if not controller.is_connected():
        print("❌ Failed to connect to uArm")
        return

    print("✅ Connected to uArm")

    # Create teach object
    teach_file = "movement_recordings/uarm/test_official.txt"
    os.makedirs("movement_recordings/uarm", exist_ok=True)

    teach = Teach(teach_file, controller.robot)
    teach.start_standby_mode()

    print("\nTesting recording...")
    print("Starting 5-second recording - move the arm manually")

    # Start recording
    teach.start_record(interval=0.05)  # 20Hz

    # Record for 5 seconds
    time.sleep(5)

    # Stop recording
    teach.stop_record()
    print("Recording stopped")

    # Check if file was created
    if os.path.exists(teach_file):
        with open(teach_file, 'r') as f:
            lines = f.readlines()
        print(f"✅ Recording saved: {len(lines)} lines")
        print("First few lines:")
        for i, line in enumerate(lines[:5]):
            print(f"  {i+1}: {line.strip()}")
    else:
        print("❌ No recording file created")
        return

    # Test playback
    print("\nTesting playback...")
    input("Press Enter to start playback...")

    teach.start_play(speed=1, times=1)

    # Wait for playback to complete
    while teach.is_playing():
        progress = teach.get_progress(wait=False)
        if progress:
            print(f"Progress: Loop {progress[0]}, {progress[1]:.1f}%")
        time.sleep(0.5)

    print("✅ Playback completed")

    # Cleanup
    teach.stop_standby_mode()
    controller.disconnect()


if __name__ == "__main__":
    test_official_teach()