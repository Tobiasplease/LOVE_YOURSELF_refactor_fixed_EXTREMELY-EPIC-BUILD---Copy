#!/usr/bin/env python3
"""
Minimal SDK Test
===============

Direct use of the uArm SDK without wrapper.
"""

import time
import os
from uarm.wrapper.swift_api import SwiftAPI
from uarm.swift.teach import Teach


def test_minimal_sdk():
    """Test SDK directly"""
    print("=== MINIMAL SDK TEST ===")

    # Connect directly using SDK
    try:
        swift = SwiftAPI(filters={'hwid': 'USB VID:PID=2341:0042'})
        swift.waiting_ready(timeout=3)
        print("✅ Connected via SDK")
    except Exception as e:
        print(f"❌ SDK connection failed: {e}")
        return

    # Get current position to verify connection
    try:
        pos = swift.get_position()
        print(f"📍 Current position: {pos}")

        angles = swift.get_servo_angle()
        print(f"🔧 Current angles: {angles}")
    except Exception as e:
        print(f"❌ Position check failed: {e}")
        return

    # Test teach functionality
    teach_file = "movement_recordings/uarm/minimal_test.txt"
    os.makedirs("movement_recordings/uarm", exist_ok=True)

    teach = Teach(teach_file, swift)
    print("📚 Teach object created")

    # Start standby mode
    teach.start_standby_mode()
    print("⏸️  Standby mode started")

    print("\n🎥 Starting 3-second recording...")
    print("Move the arm manually now!")

    # Start recording
    teach.start_record(interval=0.1)  # 10Hz

    time.sleep(3)

    # Stop recording
    teach.stop_record()
    print("⏹️  Recording stopped")

    # Check recording
    if os.path.exists(teach_file):
        with open(teach_file, 'r') as f:
            lines = f.readlines()
        print(f"✅ Recording: {len(lines)} commands")
        if lines:
            print("Sample commands:")
            for i, line in enumerate(lines[:3]):
                print(f"  {line.strip()}")
        else:
            print("❌ Empty recording")
            return
    else:
        print("❌ No recording file")
        return

    print("\n▶️  Starting playback...")

    # Start playback
    teach.start_play(speed=1, times=1)

    # Monitor playback
    while teach.is_playing():
        progress = teach.get_progress(wait=False)
        if progress:
            print(f"📊 Progress: {progress[1]:.1f}%")
        time.sleep(0.5)

    print("✅ Playback completed")

    # Cleanup
    teach.stop_standby_mode()
    swift.disconnect()


if __name__ == "__main__":
    test_minimal_sdk()