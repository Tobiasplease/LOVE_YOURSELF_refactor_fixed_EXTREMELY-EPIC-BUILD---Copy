#!/usr/bin/env python3
"""
Test Idle Movement Manager
Tests the pause/resume functionality for serial port sharing
"""

import os
import sys
import time

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from grbl.idle_movement_manager import pause_for_drawing, resume_after_drawing, start_idle_movements, stop_idle_movements, update_emotion


def test_idle_manager():
    """Test the idle movement manager functionality"""

    print("=" * 60)
    print("IDLE MOVEMENT MANAGER TEST")
    print("=" * 60)

    # Test 1: Start idle movements
    print("\n[TEST 1] Starting idle movements...")
    if start_idle_movements("calm_observant"):
        print("[SUCCESS] Idle movements started")
        time.sleep(5)
    else:
        print("[ERROR] Failed to start idle movements")
        return False

    # Test 2: Pause for drawing
    print("\n[TEST 2] Pausing idle movements for drawing...")
    if pause_for_drawing():
        print("[SUCCESS] Idle movements paused")
        print("Serial port should now be free for drawing")
        time.sleep(3)
    else:
        print("[ERROR] Failed to pause idle movements")
        return False

    # Test 3: Resume after drawing
    print("\n[TEST 3] Resuming idle movements after drawing...")
    if resume_after_drawing():
        print("[SUCCESS] Idle movements resumed")
        time.sleep(5)
    else:
        print("[ERROR] Failed to resume idle movements")
        return False

    # Test 4: Update emotion
    print("\n[TEST 4] Updating emotion to 'alert_curious'...")
    update_emotion("alert_curious")
    print("[INFO] Emotion update initiated")
    time.sleep(5)

    # Test 5: Multiple pause/resume cycles
    print("\n[TEST 5] Testing multiple pause/resume cycles...")
    for i in range(3):
        print(f"\n  Cycle {i+1}:")
        print("  - Pausing...")
        pause_for_drawing()
        time.sleep(2)
        print("  - Resuming...")
        resume_after_drawing()
        time.sleep(3)
    print("[SUCCESS] Multiple cycles completed")

    # Test 6: Stop idle movements
    print("\n[TEST 6] Stopping idle movements...")
    stop_idle_movements()
    print("[SUCCESS] Idle movements stopped")

    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED SUCCESSFULLY")
    print("=" * 60)
    return True


if __name__ == "__main__":
    try:
        success = test_idle_manager()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n[INFO] Test interrupted by user")
        stop_idle_movements()
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        stop_idle_movements()
        sys.exit(1)
