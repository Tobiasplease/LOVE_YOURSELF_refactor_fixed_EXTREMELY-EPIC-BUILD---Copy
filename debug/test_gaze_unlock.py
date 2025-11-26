#!/usr/bin/env python3
"""
Test Gaze System Lock/Unlock
============================

Tests the gaze system locking and unlocking during drawing simulation.
"""

import os
import sys
import time

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


def test_gaze_lock_unlock():
    """Test gaze system locking and unlocking."""
    try:
        from config.config import TILT_MIN, USE_SERVO
        from vision.gaze import set_drawing_mode

        if not USE_SERVO:
            print("❌ USE_SERVO is False - gaze system disabled")
            return

        print("🎯 Testing Gaze Lock/Unlock System")
        print("=" * 40)

        # Test 1: Lock gaze to drawing position
        print("\n🔒 Step 1: Locking gaze to drawing position...")
        set_drawing_mode(active=True, drawing_pan=90, drawing_tilt=TILT_MIN + 2)
        print(f"   Locked to: Pan=90°, Tilt={TILT_MIN + 2}°")
        time.sleep(3)

        # Test 2: Unlock gaze system
        print("\n🔓 Step 2: Unlocking gaze system...")
        set_drawing_mode(active=False)
        print("   Gaze unlocked - should resume normal behavior")
        time.sleep(3)

        print("\n✅ Gaze lock/unlock test complete!")
        print("Check if the gaze system is now behaving normally.")

    except Exception as e:
        print(f"❌ Error testing gaze system: {e}")
        import traceback

        traceback.print_exc()


def test_paper_detection_import():
    """Test if paper detection import is causing issues."""
    print("\n📄 Testing Paper Detection Import...")
    try:
        from config.config import ALLOW_PAPER_DETECTION_OVERRIDE, ENABLE_PAPER_DETECTION
        from safety.paper_detection import check_paper_before_drawing

        print(f"   ENABLE_PAPER_DETECTION: {ENABLE_PAPER_DETECTION}")
        print(f"   ALLOW_PAPER_DETECTION_OVERRIDE: {ALLOW_PAPER_DETECTION_OVERRIDE}")
        print("✅ Paper detection imports working")
    except ImportError as e:
        print(f"❌ Paper detection import error: {e}")
        print("   This could be causing gaze to stay locked!")
    except Exception as e:
        print(f"❌ Paper detection config error: {e}")
        print("   This could be causing gaze to stay locked!")


if __name__ == "__main__":
    test_paper_detection_import()
    test_gaze_lock_unlock()
