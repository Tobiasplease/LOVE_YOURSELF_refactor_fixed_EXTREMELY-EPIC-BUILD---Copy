#!/usr/bin/env python3
"""
Simple Paper Reference Image Capture
Capture reference image for paper detection system
"""

import os
import sys
import time

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

def main():
    print("📄 CAPTURING PAPER REFERENCE IMAGE")
    print("=" * 50)

    try:
        from safety.paper_detection import capture_paper_reference
        print("✅ Paper detection module imported")
    except ImportError as e:
        print(f"❌ Error importing paper detection: {e}")
        return False

    try:
        # Initialize camera and servos
        print("[📄] Initializing camera and servos...")

        from reactivity.camera_reactive import CameraReactivityEngine
        camera = CameraReactivityEngine()
        print("[✅] Camera initialized")

        from servo_control.servo_control import ServoController
        servos = ServoController()
        print("[✅] Servos initialized")

    except Exception as e:
        print(f"[❌] Error initializing hardware: {e}")
        return False

    try:
        print("\n📄 Capturing reference image...")
        print("   Make sure paper is properly positioned in the drawing area")
        print("   The system will automatically capture in 3 seconds...")

        # Give user time to position paper
        for i in range(3, 0, -1):
            print(f"   Capturing in {i}...")
            time.sleep(1.0)

        # Capture reference image
        success = capture_paper_reference(camera, servos)

        if success:
            print("\n✅ SUCCESS! Reference image captured")
            print("   Location: calibration/paper_reference.jpg")
            print("   Paper detection system is now ready!")
            print("\nNext steps:")
            print("   1. Test detection: python debug/test_paper_detection.py --test")
            print("   2. Check status: python debug/test_paper_detection.py --status")
        else:
            print("\n❌ FAILED to capture reference image")
            print("   Check camera and servo connections")

        return success

    except Exception as e:
        print(f"\n❌ ERROR during capture: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)