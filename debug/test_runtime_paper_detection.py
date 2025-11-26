#!/usr/bin/env python3
"""
Test script to debug runtime paper detection issues.
This simulates exactly what the runtime system does.
"""

import sys
import os
import time
import cv2

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from config.config import CAMERA_INDEX
from safety.paper_detection import PaperDetector

def test_runtime_detection():
    """Test paper detection using exact same code path as runtime."""
    print("🔧 Testing runtime paper detection code path...")

    # Initialize camera
    camera = cv2.VideoCapture(CAMERA_INDEX)
    if not camera.isOpened():
        print(f"❌ Failed to open camera {CAMERA_INDEX}")
        return

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Mock servo object (runtime expects this)
    class MockServos:
        def set_pan(self, value): pass
        def set_tilt(self, value): pass

    servos = MockServos()

    # Initialize paper detector (same as runtime)
    detector = PaperDetector()

    try:
        print("📸 Capturing frame...")
        ret, frame = camera.read()
        if not ret:
            print("❌ Failed to capture frame")
            return

        # Save test image
        timestamp = int(time.time())
        test_image_path = f"/tmp/runtime_test_{timestamp}.jpg"
        cv2.imwrite(test_image_path, frame)
        print(f"💾 Saved test image: {test_image_path}")

        # Run detection using EXACT same method as runtime
        print("🤖 Running paper detection (runtime code path)...")
        result = detector.check_paper_present(camera, servos, captioner=None)

        # Display results
        print(f"\n📄 RUNTIME RESULTS:")
        print(f"   Paper Present: {'✅ YES' if result.paper_present else '❌ NO'}")
        print(f"   Confidence: {result.confidence:.2f}")
        print(f"   Method: {result.method_used}")
        print(f"   LLM Response: '{result.llm_response}'")
        print(f"   Check Image: {result.check_image_path}")
        if result.error_message:
            print(f"   Error: {result.error_message}")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        camera.release()

if __name__ == "__main__":
    test_runtime_detection()