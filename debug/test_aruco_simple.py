#!/usr/bin/env python3
"""
Simple ArUco detection test using the machine's actual camera system.
Uses the same camera initialization as machine.py for compatibility.
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import time
from perception.camera import Camera
from config.config import CAMERA_INDEX

print("=" * 60)
print("ArUco Marker Paper Detection Test")
print("=" * 60)

# Use the actual camera class from the machine
print("\nInitializing camera (same as machine.py)...")
try:
    camera = Camera(camera_index=CAMERA_INDEX)
    print(f"✓ Camera initialized on index {CAMERA_INDEX}")
except Exception as e:
    print(f"✗ Camera initialization failed: {e}")
    sys.exit(1)

# Initialize ArUco detector
print("Initializing ArUco detector...")
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_params = cv2.aruco.DetectorParameters()

try:
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
    use_new_api = True
    print("✓ Using new OpenCV ArUco API")
except AttributeError:
    detector = None
    use_new_api = False
    print("✓ Using legacy OpenCV ArUco API")

print("\n" + "=" * 60)
print("INSTRUCTIONS:")
print("1. Place printed ArUco marker in camera view")
print("2. System will check detection every 2 seconds")
print("3. Try different positions and angles")
print("4. Press Ctrl+C to stop")
print("=" * 60 + "\n")

time.sleep(1)

detection_count = 0
check_count = 0

try:
    print("Starting detection checks...\n")

    while True:
        check_count += 1

        # Capture frame using machine's camera
        frame = camera.read_frame()

        if frame is None:
            print(f"[{check_count}] ✗ Failed to capture frame")
            time.sleep(2)
            continue

        # Detect markers
        if use_new_api:
            corners, ids, rejected = detector.detectMarkers(frame)
        else:
            corners, ids, rejected = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=aruco_params)

        # Check detection
        if ids is not None and len(ids) > 0:
            detection_count += 1

            # Get marker info
            marker_corners = corners[0][0]
            width_px = int(marker_corners[1][0] - marker_corners[0][0])
            height_px = int(marker_corners[2][1] - marker_corners[1][1])

            print(f"[{check_count}] ✓ DETECTED - ID: {ids[0][0]}, Size: {width_px}x{height_px}px, " +
                  f"Rate: {detection_count}/{check_count} ({detection_count/check_count*100:.0f}%)")

            # Save detection image for verification
            if detection_count == 1:
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                cv2.imwrite("aruco_detection_test.jpg", frame)
                print(f"    Saved detection image: aruco_detection_test.jpg")
        else:
            print(f"[{check_count}] ✗ No marker - Rate: {detection_count}/{check_count} ({detection_count/check_count*100:.0f}%)")

        time.sleep(2)

except KeyboardInterrupt:
    print("\n\nTest stopped by user")

finally:
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Total checks: {check_count}")
    print(f"Successful detections: {detection_count}")

    if check_count > 0:
        rate = detection_count / check_count * 100
        print(f"Detection rate: {rate:.1f}%")
        print()

        if rate >= 80:
            print("✅ EXCELLENT - Paper detection will work reliably!")
            print("   Ready to enable ENABLE_PAPER_DETECTION = True")
        elif rate >= 50:
            print("✓ GOOD - Paper detection should work")
            print("   Consider optimizing marker placement")
        else:
            print("⚠ POOR - Paper detection may be unreliable")
            print("   Recommendations:")
            print("   - Print marker larger (10cm+)")
            print("   - Ensure good lighting")
            print("   - Place marker flat and visible")

    print("=" * 60)
