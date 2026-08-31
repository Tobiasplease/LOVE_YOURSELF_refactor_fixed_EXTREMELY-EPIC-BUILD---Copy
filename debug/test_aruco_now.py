#!/usr/bin/env python3
"""
Simple ArUco marker detection test - LIVE VIEW
Just shows what the camera sees and if marker is detected.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time

import cv2

from config.config import CAMERA_INDEX, PAPER_DETECTION_GAZE_PAN, PAPER_DETECTION_GAZE_TILT

print("=" * 60)
print("ArUco Marker Detection - Live Test")
print("=" * 60)
print("\n1. Initializing camera and servos...")

# Initialize camera
cap = cv2.VideoCapture(CAMERA_INDEX)
ret, frame = cap.read()
if not ret:
    print("ERROR: Cannot capture from camera!")
    sys.exit(1)

print(f"✓ Camera working: {frame.shape}")

# Initialize servos
try:
    from servo_control.servo_control import ServoController

    servos = ServoController()
    print(f"✓ Servos initialized")

    # Position to look at drawing area
    print(f"\n2. Positioning camera to drawing area...")
    print(f"   Pan: {PAPER_DETECTION_GAZE_PAN}°")
    print(f"   Tilt: {PAPER_DETECTION_GAZE_TILT}°")
    servos.set_pan(PAPER_DETECTION_GAZE_PAN)
    time.sleep(0.2)
    servos.set_tilt(PAPER_DETECTION_GAZE_TILT)
    time.sleep(0.5)
    print("✓ Camera positioned")
except Exception as e:
    print(f"⚠ Servos failed: {e}")
    print("  Continuing without servo positioning...")

# Initialize ArUco detector
print("\n3. Initializing ArUco detector...")
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_params = cv2.aruco.DetectorParameters()

# Adjust for glare/lighting robustness
aruco_params.adaptiveThreshWinSizeMin = 3
aruco_params.adaptiveThreshWinSizeMax = 23
aruco_params.adaptiveThreshWinSizeStep = 10
aruco_params.adaptiveThreshConstant = 7
aruco_params.minMarkerPerimeterRate = 0.03
aruco_params.maxMarkerPerimeterRate = 4.0
aruco_params.polygonalApproxAccuracyRate = 0.05
aruco_params.minCornerDistanceRate = 0.05
aruco_params.minDistanceToBorder = 3
aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
aruco_params.cornerRefinementWinSize = 5
aruco_params.cornerRefinementMaxIterations = 30
aruco_params.cornerRefinementMinAccuracy = 0.1

try:
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
    use_new_api = True
except AttributeError:
    detector = None
    use_new_api = False

print("✓ Detector ready")

print("\n" + "=" * 60)
print("LIVE DETECTION - Press 'q' to quit")
print("=" * 60)
print("Hold your printed ArUco marker in camera view...")
print()

detection_count = 0
frame_count = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Lost camera!")
            break

        frame_count += 1

        # Detect markers
        if use_new_api:
            corners, ids, rejected = detector.detectMarkers(frame)
        else:
            corners, ids, rejected = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=aruco_params)

        # Display detection
        if ids is not None and len(ids) > 0:
            detection_count += 1

            # Draw markers on frame
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

            # Get marker size
            marker_corners = corners[0][0]
            width_px = int(marker_corners[1][0] - marker_corners[0][0])
            height_px = int(marker_corners[2][1] - marker_corners[1][1])

            # Green overlay - DETECTED
            cv2.putText(frame, "MARKER DETECTED!", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            cv2.putText(frame, f"ID: {ids[0][0]}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Size: {width_px}x{height_px}px", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            rate = detection_count / frame_count * 100
            cv2.putText(frame, f"Rate: {rate:.0f}%", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Console output every 30 frames
            if detection_count % 30 == 1:
                print(f"✓ Detecting: ID={ids[0][0]}, Size={width_px}x{height_px}px, Rate={rate:.0f}%")
        else:
            # Red overlay - NO DETECTION
            cv2.putText(frame, "NO MARKER", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

            rate = detection_count / frame_count * 100 if frame_count > 0 else 0
            cv2.putText(frame, f"Rate: {rate:.0f}%", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Show frame
        cv2.imshow("ArUco Detection Test", frame)

        # Quit on 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

except KeyboardInterrupt:
    print("\nStopped by user")

finally:
    cap.release()
    cv2.destroyAllWindows()

    # Final stats
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    if frame_count > 0:
        rate = detection_count / frame_count * 100
        print(f"Frames: {frame_count}")
        print(f"Detections: {detection_count}")
        print(f"Rate: {rate:.1f}%")
        print()

        if rate >= 80:
            print("✅ EXCELLENT - Ready for production!")
        elif rate >= 50:
            print("✓ GOOD - Should work, optimize if needed")
        else:
            print("⚠ POOR - Try larger marker or better lighting")
