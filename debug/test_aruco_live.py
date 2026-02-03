#!/usr/bin/env python3
"""
Live ArUco marker detection test for paper detection system.
Shows real-time detection status with your actual camera.
"""
import sys
import cv2
import time

# Test with the actual camera from config
try:
    from config.config import CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT
    camera_index = CAMERA_INDEX
    width = CAMERA_WIDTH
    height = CAMERA_HEIGHT
except:
    camera_index = 0
    width = 640
    height = 480

print("=" * 60)
print("ArUco Marker Paper Detection Test")
print("=" * 60)
print(f"\nCamera: Index {camera_index}, Resolution {width}x{height}")
print("\nInstructions:")
print("1. Hold the printed ArUco marker in front of camera")
print("2. Try different distances and angles")
print("3. Check detection rate")
print("4. Press 'q' to quit\n")

# Initialize camera
cap = cv2.VideoCapture(camera_index)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Actual camera resolution: {actual_width}x{actual_height}\n")

# Initialize ArUco detector (same as safety/aruco_paper_detection.py)
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_params = cv2.aruco.DetectorParameters()

# Try both old and new OpenCV API
try:
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
    use_new_api = True
    print("Using new OpenCV ArUco API")
except AttributeError:
    detector = None
    use_new_api = False
    print("Using legacy OpenCV ArUco API")

print("\nStarting detection...\n")
time.sleep(0.5)

# Detection stats
frame_count = 0
detection_count = 0
last_detection_time = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame!")
            break

        frame_count += 1

        # Detect markers
        if use_new_api:
            corners, ids, rejected = detector.detectMarkers(frame)
        else:
            corners, ids, rejected = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=aruco_params)

        # Check detection
        if ids is not None and len(ids) > 0:
            detection_count += 1
            last_detection_time = time.time()

            # Draw detected markers
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

            # Calculate marker size and center
            marker_corners = corners[0][0]
            width_px = int(marker_corners[1][0] - marker_corners[0][0])
            height_px = int(marker_corners[2][1] - marker_corners[1][1])
            center_x = int(sum([c[0] for c in marker_corners]) / 4)
            center_y = int(sum([c[1] for c in marker_corners]) / 4)

            # Display info on frame
            cv2.putText(frame, f"DETECTED: ID {ids[0][0]}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Size: {width_px}x{height_px}px", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"Center: ({center_x}, {center_y})", (10, 85),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"Detection: {detection_count}/{frame_count} ({detection_count/frame_count*100:.1f}%)",
                       (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Console output
            if detection_count % 10 == 1:  # Print every 10th detection
                print(f"✓ Detection #{detection_count}: ID={ids[0][0]}, Size={width_px}x{height_px}px, " +
                      f"Rate={detection_count/frame_count*100:.1f}%")
        else:
            # No detection
            cv2.putText(frame, "NO MARKER DETECTED", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, f"Detection: {detection_count}/{frame_count} ({detection_count/frame_count*100:.1f}%)",
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # Warn if no detection for a while
            if last_detection_time > 0 and time.time() - last_detection_time > 3:
                cv2.putText(frame, "Move marker closer or check lighting", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Display frame
        cv2.imshow("ArUco Paper Detection Test", frame)

        # Quit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\n\nTest interrupted by user")

finally:
    cap.release()
    cv2.destroyAllWindows()

    # Final statistics
    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    print(f"Total frames: {frame_count}")
    print(f"Detections: {detection_count}")
    print(f"Detection rate: {detection_count/frame_count*100:.1f}%")
    print()

    if detection_count > 0:
        if detection_count / frame_count > 0.8:
            print("✅ EXCELLENT - Detection is very reliable!")
            print("   ArUco system is ready to use for paper detection.")
        elif detection_count / frame_count > 0.5:
            print("✓ GOOD - Detection works but could be improved")
            print("   Tips: Better lighting, larger marker, flatter surface")
        else:
            print("⚠ POOR - Detection is unreliable")
            print("   Try: Larger marker print, better lighting, different marker position")
    else:
        print("✗ NO DETECTIONS - Check:")
        print("  1. Printed the correct marker (DICT_4X4_50, ID 0)")
        print("  2. Marker is visible to camera")
        print("  3. Good lighting conditions")
        print("  4. Marker is printed large enough (8cm+ recommended)")

    print("\n" + "=" * 60)
