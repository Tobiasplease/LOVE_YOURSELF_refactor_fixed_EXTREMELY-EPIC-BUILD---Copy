#!/usr/bin/env python3
"""
Live ArUco marker detection test for paper detection system.
Shows real-time detection status with your actual camera.
Includes real-time parameter tuning via trackbars.
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
print("ArUco Marker Paper Detection Test - TUNING MODE")
print("=" * 60)
print(f"\nCamera: Index {camera_index}, Resolution {width}x{height}")
print("\nInstructions:")
print("1. Hold the printed ArUco marker in front of camera")
print("2. Adjust sliders to tune detection sensitivity")
print("3. Watch detection rate improve/degrade")
print("4. Press 'q' to quit, 's' to save current params\n")

# Initialize camera
cap = cv2.VideoCapture(camera_index)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Actual camera resolution: {actual_width}x{actual_height}\n")

# Initialize ArUco detector
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_params = cv2.aruco.DetectorParameters()

# Parameter defaults (will be controlled by trackbars)
PARAM_DEFAULTS = {
    "adaptiveThreshWinSizeMin": 3,
    "adaptiveThreshWinSizeMax": 23,
    "adaptiveThreshWinSizeStep": 10,
    "adaptiveThreshConstant": 7,
    "minMarkerPerimeterRate": 3,  # x100 (0.03)
    "maxMarkerPerimeterRate": 400,  # x100 (4.0)
    "polygonalApproxAccuracyRate": 3,  # x100 (0.03)
    "minCornerDistanceRate": 5,  # x100 (0.05)
    "minDistanceToBorder": 3,
    "cornerRefinementMethod": 0,  # 0=none, 1=subpix, 2=contour
}

# Create window and trackbars
WINDOW_NAME = "ArUco Tuning"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, 900, 700)

def nothing(x):
    pass

# Create trackbars for key parameters
cv2.createTrackbar("WinSizeMin", WINDOW_NAME, PARAM_DEFAULTS["adaptiveThreshWinSizeMin"], 30, nothing)
cv2.createTrackbar("WinSizeMax", WINDOW_NAME, PARAM_DEFAULTS["adaptiveThreshWinSizeMax"], 100, nothing)
cv2.createTrackbar("WinSizeStep", WINDOW_NAME, PARAM_DEFAULTS["adaptiveThreshWinSizeStep"], 30, nothing)
cv2.createTrackbar("ThreshConst", WINDOW_NAME, PARAM_DEFAULTS["adaptiveThreshConstant"], 30, nothing)
cv2.createTrackbar("MinPerim x100", WINDOW_NAME, PARAM_DEFAULTS["minMarkerPerimeterRate"], 20, nothing)
cv2.createTrackbar("MaxPerim x100", WINDOW_NAME, PARAM_DEFAULTS["maxMarkerPerimeterRate"], 800, nothing)
cv2.createTrackbar("PolyApprox x100", WINDOW_NAME, PARAM_DEFAULTS["polygonalApproxAccuracyRate"], 20, nothing)
cv2.createTrackbar("MinCornerDist x100", WINDOW_NAME, PARAM_DEFAULTS["minCornerDistanceRate"], 20, nothing)
cv2.createTrackbar("MinBorderDist", WINDOW_NAME, PARAM_DEFAULTS["minDistanceToBorder"], 10, nothing)
cv2.createTrackbar("CornerRefine", WINDOW_NAME, PARAM_DEFAULTS["cornerRefinementMethod"], 2, nothing)

def get_params_from_trackbars():
    """Read current trackbar values and update detector parameters."""
    params = cv2.aruco.DetectorParameters()

    win_min = max(3, cv2.getTrackbarPos("WinSizeMin", WINDOW_NAME))
    win_max = max(win_min + 2, cv2.getTrackbarPos("WinSizeMax", WINDOW_NAME))
    win_step = max(2, cv2.getTrackbarPos("WinSizeStep", WINDOW_NAME))

    params.adaptiveThreshWinSizeMin = win_min
    params.adaptiveThreshWinSizeMax = win_max
    params.adaptiveThreshWinSizeStep = win_step
    params.adaptiveThreshConstant = cv2.getTrackbarPos("ThreshConst", WINDOW_NAME)
    params.minMarkerPerimeterRate = cv2.getTrackbarPos("MinPerim x100", WINDOW_NAME) / 100.0
    params.maxMarkerPerimeterRate = cv2.getTrackbarPos("MaxPerim x100", WINDOW_NAME) / 100.0
    params.polygonalApproxAccuracyRate = max(0.01, cv2.getTrackbarPos("PolyApprox x100", WINDOW_NAME) / 100.0)
    params.minCornerDistanceRate = cv2.getTrackbarPos("MinCornerDist x100", WINDOW_NAME) / 100.0
    params.minDistanceToBorder = cv2.getTrackbarPos("MinBorderDist", WINDOW_NAME)

    corner_method = cv2.getTrackbarPos("CornerRefine", WINDOW_NAME)
    if corner_method == 0:
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_NONE
    elif corner_method == 1:
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    else:
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_CONTOUR

    return params

def print_current_params():
    """Print current parameter values for copying to aruco_detector.py"""
    params = get_params_from_trackbars()
    print("\n" + "=" * 60)
    print("CURRENT PARAMETER VALUES (copy to aruco_detector.py):")
    print("=" * 60)
    print(f"self.aruco_params.adaptiveThreshWinSizeMin = {params.adaptiveThreshWinSizeMin}")
    print(f"self.aruco_params.adaptiveThreshWinSizeMax = {params.adaptiveThreshWinSizeMax}")
    print(f"self.aruco_params.adaptiveThreshWinSizeStep = {params.adaptiveThreshWinSizeStep}")
    print(f"self.aruco_params.adaptiveThreshConstant = {params.adaptiveThreshConstant}")
    print(f"self.aruco_params.minMarkerPerimeterRate = {params.minMarkerPerimeterRate}")
    print(f"self.aruco_params.maxMarkerPerimeterRate = {params.maxMarkerPerimeterRate}")
    print(f"self.aruco_params.polygonalApproxAccuracyRate = {params.polygonalApproxAccuracyRate}")
    print(f"self.aruco_params.minCornerDistanceRate = {params.minCornerDistanceRate}")
    print(f"self.aruco_params.minDistanceToBorder = {params.minDistanceToBorder}")
    corner_names = ["CORNER_REFINE_NONE", "CORNER_REFINE_SUBPIX", "CORNER_REFINE_CONTOUR"]
    corner_idx = cv2.getTrackbarPos("CornerRefine", WINDOW_NAME)
    print(f"self.aruco_params.cornerRefinementMethod = cv2.aruco.{corner_names[corner_idx]}")
    print("=" * 60 + "\n")

print("Using new OpenCV ArUco API with real-time tuning")
print("\nStarting detection...\n")
time.sleep(0.5)

# Detection stats
frame_count = 0
detection_count = 0
last_detection_time = 0
last_params_update = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame!")
            break

        frame_count += 1

        # Update parameters from trackbars every frame
        current_params = get_params_from_trackbars()
        detector = cv2.aruco.ArucoDetector(aruco_dict, current_params)

        # Detect markers
        corners, ids, rejected = detector.detectMarkers(frame)

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
                cv2.putText(frame, "Adjust sliders to improve detection", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Show rejected candidates (potential markers that failed validation)
        if rejected and len(rejected) > 0:
            cv2.putText(frame, f"Rejected candidates: {len(rejected)}", (10, actual_height - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
            # Draw rejected candidates in gray
            for rej in rejected:
                pts = rej[0].astype(int)
                cv2.polylines(frame, [pts], True, (128, 128, 128), 1)

        # Display current key parameters on frame
        y_offset = actual_height - 60
        params_text = f"WinSize: {current_params.adaptiveThreshWinSizeMin}-{current_params.adaptiveThreshWinSizeMax}"
        cv2.putText(frame, params_text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        params_text2 = f"Perim: {current_params.minMarkerPerimeterRate:.2f}-{current_params.maxMarkerPerimeterRate:.1f}"
        cv2.putText(frame, params_text2, (10, y_offset + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        params_text3 = f"ThreshConst: {current_params.adaptiveThreshConstant}"
        cv2.putText(frame, params_text3, (10, y_offset + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        # Display frame
        cv2.imshow(WINDOW_NAME, frame)

        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            print_current_params()

except KeyboardInterrupt:
    print("\n\nTest interrupted by user")

finally:
    cap.release()
    cv2.destroyAllWindows()

    # Final statistics
    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    if frame_count > 0:
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
                print("   Try adjusting the sliders or marker position")
        else:
            print("✗ NO DETECTIONS - Check:")
            print("  1. Printed the correct marker (DICT_4X4_50, ID 0)")
            print("  2. Marker is visible to camera")
            print("  3. Good lighting conditions")
            print("  4. Marker is printed large enough (8cm+ recommended)")

        # Print final parameters
        print_current_params()
    else:
        print("No frames captured.")

    print("=" * 60)
