#!/usr/bin/env python3
"""Camera test - shows native firmware defaults before touching anything."""

import cv2
import sys

CAMERA_INDEX = 0
TARGET_WIDTH = 2560
TARGET_HEIGHT = 1440

cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print(f"Failed to open camera {CAMERA_INDEX}")
    sys.exit(1)

# Query camera's NATIVE values before we touch anything
native_brightness = cap.get(cv2.CAP_PROP_BRIGHTNESS)
native_contrast = cap.get(cv2.CAP_PROP_CONTRAST)
native_saturation = cap.get(cv2.CAP_PROP_SATURATION)
native_sharpness = cap.get(cv2.CAP_PROP_SHARPNESS)

print("=" * 50)
print("CAMERA NATIVE FIRMWARE DEFAULTS (before any changes):")
print(f"  Brightness: {native_brightness}")
print(f"  Contrast:   {native_contrast}")
print(f"  Saturation: {native_saturation}")
print(f"  Sharpness:  {native_sharpness}")
print("=" * 50)

# Only set resolution (shouldn't affect image quality)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_HEIGHT)

actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Resolution: {actual_w}x{actual_h}")
print("Controls: q=quit, d=show current settings")
print("This shows RAW camera output - no settings modified")
print("-" * 50)

cv2.namedWindow("Camera Test - RAW", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Camera Test - RAW", 1280, 720)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame")
        break

    cv2.imshow("Camera Test - RAW", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('d'):
        print(f"Brightness: {cap.get(cv2.CAP_PROP_BRIGHTNESS)}")
        print(f"Contrast: {cap.get(cv2.CAP_PROP_CONTRAST)}")
        print(f"Saturation: {cap.get(cv2.CAP_PROP_SATURATION)}")
        print(f"Sharpness: {cap.get(cv2.CAP_PROP_SHARPNESS)}")

cap.release()
cv2.destroyAllWindows()
