#!/usr/bin/env python3
"""
Interactive Servo Angle Finder for Paper Detection
Use keyboard to adjust pan/tilt until camera points at drawing surface.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from servo_control.servo_control import ServoController
from config.config import CAMERA_INDEX
import cv2
import time

print("=" * 60)
print("Interactive Paper Detection Angle Finder")
print("=" * 60)

# Initialize
servos = ServoController(port="/dev/arduino_lunggaze", baudrate=9600)
cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print("Failed to open camera!")
    exit(1)
print("✓ Servos and camera initialized\n")

# Starting position (current config values)
pan = 80
tilt = 50

print("Controls:")
print("  A/D = Pan left/right")
print("  W/S = Tilt up/down")
print("  Q   = Quit and save angles")
print("  ESC = Quit without saving")
print("\nStarting position: pan=80, tilt=50")
print("Adjust until ArUco marker is visible and centered!\n")

# Position servos
servos.set_pan(pan)
time.sleep(0.1)
servos.set_tilt(tilt)

try:
    while True:
        # Capture and display
        ret, frame = cap.read()
        if ret and frame is not None:
            # Add angle info to display
            display = frame.copy()
            cv2.putText(display, f"Pan: {pan}  Tilt: {tilt}", (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(display, "A/D=pan W/S=tilt Q=save ESC=quit", (20, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            cv2.imshow("Paper Detection Angle Finder", display)

        # Handle keyboard (longer delay for better key capture)
        key = cv2.waitKey(50) & 0xFF

        # Debug: show which key was pressed
        if key != 255:  # 255 means no key pressed
            print(f"Key pressed: {key} ({chr(key) if key < 128 else 'special'})")

        if key == ord('a'):  # Pan left
            pan -= 5
            servos.set_pan(pan)
            print(f"→ Pan LEFT: {pan}")
            time.sleep(0.1)
        elif key == ord('d'):  # Pan right
            pan += 5
            servos.set_pan(pan)
            print(f"→ Pan RIGHT: {pan}")
            time.sleep(0.1)
        elif key == ord('w'):  # Tilt up
            tilt -= 5
            servos.set_tilt(tilt)
            print(f"→ Tilt UP: {tilt}")
            time.sleep(0.1)
        elif key == ord('s'):  # Tilt down
            tilt += 5
            servos.set_tilt(tilt)
            print(f"→ Tilt DOWN: {tilt}")
            time.sleep(0.1)
        elif key == ord('q'):  # Save and quit
            print("\n" + "=" * 60)
            print(f"FINAL ANGLES: pan={pan}, tilt={tilt}")
            print("\nUpdate config.py:")
            print(f"  PAPER_DETECTION_GAZE_PAN = {pan}")
            print(f"  PAPER_DETECTION_GAZE_TILT = {tilt}")
            print("=" * 60)
            break
        elif key == 27:  # ESC - quit without saving
            print("\nCancelled - no changes saved")
            break

except KeyboardInterrupt:
    print("\n\nInterrupted")
finally:
    cv2.destroyAllWindows()
