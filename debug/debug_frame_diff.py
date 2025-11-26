#!/usr/bin/env python3
"""
Debug frame difference calculation to see actual values
"""
import os
import sys
import time

import cv2

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import LIGHTBULB_SENSITIVITY


def debug_frame_diff():
    print("Debugging frame difference calculation...")
    print(f"LIGHTBULB_SENSITIVITY = {LIGHTBULB_SENSITIVITY}")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    prev_gray = None
    frame_count = 0

    print("\nMove around in front of the camera to see frame diff values...")
    print("Press 'q' to quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_gray is not None:
            diff = cv2.absdiff(prev_gray, gray)
            raw_diff_mean = diff.mean()

            # Current calculation from machine.py
            diff_score = raw_diff_mean * LIGHTBULB_SENSITIVITY * 10.0
            base_pwm = int(min(255, diff_score * 4))

            # Show the values every 10 frames
            if frame_count % 10 == 0:
                print(f"raw_diff_mean: {raw_diff_mean:6.2f}, diff_score: {diff_score:6.2f}, base_pwm: {base_pwm:3d}")

        prev_gray = gray.copy()
        frame_count += 1

        cv2.imshow("Debug Frame Diff", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    debug_frame_diff()
