#!/usr/bin/env python3
"""
End-to-end test of the runtime paper check (safety/paper_detection.py) with
PAPER_CHECK_METHOD as configured — exercises the same check_paper_before_drawing
entry the drawing pipeline calls (early pre-ComfyUI, post-home, image monitor).

The gaze loop isn't running standalone, so servos are positioned directly
before the check (set_paper_search_mode is a no-op here).

Usage: python debug/test_paper_check_runtime.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2

from config import config as cfg
from config.config import BAUD_RATE, CAMERA_INDEX, PAPER_DETECTION_GAZE_PAN, PAPER_DETECTION_GAZE_TILT, SERIAL_PORT
from safety.paper_detection import _parse_paper_state, check_paper_before_drawing, get_paper_detection_status, paper_detector

# Parser sanity before touching hardware
CASES = [
    ("PAPER: YES\nMARKS: NO\nA blank white sheet.", "blank_paper"),
    ("PAPER: YES\nMARKS: YES\nA sheet with a drawing.", "drawn_paper"),
    ("PAPER: NO\nMARKS: N/A\nBare wood.", "no_paper"),
    ("paper: yes\nmarks: na\nlowercase variant", "blank_paper"),
    ("The table is empty.", "unclear"),
]
for text, expected in CASES:
    got = _parse_paper_state(text)
    assert got == expected, f"parser: expected {expected}, got {got} for {text!r}"
print(f"✓ Parser sanity: {len(CASES)}/{len(CASES)}")
print(f"✓ Status: {get_paper_detection_status()}")

cap = cv2.VideoCapture(CAMERA_INDEX)
ret, _ = cap.read()
if not ret:
    print("ERROR: cannot capture from camera (is machine.py running and holding it?)")
    sys.exit(1)

try:
    from servo_control.servo_control import ServoController

    servos = ServoController(port=SERIAL_PORT, baudrate=BAUD_RATE)
    if servos.ser is None:
        raise RuntimeError(f"no serial connection on {SERIAL_PORT}")
    servos.set_pan(PAPER_DETECTION_GAZE_PAN)
    time.sleep(0.2)
    servos.set_tilt(PAPER_DETECTION_GAZE_TILT)
    time.sleep(0.8)
    print(f"✓ Camera at paper-check angle (pan {PAPER_DETECTION_GAZE_PAN}°, tilt {PAPER_DETECTION_GAZE_TILT}°)")
except Exception as e:
    print(f"⚠ Servos unavailable ({e}) — using current camera angle")
    servos = None

for _ in range(5):
    cap.read()

print(f"\nRunning check_paper_before_drawing (method={cfg.PAPER_CHECK_METHOD})...")
t0 = time.time()
allowed = check_paper_before_drawing(cap, servos, None)
print(f"\nRESULT: {'ALLOW drawing' if allowed else 'BLOCK drawing'}  ({time.time() - t0:.1f}s)")
print(f"Check images in {os.path.join(cfg.MOOD_SNAPSHOT_FOLDER, 'paper_checks')}/")

cap.release()
