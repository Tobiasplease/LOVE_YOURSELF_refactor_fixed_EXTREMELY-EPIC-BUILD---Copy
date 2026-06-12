"""Calibrate the ego-compensated scene motion estimator (vision/scene_motion.py).

Run while machine.py is STOPPED (it needs the camera):

    python debug/test_scene_motion.py

Then experiment in front of the camera and watch the numbers:
  - hold everything still           -> residual ~0.00
  - move the camera by hand         -> camera_shift rises, residual stays low
  - wave / walk through the frame   -> residual rises (this is scene motion)
  - both at once                    -> residual still rises

The decision threshold is SCENE_MOTION_RESIDUAL_THRESHOLD in config.py
(currently 0.04). Pick a value between what stillness shows and what real
movement shows.
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2

from config.config import SCENE_MOTION_RESIDUAL_THRESHOLD
from vision.scene_motion import SceneMotionEstimator


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open camera — is machine.py still running?")
        return 1

    estimator = SceneMotionEstimator()
    print(f"Threshold: {SCENE_MOTION_RESIDUAL_THRESHOLD}  (Ctrl+C to stop)\n")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (320, 240))
            r = estimator.update(gray)
            verdict = "SCENE MOTION" if r["valid"] and r["residual_fraction"] > SCENE_MOTION_RESIDUAL_THRESHOLD else ""
            bar = "#" * int(r["residual_fraction"] * 200)
            print(f"valid={str(r['valid']):5}  camera_shift={r['camera_shift_px']:5.1f}px  residual={r['residual_fraction']:.3f} {bar:<20} {verdict}")
            time.sleep(0.5)  # match the frame buffer's 2fps cadence
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
    return 0


if __name__ == "__main__":
    sys.exit(main())
