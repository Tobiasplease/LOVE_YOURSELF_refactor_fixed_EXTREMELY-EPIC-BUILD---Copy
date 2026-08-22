#!/usr/bin/env python3
"""
Can the VLM replace the ArUco paper check?

Positions the camera at the paper-check angle (PAPER_DETECTION_GAZE_PAN/TILT),
grabs frames, and for each frame runs BOTH checks side by side:
  - the production ArUco detection (marker visible = no paper)
  - a plain question to the loaded model: is there a white paper sheet on the table?

Prints a per-trial comparison and saves each frame + raw model response to
debug/paper_vlm_frames/ for inspection.

Requires llama-server up (starts it if not) and the camera free — stop
machine.py first. The CNC arm is NOT tucked (the real check plays the kinetic
get-clear move first), so make sure the arm isn't blocking the view.

Usage: python debug/test_paper_vlm_check.py [num_trials]
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2

from config.config import BAUD_RATE, CAMERA_INDEX, PAPER_DETECTION_GAZE_PAN, PAPER_DETECTION_GAZE_TILT, SERIAL_PORT
from utils.inference import is_failed_response, query_model

NUM_TRIALS = int(sys.argv[1]) if len(sys.argv) > 1 else 3
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "paper_vlm_frames")

PROMPT = (
    "You are looking down at the table surface where you draw. "
    "Is there a white sheet of paper lying on the table right now? "
    "Answer YES or NO on the first line, then one sentence describing what you see on the surface."
)

# Same acceptance rules as safety/aruco_detector.py
ARUCO_VALID_ID = 0
ARUCO_MIN_PIXELS = 40


def aruco_paper_present(frame):
    """Production-equivalent single-frame check. Marker visible = NO paper."""
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())
    corners, ids, _ = detector.detectMarkers(frame)
    if ids is None:
        return True
    for i, marker_id in enumerate(ids.flatten()):
        mc = corners[i][0]
        side1 = ((mc[0][0] - mc[1][0]) ** 2 + (mc[0][1] - mc[1][1]) ** 2) ** 0.5
        side2 = ((mc[1][0] - mc[2][0]) ** 2 + (mc[1][1] - mc[2][1]) ** 2) ** 0.5
        if int(marker_id) == ARUCO_VALID_ID and (side1 + side2) / 2 >= ARUCO_MIN_PIXELS:
            return False
    return True


def parse_verdict(text):
    """Structural yes/no parse of the model's free words. Returns True/False/None."""
    first_line = text.strip().split("\n")[0].lower()
    words = [w.strip(".,:;!*") for w in first_line.split()]
    if "yes" in words[:3]:
        return True
    if "no" in words[:3]:
        return False
    lowered = text.lower()
    if "no paper" in lowered or "there is no" in lowered or "isn't a" in lowered or "is not a" in lowered:
        return False
    if "paper" in lowered and ("there is a" in lowered or "i see a" in lowered or "i can see a" in lowered):
        return True
    return None


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    ret, _ = cap.read()
    if not ret:
        print("ERROR: cannot capture from camera (is machine.py running and holding it?)")
        sys.exit(1)
    print("✓ Camera open")

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

    for _ in range(5):
        cap.read()

    results = []
    for trial in range(1, NUM_TRIALS + 1):
        print(f"\n--- Trial {trial}/{NUM_TRIALS} ---")
        ret, frame = cap.read()
        if not ret:
            print("Lost camera, stopping.")
            break

        frame_path = os.path.join(OUT_DIR, f"trial_{trial}.jpg")
        cv2.imwrite(frame_path, frame)

        aruco_verdict = aruco_paper_present(frame)
        print(f"ArUco:  paper_present={aruco_verdict}  (marker {'NOT visible' if aruco_verdict else 'visible'})")

        ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not ok:
            print("JPEG encode failed, skipping trial")
            continue

        t0 = time.time()
        response = query_model(
            PROMPT,
            image=jpg.tobytes(),
            timeout=120,
            options={"temperature": 0.1, "num_predict": 120},
            prompt_type="paper_vlm_test",
            skip_generation_wait=True,
        )
        elapsed = time.time() - t0

        if is_failed_response(response):
            print(f"Model:  QUERY FAILED ({elapsed:.1f}s): {response}")
            results.append((trial, aruco_verdict, None, "FAILED"))
            continue

        with open(os.path.join(OUT_DIR, f"trial_{trial}_response.txt"), "w") as f:
            f.write(response)

        vlm_verdict = parse_verdict(response)
        print(f"Model:  paper_present={vlm_verdict}  ({elapsed:.1f}s)")
        print(f'Model says: "{response.strip()}"')
        results.append((trial, aruco_verdict, vlm_verdict, response.strip().split(chr(10))[0]))

        time.sleep(0.5)

    cap.release()

    print("\n" + "=" * 60)
    print("SUMMARY  (paper_present: ArUco vs model)")
    print("=" * 60)
    agree = 0
    for trial, aruco_v, vlm_v, note in results:
        match = "AGREE" if aruco_v == vlm_v else ("?" if vlm_v is None else "DISAGREE")
        if aruco_v == vlm_v:
            agree += 1
        print(f"  trial {trial}: aruco={aruco_v}  model={vlm_v}  [{match}]  {note[:60]}")
    if results:
        print(f"\nAgreement: {agree}/{len(results)}")
    print(f"Frames + raw responses in {OUT_DIR}/")


if __name__ == "__main__":
    main()
