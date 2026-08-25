"""Pose-model swap + skeleton gate against real studio frames.

Runs yolo11m-pose on given images, prints every person-candidate with its
keypoint count, body-region spread, and the gate's verdict.

    python debug/test_pose_gate.py <image> [more images...]
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
from ultralytics import YOLO

from config.config import YOLO_CONFIDENCE_THRESHOLD, YOLO_MODEL_PATH, YOLO_PERSON_MIN_AREA_FRAC
from perception.object_detection import ObjectDetectionThread

model = YOLO(YOLO_MODEL_PATH)
print(f"model: {os.path.basename(YOLO_MODEL_PATH)} task={model.task}\n")

for path in sys.argv[1:]:
    frame = cv2.imread(path)
    if frame is None:
        print(f"{path}: unreadable")
        continue
    results = model.predict(frame, verbose=False, imgsz=512)[0]
    kps = getattr(results, "keypoints", None)
    print(f"=== {os.path.basename(path)} ({frame.shape[1]}x{frame.shape[0]})")
    n = 0
    for i, box in enumerate(results.boxes):
        if int(box.cls[0]) != 0:
            continue
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        area_frac = (x2 - x1) * (y2 - y1) / (frame.shape[0] * frame.shape[1])
        small = area_frac < YOLO_PERSON_MIN_AREA_FRAC
        low = conf < YOLO_CONFIDENCE_THRESHOLD
        verdict = "?"
        detail = ""
        if kps is not None and kps.conf is not None and i < len(kps.conf):
            ok, n_kp, n_reg = ObjectDetectionThread._skeleton_coherent(kps.conf[i])
            verdict = "PERSON" if ok and not small and not low else "rejected"
            detail = f"{n_kp} kps, {n_reg} regions"
        flags = ("small " if small else "") + ("lowconf" if low else "")
        print(f"  conf {conf:.2f}  box ({x1},{y1})-({x2},{y2})  {detail:18s} -> {verdict} {flags}")
        n += 1
    if n == 0:
        print("  no person-candidates")
    print()
