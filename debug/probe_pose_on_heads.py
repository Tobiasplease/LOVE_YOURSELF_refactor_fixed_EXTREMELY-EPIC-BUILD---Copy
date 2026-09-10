"""Probe (Sep 10 2026): run the pose model on saved frames and show, per
person-box, which keypoints it finds and whether the skeleton gate passes —
to see WHY a silicone head on a desk gets through a gate built to require a
body. CPU only, so it never competes with the running machine.

Run:  python debug/probe_pose_on_heads.py frame1.jpg [frame2.jpg ...]
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import numpy as np
from ultralytics import YOLO
from config.config import YOLO_MODEL_PATH, YOLO_SKELETON_KP_CONF, YOLO_SKELETON_MIN_KEYPOINTS, YOLO_SKELETON_MIN_REGIONS, YOLO_CONFIDENCE_THRESHOLD
from perception.object_detection import _KP_REGIONS

NAMES = ["nose","l_eye","r_eye","l_ear","r_ear","l_shoulder","r_shoulder","l_elbow","r_elbow","l_wrist","r_wrist","l_hip","r_hip","l_knee","r_knee","l_ankle","r_ankle"]
REGION_NAMES = ["head","arms","hips","legs"][:len(_KP_REGIONS)]
model = YOLO(YOLO_MODEL_PATH)
print(f"weights: {YOLO_MODEL_PATH}  gate: >={YOLO_SKELETON_MIN_KEYPOINTS} kp @conf>{YOLO_SKELETON_KP_CONF} in >={YOLO_SKELETON_MIN_REGIONS} regions")
for path in sys.argv[1:]:
    r = model(path, imgsz=512, device="cpu", verbose=False)[0]
    kps = getattr(r, "keypoints", None)
    print(f"\n{os.path.basename(path)}: {len(r.boxes)} boxes")
    for i, box in enumerate(r.boxes):
        if int(box.cls[0]) != 0: continue
        conf = float(box.conf[0]); x1,y1,x2,y2 = map(int, box.xyxy[0])
        if kps is None or kps.conf is None:
            print(f"  person conf={conf:.2f} box=({x1},{y1},{x2},{y2})  [no keypoints — not a pose model: GATE SKIPPED]"); continue
        c = kps.conf[i].cpu().numpy(); strong = c > YOLO_SKELETON_KP_CONF
        total = int(strong.sum()); regs = [sum(bool(strong[j]) for j in idxs) for idxs in _KP_REGIONS]
        nreg = sum(1 for n in regs if n >= 2)
        ok = total >= YOLO_SKELETON_MIN_KEYPOINTS and nreg >= YOLO_SKELETON_MIN_REGIONS
        found = [f"{NAMES[j]}:{c[j]:.2f}" for j in range(len(c)) if strong[j]]
        print(f"  person conf={conf:.2f} box=({x1},{y1},{x2},{y2}) {'PASSES gate' if ok else 'rejected'}  kp={total} regions={dict(zip(REGION_NAMES,regs))}")
        print(f"     confident keypoints: {found}")
