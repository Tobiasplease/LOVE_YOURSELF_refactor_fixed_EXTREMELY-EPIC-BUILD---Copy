"""Compare YOLO model variants on studio frames for person false positives.

The studio is full of mannequin parts and robot limbs that yolov8n reads as
"person". This samples event_log frames, runs two models, and reports frames
where they disagree so the disagreements can be inspected by eye.

Usage:
    python debug/compare_yolo_models.py [n_samples]
"""

import glob
import json
import sys
import time

import cv2
from ultralytics import YOLO

CONF = 0.55  # matches YOLO_CONFIDENCE_THRESHOLD
MODEL_A = "models/yolov8n.pt"
MODEL_B = sys.argv[2] if len(sys.argv) > 2 else "yolov8m.pt"
N_SAMPLES = int(sys.argv[1]) if len(sys.argv) > 1 else 300


def person_hits(model, img):
    r = model(img, verbose=False, imgsz=512)[0]
    return [(round(float(b.conf[0]), 3), [int(v) for v in b.xyxy[0]]) for b in r.boxes if int(b.cls[0]) == 0 and float(b.conf[0]) >= CONF]


def main():
    model_a, model_b = YOLO(MODEL_A), YOLO(MODEL_B)
    for name, m in ((MODEL_A, model_a), (MODEL_B, model_b)):
        params = sum(p.numel() for p in m.model.parameters())
        print(f"{name}: {params/1e6:.1f}M params")

    paths = sorted(glob.glob("event_log/*-images/*.jpg"))
    step = max(1, len(paths) // N_SAMPLES)
    sample = paths[::step][:N_SAMPLES]
    print(f"sampling {len(sample)} of {len(paths)} frames, conf>={CONF}")

    results = {"both": [], "a_only": [], "b_only": [], "neither": 0}
    t_a = t_b = 0.0
    for p in sample:
        img = cv2.imread(p)
        if img is None:
            continue
        t0 = time.perf_counter()
        hits_a = person_hits(model_a, img)
        t_a += time.perf_counter() - t0
        t0 = time.perf_counter()
        hits_b = person_hits(model_b, img)
        t_b += time.perf_counter() - t0

        entry = {"path": p, "a": hits_a, "b": hits_b}
        if hits_a and hits_b:
            results["both"].append(entry)
        elif hits_a:
            results["a_only"].append(entry)
        elif hits_b:
            results["b_only"].append(entry)
        else:
            results["neither"] += 1

    n = len(sample)
    print(f"\nper-frame inference: {MODEL_A} {t_a/n*1000:.1f}ms | {MODEL_B} {t_b/n*1000:.1f}ms")
    print(f"both detect person: {len(results['both'])}")
    print(f"{MODEL_A} only:     {len(results['a_only'])}")
    print(f"{MODEL_B} only:     {len(results['b_only'])}")
    print(f"neither:            {results['neither']}")

    out = "debug/yolo_compare_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=1)
    print(f"\nfull results -> {out}")
    for key in ("a_only", "b_only"):
        print(f"\n{key} frames to eyeball:")
        for e in results[key][:10]:
            print(f"  {e['path']}  a={e['a']} b={e['b']}")


if __name__ == "__main__":
    main()
