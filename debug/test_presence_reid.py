"""Threshold-tuning check for session re-ID (perception/presence_identity.py).

Finds person crops in saved frames with YOLO, embeds them with the same CLIP
path the live system uses, and prints the pairwise cosine similarity matrix.
Same-person pairs should sit clearly above PRESENCE_REID_THRESHOLD and
different-person pairs below it. Crops are saved for eyeballing who is who.

Usage:
    python debug/test_presence_reid.py frame1.jpg frame2.jpg ...
"""

import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import PRESENCE_REID_THRESHOLD
from perception.presence_identity import PresenceIdentity

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_FRAMES = [
    "event_log/873a6770-images/mood_1785938232.jpg",
    "event_log/873a6770-images/mood_1785938024.jpg",
    "event_log/ccd4cfbc-images/mood_1785928841.jpg",
    "event_log/ccd4cfbc-images/mood_1785929221.jpg",
]


def main():
    frames = sys.argv[1:] or [os.path.join(REPO, f) for f in DEFAULT_FRAMES]
    out_dir = os.path.join(REPO, "debug", "reid_test_crops")
    os.makedirs(out_dir, exist_ok=True)

    from ultralytics import YOLO

    yolo = YOLO(os.path.join(REPO, "models", "yolov8m.pt"))
    pid = PresenceIdentity()

    crops = []  # (label, embedding)
    for fp in frames:
        img = cv2.imread(fp)
        if img is None:
            print(f"skipping unreadable {fp}")
            continue
        r = yolo.predict(img, conf=0.5, classes=[0], device="cpu", verbose=False)[0]
        for i, box in enumerate(r.boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            crop = img[y1:y2, x1:x2]
            if crop.shape[0] < 40 or crop.shape[1] < 20:
                continue
            label = f"{os.path.basename(fp).replace('mood_', '').replace('.jpg', '')}#{i}"
            cv2.imwrite(os.path.join(out_dir, f"{label}.jpg"), crop)
            crops.append((label, pid.embed_crop(crop)))
            print(f"person crop: {label}  ({x2 - x1}x{y2 - y1}px)")

    if len(crops) < 2:
        print("need at least two person crops")
        return

    print(f"\npairwise cosine similarity (threshold {PRESENCE_REID_THRESHOLD}):\n")
    width = max(len(l) for l, _ in crops)
    print(" " * (width + 2) + "  ".join(f"{l[-6:]:>6s}" for l, _ in crops))
    for la, ea in crops:
        row = "  ".join(f"{float(ea @ eb):6.3f}" for _, eb in crops)
        print(f"{la:>{width}s}  {row}")
    print(f"\ncrops saved to {out_dir} — check who is who")


if __name__ == "__main__":
    main()
