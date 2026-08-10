"""Standalone check of the Phase 1 open-vocab detector service (no camera, no thread).

Feeds a saved frame through detect_once and the hot-swap path, prints what the machine
would see. Optionally draws the boxes to an output image.

Usage:
    python debug/test_open_vocab_detector.py [path/to/frame.jpg]
"""

import glob
import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from perception.open_vocab_detector import OpenVocabDetectorThread

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    if len(sys.argv) > 1:
        frame_path = sys.argv[1]
    else:
        candidates = sorted(glob.glob(os.path.join(REPO, "event_log", "*-images", "mood_*.jpg")), key=os.path.getmtime)
        frame_path = candidates[-1]
    print(f"frame: {frame_path}")

    frame = cv2.imread(frame_path)
    detector = OpenVocabDetectorThread()

    detections = detector.detect_once(frame)
    print(f"\n{len(detections)} detections:")
    for d in sorted(detections, key=lambda d: -d["conf"]):
        print(f"  {d['conf']:.2f}  {d['term']:24s} {d['box']}")

    out = frame.copy()
    for d in detections:
        x1, y1, x2, y2 = d["box"]
        cv2.rectangle(out, (x1, y1), (x2, y2), (255, 0, 255), 1)
        cv2.putText(out, f"{d['term']} {d['conf']:.2f}", (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
    out_path = os.path.join(REPO, "debug", "open_vocab_test_overlay.jpg")
    cv2.imwrite(out_path, out)
    print(f"\noverlay written to {out_path}")

    print("\nhot-swap check: shrinking vocabulary to 3 terms...")
    detector.set_vocabulary(["mannequin head", "pink shelf", "red foam finger"])
    with detector.lock:
        detector._vocabulary = detector._pending_vocabulary
        detector._pending_vocabulary = None
    detector.model.set_classes(detector._vocabulary)
    detections = detector.detect_once(frame)
    print(f"{len(detections)} detections after swap:")
    for d in sorted(detections, key=lambda d: -d["conf"]):
        print(f"  {d['conf']:.2f}  {d['term']:24s} {d['box']}")


if __name__ == "__main__":
    main()
