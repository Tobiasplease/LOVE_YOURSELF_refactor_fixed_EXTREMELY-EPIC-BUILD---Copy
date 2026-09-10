#!/usr/bin/env python3
"""Run the drawing review over real captures, without a drawing or the rig.

Crops each capture the way the ritual does, then asks the review pass what
landed. Pass an intent to judge against, or let it use the pairs below — the
point of those is that the SAME sheet gets a matching intent and a missed one,
because a review that only ever says "rough sketch" is not judging anything.

Needs llama-server up. Writes its crops to a temp dir, never into the capture
folder (an over-broad cleanup glob ate four real captures on Sep 10).

Usage: python debug/test_finished_review.py [image] [intent]
"""

import os
import sys
import tempfile

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from drawing.finished_review import review_finished_drawing  # noqa: E402
from drawing.sheet_crop import crop_to_sheet, enhance  # noqa: E402

# (capture, intent) — the middle one deliberately misses, to prove the pass can
# say so. Its real intent that evening was a low dense pool of hatching; what
# landed was clustered high inside a rounded shape.
CASES = [
    ("finished_20260910_223751_1.jpg", "A rounded head or hood in outline, with fine hatching packed inside its lower half"),
    ("finished_20260910_223751_1.jpg", "A tight cluster of fine black ink hatching sits low on the paper, a dense pool of shadow, not an object"),
    ("finished_20260910_221152_1.jpg", "A seated figure at a desk, and the screen he is looking at"),
]

tmp = tempfile.mkdtemp(prefix="review_")
cases = [(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "")] if len(sys.argv) > 1 else CASES

for name, intent in cases:
    path = name if os.path.isabs(name) or os.path.exists(name) else os.path.join("event_log/finished_drawings", name)
    frame = cv2.imread(path)
    if frame is None:
        print(f"{name}: unreadable")
        continue
    crop, box, method = crop_to_sheet(frame)
    shot = os.path.join(tmp, os.path.basename(path))
    cv2.imwrite(shot, enhance(crop), [cv2.IMWRITE_JPEG_QUALITY, 95])

    print(f"\n=== {os.path.basename(path)}  [{method}] ===")
    print(f"intent: {intent}")
    print(f"review: {review_finished_drawing(image_path=shot, intent=intent)}")

print(f"\nCrops: {tmp}")
