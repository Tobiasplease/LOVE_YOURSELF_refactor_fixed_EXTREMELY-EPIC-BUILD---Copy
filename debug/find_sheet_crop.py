#!/usr/bin/env python3
"""Check the finished-drawing sheet crop against real captures, and re-derive
the nominal box when the rig moves.

Runs drawing/sheet_crop over every capture in event_log/finished_drawings (or
whatever paths you pass), reports the box each one picked and how much of the
frame it keeps, and prints a FINISHED_CAPTURE_SHEET_BOX line covering all of
them. Writes a preview strip — raw crop above sharpened crop — to a temp dir,
never into the capture folder: an over-broad cleanup glob ate four real
captures on Sep 10 and they were not recoverable.

Usage: python debug/find_sheet_crop.py [image ...]
"""

import glob
import os
import sys
import tempfile

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from drawing.sheet_crop import _detect, crop_to_sheet, enhance  # noqa: E402

paths = sys.argv[1:] or sorted(glob.glob("event_log/finished_drawings/finished_*.jpg"))
paths = [p for p in paths if not p.endswith("_sheet.jpg")]
if not paths:
    print("No captures found. Pass image paths, or run a drawing first.")
    raise SystemExit(1)

out_dir = tempfile.mkdtemp(prefix="sheet_crop_")
spans = []

for p in paths:
    frame = cv2.imread(p)
    if frame is None:
        print(f"{os.path.basename(p)}: unreadable")
        continue
    h, w = frame.shape[:2]
    raw, box, method = crop_to_sheet(frame)
    share = ((box[2] - box[0]) * (box[3] - box[1])) / float(w * h)
    found = _detect(frame)
    spans.append(found or box)

    gray_raw = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray_sharp = cv2.cvtColor(enhance(raw), cv2.COLOR_BGR2GRAY).astype(np.float32)
    print(f"{os.path.basename(p)}: box={box} {method} keeps {share:.0%} of frame, " f"crop {raw.shape[1]}x{raw.shape[0]}, detected={found}")
    print(
        f"    darkest 2% {np.percentile(gray_raw, 2):.0f} -> {np.percentile(gray_sharp, 2):.0f}   "
        f"paper p50 {np.percentile(gray_raw, 50):.0f} -> {np.percentile(gray_sharp, 50):.0f}"
        "   (paper should barely move)"
    )

    strip = np.vstack([raw, enhance(raw)])
    cv2.imwrite(os.path.join(out_dir, os.path.basename(p).replace(".jpg", "_strip.jpg")), strip)

if spans:
    # Frame fractions covering every sheet seen, which is what the nominal box
    # has to do — it is unioned in so a bad detection can never clip a drawing.
    x1 = min(s[0] for s in spans) / w
    y1 = min(s[1] for s in spans) / h
    x2 = max(s[2] for s in spans) / w
    y2 = max(s[3] for s in spans) / h
    pad = 25.0
    print(f"\nCovering all {len(spans)} sheets, with {pad:.0f}px margin:")
    print(
        f"FINISHED_CAPTURE_SHEET_BOX = ({max(0, x1 - pad / w):.3f}, {max(0, y1 - pad / h):.3f}, "
        f"{min(1, x2 + pad / w):.3f}, {min(1, y2 + pad / h):.3f})"
    )

print(f"\nPreviews (raw above sharpened): {out_dir}")
