#!/usr/bin/env python3
"""Check the finished-drawing sheet crop against real captures, and re-derive
the nominal box when the rig moves.

Runs drawing/sheet_crop over every capture in event_log/finished_drawings (or
whatever paths you pass), reports the box each one picked and how much of the
frame it keeps, and prints a FINISHED_CAPTURE_SHEET_BOX line covering all of
them. Writes a preview strip — raw crop above sharpened crop — to a temp dir,
never into the capture folder: an over-broad cleanup glob ate four real
captures on Sep 10 and they were not recoverable.

--live N grabs N frames from the camera at the paper-look angle instead, for
when the rig has moved and there are no captures from the new position yet. It
needs machine.py stopped, since that holds the camera.

--drift widens the box for how far the SHEET wanders between drawings — it is
placed by hand, and across the Sep 10 captures it moved 61px in x and 27px in y.
A box derived from frames taken in one sitting has not seen that, so without the
allowance the first re-placed sheet falls outside it.

Usage: python debug/find_sheet_crop.py [image ...] [--live N] [--drift X,Y]
"""

import glob
import os
import sys
import tempfile

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from drawing.sheet_crop import _detect, crop_to_sheet, enhance  # noqa: E402

argv = sys.argv[1:]


def _opt(name, default):
    if name in argv:
        i = argv.index(name)
        val = argv[i + 1]
        del argv[i : i + 2]
        return val
    return default


live = int(_opt("--live", 0))
drift = tuple(float(v) for v in _opt("--drift", "61,27").split(","))

if live:
    from config.config import CAMERA_HEIGHT, CAMERA_INDEX, CAMERA_WIDTH, PAPER_DETECTION_GAZE_PAN, PAPER_DETECTION_GAZE_TILT
    from servo_control.servo_control import ServoController

    import time

    servos = ServoController(port="/dev/arduino_lunggaze", baudrate=9600)
    time.sleep(2.0)
    servos.set_pan(PAPER_DETECTION_GAZE_PAN)
    time.sleep(0.4)
    servos.set_tilt(PAPER_DETECTION_GAZE_TILT)
    time.sleep(1.8)
    cam = cv2.VideoCapture(CAMERA_INDEX)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    live_dir = tempfile.mkdtemp(prefix="sheet_live_")
    paths = []
    for i in range(live):
        for _ in range(8):
            cam.read()
            time.sleep(0.1)
        ok, fr = cam.read()
        if ok:
            fp = os.path.join(live_dir, f"live_{i}.jpg")
            cv2.imwrite(fp, fr, [cv2.IMWRITE_JPEG_QUALITY, 95])
            paths.append(fp)
    cam.release()
    print(f"Captured {len(paths)} frames at pan {PAPER_DETECTION_GAZE_PAN} / tilt {PAPER_DETECTION_GAZE_TILT} -> {live_dir}\n")
else:
    paths = argv or sorted(glob.glob("event_log/finished_drawings/finished_*.jpg"))
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
    dx, dy = (pad + drift[0]) / w, (pad + drift[1]) / h
    print(f"\nCovering all {len(spans)} sheets, {pad:.0f}px margin + {drift[0]:.0f}x{drift[1]:.0f}px sheet drift:")
    print(f"FINISHED_CAPTURE_SHEET_BOX = ({max(0, x1 - dx):.3f}, {max(0, y1 - dy):.3f}, " f"{min(1, x2 + dx):.3f}, {min(1, y2 + dy):.3f})")

print(f"\nPreviews (raw above sharpened): {out_dir}")
