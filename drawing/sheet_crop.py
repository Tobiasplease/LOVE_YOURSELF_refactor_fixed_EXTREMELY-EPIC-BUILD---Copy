#!/usr/bin/env python3
"""
Crop a finished-drawing capture down to the sheet (Sep 10 2026).

The vision tower reads an image as 32x32-pixel cells — patch_size 16 with
spatial_merge_size 2, per mmproj-F16.gguf — and `--image-min-tokens 1024` puts
a floor of ~1024 of them on every image. A full 1280x720 table view spends
about 160 of those cells on the paper and about 13 on the marks themselves;
the rest go to the floor, the chair and the curtain. Cropping to the sheet
hands the whole budget to the drawing, which is worth more than any amount of
straightening.

Deliberately NOT rectified and NOT contrast-stretched. The sheet is a trapezoid
from this angle and the model reads it fine; more to the point, plenty of these
drawings come out faint, and how legible the marks are is something the machine
needs to know when it judges what it made. Cleaning that up would be lying to
it about its own hand.

The box is FIXED (FINISHED_CAPTURE_SHEET_BOX), sized to cover the sheet's
hand-placed wander — 61px in x and 27px in y across the Sep 10 captures.
Detection no longer feeds the crop: it was unioned with the nominal box, so it
could only widen it, and once the nominal covered the wander it contributed
nothing at all. It survives as sheet_warning(), which says the box has gone
stale rather than silently changing the crop. Re-derive after the rig moves:
python debug/find_sheet_crop.py --live 5
"""

from typing import Optional, Tuple

import cv2
import numpy as np

from config import config as _cfg

Box = Tuple[int, int, int, int]  # x1, y1, x2, y2


def _nominal(w: int, h: int) -> Box:
    """The configured sheet box, stored as frame fractions so it survives a
    capture-resolution change."""
    fx1, fy1, fx2, fy2 = getattr(_cfg, "FINISHED_CAPTURE_SHEET_BOX", (0.20, 0.50, 0.74, 0.99))
    return (int(fx1 * w), int(fy1 * h), int(fx2 * w), int(fy2 * h))


def _detect(frame: np.ndarray) -> Optional[Box]:
    """Largest bright, desaturated blob in the lower half — the paper.

    Returns None when nothing passes the sanity gates, which is the signal to
    fall back to the nominal box rather than to trust a bad mask.
    """
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    value = hsv[..., 2].astype(np.int16)
    sat = hsv[..., 1].astype(np.int16)

    mask = ((value > np.percentile(value, 90) * 0.82) & (sat < 60)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    frame_area = float(w * h)
    best = None
    for contour in contours:
        x, y, cw, ch = cv2.boundingRect(contour)
        if ch <= 0:
            continue
        area = cv2.contourArea(contour)
        if not (0.06 * frame_area <= area <= 0.50 * frame_area):
            continue
        if y + ch / 2.0 < 0.45 * h:  # the table is always in the lower half
            continue
        if not (1.2 <= cw / float(ch) <= 3.5):
            continue
        if best is None or area > best[0]:
            best = (area, x, y, x + cw, y + ch)
    return best[1:] if best else None


def sheet_box(frame: np.ndarray) -> Tuple[Box, str]:
    """Where to crop. The configured box, always.

    Detection used to feed this, unioned with the nominal box so it could only
    ever widen the crop — which meant that once the nominal was sized to cover
    the sheet's hand-placed wander, detection contributed nothing. Measured on
    the Sep 14 frames: detected+margin (244, 299, 854, 720) inside nominal
    (183, 272, 915, 720), final crop identical to nominal. It was also the
    fragile half, keying on bright desaturated paper at exactly the moment the
    paper is covered in ink — on Sep 10 it returned a box 742px wide instead of
    919 when a shadow crossed the sheet, and the nominal rescued it.

    So detection is demoted to a staleness check (see sheet_warning): it says
    the box needs re-deriving, rather than silently changing the crop.
    """
    h, w = frame.shape[:2]
    return _nominal(w, h), "nominal"


def sheet_warning(frame: np.ndarray) -> Optional[str]:
    """Is the sheet still inside the configured box? Never raises.

    Returns a line worth logging when the sheet has moved out from under the
    crop — camera knocked, table shifted, paper badly placed — which is the
    signal to re-derive with debug/find_sheet_crop.py --live. Returns None when
    all is well, or when the sheet simply could not be found (an inked sheet is
    hard to detect, and that is not evidence the box is wrong).
    """
    try:
        h, w = frame.shape[:2]
        found = _detect(frame)
        if found is None:
            return None
        x1, y1, x2, y2 = _nominal(w, h)
        out = []
        if found[0] < x1:
            out.append(f"{x1 - found[0]}px past the left")
        if found[1] < y1:
            out.append(f"{y1 - found[1]}px past the top")
        if found[2] > x2:
            out.append(f"{found[2] - x2}px past the right")
        if found[3] > y2:
            out.append(f"{found[3] - y2}px past the bottom")
        if not out:
            return None
        return f"sheet sits outside the crop box ({', '.join(out)}) — re-derive with debug/find_sheet_crop.py --live 5"
    except Exception:
        return None


def enhance(crop: np.ndarray) -> np.ndarray:
    """Undo what the lens did — not what the pen did.

    An unsharp mask, and nothing else. The tower resamples this into 32x32
    cells and hatching about 12px wide aliases into mush on the way down; a
    mild pre-sharpen is what survives the trip. Measured on the Sep 10
    captures, it deepens the ink (darkest 2% went 66 -> 59 at 0.6) while the
    paper median does not move (164 -> 163). That is the property that matters:
    the paper stays the reference point, so a drawing that came out faint still
    reads as faint against its own sheet.

    Two things were tried and rejected, both because they destroy exactly that:

    - percentile contrast stretch: maps THIS frame's darkest ink to black, so a
      barely-there drawing arrives looking confident. It normalizes away the
      one judgement that matters most — whether the mark actually landed.
    - illumination flatten (divide by a blurred background): a dense hatching
      cluster drags its own local background down, so dividing by it brightens
      precisely the passages we care about. Measured: darkest 2% went 66 -> 125,
      i.e. it washed the drawing out to fix a shadow that was not the problem.
    """
    strength = float(getattr(_cfg, "FINISHED_CAPTURE_SHARPEN", 0.6))
    if strength <= 0:
        return crop

    out = crop.astype(np.float32)
    blur = cv2.GaussianBlur(out, (0, 0), 1.4)
    return np.clip(out * (1.0 + strength) - blur * strength, 0, 255).astype(np.uint8)


def crop_to_sheet(frame: np.ndarray) -> Tuple[np.ndarray, Box, str]:
    """Crop to the sheet. Returns the crop, the box used, and the method.

    Never raises and never returns an empty image — a degenerate box falls back
    to the whole frame, since an uncropped picture is still worth having.
    """
    try:
        box, method = sheet_box(frame)
        x1, y1, x2, y2 = box
        if x2 - x1 < 32 or y2 - y1 < 32:
            raise ValueError(f"degenerate sheet box {box}")
        return frame[y1:y2, x1:x2].copy(), box, method
    except Exception as e:
        h, w = frame.shape[:2]
        print(f"[📷] Sheet crop failed ({e}) — keeping the full frame")
        return frame, (0, 0, w, h), "full-frame"
