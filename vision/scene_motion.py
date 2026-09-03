"""Ego-motion-compensated scene motion.

The servo camera sways (breathing) and saccades (gaze nudges) almost
constantly, so raw pixel diff and per-frame servo deltas cannot separate
"I moved" from "something moved". This module does it optically:

  1. Track corner features between consecutive frames (Lucas-Kanade flow).
  2. Fit the dominant rigid transform with RANSAC — when the camera moves,
     the whole frame shifts together, so that transform IS the ego-motion.
  3. Warp the previous frame through the transform (undoing the camera's
     movement) and diff against the current frame.
  4. The fraction of pixels still changing is true scene motion — a person
     shifting, a door opening — regardless of what the camera was doing.

Unlike YOLO person-angle, this also sees non-person motion. Both signals
feed captioner._assess_scene.

Pure OpenCV (offline-safe), ~2ms per call at 320x240, called at the frame
buffer's 2fps push cadence.
"""

from typing import Optional

import cv2
import numpy as np

# Pixels of post-compensation intensity change that count as "moving"
_DIFF_THRESHOLD = 28
# Feature tracking parameters
_MAX_CORNERS = 200
_MIN_TRACKED = 20
# Beyond this the camera made a saccade (gaze nudge ~12° ≈ 60px at 320w):
# frames are motion-blurred and compensation residue swamps real motion —
# refuse to measure rather than report noise (live run June 12: constant
# 0.07-0.24 false residual kept salience permanently hot). Breathing sway
# (~1° ≈ 5px) stays well inside the trusted range.
_MAX_TRUSTED_SHIFT_PX = 25.0


class SceneMotionEstimator:
    def __init__(self) -> None:
        self._prev_gray: Optional[np.ndarray] = None

    def reset(self) -> None:
        self._prev_gray = None

    def update(self, gray: np.ndarray) -> dict:
        """Feed the next (downscaled, grayscale) frame.

        Returns:
            valid: bool — False when flow could not be estimated (too few
                features, first frame); callers should fall back gracefully
            camera_shift_px: float — magnitude of the camera's own movement
            residual_fraction: float 0..1 — share of the frame still moving
                after the camera's movement is mathematically undone
            reason: str|None — why valid is False. "saccade" is load-bearing
                provenance (Sep 3, world-anchor): it means "the CAMERA jumped,
                not the world" — downstream must not treat it like the other
                can't-measure cases
        """
        result = {"valid": False, "camera_shift_px": 0.0, "residual_fraction": 0.0, "reason": None}
        prev = self._prev_gray
        self._prev_gray = gray
        if prev is None or prev.shape != gray.shape:
            result["reason"] = "first_frame"
            return result

        corners = cv2.goodFeaturesToTrack(prev, maxCorners=_MAX_CORNERS, qualityLevel=0.01, minDistance=8)
        if corners is None or len(corners) < _MIN_TRACKED:
            result["reason"] = "few_features"
            return result

        moved, status, _err = cv2.calcOpticalFlowPyrLK(prev, gray, corners, None)
        ok = status.ravel() == 1
        src, dst = corners[ok], moved[ok]
        if len(src) < _MIN_TRACKED:
            result["reason"] = "few_features"
            return result

        transform, _inliers = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=2.0)
        if transform is None:
            result["reason"] = "no_transform"
            return result

        result["camera_shift_px"] = float(np.hypot(transform[0, 2], transform[1, 2]))
        if result["camera_shift_px"] > _MAX_TRUSTED_SHIFT_PX:
            result["reason"] = "saccade"
            return result  # blurred frames, measurement would be noise

        # Undo the camera's movement, then diff. Borders the warp can't fill
        # are masked out; means are matched so auto-exposure shifts during
        # pans don't read as motion.
        h, w = gray.shape
        warped = cv2.warpAffine(prev, transform, (w, h))
        coverage = cv2.warpAffine(np.full((h, w), 255, dtype=np.uint8), transform, (w, h))
        mask = coverage > 250
        if mask.sum() < (h * w) * 0.5:
            result["reason"] = "low_coverage"
            return result  # camera moved too far to compare

        a = gray.astype(np.int16)
        b = warped.astype(np.int16)
        b = b + int(a[mask].mean() - b[mask].mean())
        diff = np.abs(a - b)
        moving = ((diff > _DIFF_THRESHOLD) & mask).astype(np.uint8)
        # Erode away single-pixel speckle (sensor noise, compensation residue
        # along edges) — a real moving body survives, noise doesn't
        moving = cv2.erode(moving, np.ones((3, 3), np.uint8))
        result["residual_fraction"] = float(moving.sum()) / float(mask.sum())
        result["valid"] = True
        return result
