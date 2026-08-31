"""
captioner/frame_buffer.py
-------------------------
Ring buffer that collects camera frames at ~2fps for temporal video analysis.
The main loop feeds frames continuously; on each caption cycle, the buffer
provides the last N seconds of frames for the super-frame pipeline.

Usage:
    from captioner.frame_buffer import frame_buffer

    # In main camera loop (every frame):
    frame_buffer.push(frame)

    # In caption cycle (every 10s):
    frames = frame_buffer.get_recent(seconds=10)
    # frames is a list of JPEG-encoded bytes, chronological order
"""

import threading
import time
from collections import deque
from typing import List, Optional, Tuple

import cv2
import numpy as np


class FrameBuffer:
    """Thread-safe ring buffer for camera frames with frame-diff gating."""

    def __init__(self, target_fps: float = 2.0, max_seconds: float = 30.0):
        self._min_interval = 1.0 / target_fps
        self._max_frames = int(target_fps * max_seconds)

        # Storage: (timestamp, jpeg_bytes, diff_score, detection_snapshot)
        self._buffer: deque[Tuple[float, bytes, float, dict]] = deque(maxlen=self._max_frames)
        self._lock = threading.Lock()

        self._last_push_time = 0.0
        self._last_frame_gray: Optional[np.ndarray] = None
        self._last_pan: Optional[float] = None
        self._last_tilt: Optional[float] = None

        from vision.scene_motion import SceneMotionEstimator

        self._motion_estimator = SceneMotionEstimator()

    def push(self, frame: np.ndarray, detection: Optional[dict] = None) -> None:
        """Push a frame into the buffer. Rate-limited to target_fps.
        Frames are stored as JPEG bytes to keep memory bounded.

        Args:
            frame: BGR numpy array
            detection: optional snapshot from detection layer, e.g.
                {"face": True, "person": True, "track_id": 2, "person_count": 1}
        """
        now = time.time()
        if now - self._last_push_time < self._min_interval:
            return

        # Compute frame diff score for "interestingness"
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (320, 240))
        diff_score = 0.0
        if self._last_frame_gray is not None:
            diff = cv2.absdiff(small, self._last_frame_gray)
            diff_score = float(diff.mean()) / 255.0
        self._last_frame_gray = small

        det = dict(detection or {})

        # Ego-compensated scene motion: optical flow estimates the camera's
        # own movement between pushes and undoes it; what still changes is
        # true scene motion, measurable even mid-sway. residual_motion is
        # None when flow couldn't be estimated — consumers fall back to the
        # servo-delta heuristic.
        try:
            flow = self._motion_estimator.update(small)
            det["residual_motion"] = flow["residual_fraction"] if flow["valid"] else None
        except Exception:
            det["residual_motion"] = None

        # Ego-motion: if the servo moved since the last push, this frame's diff
        # is (mostly) self-caused — the camera looked around, the room may be
        # still. Still used to pick steady frames for superframe pairing.
        pan, tilt = det.get("pan"), det.get("tilt")
        if pan is not None and tilt is not None and self._last_pan is not None:
            angular_delta = abs(pan - self._last_pan) + abs(tilt - self._last_tilt)
            det["ego_motion"] = angular_delta > 2.0
        else:
            det["ego_motion"] = False
        if pan is not None:
            self._last_pan, self._last_tilt = pan, tilt

        # Encode as JPEG
        _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        jpeg_bytes = jpeg.tobytes()

        with self._lock:
            self._buffer.append((now, jpeg_bytes, diff_score, det))
        self._last_push_time = now

    def _select_frames(self, seconds: float, max_frames: int) -> list:
        """Shared selection logic for recent frame retrieval."""
        cutoff = time.time() - seconds
        with self._lock:
            recent = [(ts, jpg, diff, det) for ts, jpg, diff, det in self._buffer if ts >= cutoff]

        if not recent:
            return []

        if len(recent) > max_frames:
            first = recent[0]
            last = recent[-1]
            middle = sorted(recent[1:-1], key=lambda x: x[2], reverse=True)
            selected = [first] + middle[: max_frames - 2] + [last]
            selected.sort(key=lambda x: x[0])
            recent = selected

        return recent

    def get_recent(self, seconds: float = 10.0, max_frames: int = 20) -> List[bytes]:
        """Get the most recent frames spanning `seconds`, up to `max_frames`.
        Returns JPEG bytes in chronological order.
        """
        return [jpg for _, jpg, _, _ in self._select_frames(seconds, max_frames)]

    def get_recent_with_metadata(self, seconds: float = 10.0, max_frames: int = 20) -> List[dict]:
        """Get recent frames with timestamps, diff scores, and detection snapshots."""
        return [
            {"timestamp": ts, "jpeg": jpg, "diff_score": diff, "detection": det} for ts, jpg, diff, det in self._select_frames(seconds, max_frames)
        ]

    @property
    def frame_count(self) -> int:
        with self._lock:
            return len(self._buffer)

    @property
    def seconds_buffered(self) -> float:
        with self._lock:
            if len(self._buffer) < 2:
                return 0.0
            return self._buffer[-1][0] - self._buffer[0][0]

    def clear(self) -> None:
        with self._lock:
            self._buffer.clear()
        self._last_frame_gray = None
        self._last_pan = None
        self._last_tilt = None
        self._motion_estimator.reset()


# Singleton
frame_buffer = FrameBuffer()
