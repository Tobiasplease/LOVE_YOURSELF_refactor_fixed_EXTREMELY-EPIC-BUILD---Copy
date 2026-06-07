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
        self._target_fps = target_fps
        self._min_interval = 1.0 / target_fps
        self._max_frames = int(target_fps * max_seconds)

        # Storage: (timestamp, jpeg_bytes, diff_score)
        self._buffer: deque[Tuple[float, bytes, float]] = deque(maxlen=self._max_frames)
        self._lock = threading.Lock()

        self._last_push_time = 0.0
        self._last_frame_gray: Optional[np.ndarray] = None

    def push(self, frame: np.ndarray) -> None:
        """Push a frame into the buffer. Rate-limited to target_fps.
        Frames are stored as JPEG bytes to keep memory bounded.
        """
        now = time.time()
        if now - self._last_push_time < self._min_interval:
            return

        # Compute frame diff score for "interestingness"
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (160, 120))
        diff_score = 0.0
        if self._last_frame_gray is not None:
            diff = cv2.absdiff(small, self._last_frame_gray)
            diff_score = float(diff.mean()) / 255.0
        self._last_frame_gray = small

        # Encode as JPEG
        _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        jpeg_bytes = jpeg.tobytes()

        with self._lock:
            self._buffer.append((now, jpeg_bytes, diff_score))
        self._last_push_time = now

    def get_recent(self, seconds: float = 10.0, max_frames: int = 20) -> List[bytes]:
        """Get the most recent frames spanning `seconds`, up to `max_frames`.
        Returns JPEG bytes in chronological order.
        """
        cutoff = time.time() - seconds
        with self._lock:
            recent = [(ts, jpg, diff) for ts, jpg, diff in self._buffer if ts >= cutoff]

        if not recent:
            return []

        # If we have more frames than max_frames, keep the most "interesting" ones
        # but always keep first and last for temporal coverage
        if len(recent) > max_frames:
            first = recent[0]
            last = recent[-1]
            middle = sorted(recent[1:-1], key=lambda x: x[2], reverse=True)
            selected = [first] + middle[: max_frames - 2] + [last]
            selected.sort(key=lambda x: x[0])  # re-sort by time
            recent = selected

        return [jpg for _, jpg, _ in recent]

    def get_recent_with_metadata(self, seconds: float = 10.0, max_frames: int = 20) -> List[dict]:
        """Get recent frames with timestamps and diff scores."""
        cutoff = time.time() - seconds
        with self._lock:
            recent = [(ts, jpg, diff) for ts, jpg, diff in self._buffer if ts >= cutoff]

        if len(recent) > max_frames:
            first = recent[0]
            last = recent[-1]
            middle = sorted(recent[1:-1], key=lambda x: x[2], reverse=True)
            selected = [first] + middle[: max_frames - 2] + [last]
            selected.sort(key=lambda x: x[0])
            recent = selected

        return [
            {"timestamp": ts, "jpeg": jpg, "diff_score": diff}
            for ts, jpg, diff in recent
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


# Singleton
frame_buffer = FrameBuffer()
