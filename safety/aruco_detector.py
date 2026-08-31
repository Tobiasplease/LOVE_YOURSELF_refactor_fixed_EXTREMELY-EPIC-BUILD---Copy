"""
Real-time ArUco marker detection thread.
Runs continuously like YOLO/face detection - just check the result when needed.
"""

import threading
import time

import cv2


class ArucoDetectorThread(threading.Thread):
    """Background thread for continuous ArUco marker detection."""

    def __init__(self, update_interval: float = 0.1):
        super().__init__(daemon=True)
        self.update_interval = update_interval
        self.running = True
        self.shared_frame = None
        self.lock = threading.Lock()

        # Detection state - read this to check if marker is visible
        self.marker_visible = False
        self.last_detection_time = 0
        self.detected_ids = set()
        self.detection_confidence = 0.0  # Rolling average of recent detections
        self.detected_corners = []  # For visualization: list of (corners, marker_id)
        self.all_raw_corners = []  # ALL detected corners before ID filtering (for debug)

        # Rolling window for detection rate
        self._recent_detections = []  # List of (timestamp, detected_bool)
        self._window_seconds = 2.0  # 2 second rolling window

        # Only accept this specific marker ID (reduces false positives)
        self.valid_marker_id = 0  # The actual printed marker is ID 0
        self.min_marker_pixels = 40  # Minimum marker side length in pixels

        # Initialize ArUco detector
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters()

        # Default parameters — new camera is clean, no noise compensation needed

        # Use new ArucoDetector API
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        self.use_new_api = True

    def set_frame(self, frame):
        """Called by main loop to provide latest camera frame."""
        with self.lock:
            self.shared_frame = frame.copy() if frame is not None else None

    def run(self):
        """Continuous detection loop."""
        print("[ArUco] Marker detection thread started.")
        while self.running:
            frame = None
            with self.lock:
                if self.shared_frame is not None:
                    frame = self.shared_frame.copy()

            if frame is None:
                time.sleep(0.05)
                continue

            # Run detection
            try:
                if self.use_new_api:
                    corners, ids, _ = self.detector.detectMarkers(frame)
                else:
                    corners, ids, _ = cv2.aruco.detectMarkers(frame, self.aruco_dict, parameters=self.aruco_params)

                # Filter detections: only accept valid ID and minimum size
                detected = False
                valid_ids = set()
                valid_corners = []
                raw_corners = []  # All corners before filtering

                # Track total frames for debug
                if not hasattr(self, "_total_frames"):
                    self._total_frames = 0
                self._total_frames += 1

                # Save a frame after 100 total frames (one-time debug)
                if self._total_frames == 100:
                    h, w = frame.shape[:2] if frame is not None else (0, 0)
                    cv2.imwrite("/tmp/aruco_debug_frame.png", frame)
                    print(f"[ArUco] DEBUG: Saved {w}x{h} frame to /tmp/aruco_debug_frame.png")

                if ids is not None and len(ids) > 0:
                    for i, marker_id in enumerate(ids.flatten()):
                        if corners and len(corners) > i:
                            marker_corners = corners[i][0]
                            raw_corners.append((marker_corners.copy(), int(marker_id)))

                            side1 = ((marker_corners[0][0] - marker_corners[1][0]) ** 2 + (marker_corners[0][1] - marker_corners[1][1]) ** 2) ** 0.5
                            side2 = ((marker_corners[1][0] - marker_corners[2][0]) ** 2 + (marker_corners[1][1] - marker_corners[2][1]) ** 2) ** 0.5
                            avg_side = (side1 + side2) / 2

                            if int(marker_id) == self.valid_marker_id and avg_side >= self.min_marker_pixels:
                                detected = True
                                valid_ids.add(int(marker_id))
                                valid_corners.append((marker_corners.copy(), int(marker_id)))

                now = time.time()

                # Update rolling window
                self._recent_detections.append((now, detected))
                # Remove old entries
                self._recent_detections = [(t, d) for t, d in self._recent_detections if now - t < self._window_seconds]

                # Calculate detection rate over window
                if self._recent_detections:
                    detection_rate = sum(1 for _, d in self._recent_detections if d) / len(self._recent_detections)
                else:
                    detection_rate = 0.0

                # Update state
                with self.lock:
                    self.detection_confidence = detection_rate
                    # Marker considered "visible" if detected in >5% of recent frames (more sensitive)
                    self.marker_visible = detection_rate > 0.05
                    self.all_raw_corners = raw_corners  # ALL detections for debug viz
                    if detected:
                        self.last_detection_time = now
                        self.detected_ids = valid_ids
                        self.detected_corners = valid_corners
                    else:
                        self.detected_corners = []

            except Exception as e:
                print(f"[ArUco] Detection error: {e}")

            time.sleep(self.update_interval)

    def reset_detection_state(self) -> None:
        """Clear rolling window — call before a paper check so stale detections don't carry over."""
        with self.lock:
            self._recent_detections.clear()
            self.marker_visible = False
            self.detection_confidence = 0.0

    def get_status(self) -> dict:
        """Get full detection status."""
        with self.lock:
            return {
                "marker_visible": self.marker_visible,
                "confidence": self.detection_confidence,
                "last_detection": self.last_detection_time,
                "detected_ids": list(self.detected_ids),
                "window_size": len(self._recent_detections),
            }

    def get_corners_for_drawing(self, include_rejected: bool = True):
        """Get marker corners for visualization.
        Returns: list of (corners, marker_id, is_valid) tuples
        """
        with self.lock:
            result = []
            # Add all raw detections (for debug)
            if include_rejected:
                for corners, marker_id in self.all_raw_corners:
                    is_valid = marker_id == self.valid_marker_id
                    result.append((corners, marker_id, is_valid))
            else:
                # Only valid markers
                for corners, marker_id in self.detected_corners:
                    result.append((corners, marker_id, True))
            return result

    def stop(self):
        """Stop the detection thread."""
        print("[ArUco] Stopping marker detection thread...")
        self.running = False


# Global instance
_aruco_detector = None


def get_aruco_detector() -> ArucoDetectorThread:
    """Get or create the global ArUco detector instance."""
    global _aruco_detector
    if _aruco_detector is None:
        _aruco_detector = ArucoDetectorThread()
        _aruco_detector.start()
    return _aruco_detector
