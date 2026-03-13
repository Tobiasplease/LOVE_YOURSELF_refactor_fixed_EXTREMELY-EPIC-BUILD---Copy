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

        # Rolling window for detection rate
        self._recent_detections = []  # List of (timestamp, detected_bool)
        self._window_seconds = 2.0  # 2 second rolling window

        # Initialize ArUco detector
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters()

        # Optimize for low-light/variable conditions
        self.aruco_params.adaptiveThreshWinSizeMin = 3
        self.aruco_params.adaptiveThreshWinSizeMax = 27
        self.aruco_params.adaptiveThreshWinSizeStep = 3
        self.aruco_params.adaptiveThreshConstant = 5
        self.aruco_params.minMarkerPerimeterRate = 0.005
        self.aruco_params.maxMarkerPerimeterRate = 4.0
        self.aruco_params.polygonalApproxAccuracyRate = 0.08
        self.aruco_params.minCornerDistanceRate = 0.015
        self.aruco_params.minDistanceToBorder = 1

        # Try new API, fall back to legacy
        try:
            self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
            self.use_new_api = True
        except AttributeError:
            self.detector = None
            self.use_new_api = False

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
                    corners, ids, _ = cv2.aruco.detectMarkers(
                        frame, self.aruco_dict, parameters=self.aruco_params
                    )

                detected = ids is not None and len(ids) > 0
                now = time.time()

                # Update rolling window
                self._recent_detections.append((now, detected))
                # Remove old entries
                self._recent_detections = [
                    (t, d) for t, d in self._recent_detections
                    if now - t < self._window_seconds
                ]

                # Calculate detection rate over window
                if self._recent_detections:
                    detection_rate = sum(1 for _, d in self._recent_detections if d) / len(self._recent_detections)
                else:
                    detection_rate = 0.0

                # Update state
                with self.lock:
                    self.detection_confidence = detection_rate
                    # Marker considered "visible" if detected in >10% of recent frames
                    self.marker_visible = detection_rate > 0.10
                    if detected:
                        self.last_detection_time = now
                        self.detected_ids = set(ids.flatten().tolist())

            except Exception as e:
                print(f"[ArUco] Detection error: {e}")

            time.sleep(self.update_interval)

    def is_marker_visible(self) -> bool:
        """Check if marker is currently visible (paper NOT present)."""
        with self.lock:
            return self.marker_visible

    def get_detection_confidence(self) -> float:
        """Get rolling detection rate (0.0 to 1.0)."""
        with self.lock:
            return self.detection_confidence

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


def is_paper_present() -> bool:
    """
    Quick check: is paper covering the marker?
    Returns True if paper is present (marker NOT visible).
    Returns False if no paper (marker IS visible).
    """
    detector = get_aruco_detector()
    marker_visible = detector.is_marker_visible()
    # Marker visible = no paper covering it
    return not marker_visible
