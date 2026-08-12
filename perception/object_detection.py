# object_detection.py

import threading
import time
import warnings

import cv2
from ultralytics import YOLO

from config.config import YOLO_CONFIDENCE_THRESHOLD, YOLO_INTERVAL_IDLE, YOLO_INTERVAL_TRACKING, YOLO_MODEL_PATH, YOLO_PERSON_MIN_AREA_FRAC
from perception.detection_memory import DetectionMemory

# Suppress ultralytics config warnings
warnings.filterwarnings("ignore", message=".*attempted relative import.*")


class ObjectDetectionThread(threading.Thread):
    def __init__(self, model_path: str = YOLO_MODEL_PATH, update_interval: float = YOLO_INTERVAL_IDLE):
        super().__init__()
        self.model = YOLO(model_path)
        self.update_interval = update_interval
        self.running = True
        self.shared_frame = None
        self.lock = threading.Lock()
        self.force_cpu = False  # fallback to CPU on CUDA OOM
        self._tracking_mode = False  # When True, use fast interval
        self._target_track_id = None  # sticky gaze target across detection cycles

    def set_frame(self, frame):
        with self.lock:
            self.shared_frame = frame.copy()

    def set_tracking_mode(self, is_tracking: bool):
        """Switch between fast tracking mode and idle mode."""
        if is_tracking != self._tracking_mode:
            self._tracking_mode = is_tracking
            self.update_interval = YOLO_INTERVAL_TRACKING if is_tracking else YOLO_INTERVAL_IDLE
            mode_name = "TRACKING" if is_tracking else "IDLE"
            print(f"[YOLOv8] Switched to {mode_name} mode (interval: {self.update_interval}s)")

    def run(self):
        print("[YOLOv8] Object detection thread started.")
        while self.running:
            cycle_start = time.time()
            with self.lock:
                frame = self.shared_frame.copy() if self.shared_frame is not None else None

            if frame is None:
                time.sleep(0.1)
                continue

            # Check if model is still available
            if self.model is None:
                time.sleep(0.1)
                continue

            clean_frame = frame.copy()
            try:
                # ByteTrack tracking: persist=True maintains track IDs across frames
                if self.force_cpu:
                    results = self.model.track(frame, persist=True, tracker="config/bytetrack_custom.yaml", verbose=False, imgsz=512, device="cpu")[0]
                else:
                    results = self.model.track(frame, persist=True, tracker="config/bytetrack_custom.yaml", verbose=False, imgsz=512)[0]
            except Exception as e:
                if "CUDA out of memory" in str(e) or "CUDA" in str(e):
                    print("[YOLOv8] CUDA OOM detected. Falling back to CPU for detection.")
                    self.force_cpu = True
                    time.sleep(self.update_interval)
                    continue
                else:
                    print(f"[YOLOv8] Detection error: {e}")
                    time.sleep(self.update_interval)
                    continue
            detected = set()
            person_count = 0
            person_tracks = {}  # {track_id: (bbox, confidence)}
            untracked_bbox = None
            untracked_conf = 0.0

            for box in results.boxes:
                cls_id = int(box.cls[0])
                label = self.model.names[cls_id]
                conf = float(box.conf[0])

                if cls_id != 0:
                    continue

                if conf < YOLO_CONFIDENCE_THRESHOLD:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                # Tiny "persons" are phantoms (a real person in this room is
                # never this small); filtering by size beats raising the conf
                # threshold, which worsens the seated-still-person misses.
                if (x2 - x1) * (y2 - y1) < YOLO_PERSON_MIN_AREA_FRAC * frame.shape[0] * frame.shape[1]:
                    continue

                # Get persistent tracking ID from ByteTrack
                track_id = int(box.id[0]) if box.id is not None else None

                detected.add(label)
                person_count += 1

                id_label = f"#{track_id}" if track_id is not None else "?"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"person {id_label} ({conf:.2f})", (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                if track_id is not None:
                    person_tracks[track_id] = ((x1, y1, x2, y2), conf)
                elif conf > untracked_conf:
                    untracked_bbox, untracked_conf = (x1, y1, x2, y2), conf

            # Sticky target: keep following the same ByteTrack ID while its track
            # lives — per-frame confidence argmax made gaze jump between people.
            if self._target_track_id not in person_tracks:
                self._target_track_id = max(person_tracks, key=lambda t: person_tracks[t][1]) if person_tracks else None

            if self._target_track_id is not None:
                best_person_bbox, best_person_conf = person_tracks[self._target_track_id]
            else:
                best_person_bbox, best_person_conf = untracked_bbox, untracked_conf

            DetectionMemory.update(
                list(detected), time.time(), clean_frame,
                best_person_bbox, best_person_conf,
                person_count=person_count,
                best_track_id=self._target_track_id,
                person_tracks=person_tracks,
            )

            # Chunked sleep: inference time counts toward the interval, and a
            # mode switch takes effect now instead of after a full idle sleep.
            # CPU fallback runs ~25x slower — cap its cadence and always yield
            # at least once, or a fallen-back model saturates every core and
            # starves the camera loop (gaze + lung freeze).
            interval = max(self.update_interval, 1.0) if self.force_cpu else self.update_interval
            time.sleep(0.05)
            while self.running and (time.time() - cycle_start) < interval:
                time.sleep(0.05)

    def stop(self):
        print("[YOLOv8] Stopping object detection thread...")
        self.running = False

        # Clean up YOLO model resources
        if hasattr(self, "model") and self.model is not None:
            try:
                # Clear YOLO model cache and free resources
                if hasattr(self.model, "model"):
                    del self.model.model
                del self.model
                self.model = None
                print("[YOLOv8] Model resources cleaned up")
            except Exception as e:
                print(f"[YOLOv8] Warning: Error cleaning up model: {e}")

        # Clear shared frame
        with self.lock:
            self.shared_frame = None
