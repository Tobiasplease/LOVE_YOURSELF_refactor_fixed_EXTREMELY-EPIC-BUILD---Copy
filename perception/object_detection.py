# object_detection.py

import threading
import time
import warnings

import cv2
from ultralytics import YOLO

from config.config import YOLO_CONFIDENCE_THRESHOLD, YOLO_MODEL_PATH
from perception.detection_memory import DetectionMemory

# Suppress ultralytics config warnings
warnings.filterwarnings("ignore", message=".*attempted relative import.*")


class ObjectDetectionThread(threading.Thread):
    def __init__(self, model_path: str = YOLO_MODEL_PATH, update_interval: int = 5):  # default to small model
        super().__init__()
        self.model = YOLO(model_path)
        self.update_interval = update_interval
        self.running = True
        self.shared_frame = None
        self.lock = threading.Lock()
        self.force_cpu = False  # fallback to CPU on CUDA OOM

    def set_frame(self, frame):
        with self.lock:
            self.shared_frame = frame.copy()

    def run(self):
        print("[YOLOv8] Object detection thread started.")
        while self.running:
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
                # Use smaller inference size and CPU fallback when needed
                if self.force_cpu:
                    results = self.model(frame, verbose=False, imgsz=512, device="cpu")[0]
                else:
                    results = self.model(frame, verbose=False, imgsz=512)[0]
            except Exception as e:
                if "CUDA out of memory" in str(e) or "CUDA" in str(e):
                    print("[YOLOv8] CUDA OOM detected. Falling back to CPU for detection.")
                    self.force_cpu = True
                    time.sleep(self.update_interval)
                    continue
                else:
                    # Non-fatal: log and skip this cycle
                    print(f"[YOLOv8] Detection error: {e}")
                    time.sleep(self.update_interval)
                    continue
            detected = set()

            for box in results.boxes:
                cls_id = int(box.cls[0])
                label = self.model.names[cls_id]
                conf = float(box.conf[0])

                # Only process person detections (class ID 0 in COCO dataset)
                if cls_id != 0:  # 0 = person in YOLO COCO classes
                    continue

                if conf < YOLO_CONFIDENCE_THRESHOLD:
                    continue

                detected.add(label)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} ({conf:.2f})", (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            DetectionMemory.update(list(detected), time.time(), clean_frame)

            time.sleep(self.update_interval)

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
