# object_detection.py

import threading
import time
from ultralytics import YOLO
import cv2
from config.config import YOLO_CONFIDENCE_THRESHOLD
from perception.detection_memory import DetectionMemory


class ObjectDetectionThread(threading.Thread):
    def __init__(self, model_path="models/yolov8m.pt", update_interval=5):  # ✨ Changed to yolov8m.pt
        super().__init__()
        self.model = YOLO(model_path)
        self.update_interval = update_interval
        self.running = True
        self.shared_frame = None
        self.lock = threading.Lock()

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

            clean_frame = frame.copy()
            results = self.model(frame, verbose=False)[0]
            detected = set()

            for box in results.boxes:
                cls_id = int(box.cls[0])
                label = self.model.names[cls_id]
                conf = float(box.conf[0])
                if conf < YOLO_CONFIDENCE_THRESHOLD:
                    continue
                detected.add(label)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} ({conf:.2f})", (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            DetectionMemory.update(list(detected), time.time(), clean_frame)

            time.sleep(self.update_interval)

    def stop(self):
        self.running = False