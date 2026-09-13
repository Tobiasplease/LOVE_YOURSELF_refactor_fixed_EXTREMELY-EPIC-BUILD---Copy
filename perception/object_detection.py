# object_detection.py

import subprocess
import threading
import time
import warnings

import cv2
import numpy as np
from ultralytics import YOLO

from config.config import (
    YOLO_CONFIDENCE_THRESHOLD,
    YOLO_CUDA_RETRY_S,
    YOLO_INTERVAL_IDLE,
    YOLO_INTERVAL_TRACKING,
    YOLO_MODEL_PATH,
    YOLO_PERSON_MIN_AREA_FRAC,
    YOLO_SKELETON_KP_CONF,
    YOLO_SKELETON_MIN_KEYPOINTS,
    YOLO_SKELETON_MIN_REGIONS,
    YOLO_VRAM_MIN_MIB,
)
from perception.detection_memory import DetectionMemory

# COCO keypoint regions: a person is a head AND a body, not a head-like blob.
_KP_REGIONS = ((0, 1, 2, 3, 4), (5, 6, 11, 12), (7, 8, 9, 10, 13, 14, 15, 16))  # head / torso / limbs

# Suppress ultralytics config warnings
warnings.filterwarnings("ignore", message=".*attempted relative import.*")


def _free_vram_mib():
    """Free VRAM on device 0 via nvidia-smi (None if unknown). Deliberately not
    torch: after one failed CUDA allocation, torch.cuda.mem_get_info() itself
    raises in this process (debug/test_yolo_cpu_fallback.py --recover-test)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits", "-i", "0"],
            capture_output=True, text=True, timeout=5,
        ).stdout
        return int(float(out.strip().splitlines()[0]))
    except Exception:
        return None


class ObjectDetectionThread(threading.Thread):
    def __init__(self, model_path: str = YOLO_MODEL_PATH, update_interval: float = YOLO_INTERVAL_IDLE):
        super().__init__()
        self.model_path = model_path
        self.model = YOLO(model_path)  # lands on CPU; the first track() call moves it to the device
        self.update_interval = update_interval
        self.running = True
        self.shared_frame = None
        self.lock = threading.Lock()
        self.force_cpu = False  # fallback to CPU on CUDA OOM
        self._last_cuda_retry = 0.0
        self._cuda_retry_wait = float(YOLO_CUDA_RETRY_S)
        # Sep 13: decide the device BEFORE touching CUDA. On a full card (the
        # llama-server had grown to 22.5 GB) the first .to("cuda") fails halfway
        # and leaves the model unusable on either device — see _fall_back_to_cpu.
        free = _free_vram_mib()
        if free is not None and free < YOLO_VRAM_MIN_MIB:
            self.force_cpu = True
            self._last_cuda_retry = time.time()
            print(f"[YOLOv8] only {free} MiB of VRAM free at start — detecting on CPU until >= {YOLO_VRAM_MIN_MIB} MiB frees up.")
        self._tracking_mode = False  # When True, use fast interval
        self._target_track_id = None  # sticky gaze target across detection cycles

    @staticmethod
    def _skeleton_coherent(conf_row):
        """(ok, n_confident, n_regions) — the structural person test: enough
        confident keypoints, spread over enough distinct body regions."""
        c = conf_row.cpu().numpy() if hasattr(conf_row, "cpu") else np.asarray(conf_row)
        strong = c > YOLO_SKELETON_KP_CONF
        total = int(strong.sum())
        regions = sum(1 for idxs in _KP_REGIONS if sum(bool(strong[j]) for j in idxs) >= 2)
        return total >= YOLO_SKELETON_MIN_KEYPOINTS and regions >= YOLO_SKELETON_MIN_REGIONS, total, regions

    def set_frame(self, frame):
        with self.lock:
            self.shared_frame = frame.copy()

    def _fall_back_to_cpu(self, now: float) -> None:
        """Sep 13: a failed .to("cuda") leaves the model half-moved, and every
        later call — even with device="cpu" — re-raises the CUDA error (163
        fallback lines in one boot; detection dead, nobody could be seen).
        Reload a fresh model instead: a CPU model keeps working after a failed
        CUDA attempt by another object (debug/test_yolo_cpu_fallback.py)."""
        self.force_cpu = True
        self._last_cuda_retry = now
        try:
            self.model = YOLO(self.model_path)
        except Exception as e:
            print(f"[YOLOv8] CPU reload failed: {e}")
        self._target_track_id = None
        print(f"[YOLOv8] CUDA out of memory — detecting on CPU; retrying CUDA every {self._cuda_retry_wait:.0f}s once >= {YOLO_VRAM_MIN_MIB} MiB is free.")

    def _maybe_retry_cuda(self, now: float) -> None:
        """While on CPU, try the card again every YOLO_CUDA_RETRY_S when nvidia-smi
        shows room (the llama-server reloads fresh after every drawing). A NEW
        model object makes the attempt, so a failure never touches the one in
        use; the wait doubles on failure (cap 20 min) and is kept on success so
        a marginal card cannot flap every two minutes."""
        if not self.force_cpu or now - self._last_cuda_retry < self._cuda_retry_wait:
            return
        self._last_cuda_retry = now
        free = _free_vram_mib()
        if free is None or free < YOLO_VRAM_MIN_MIB:
            return
        cand = None
        try:
            cand = YOLO(self.model_path)
            cand.model.to("cuda")
            self.model, self.force_cpu, self._target_track_id = cand, False, None
            print(f"[YOLOv8] {free} MiB free — back on CUDA.")
        except Exception as e:
            cand = None
            self._cuda_retry_wait = min(self._cuda_retry_wait * 2, 1200.0)
            print(f"[YOLOv8] CUDA retry failed ({str(e)[:60]!r}); next try in {self._cuda_retry_wait:.0f}s.")

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
            self._maybe_retry_cuda(cycle_start)
            try:
                # ByteTrack tracking: persist=True maintains track IDs across frames
                if self.force_cpu:
                    results = self.model.track(frame, persist=True, tracker="config/bytetrack_custom.yaml", verbose=False, imgsz=512, device="cpu")[0]
                else:
                    results = self.model.track(frame, persist=True, tracker="config/bytetrack_custom.yaml", verbose=False, imgsz=512)[0]
            except Exception as e:
                if "CUDA out of memory" in str(e) or "CUDA" in str(e):
                    self._fall_back_to_cpu(time.time())
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
            kps = getattr(results, "keypoints", None)  # pose models only; None on plain detectors

            for i, box in enumerate(results.boxes):
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

                # Skeleton coherence gate (Aug 25): a person is a head AND a
                # body. A mannequin head yields head-keypoints only and fails;
                # appearance can lie about person-ness, geometry mostly can't.
                if kps is not None and kps.conf is not None and i < len(kps.conf):
                    ok, n_kp, n_reg = self._skeleton_coherent(kps.conf[i])
                    if not ok:
                        now_t = time.time()
                        if now_t - getattr(self, "_last_gate_log", 0.0) > 5.0:
                            self._last_gate_log = now_t
                            print(f"[YOLOv8] person-shape rejected by skeleton gate ({n_kp} keypoints, {n_reg} body regions)")
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (128, 128, 128), 1)
                        cv2.putText(frame, "no skeleton", (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
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
                list(detected),
                clean_frame,
                best_person_bbox,
                best_person_conf,
                person_count=person_count,
                best_track_id=self._target_track_id,
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
