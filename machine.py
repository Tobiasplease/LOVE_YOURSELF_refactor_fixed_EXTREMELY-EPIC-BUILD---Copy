import time
import argparse
import sys
import cv2
import threading


def parse_args():
    parser = argparse.ArgumentParser(description="AI Mirror System")
    parser.add_argument("--config_override", type=str, help="Path to JSON config override file")
    return parser.parse_args()


args = parse_args()

if args.config_override:
    try:
        from config.loader import load_config_override, apply_config_overrides
        import config.config as config_module

        overrides = load_config_override(args.config_override)
        apply_config_overrides(config_module, overrides)
        print(f"[CONFIG] Applied overrides from: {args.config_override}")
    except Exception as e:
        print(f"[CONFIG] Error loading config override: {e}")
        sys.exit(1)

from perception.object_detection import ObjectDetectionThread
from captioner.captioner import Captioner
from vision.gaze import update_gaze
from mood.mood import MoodEngine
from breathing.breathing import update_lung_position
from config.config import (
    USE_SERVO,
    CAMERA_INDEX,
    SERIAL_PORT,
    BAUD_RATE,
    CONFIDENCE_THRESHOLD,
    MOOD_SNAPSHOT_FOLDER,
    MOOD_EVALUATION_INTERVAL,
    PAUSE_DURATION,
    MODEL_PATH,
)
from event_logging.run_manager import get_run_image_path
from event_logging.event_logger import get_current_run_id, set_start_time, log_json_entry

if USE_SERVO:
    from servo_control.servo_control import ServoController

VERBOSE = False

# === INIT ===
cap = cv2.VideoCapture(CAMERA_INDEX if "CAMERA_INDEX" in globals() else 0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

proto = f"{MODEL_PATH}/deploy.prototxt"
model = f"{MODEL_PATH}/res10_300x300_ssd_iter_140000.caffemodel"

net = cv2.dnn.readNetFromCaffe(proto, model)
if USE_SERVO:
    servos = ServoController(port=SERIAL_PORT, baudrate=BAUD_RATE)
    servos.serial.setDTR(False)  # type: ignore
    time.sleep(1)
    servos.serial.setDTR(True)  # type: ignore
    time.sleep(2)
else:
    servos = None

lung_angle = 0.0
breath_speed = 4.0
breath_paused = False
pause_start_time = 0
last_breath_direction = None

last_mood_time = 0
last_seen_time = time.time()
last_time = time.time()

last_snapshot_time = 0
object_detector = ObjectDetectionThread()
object_detector.start()

# Initialize run ID and start time for this session
start_time = time.time()
set_start_time(start_time)
run_id = get_current_run_id()
log_json_entry("session_start", {"run_id": run_id}, MOOD_SNAPSHOT_FOLDER, auto_print=True, print_message=f"🚀 Starting session with run ID: {run_id}")
log_json_entry("info", {"message": f"Event log: {run_id}-event-log.json"}, MOOD_SNAPSHOT_FOLDER, auto_print=True, print_message=f"📁 Event log: {run_id}-event-log.json")
log_json_entry("info", {"message": f"Images folder: {run_id}-images/"}, MOOD_SNAPSHOT_FOLDER, auto_print=True, print_message=f"🖼️ Images folder: {run_id}-images/")

mood_engine = MoodEngine()
captioner = Captioner()

best_box = None


def mood_update_thread(frame, timestamp):
    global last_snapshot_time, best_box
    if not captioner.is_processing:
        now = time.time()
        if now - last_snapshot_time >= 10:
            snapshot_path = get_run_image_path(MOOD_SNAPSHOT_FOLDER, f"mood_{int(now)}.jpg")
            cv2.imwrite(snapshot_path, frame)
            mood_engine.update_feeling_brain(frame, image_path=snapshot_path)
            try:
                captioner.update(
                    frame=frame,
                    person_present=best_box is not None,
                    mood=mood_engine.get_current_mood(),
                )
            except Exception:
                pass
            last_snapshot_time = now


try:
    while True:
        ret, frame = cap.read()
        object_detector.set_frame(frame)
        if not ret:
            continue

        frame = cv2.resize(frame, (320, 240))
        frame = cv2.flip(frame, 1)

        now = time.time()
        delta = now - last_time
        last_time = now

        # === FACE DETECTION ===
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104, 177, 123), False, False)
        net.setInput(blob)
        detections = net.forward()

        best_box = None
        best_conf = 0.0
        for i in range(detections.shape[2]):
            conf = detections[0, 0, i, 2]
            if conf > CONFIDENCE_THRESHOLD and conf > best_conf:
                box = detections[0, 0, i, 3:7] * [w, h, w, h]
                best_box = box.astype("int")
                best_conf = conf

        if now - last_mood_time > MOOD_EVALUATION_INTERVAL:
            threading.Thread(target=mood_update_thread, args=(frame.copy(), int(now)), daemon=True).start()
            last_mood_time = now

        current_mood = mood_engine.get_current_mood()

        face_box = tuple(best_box) if best_box is not None else None
        person_present, pan, tilt = update_gaze(frame, face_box, current_mood)

        (
            lung_pos,
            lung_angle,
            breath_speed,
            breath_paused,
            last_breath_direction,
            pause_start_time,
        ) = update_lung_position(
            current_mood=current_mood,
            person_present=person_present,
            delta=delta,
            lung_angle=lung_angle,
            breath_speed=breath_speed,
            breath_paused=breath_paused,
            last_breath_direction=last_breath_direction,
            pause_start_time=pause_start_time,
            pause_duration=PAUSE_DURATION,
            servo_controller=servos,
        )

        if USE_SERVO:
            servos.set_pan(pan)  # type: ignore
            servos.set_tilt(tilt)  # type: ignore

        if face_box:
            (x1, y1, x2, y2) = face_box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # === DISPLAY OVERLAYS ===
        debug = f"Mood: {current_mood:.2f} | Lung: {lung_pos} | Pan/Tilt: {pan}/{tilt}"
        cv2.putText(frame, debug, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        from perception.detection_memory import DetectionMemory

        labels = DetectionMemory.get_labels()
        label_text = ", ".join(labels) if labels else "no objects"
        cv2.putText(
            frame,
            f"Seen: {label_text}",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
        )

        cv2.imshow("mslint camera", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

except KeyboardInterrupt:
    object_detector.stop()
    object_detector.join()
    cap.release()
    cv2.destroyAllWindows()
