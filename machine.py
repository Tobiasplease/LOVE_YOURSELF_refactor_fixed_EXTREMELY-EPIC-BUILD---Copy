import time
import argparse
import sys
import cv2
import threading
import subprocess
import os


def parse_args():
    parser = argparse.ArgumentParser(description="AI Mirror System")
    parser.add_argument("--config_override", type=str, help="Path to JSON config override file")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode with verbose output")
    return parser.parse_args()


args = parse_args()

# Debug mode setup
DEBUG_MODE = args.debug


def debug_print(message, level="INFO"):
    """Print debug messages only when debug mode is enabled."""
    if DEBUG_MODE:
        timestamp = time.strftime("%H:%M:%S")
        print(f"[DEBUG {timestamp}] {level}: {message}")


if DEBUG_MODE:
    print("🐛 DEBUG MODE ENABLED - Verbose output active")

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
from image_monitor import ImageMonitor
from utils.state_manager import state_manager
from utils.continuity import describe_duration
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
    PRINT_CLEAN_CAPTIONS,
)
from event_logging.run_manager import get_run_image_path
from event_logging.event_logger import get_current_run_id, set_start_time, log_json_entry
from event_logging.log_type import LogType
from hand_controller_bridge import HandControllerBridge
from reactivity.camera_reactive import CameraReactivityEngine


def launch_hand_controller():
    """Launch the hand controller interface in a separate process."""
    hand_controller_path = r"C:\Users\tobia\Downloads\HandControlStandalone\hand_control_interface.py"
    
    if os.path.exists(hand_controller_path):
        try:
            # Launch hand controller in a separate process - show output for debugging
            result = subprocess.Popen([
                "python", 
                hand_controller_path
            ], 
            cwd=r"C:\Users\tobia\Downloads\HandControlStandalone",
            creationflags=subprocess.CREATE_NEW_CONSOLE if os.name == 'nt' else 0  # New console on Windows
            )
            print(f"🤲 Hand controller interface launched successfully (PID: {result.pid})")
            return True
        except Exception as e:
            print(f"❌ Failed to launch hand controller: {e}")
            return False
    else:
        print(f"❌ Hand controller not found at: {hand_controller_path}")
        return False


if USE_SERVO:
    from servo_control.servo_control import ServoController

VERBOSE = False

# === INIT ===
debug_print("Opening camera", "INIT")
cap = cv2.VideoCapture(CAMERA_INDEX if "CAMERA_INDEX" in globals() else 0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()
debug_print("Camera opened successfully", "INIT")

proto = f"{MODEL_PATH}/deploy.prototxt"
model = f"{MODEL_PATH}/res10_300x300_ssd_iter_140000.caffemodel"

debug_print("Loading face detection model", "INIT")
net = cv2.dnn.readNetFromCaffe(proto, model)
debug_print("Face detection model loaded", "INIT")
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
debug_print("Starting object detection thread", "INIT")
object_detector = ObjectDetectionThread()
object_detector.start()

# Start image monitoring
debug_print("Starting image monitor", "INIT")
image_monitor = ImageMonitor(log_folder=MOOD_SNAPSHOT_FOLDER)

# Initialize run ID and start time for this session
start_time = time.time()
set_start_time(start_time)
run_id = get_current_run_id()

log_json_entry(
    LogType.SESSION_START,
    {"run_id": run_id},
    print_message=f"🚀 Starting session with run ID: {run_id}",
)
log_json_entry(
    LogType.INFO,
    {"message": f"Event log: {run_id}-event-log.json"},
    MOOD_SNAPSHOT_FOLDER,
    print_message=f"📁 Event log: {run_id}-event-log.json",
)
log_json_entry(
    LogType.INFO,
    {"message": f"Images folder: {run_id}-images/"},
    print_message=f"🖼️ Images folder: {run_id}-images/",
)

debug_print("Initializing mood engine", "INIT")
mood_engine = MoodEngine()
debug_print("Initializing captioner", "INIT")
captioner = Captioner()

# Initialize hand controller bridge
debug_print("Initializing hand controller bridge", "INIT")
hand_bridge = HandControllerBridge("file")  # Use file-based communication

# Initialize camera reactivity engine
debug_print("Initializing camera reactivity engine", "INIT")
reactivity_engine = CameraReactivityEngine(sensitivity=1.2, smoothing_factor=0.8)
debug_print("Camera reactivity enabled - hand will respond to environmental changes", "INIT")

# Launch hand controller interface
debug_print("Launching hand controller from proper directory to access datasets", "INIT")
launch_hand_controller()  # Launch from HandControlStandalone directory

# Set up image monitor with self-critique callback
def on_drawing_complete(image_path: str):
    """Handle drawing completion with self-critique."""
    captioner.drawing.critique_drawing(image_path)

image_monitor.on_image_complete = on_drawing_complete
image_monitor.start()

# Load previous session state if available
debug_print("Loading previous session state", "INIT")
previous_state = state_manager.load_session_state()
if previous_state:
    # Apply state to components
    state_manager.apply_state_to_captioner(previous_state, captioner)
    state_manager.apply_state_to_mood_engine(previous_state, mood_engine)
    
    # Send immediate mood update to hand controller with restored state
    debug_print("Sending restored mood to hand controller", "INIT")
    initial_reactivity = {'chaos_multiplier': 1.0, 'speed_multiplier': 1.0, 'activity_level': 0.0}
    hand_bridge.update_hand_controller(mood_engine, force_update=True, reactivity_data=initial_reactivity)
    
    # Reset last_caption so remnants from previous session are not printed
    captioner.last_caption = ""
    # Set memory loaded flag BEFORE generating awakening message
    captioner.memory_loaded_from_previous = True

    # Generate awakening message with continuity
    save_time = previous_state["metadata"]["save_time"]
    time_since_last = describe_duration(save_time)
    previous_beliefs = previous_state["captioner"].get("beliefs", {})

    awakening_msg = captioner.generate_awakening_message(time_since_last, previous_beliefs)

    log_json_entry(
        LogType.INFO,
        {"message": awakening_msg, "continuity": True, "time_since_last": time_since_last},
        print_message=f'"{awakening_msg}"',
    )
    # Mark awakening complete to avoid duplicate environmental description
    captioner.mark_awakening_complete()
else:
    # Fresh start
    captioner.memory_loaded_from_previous = False
    awakening_msg = captioner.generate_awakening_message()

    log_json_entry(
        LogType.INFO,
        {"message": awakening_msg, "continuity": False},
        print_message=f'"{awakening_msg}"',
    )
    # Mark awakening complete to avoid duplicate environmental description
    captioner.mark_awakening_complete()

debug_print("System initialization complete", "INIT")

best_box = None


def mood_update_thread(frame, timestamp):
    global last_snapshot_time, best_box
    debug_print("Mood update thread started", "MOOD")
    if not captioner.is_processing:
        debug_print("Captioner is not processing, proceeding with mood update", "MOOD")
        now = time.time()
        if now - last_snapshot_time >= 10:
            snapshot_path = get_run_image_path(MOOD_SNAPSHOT_FOLDER, f"mood_{int(now)}.jpg")
            cv2.imwrite(snapshot_path, frame)
            debug_print(f"Snapshot saved: {snapshot_path}", "MOOD")

            try:
                # First: Update captioner to generate new captions
                captioner.update(
                    frame=frame,
                    person_present=best_box is not None,
                    mood=mood_engine.get_current_mood(),  # Use previous mood for now
                )
                debug_print("Captioner updated successfully", "CAPTIONER")

                # Second: Analyze mood from captioner's latest caption
                if captioner.last_caption:
                    clean_caption = captioner.last_caption
                    if clean_caption.lower().startswith("caption:"):
                        clean_caption = clean_caption[len("caption:") :].strip()

                    if PRINT_CLEAN_CAPTIONS:
                        print(f"\n{clean_caption}\n")

                    current_mood = mood_engine.analyze_mood(clean_caption, image_path=snapshot_path)
                    debug_print(f"Mood analyzed from caption: {current_mood:.2f}", "MOOD")

                    # Update hand controller with new emotional state + camera reactivity
                    hand_bridge.update_hand_controller(mood_engine, reactivity_data=reactivity_metrics)

                    # Third: Update captioner's mood state for next cycle
                    captioner.current_mood = current_mood

            except Exception as e:
                debug_print(f"Captioner update failed: {e}", "ERROR")
            last_snapshot_time = now
    else:
        debug_print("Captioner is processing, skipping mood update", "MOOD")


try:
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        
        object_detector.set_frame(frame)

        frame = cv2.resize(frame, (320, 240))
        frame = cv2.flip(frame, 1)

        # === CAMERA REACTIVITY PROCESSING ===
        # Process frame for real-time behavioral reactivity
        reactivity_metrics = reactivity_engine.process_frame(frame)

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

        if best_box is not None:
            # Suppress frequent face detection messages - only show occasionally
            if int(now) % 5 == 0:  # Every 5 seconds
                debug_print(f"Face detected with confidence: {best_conf:.2f}", "FACE")
        else:
            # Only show "no face" occasionally to avoid spam
            if int(now) % 10 == 0:  # Every 10 seconds
                debug_print("No face detected", "FACE")

        if now - last_mood_time > MOOD_EVALUATION_INTERVAL:
            debug_print(f"Starting mood update thread - interval: {MOOD_EVALUATION_INTERVAL}s", "MOOD")
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

        # === CAMERA REACTIVITY OVERLAY ===
        if reactivity_metrics:
            # Compact reactivity bar at bottom
            bar_x, bar_y = 10, frame.shape[0] - 60
            bar_w, bar_h = 200, 50
            
            # Background
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (0, 0, 0), -1)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (100, 100, 100), 1)
            
            # Activity bar (green)
            activity_len = int(reactivity_metrics['activity_level'] * (bar_w - 20))
            cv2.rectangle(frame, (bar_x + 10, bar_y + 8), (bar_x + 10 + activity_len, bar_y + 16), (0, 255, 0), -1)
            cv2.putText(frame, "ACT", (bar_x + 2, bar_y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
            
            # Speed bar (red) - normalize to 0-1 range
            speed_norm = min(1.0, (reactivity_metrics['speed_multiplier'] - 0.2) / 4.3)  # 0.2-4.5 -> 0-1
            speed_len = int(speed_norm * (bar_w - 20))
            cv2.rectangle(frame, (bar_x + 10, bar_y + 20), (bar_x + 10 + speed_len, bar_y + 28), (0, 0, 255), -1)
            cv2.putText(frame, "SPD", (bar_x + 2, bar_y + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
            
            # Chaos bar (blue) - normalize to 0-1 range  
            chaos_norm = min(1.0, (reactivity_metrics['chaos_multiplier'] - 0.3) / 3.2)  # 0.3-3.5 -> 0-1
            chaos_len = int(chaos_norm * (bar_w - 20))
            cv2.rectangle(frame, (bar_x + 10, bar_y + 32), (bar_x + 10 + chaos_len, bar_y + 40), (255, 0, 0), -1)
            cv2.putText(frame, "CHS", (bar_x + 2, bar_y + 38), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
            
            # Pause indicator
            if reactivity_metrics.get('paused', False):
                cv2.rectangle(frame, (bar_x + bar_w - 30, bar_y + 10), (bar_x + bar_w - 10, bar_y + 30), (0, 0, 255), -1)
                cv2.putText(frame, "PAUSE", (bar_x + bar_w - 28, bar_y + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

        cv2.imshow("mslint camera", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

except KeyboardInterrupt:
    debug_print("Shutting down gracefully", "SHUTDOWN")

    # Save session state before shutdown
    print("[💾] Saving session state...")
    success = state_manager.save_session_state(captioner, mood_engine)
    if success:
        print("[✅] Session state saved successfully")
    else:
        print("[❌] Failed to save session state")

    log_json_entry(
        LogType.INFO,
        {"message": "Session ended", "run_id": run_id, "duration": time.time() - start_time},
        MOOD_SNAPSHOT_FOLDER,
        print_message=f"[👋] Session ended. Duration: {time.time() - start_time:.1f}s",
    )

    object_detector.stop()
    object_detector.join()
    image_monitor.stop()
    cap.release()
    cv2.destroyAllWindows()
