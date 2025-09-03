import argparse
import atexit

# import subprocess
import os
import signal
import sys
import threading
import time

import cv2

from config.config import USE_LIGHTBULB_PWM


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
    print("DEBUG MODE ENABLED - Verbose output active")

if args.config_override:
    try:
        import config.config as config_module
        from config.loader import apply_config_overrides, load_config_override

        overrides = load_config_override(args.config_override)
        apply_config_overrides(config_module, overrides)
        print(f"[CONFIG] Applied overrides from: {args.config_override}")
    except Exception as e:
        print(f"[CONFIG] Error loading config override: {e}")
        sys.exit(1)


from breathing.breathing import update_lung_position

# from perception.object_detection import ObjectDetectionThread
from captioner.captioner import Captioner
from config.config import (
    BAUD_RATE,
    CAMERA_INDEX,
    CONFIDENCE_THRESHOLD,
    DEBUG_REACTIVITY_PAUSE,
    MODEL_PATH,
    MOOD_EVALUATION_INTERVAL,
    MOOD_SNAPSHOT_FOLDER,
    PAUSE_DURATION,
    REACTIVITY_PAUSE_COOLDOWN,
    REACTIVITY_PAUSE_DURATION,
    REACTIVITY_PAUSE_THRESHOLD,
    USE_HAND_CONTROLLER,
    USE_SERVO,
)
from event_logging.event_logger import get_current_run_id, log_json_entry, set_start_time
from event_logging.log_type import LogType
from event_logging.run_manager import get_run_image_path
from image_monitor import ImageMonitor
from mood.mood import MoodEngine
from utils.continuity import describe_duration
from utils.error_tracking import get_failure_tracker
from utils.state_manager import state_manager
from vision.gaze import update_gaze

try:
    import config.config as config_module

    if getattr(config_module, "NO_HANDS", False):
        raise ImportError("Hand control disabled by NO_HANDS config")

    from hand_control.direct_hand_control import change_to_emotion  # type: ignore
    from hand_control.direct_hand_control import get_status  # type: ignore
    from hand_control.direct_hand_control import send_reactivity_data  # type: ignore
    from hand_control.direct_hand_control import start_autonomous_mode  # type: ignore
    from hand_control.direct_hand_control import start_hand_controller  # type: ignore
    from hand_control.direct_hand_control import (  # set_emotion,
        stop_hand_controller,
    )

    HAND_CONTROL_AVAILABLE = True
except ImportError:
    print("Warning: hand_control module not available (tkinter dependency missing). Hand control features disabled.")
    HAND_CONTROL_AVAILABLE = False

    # Define stub functions
    def start_hand_controller(headless=False):
        return False

    def stop_hand_controller():
        pass

    def send_reactivity_data(*args, **kwargs):
        pass

    def get_status():
        return "disabled"

    def change_to_emotion(*args, **kwargs):
        pass

    def start_autonomous_mode(*args, **kwargs):
        pass


from reactivity.camera_reactive import CameraReactivityEngine

if USE_SERVO:
    from servo_control.servo_control import ServoController

if USE_LIGHTBULB_PWM:
    from servo_control.lightbulb_controller_nonblocking import NonBlockingLightbulbController

VERBOSE = False

# Fixed udev device paths for each Arduino
ARDUINO_DEVICES = {
    "LIGHTBULB": "/dev/arduino_lightbulb",  # Lightbulb PWM controller
    "LUNGGAZE": "/dev/arduino_lunggaze",  # Gaze pan/tilt and breath controller
    "LEFTHAND": "/dev/arduino_lefthand",  # Hand gesture controller
    "CNC": "/dev/arduino_cnc",  # GRBL CNC controller (for bCNC)
    "UARM": "/dev/arduino_uarm",  # uArm (not integrated yet)
}

# === INIT ===
debug_print("Using fixed udev Arduino device paths", "INIT")
for device_name, device_path in ARDUINO_DEVICES.items():
    if os.path.exists(device_path):
        debug_print(f"  {device_name}: {device_path} [FOUND]", "INIT")
    else:
        debug_print(f"  {device_name}: {device_path} [NOT CONNECTED]", "INIT")

debug_print("Opening camera", "INIT")
lightbulb = None
if USE_LIGHTBULB_PWM:
    lightbulb_port = ARDUINO_DEVICES["LIGHTBULB"]
    if os.path.exists(lightbulb_port):
        try:
            lightbulb = NonBlockingLightbulbController(lightbulb_port, debug=False)
            debug_print(f"Lightbulb controller initialized on {lightbulb_port}", "INIT")
        except Exception as e:
            debug_print(f"Lightbulb controller init failed on {lightbulb_port}: {e}", "ERROR")
            print("  Device may not be ready or firmware mismatch")
            lightbulb = None
    else:
        debug_print(f"Lightbulb controller not found at {lightbulb_port}", "WARN")
        print("  Device may not be connected")
cap = cv2.VideoCapture(CAMERA_INDEX if "CAMERA_INDEX" in globals() else 0)
_global_cap = cap
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()
debug_print("Camera opened successfully", "INIT")
# Set camera resolution for better image quality
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

proto = f"{MODEL_PATH}/deploy.prototxt"
model = f"{MODEL_PATH}/res10_300x300_ssd_iter_140000.caffemodel"

debug_print("Loading face detection model", "INIT")
net = cv2.dnn.readNetFromCaffe(proto, model)
debug_print("Face detection model loaded", "INIT")
if USE_SERVO:
    servo_port = ARDUINO_DEVICES["LUNGGAZE"]  # Gaze pan/tilt and breath controller
    if os.path.exists(servo_port):
        try:
            servos = ServoController(port=servo_port, baudrate=BAUD_RATE)
            debug_print(f"Servo controller initialized on {servo_port}", "INIT")
        except Exception as e:
            print(f"ERROR: Servo controller init failed on {servo_port}: {e}")
            print("  Device may not be ready or firmware mismatch")
            servos = None
    else:
        print(f"WARNING: Servo controller not found at {servo_port}")
        print("  Device may not be connected")
        servos = None
else:
    debug_print("Servo control disabled in config", "INIT")
    servos = None

# Initialize breathing variables regardless of servo setting
lung_angle = 0.0
breath_speed = 4.0
breath_paused = False
pause_start_time = 0
last_breath_direction = None

last_mood_time = 0
last_seen_time = time.time()
last_time = time.time()

# Global cleanup state
shutdown_in_progress = False
cleanup_completed = False

# Global references for cleanup
_global_cap = None
_global_object_detector = None
_global_image_monitor = None
_global_captioner = None
_global_mood_engine = None
_global_state_manager = None
_global_start_time = None
_global_run_id = None


def emergency_cleanup():
    """Emergency cleanup function that can be called from signal handlers."""
    global shutdown_in_progress, cleanup_completed

    if shutdown_in_progress:
        print("[WARNING] Multiple shutdown signals - forcing exit")
        os._exit(1)

    shutdown_in_progress = True
    print("[EMERGENCY] Emergency shutdown initiated...")

    try:
        # Quick cleanup - no waiting
        if _global_object_detector:
            _global_object_detector.stop()
        if _global_image_monitor:
            _global_image_monitor.stop()
        if _global_cap:
            _global_cap.release()

        # Emergency hand controller stop
        try:
            stop_hand_controller()
        except Exception:
            pass

        cv2.destroyAllWindows()

        # Clear PyTorch cache if available (helps with YOLO cleanup)
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                # Only print debug if clean output is disabled
                try:
                    from config.config import CLEAN_LLM_OUTPUT

                    if not CLEAN_LLM_OUTPUT:
                        print("[🔧] GPU cache cleared")
                except ImportError:
                    print("[🔧] GPU cache cleared")
        except ImportError:
            pass  # PyTorch not available, skip
        except Exception as e:
            print(f"[WARNING] Warning: Could not clear GPU cache: {e}")

        print("[SUCCESS] Emergency cleanup completed")
        cleanup_completed = True

    except Exception as e:
        print(f"[ERROR] Emergency cleanup error: {e}")
    finally:
        os._exit(0)


def graceful_cleanup():
    """Graceful cleanup with timeouts and error handling."""
    global shutdown_in_progress, cleanup_completed

    if cleanup_completed:
        return

    shutdown_in_progress = True
    print("[SHUTDOWN] Graceful shutdown initiated...")

    # Save session state first (most important)
    if _global_captioner and _global_mood_engine and _global_state_manager:
        try:
            # Only print debug if clean output is disabled
            try:
                from config.config import CLEAN_LLM_OUTPUT

                if not CLEAN_LLM_OUTPUT:
                    print("[💾] Saving session state...")
            except ImportError:
                print("[💾] Saving session state...")
            success = _global_state_manager.save_session_state(_global_captioner, _global_mood_engine)
            print("[SUCCESS] Session state saved successfully" if success else "[ERROR] Failed to save session state")
        except Exception as e:
            print(f"[ERROR] Error saving session state: {e}")

    # Log session end
    if _global_start_time and _global_run_id:
        try:
            log_json_entry(
                LogType.INFO,
                {"message": "Session ended", "run_id": _global_run_id, "duration": time.time() - _global_start_time},
                MOOD_SNAPSHOT_FOLDER,
                print_message=f"[👋] Session ended. Duration: {time.time() - _global_start_time:.1f}s",
            )
        except Exception as e:
            print(f"[ERROR] Error logging session end: {e}")

    # Stop threads with timeouts
    if _global_object_detector:
        try:
            _global_object_detector.stop()
            _global_object_detector.join(timeout=2.0)
            if _global_object_detector.is_alive():
                print("[WARNING] Object detector thread didn't stop cleanly - forcing termination")
                # Force terminate if it's still alive
                import ctypes

                ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(_global_object_detector.ident), ctypes.py_object(SystemExit))
        except Exception as e:
            print(f"[ERROR] Error stopping object detector: {e}")

    if _global_image_monitor:
        try:
            _global_image_monitor.stop()
        except Exception as e:
            print(f"[ERROR] Error stopping image monitor: {e}")

    # Stop hand controller
    try:
        stop_hand_controller()
    except Exception as e:
        print(f"[ERROR] Error stopping hand controller: {e}")

    # Release camera
    if _global_cap:
        try:
            _global_cap.release()
        except Exception as e:
            print(f"[ERROR] Error releasing camera: {e}")

    try:
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"[ERROR] Error destroying windows: {e}")

    # Clear PyTorch cache if available (helps with YOLO cleanup)
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            # Only print debug if clean output is disabled
            try:
                from config.config import CLEAN_LLM_OUTPUT

                if not CLEAN_LLM_OUTPUT:
                    print("[🔧] GPU cache cleared")
            except ImportError:
                print("[🔧] GPU cache cleared")
    except ImportError:
        pass  # PyTorch not available, skip
    except Exception as e:
        print(f"[WARNING] Warning: Could not clear GPU cache: {e}")

    # Shutdown error tracker
    try:
        get_failure_tracker().shutdown()
        # Only print debug if clean output is disabled
        try:
            from config.config import CLEAN_LLM_OUTPUT

            if not CLEAN_LLM_OUTPUT:
                print("[📊] Error tracker shutdown")
        except ImportError:
            print("[📊] Error tracker shutdown")
    except Exception as e:
        print(f"[WARNING] Error shutting down error tracker: {e}")

    cleanup_completed = True
    print("[SUCCESS] Graceful shutdown completed")


def signal_handler(signum, frame):
    """Handle interrupt signals."""
    print(f"\n[🔄] Received signal {signum}")
    graceful_cleanup()
    sys.exit(0)


# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Register emergency cleanup for exit
atexit.register(emergency_cleanup)

last_snapshot_time = 0
mood_thread_running = False
mood_thread_lock = threading.Lock()
debug_print("YOLO person detection enabled", "INIT")
from perception.object_detection import ObjectDetectionThread

object_detector = ObjectDetectionThread()
_global_object_detector = object_detector
object_detector.start()

# Start image monitoring
debug_print("Starting image monitor", "INIT")
image_monitor = ImageMonitor(log_folder=MOOD_SNAPSHOT_FOLDER)
_global_image_monitor = image_monitor

# Initialize run ID and start time for this session
start_time = time.time()
_global_start_time = start_time
set_start_time(start_time)
run_id = get_current_run_id()
_global_run_id = run_id

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
_global_mood_engine = mood_engine
debug_print("Initializing captioner", "INIT")
captioner = Captioner()
_global_captioner = captioner
_global_state_manager = state_manager

# Initialize direct hand controller integration
if USE_HAND_CONTROLLER:
    debug_print("Initializing direct hand controller integration", "INIT")
    hand_controller_started = start_hand_controller(headless=False)
else:
    debug_print("Hand controller disabled in config", "INIT")
    hand_controller_started = False

if hand_controller_started:
    debug_print("Hand controller started with UI", "INIT")

    # Start autonomous mode (Markov generation) after successful initialization
    time.sleep(1)  # Give time for datasets to load
    if start_autonomous_mode():
        debug_print("Hand controller autonomous mode activated", "INIT")
    else:
        debug_print("Failed to start autonomous mode", "INIT")
    status = get_status()
    debug_print(f"Hand controller status: {status}", "INIT")
else:
    debug_print("Hand controller failed to start", "WARN")

# Initialize camera reactivity engine
debug_print("Initializing camera reactivity engine", "INIT")
reactivity_engine = CameraReactivityEngine(sensitivity=1.8, smoothing_factor=0.85, pause_threshold=0.20, pause_duration=3.0)
debug_print("Camera reactivity enabled - hand will respond to environmental changes", "INIT")


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

    # Direct integration - set emotion based on mood
    emotion = mood_engine.get_emotion_for_hand_controller()
    change_to_emotion(emotion)
    debug_print(f"Set hand controller emotion: {emotion}", "INIT")
    debug_print(f"Set hand controller emotion: {emotion}", "INIT")

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

# Track last printed caption timestamp to prevent duplicates
last_printed_caption_time = 0.0
last_state_save_time = 0.0


# Redundant functions removed - using existing MoodEngine functionality


def mood_update_thread(frame, timestamp):
    global last_snapshot_time, last_state_save_time, mood_thread_running

    # Set running flag at start
    with mood_thread_lock:
        mood_thread_running = True

    try:
        debug_print("Mood update thread started", "MOOD")
        now = time.time()
        if now - last_snapshot_time >= 10:
            snapshot_path = get_run_image_path(MOOD_SNAPSHOT_FOLDER, f"mood_{int(now)}.jpg")
            cv2.imwrite(snapshot_path, frame)
            debug_print(f"Snapshot saved: {snapshot_path}", "MOOD")

            try:
                # Use existing MoodEngine for emotional state - single source of truth
                current_emotion = mood_engine.get_emotion_for_hand_controller()
                current_mood = mood_engine.get_current_mood()

                debug_print(f"Current emotion: {current_emotion}, mood: {current_mood:.2f}", "EMOTION")

                # Captioner is now updated in main loop - mood system just reads captions
                debug_print("Captioner updated successfully", "CAPTIONER")

                # Second: Process caption and update physical systems
                if captioner.last_caption:
                    clean_caption = captioner.last_caption
                    if clean_caption.lower().startswith("caption:"):
                        clean_caption = clean_caption[len("caption:") :].strip()
                    # Lightbulb flash on caption print
                    if USE_LIGHTBULB_PWM and lightbulb:
                        try:
                            lightbulb.caption_flash()
                        except Exception as e:
                            debug_print(f"Lightbulb caption flash failed: {e}", "ERROR")

                    # PRINT_CLEAN_CAPTIONS? chuck into logging func?
                    # if captioner.last_caption_time > last_printed_caption_time:
                    #     print(f"\n{clean_caption}\n")
                    #     last_printed_caption_time = captioner.last_caption_time

                    # SimpleLightbulbController doesn't have mood parameters - uses frame diff only

                    # Periodic state saving (every 2 minutes)
                    if now - last_state_save_time > 120:  # 2 minutes
                        if _global_state_manager:
                            try:
                                _global_state_manager.save_session_state(captioner, mood_engine)
                                last_state_save_time = now
                            except Exception as e:
                                print(f"[ERROR] Periodic state save failed: {e}")

                    # Update hand controller with new emotional state
                    change_to_emotion(current_emotion)
                    debug_print(f"Updated hand controller emotion: {current_emotion}", "HAND")

                    # Third: Update captioner's mood state and pattern data for next cycle
                    captioner.current_mood = current_mood
                    pattern_data = mood_engine.get_pattern_data()
                    captioner.set_novelty_score(pattern_data["novelty_score"])
                    # Pass recent motifs to captioner for memory integration
                    captioner.current_motifs_from_mood = pattern_data["recent_motifs"]

            except Exception as e:
                debug_print(f"Captioner update failed: {e}", "ERROR")
            last_snapshot_time = now
        debug_print("Mood update thread completed successfully", "MOOD")
    finally:
        # Always clear running flag when thread completes
        with mood_thread_lock:
            mood_thread_running = False
        debug_print("Mood update thread flag cleared", "MOOD")


# === REACTIVITY PAUSE SYSTEM STATE ===
last_pause_time = 0.0
pause_is_active = False
frame_count = 0

# Add frame display throttling to prevent memory issues
last_display_time = 0
DISPLAY_THROTTLE_INTERVAL = 0.1  # Show camera feed max 10 FPS to save memory

try:
    prev_gray = None
    smoothed_pwm = 0
    debug_print("Entering main camera processing loop", "MAIN")
    while True:
        # Check for shutdown signal
        if shutdown_in_progress:
            print("[SHUTDOWN] Shutdown signal received - breaking main loop")
            break

        debug_print("Reading camera frame", "MAIN")
        ret, frame = cap.read()
        if not ret:
            continue

        object_detector.set_frame(frame)  # YOLO person detection enabled

        frame = cv2.resize(frame, (320, 240))
        frame = cv2.flip(frame, 1)

        # Force garbage collection periodically to prevent memory accumulation
        frame_count += 1
        if frame_count % 100 == 0:  # Every 100 frames
            import gc

            gc.collect()

        # === CAMERA REACTIVITY PROCESSING ===
        # Process frame for real-time behavioral reactivity
        reactivity_metrics = reactivity_engine.process_frame(frame)
        frame_count += 1

        # === LIGHTBULB BRIGHTNESS FROM REACTIVITY DATA ===
        # Use the same activity level that drives the reactivity bar
        if USE_LIGHTBULB_PWM and lightbulb and reactivity_metrics:
            try:
                # Get activity level from reactivity engine (0.0-1.0 scale)
                activity_level = reactivity_metrics.get("activity_level", 0.0)

                # Convert to 0-255 brightness scale
                brightness = int(min(255, activity_level * 255))

                lightbulb.set_frame_diff_brightness(brightness)
            except Exception as e:
                debug_print(f"Lightbulb brightness failed: {e}", "ERROR")

        # Get current time for pause/cooldown calculations
        now = time.time()

        # Add pause/cooldown information to metrics for display
        if pause_is_active:
            # Currently paused - show remaining pause time
            pause_elapsed = now - last_pause_time
            pause_remaining = max(0, REACTIVITY_PAUSE_DURATION - pause_elapsed)
            reactivity_metrics["is_paused"] = True
            reactivity_metrics["pause_remaining"] = pause_remaining
        else:
            # Not paused - check cooldown
            cooldown_elapsed = now - last_pause_time
            cooldown_remaining = max(0, REACTIVITY_PAUSE_COOLDOWN - cooldown_elapsed)
            reactivity_metrics["is_paused"] = False
            reactivity_metrics["cooldown_remaining"] = cooldown_remaining

        # === REACTIVITY PAUSE SYSTEM ===
        # Check if activity level crosses threshold for pausing Markov generation
        current_activity = reactivity_metrics.get("activity_level", 0.0)

        # Debug: Show activity level occasionally
        if frame_count % 180 == 0:  # Every ~6 seconds at 30fps
            debug_print(f"Current activity level: {current_activity:.3f} (threshold: {REACTIVITY_PAUSE_THRESHOLD})", "ACTIVITY")

        if current_activity > REACTIVITY_PAUSE_THRESHOLD and not pause_is_active and now - last_pause_time > REACTIVITY_PAUSE_COOLDOWN:

            # High activity detected - pause Markov generation
            pause_data = {
                "action": "pause",
                "duration": REACTIVITY_PAUSE_DURATION,
                "activity_level": float(current_activity),  # Convert numpy to Python float
            }
            if DEBUG_REACTIVITY_PAUSE:
                debug_print(f"🚨 Sending pause command: {pause_data}", "REACTIVITY")

            send_reactivity_data(pause_data)
            pause_is_active = True
            last_pause_time = now
            if DEBUG_REACTIVITY_PAUSE:
                debug_print(
                    f"🚨 High activity detected ({current_activity:.2f}) - pausing hand controller for {REACTIVITY_PAUSE_DURATION}s", "REACTIVITY"
                )

        # Check if pause duration has expired OR activity dropped significantly
        elif pause_is_active and (
            now - last_pause_time > REACTIVITY_PAUSE_DURATION or (current_activity < REACTIVITY_PAUSE_THRESHOLD * 0.1 and now - last_pause_time > 2.0)
        ):  # Only resume early if activity is very low AND at least 2s have passed
            # Pause duration expired or activity dropped - resume Markov generation
            resume_reason = "duration expired" if now - last_pause_time > REACTIVITY_PAUSE_DURATION else "very low activity"
            resume_data = {"action": "resume", "activity_level": float(current_activity)}  # Convert numpy to Python float
            if DEBUG_REACTIVITY_PAUSE:
                debug_print(f"RESUME Sending resume command ({resume_reason}): {resume_data}", "REACTIVITY")

            send_reactivity_data(resume_data)
            pause_is_active = False
            if DEBUG_REACTIVITY_PAUSE:
                debug_print(f"SUCCESS Resume triggered by {resume_reason} - resuming hand controller", "REACTIVITY")

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

        # Also check YOLO person detections (if face detection didn't find anything)
        if best_box is None:
            from perception.detection_memory import DetectionMemory

            labels = DetectionMemory.get_labels()
            if "person" in labels:
                # Person detected by YOLO - create a dummy box for person presence
                best_box = [50, 50, 100, 100]  # Dummy box to indicate person present
                best_conf = 0.8  # Dummy confidence for YOLO person detection
                if int(now) % 5 == 0:  # Every 5 seconds
                    debug_print("Person detected by YOLO", "PERSON")

        if best_box is not None:
            # Suppress frequent detection messages - only show occasionally
            if int(now) % 5 == 0:  # Every 5 seconds
                source = "Face" if best_conf < 0.7 else "Person"  # Distinguish source
                debug_print(f"{source} detected with confidence: {best_conf:.2f}", "DETECTION")
        else:
            # Only show "no person" occasionally to avoid spam
            if int(now) % 10 == 0:  # Every 10 seconds
                debug_print("No person detected", "DETECTION")

        # Update captioner with every frame (decoupled from mood system)
        captioner.update(
            frame=frame,
            person_present=best_box is not None,
            mood=mood_engine.get_current_mood() if mood_engine else 0.5,
            reactivity_data=reactivity_metrics,
        )

        if now - last_mood_time > MOOD_EVALUATION_INTERVAL:
            # Check if mood analysis is already running to prevent overlapping threads
            with mood_thread_lock:
                if not mood_thread_running:
                    debug_print(f"Starting mood update thread - interval: {MOOD_EVALUATION_INTERVAL}s", "MOOD")
                    threading.Thread(target=mood_update_thread, args=(frame.copy(), int(now)), daemon=True).start()
                    last_mood_time = now
                else:
                    debug_print("Mood analysis already running - skipping this interval", "MOOD")

        current_mood = mood_engine.get_current_mood()

        face_box = tuple(best_box) if best_box is not None else None
        person_present, pan, tilt = update_gaze(frame, face_box, mood_engine.get_emotion_for_hand_controller())

        (
            lung_pos,
            lung_angle,
            breath_speed,
            breath_paused,
            last_breath_direction,
            pause_start_time,
        ) = update_lung_position(
            current_emotion_state=mood_engine.get_emotion_for_hand_controller(),
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

        if USE_SERVO and servos:
            try:
                servos.set_pan(pan)  # type: ignore
                servos.set_tilt(tilt)  # type: ignore
            except Exception as e:
                debug_print(f"Servo command failed: {e}", "ERROR")
                # Don't crash the whole system for servo errors

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
            # Small pause bar at bottom of screen
            bar_w, bar_h = 200, 12
            bar_x = 10  # Left side
            bar_y = frame.shape[0] - 30  # Bottom with small margin

            # Background (black with white border)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (0, 0, 0), -1)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (255, 255, 255), 2)

            # Progress to pause bar (color changes with proximity)
            progress = reactivity_metrics.get("progress_to_pause", 0)
            progress_len = int(min(progress, 100) * bar_w / 100)

            # Color: green -> yellow -> red as it approaches pause
            if progress >= 100:
                color = (0, 0, 255)  # Red (paused)
            elif progress >= 80:
                color = (0, 165, 255)  # Orange
            elif progress >= 60:
                color = (0, 255, 255)  # Yellow
            else:
                color = (0, 255, 0)  # Green

            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + progress_len, bar_y + bar_h), color, -1)

            # Pause threshold line (always at 100%)
            threshold_x = bar_x + bar_w
            cv2.line(frame, (threshold_x, bar_y), (threshold_x, bar_y + bar_h), (0, 0, 255), 3)

            # Text
            text = f"Pause: {progress:.0f}%"
            if reactivity_metrics.get("is_paused", False):
                text = f"PAUSED: {reactivity_metrics.get('pause_remaining', 0):.1f}s"
            elif reactivity_metrics.get("cooldown_remaining", 0) > 0:
                text = f"Cooldown: {reactivity_metrics.get('cooldown_remaining', 0):.1f}s"

            cv2.putText(frame, text, (bar_x, bar_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Chaos bar (blue) - normalize to 0-1 range
            chaos_norm = min(1.0, (reactivity_metrics["chaos_multiplier"] - 0.3) / 3.2)  # 0.3-3.5 -> 0-1
            chaos_len = int(chaos_norm * (bar_w - 20))
            cv2.rectangle(frame, (bar_x + 10, bar_y + 32), (bar_x + 10 + chaos_len, bar_y + 40), (255, 0, 0), -1)
            cv2.putText(frame, "CHS", (bar_x + 2, bar_y + 38), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

            # Pause indicator
            if reactivity_metrics.get("paused", False):
                cv2.rectangle(frame, (bar_x + bar_w - 30, bar_y + 10), (bar_x + bar_w - 10, bar_y + 30), (0, 0, 255), -1)
                cv2.putText(frame, "PAUSE", (bar_x + bar_w - 28, bar_y + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

                # SimpleLightbulbController uses frame diff for brightness, not mood fluctuation

        # MEMORY FIX: Throttle camera display to prevent graphics memory exhaustion
        current_time = time.time()
        if current_time - last_display_time > DISPLAY_THROTTLE_INTERVAL:
            cv2.imshow("mslint camera", frame)
            last_display_time = current_time

        # Hand controller now runs completely autonomously in its own thread
        # No GUI updates needed from machine.py

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

except KeyboardInterrupt:
    graceful_cleanup()
    sys.exit(0)
