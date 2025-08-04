import time
import argparse
import sys
import cv2
import threading
import signal


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
from servo_control.emotional_hand_controller import update_hand_position
from image_monitor import ImageMonitor
from utils.state_manager import state_manager
from utils.continuity import describe_duration
from servo_control.hand_expression import HandExpressionController
from config.config import (
    USE_SERVO,
    USE_HAND_SERVO,
    CAMERA_INDEX,
    SERIAL_PORT,
    HAND_SERIAL_PORT,
    BAUD_RATE,
    CONFIDENCE_THRESHOLD,
    MOOD_SNAPSHOT_FOLDER,
    MOOD_EVALUATION_INTERVAL,
    CAPTION_INTERVAL,
    MIN_SNAPSHOT_INTERVAL,
    PAUSE_DURATION,
    MODEL_PATH,
    CLEAN_CAPTION_OUTPUT,
    TEMPORAL_CONTEXT_CACHE_SECONDS,
    ENABLE_FRAME_CHANGE_DETECTION,
    FRAME_CHANGE_THRESHOLD,
    ATTENTION_SHIFT_OVERRIDE,
)
from event_logging.run_manager import get_run_image_path
from event_logging.event_logger import get_current_run_id, set_start_time, log_json_entry, LogType

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
breath_speed = 0.7  # Start closer to MIN_LUNG_SPEED for faster initial breathing
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
image_monitor.start()

# Initialize hand expression controller (if enabled)
hand_controller = None
if USE_HAND_SERVO:
    debug_print("Initializing hand expression controller", "INIT")
    hand_controller = HandExpressionController(port=HAND_SERIAL_PORT, baudrate=BAUD_RATE, clean_output=CLEAN_CAPTION_OUTPUT)
else:
    debug_print("Hand servo disabled - skipping hand controller initialization", "INIT")
    hand_controller = None  # Explicitly ensure it's None for clean captioning

# Face detection state for startle reactions
face_detected_last_frame = False

# Initialize run ID and start time for this session
start_time = time.time()
set_start_time(start_time)
run_id = get_current_run_id()
debug_print(f"Session initialized with run ID: {run_id}", "INIT")

log_json_entry(
    LogType.SESSION_START,
    {"run_id": run_id},
    MOOD_SNAPSHOT_FOLDER,
    auto_print=not CLEAN_CAPTION_OUTPUT,
    print_message=f"🚀 Starting session with run ID: {run_id}" if not CLEAN_CAPTION_OUTPUT else None,
)
log_json_entry(
    LogType.INFO,
    {"message": f"Event log: {run_id}-event-log.json"},
    MOOD_SNAPSHOT_FOLDER,
    auto_print=not CLEAN_CAPTION_OUTPUT,
    print_message=f"📁 Event log: {run_id}-event-log.json" if not CLEAN_CAPTION_OUTPUT else None,
)
log_json_entry(
    LogType.INFO,
    {"message": f"Images folder: {run_id}-images/"},
    MOOD_SNAPSHOT_FOLDER,
    auto_print=not CLEAN_CAPTION_OUTPUT,
    print_message=f"🖼️ Images folder: {run_id}-images/" if not CLEAN_CAPTION_OUTPUT else None,
)

debug_print("Initializing mood engine", "INIT")
mood_engine = MoodEngine()
debug_print("Initializing captioner", "INIT")
captioner = Captioner()

# Load previous session state if available
debug_print("Loading previous session state", "INIT")
previous_state = state_manager.load_session_state()
if previous_state:
    # Apply state to components
    state_manager.apply_state_to_captioner(previous_state, captioner)
    state_manager.apply_state_to_mood_engine(previous_state, mood_engine)
    # Reset last_caption so remnants from previous session are not printed
    captioner.last_caption = ""
    # Set memory loaded flag BEFORE generating awakening message
    captioner.memory_loaded_from_previous = True

    # Generate awakening message with continuity
    save_time = previous_state["metadata"]["save_time"]
    time_since_last = describe_duration(save_time)
    captioner.time_since_last_session = time_since_last  # Store for first caption
    previous_beliefs = previous_state["captioner"].get("beliefs", {})

    awakening_msg = captioner.generate_awakening_message(time_since_last, previous_beliefs)

    # Don't log the awakening message since the animation will show it
    # log_json_entry(
    #     LogType.INFO,
    #     {"message": awakening_msg, "continuity": True, "time_since_last": time_since_last},
    #     MOOD_SNAPSHOT_FOLDER,
    #     auto_print=CLEAN_CAPTION_OUTPUT,
    #     print_message=None,
    # )
    # Mark awakening complete to avoid duplicate environmental description
    captioner.mark_awakening_complete()
else:
    # Fresh start
    captioner.memory_loaded_from_previous = False
    awakening_msg = captioner.generate_awakening_message()

    # Don't log the awakening message since the animation will show it
    # log_json_entry(
    #     LogType.INFO,
    #     {"message": awakening_msg, "continuity": False},
    #     MOOD_SNAPSHOT_FOLDER,
    #     auto_print=CLEAN_CAPTION_OUTPUT,
    #     print_message=None,
    # )
    # Mark awakening complete to avoid duplicate environmental description
    captioner.mark_awakening_complete()

debug_print("System initialization complete", "INIT")

# Global shutdown flag
shutdown_requested = False

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    global shutdown_requested
    print(f"\n[🛑] Received shutdown signal {signum}")
    shutdown_requested = True

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

best_box = None


def mood_update_thread(frame, timestamp):
    global last_snapshot_time, best_box
    debug_print("Mood update thread started", "MOOD")
    if not captioner.is_processing:
        debug_print("Captioner is not processing, proceeding with mood update", "MOOD")
        now = time.time()
        if now - last_snapshot_time >= MIN_SNAPSHOT_INTERVAL:
            # Enhanced frame change detection to prevent repetitive processing
            should_process = True
            
            if ENABLE_FRAME_CHANGE_DETECTION:
                # Import numpy for frame comparison
                import numpy as np
                
                # Convert frame to grayscale for comparison
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Check if we have a previous frame to compare
                if hasattr(mood_update_thread, '_last_gray_frame'):
                    try:
                        # Try to use structural similarity if available
                        from skimage.metrics import structural_similarity as ssim
                        similarity = ssim(mood_update_thread._last_gray_frame, gray_frame)
                    except ImportError:
                        # Fallback to mean squared error comparison
                        mse = np.mean((mood_update_thread._last_gray_frame.astype(float) - gray_frame.astype(float)) ** 2)
                        # Convert MSE to similarity (0-1 scale, higher = more similar)
                        max_mse = 255 ** 2  # Maximum possible MSE for 8-bit images
                        similarity = 1.0 - (mse / max_mse)
                    
                    # Check if frame has changed significantly or if enough time has passed for attention shifting
                    time_since_last_processing = now - getattr(mood_update_thread, '_last_processing_time', 0)
                    frame_changed = similarity < FRAME_CHANGE_THRESHOLD
                    attention_shift_needed = time_since_last_processing > ATTENTION_SHIFT_OVERRIDE
                    
                    should_process = frame_changed or attention_shift_needed
                    
                    if not should_process:
                        debug_print(f"Frame similar ({similarity:.3f}) and recent ({time_since_last_processing:.1f}s), skipping", "CAPTIONER")
                        return
                    elif frame_changed:
                        debug_print(f"Frame change detected (similarity: {similarity:.3f})", "CAPTIONER")
                    else:
                        debug_print(f"Attention shift override after {time_since_last_processing:.1f}s", "CAPTIONER")
                
                mood_update_thread._last_gray_frame = gray_frame.copy()
                mood_update_thread._last_processing_time = now
            
            snapshot_path = get_run_image_path(MOOD_SNAPSHOT_FOLDER, f"mood_{int(now)}.jpg")
            cv2.imwrite(snapshot_path, frame)
            debug_print(f"Snapshot saved: {snapshot_path}", "MOOD")

            try:
                # First: Update captioner to generate new captions with temporal context (cached)
                from utils.temporal_context import TemporalContextManager
                if not hasattr(mood_update_thread, '_temporal_manager'):
                    mood_update_thread._temporal_manager = TemporalContextManager()
                
                # Only regenerate temporal context every configurable interval to reduce load
                current_time = time.time()
                if not hasattr(mood_update_thread, '_last_temporal_update') or \
                   current_time - mood_update_thread._last_temporal_update > TEMPORAL_CONTEXT_CACHE_SECONDS:
                    temporal_context = mood_update_thread._temporal_manager.get_temporal_context(captioner)
                    mood_update_thread._cached_temporal_context = temporal_context
                    mood_update_thread._last_temporal_update = current_time
                else:
                    temporal_context = getattr(mood_update_thread, '_cached_temporal_context', {})
                
                captioner.update(
                    frame=frame,
                    person_present=best_box is not None,
                    mood=mood_engine.get_current_mood(),
                    temporal_context=temporal_context,  # Add temporal awareness
                )
                debug_print("Captioner updated successfully", "CAPTIONER")

                # Second: Analyze mood from captioner's latest caption with temporal awareness
                if captioner.last_caption:
                    # Remove 'Caption:' prefix if present and print with line spacing
                    clean_caption = captioner.last_caption
                    if clean_caption.lower().startswith("caption:"):
                        clean_caption = clean_caption[len("caption:") :].strip()
                    print(f"\n{clean_caption}\n")

                    # Include temporal context in mood analysis (use cached version)
                    current_mood = mood_engine.analyze_mood(
                        clean_caption, 
                        image_path=snapshot_path,
                        temporal_context=getattr(mood_update_thread, '_cached_temporal_context', None)
                    )
                    debug_print(f"Mood analyzed from caption: {current_mood:.2f}", "MOOD")

                    # Third: Update captioner's mood state for next cycle
                    captioner.current_mood = current_mood

            except Exception as e:
                debug_print(f"Captioner update failed: {e}", "ERROR")
            last_snapshot_time = now
    else:
        debug_print("Captioner is processing, skipping mood update", "MOOD")


try:
    while True:
        # Check for shutdown request
        if shutdown_requested:
            debug_print("Shutdown requested, breaking main loop", "SHUTDOWN")
            break
            
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

        if best_box is not None:
            # Check for NEW face detection (startle reaction)
            face_detected_this_frame = True
            if not face_detected_last_frame and USE_HAND_SERVO:  # Only process startle if hand servo enabled
                # NEW FACE DETECTED - TRIGGER STARTLE WITH COOLDOWN!
                if not CLEAN_CAPTION_OUTPUT:
                    print("😲 STARTLE! Face detected - triggering coordinated hand reaction")
                debug_print("😲 STARTLE! Face detected - triggering coordinated hand reaction", "STARTLE")
                if hand_controller:  # Only trigger if hand system is enabled
                    try:
                        hand_controller.trigger_startle()  # Now includes cooldown logic
                        if not CLEAN_CAPTION_OUTPUT:
                            print("✅ Startle command sent to hand controller")
                        debug_print("✅ Startle command sent successfully", "STARTLE")
                    except Exception as e:
                        if not CLEAN_CAPTION_OUTPUT:
                            print(f"❌ Startle failed: {e}")
                        debug_print(f"❌ Startle command failed: {e}", "ERROR")
                else:
                    if not CLEAN_CAPTION_OUTPUT:
                        print("🚫 Hand servo disabled - skipping startle reaction")
                    debug_print("🚫 Hand servo disabled - skipping startle reaction", "STARTLE")
            
            # Suppress frequent face detection messages - only show occasionally
            if int(now) % 5 == 0:  # Every 5 seconds
                debug_print(f"Face detected with confidence: {best_conf:.2f}", "FACE")
            
            # Track person presence for temporal relationships (with error handling)
            try:
                captioner.update_person_presence(True, now)
            except Exception as e:
                debug_print(f"Person presence tracking error: {e}", "ERROR")
        else:
            face_detected_this_frame = False
            # Only show "no face" occasionally to avoid spam
            if int(now) % 10 == 0:  # Every 10 seconds
                debug_print("No face detected", "FACE")
            
            # Track absence for temporal context (with error handling)
            try:
                captioner.update_person_presence(False, now)
            except Exception as e:
                debug_print(f"Person presence tracking error: {e}", "ERROR")

        # Update face detection state for next frame
        face_detected_last_frame = face_detected_this_frame

        if now - last_mood_time > MOOD_EVALUATION_INTERVAL:
            debug_print(f"Starting mood update thread - interval: {MOOD_EVALUATION_INTERVAL}s", "MOOD")
            threading.Thread(target=mood_update_thread, args=(frame.copy(), int(now)), daemon=True).start()
            last_mood_time = now

        current_mood = mood_engine.get_current_mood()

        face_box = tuple(best_box) if best_box is not None else None
        person_present, pan, tilt = update_gaze(frame, face_box, current_mood, delta)
        
        # Update hand expression based on consciousness state (if enabled)
        hand_positions = None
        if hand_controller and USE_HAND_SERVO:  # Double check both conditions
            temporal_context = getattr(mood_update_thread, '_cached_temporal_context', {})
            hand_positions = hand_controller.update_from_consciousness(
                mood=current_mood,
                novelty=captioner.novelty_score,
                boredom=captioner.boredom,
                person_present=person_present,
                temporal_context=temporal_context
            )
        
        # Debug servo values occasionally to avoid spam
        if int(now) % 8 == 0:  # Every 8 seconds to include hand info
            if USE_HAND_SERVO and hand_controller:
                gesture_desc = hand_controller.get_current_gesture_description()
            else:
                gesture_desc = "Hand servo disabled"
            debug_print(f"Servo values: pan={pan}, tilt={tilt}, person_present={person_present}", "SERVO")
            debug_print(f"Hand gesture: {gesture_desc}", "HAND")

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

        # === REMOVED OLD AUTONOMOUS HAND CONTROL ===
        # The old update_hand_position() system has been replaced by
        # hand_controller.update_from_consciousness() above (line ~420)
        # Default hand state for display overlay compatibility
        hand_state = {
            'emotional_state': 'consciousness-driven' if hand_controller else 'disabled',
            'is_frozen': False,
            'idle_mode': False
        }

        if face_box:
            (x1, y1, x2, y2) = face_box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # === DISPLAY OVERLAYS ===
        freeze_status = ""
        if 'is_frozen' in hand_state and hand_state['is_frozen']:
            freeze_status = " [FROZEN]"
        elif 'idle_mode' in hand_state and hand_state['idle_mode']:
            freeze_status = " [IDLE]"
        
        debug = f"Mood: {current_mood:.2f} | Lung: {lung_pos} | Hand: {hand_state['emotional_state']}{freeze_status} | Pan/Tilt: {pan}/{tilt}"
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

        if cv2.waitKey(1) & 0xFF == ord("q") or shutdown_requested:
            debug_print("Exit requested", "SHUTDOWN")
            break

except KeyboardInterrupt:
    debug_print("Shutting down gracefully", "SHUTDOWN")
    print("[🛑] Shutdown initiated...")

except Exception as e:
    debug_print(f"Unexpected error: {e}", "ERROR")
    print(f"[❌] Unexpected error: {e}")

finally:
    # Ensure cleanup happens regardless of how we exit
    try:
        # Save session state before shutdown
        print("[💾] Saving session state...")
        success = state_manager.save_session_state(captioner, mood_engine)
        if success:
            print("[✅] Session state saved successfully")
        else:
            print("[❌] Failed to save session state")

        # Log session end
        log_json_entry(
            LogType.INFO,
            {"message": "Session ended", "run_id": run_id, "duration": time.time() - start_time},
            MOOD_SNAPSHOT_FOLDER,
            auto_print=not CLEAN_CAPTION_OUTPUT,
            print_message=f"[👋] Session ended. Duration: {time.time() - start_time:.1f}s" if not CLEAN_CAPTION_OUTPUT else None,
        )
    except Exception as e:
        print(f"[⚠️] Error during state saving: {e}")

    # Cleanup resources
    try:
        print("[🧹] Cleaning up resources...")
        if 'object_detector' in locals():
            object_detector.stop()
            object_detector.join(timeout=2.0)  # Add timeout
        if 'image_monitor' in locals():
            image_monitor.stop()
        if 'hand_controller' in locals() and hand_controller and USE_HAND_SERVO:
            hand_controller.cleanup()
        if 'cap' in locals():
            cap.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)  # Force window cleanup
        print("[✅] Cleanup complete")
    except Exception as e:
        print(f"[⚠️] Error during cleanup: {e}")
        cv2.destroyAllWindows()  # Force cleanup even if error
