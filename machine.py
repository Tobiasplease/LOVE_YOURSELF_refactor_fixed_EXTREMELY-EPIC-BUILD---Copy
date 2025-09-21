#!/usr/bin/env python3
print("🚀 [DEBUG] MACHINE.PY STARTING UP")

import argparse
import atexit
import glob
import os
import signal
import sys
import threading
import time

print("🚀 [DEBUG] BASIC IMPORTS COMPLETE")

import cv2

try:
    import torch
except ImportError:
    torch = None

from breathing.breathing import update_lung_position
from captioner.captioner import Captioner
from config.config import (
    BAUD_RATE,
    CAMERA_INDEX,
    CAMERA_WIDTH,
    CAMERA_HEIGHT,
    CAMERA_SHARPNESS,
    CAMERA_SATURATION,
    CAMERA_CONTRAST,
    CAMERA_BRIGHTNESS,
    CAMERA_EXPOSURE,
    CAMERA_AUTO_FOCUS,
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
    USE_LIGHTBULB_PWM,
    USE_SERVO,
    USE_UARM,
    USE_CAPTION_DISPLAY,
    CAPTION_DISPLAY_PORT,
    UARM_PORT,
    UARM_CONNECT_ON_STARTUP,
    UARM_HOME_ON_CONNECT,
    UARM_DEFAULT_SPEED,
    UARM_MOTION_STORAGE,
)
from event_logging.event_logger import get_current_run_id, log_json_entry, set_start_time
from event_logging.log_type import LogType
from event_logging.run_manager import get_run_image_path
from grbl.idle_movement_manager import start_idle_movements, stop_idle_movements, update_emotion
from image_monitor import ImageMonitor
from mood.mood import MoodEngine
from perception.detection_memory import DetectionMemory
from perception.object_detection import ObjectDetectionThread
from perception.person_detection_state import get_person_detection_state
from reactivity.camera_reactive import CameraReactivityEngine
from utils.continuity import describe_duration
from utils.error_tracking import get_failure_tracker
from utils.state_manager import state_manager
from vision.gaze import update_gaze


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


# Imports moved to top of file

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

    def send_reactivity_data(*stub_args, **stub_kwargs):
        pass

    def get_status():
        return "disabled"

    def change_to_emotion(*stub_args, **stub_kwargs):
        pass

    def start_autonomous_mode(*stub_args, **stub_kwargs):
        pass


# Moved to top

if USE_SERVO:
    from servo_control.servo_control import ServoController

if USE_LIGHTBULB_PWM:
    from servo_control.lightbulb_controller_nonblocking import NonBlockingLightbulbController

if USE_UARM:
    # Try native controller API; if unavailable, fall back to official Teach API
    UARM_BACKEND = "controller"
    try:
        from uarm_control.uarm_controller import UarmController
        from uarm_control.motion_manager import MotionManager
        from uarm_control.simple_api import UarmSimpleAPI
        import uarm_control.simple_api as uarm_api
    except Exception:
        UARM_BACKEND = "teach"
        try:
            from uarm_control.teach_menu import UArmTeachApp  # Official SDK path
        except Exception as e:
            print(f"Warning: uArm controller imports unavailable: {e}")
            UARM_BACKEND = None

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
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

# Set camera image quality properties
def set_camera_property_safe(cap, prop, value, name):
    """Safely set camera property with error handling"""
    if value != -1:  # -1 means use default/auto
        try:
            result = cap.set(prop, value)
            if result:
                debug_print(f"Camera {name} set to {value}", "INIT")
            else:
                debug_print(f"Camera {name} setting failed (not supported)", "WARN")
        except Exception as e:
            debug_print(f"Camera {name} error: {e}", "ERROR")

# Apply camera quality settings
set_camera_property_safe(cap, cv2.CAP_PROP_SHARPNESS, CAMERA_SHARPNESS, "sharpness")
set_camera_property_safe(cap, cv2.CAP_PROP_SATURATION, CAMERA_SATURATION, "saturation")
set_camera_property_safe(cap, cv2.CAP_PROP_CONTRAST, CAMERA_CONTRAST, "contrast")
set_camera_property_safe(cap, cv2.CAP_PROP_BRIGHTNESS, CAMERA_BRIGHTNESS, "brightness")
set_camera_property_safe(cap, cv2.CAP_PROP_EXPOSURE, CAMERA_EXPOSURE, "exposure")
set_camera_property_safe(cap, cv2.CAP_PROP_AUTOFOCUS, 1 if CAMERA_AUTO_FOCUS else 0, "autofocus")

proto = f"{MODEL_PATH}/deploy.prototxt"
model = f"{MODEL_PATH}/res10_300x300_ssd_iter_140000.caffemodel"

debug_print("Loading face detection model", "INIT")
net = cv2.dnn.readNetFromCaffe(proto, model)
debug_print("Face detection model loaded", "INIT")

# Initialize person detection state for consciousness context
person_detection = get_person_detection_state()
debug_print("Person detection state initialized", "INIT")
if USE_SERVO:
    servo_port = ARDUINO_DEVICES["LUNGGAZE"]  # Gaze pan/tilt and breath controller
    if os.path.exists(servo_port):
        try:
            servos = ServoController(port=servo_port, baudrate=BAUD_RATE)
            debug_print(f"Servo controller initialized on {servo_port}", "INIT")
            
            # Startup movement sequence disabled to prevent timing conflicts
            debug_print("Servo controller ready - skipping startup sequence", "INIT")
            
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

# Initialize uArm Swift Pro robotic arm (separate from servos)
uarm_controller = None
motion_manager = None
uarm_teach_app = None
if USE_UARM and UARM_BACKEND:
    try:
        uarm_port = ARDUINO_DEVICES["UARM"]
        debug_print(f"Initializing uArm on {uarm_port} using backend={UARM_BACKEND}", "INIT")

        if not os.path.exists(uarm_port):
            print(f"WARNING: uArm device not found at {uarm_port}")
            print("  Check USB connection or udev symlink")
        else:
            if UARM_BACKEND == "controller":
                uarm_controller = UarmController(port=uarm_port, connect_on_init=UARM_CONNECT_ON_STARTUP)
                if uarm_controller and uarm_controller.is_connected():
                    debug_print("uArm connected successfully", "INIT")
                    motion_manager = MotionManager(storage_path=UARM_MOTION_STORAGE, controller=uarm_controller)
                    debug_print("uArm motion manager initialized", "INIT")
                    simple_api = UarmSimpleAPI(controller=uarm_controller, motion_manager=motion_manager)
                    uarm_api.uarm_api = simple_api
                    debug_print("uArm simple API initialized", "INIT")
                    if UARM_HOME_ON_CONNECT:
                        debug_print("Performing uArm homing sequence", "INIT")
                        uarm_controller.home()
                else:
                    debug_print("uArm connection failed - check USB connection and firmware", "WARN")
                    if uarm_controller and uarm_controller.last_error:
                        debug_print(f"uArm error: {uarm_controller.last_error}", "WARN")
            elif UARM_BACKEND == "teach":
                try:
                    uarm_teach_app = UArmTeachApp()
                    uarm_teach_app.connect()
                    debug_print("uArm (Teach API) connected", "INIT")
                    # Startup reassurance play (non-blocking)
                    try:
                        from config.config import UARM_PLAY_ON_START, UARM_START_PLAY_FILE
                        if UARM_PLAY_ON_START:
                            import threading

                            def _uarm_start_play():
                                try:
                                    target = UARM_START_PLAY_FILE
                                    # Fast playback; avoid homing around play
                                    try:
                                        uarm_teach_app.smoothing_enabled = False
                                        uarm_teach_app.use_home_before_play = False
                                        uarm_teach_app.use_home_after_play = False
                                    except Exception:
                                        pass
                                    if os.path.exists(target):
                                        uarm_teach_app.play_file = target
                                    print(f"[uArm] Startup play: {os.path.basename(getattr(uarm_teach_app, 'play_file', target))}")
                                    uarm_teach_app.play()
                                except Exception as e:
                                    print(f"[uArm] Startup play failed: {e}")

                            threading.Thread(target=_uarm_start_play, daemon=True, name="UArmStartupPlay").start()
                    except Exception:
                        pass
                except Exception as e:
                    print(f"WARNING: Failed to init uArm Teach API: {e}")
            else:
                debug_print("uArm backend not available", "WARN")

    except Exception as e:
        debug_print(f"uArm initialization failed: {e}", "ERROR")
        uarm_controller = None
        motion_manager = None
else:
    debug_print("uArm control disabled in config", "INIT")

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
organic_left_arm = None


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

        # Emergency organic left arm stop
        try:
            if 'organic_left_arm' in globals() and organic_left_arm:
                organic_left_arm.shutdown()
        except Exception:
            pass

        # Stop idle movements on shutdown
        try:
            stop_idle_movements()
        except Exception as e:
            print(f"[WARNING] Failed to stop idle movements: {e}")

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

    # Stop organic left arm controller
    try:
        if 'organic_left_arm' in globals() and organic_left_arm:
            organic_left_arm.shutdown()
            print("[🦾] Organic left arm controller shutdown")
    except Exception as e:
        print(f"[ERROR] Error stopping organic left arm: {e}")

    # Stop uArm controller
    if uarm_controller:
        try:
            debug_print("Disconnecting uArm controller", "CLEANUP")
            uarm_controller.disconnect()
        except Exception as e:
            print(f"[ERROR] Error stopping uArm controller: {e}")
    elif 'uarm_teach_app' in globals() and uarm_teach_app:
        try:
            debug_print("Disconnecting uArm (Teach API)", "CLEANUP")
            uarm_teach_app.disconnect()
        except Exception as e:
            print(f"[ERROR] Error stopping uArm (Teach API): {e}")

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

    # Close LCD caption display
    if USE_CAPTION_DISPLAY:
        try:
            from utils.caption_display import close_caption_display
            close_caption_display()
            print("[📟] LCD caption display closed")
        except Exception as e:
            print(f"[WARNING] Error closing LCD caption display: {e}")

    cleanup_completed = True
    print("[SUCCESS] Graceful shutdown completed")


def signal_handler(signum, signal_frame):
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
# Import moved to top

object_detector = ObjectDetectionThread()
_global_object_detector = object_detector
object_detector.start()

# Start image monitoring
debug_print("Starting image monitor", "INIT")
image_monitor = ImageMonitor(log_folder=MOOD_SNAPSHOT_FOLDER)
_global_image_monitor = image_monitor

# Idle movements manager will start after emotion state is determined

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

# Initialize LCD caption display
if USE_CAPTION_DISPLAY:
    debug_print("Initializing LCD caption display", "INIT")
    try:
        from utils.caption_display import init_caption_display
        init_caption_display(CAPTION_DISPLAY_PORT)
        log_json_entry(
            LogType.INFO,
            {"message": f"LCD caption display initialized on {CAPTION_DISPLAY_PORT}"},
            print_message=f"📟 LCD caption display initialized on {CAPTION_DISPLAY_PORT}",
        )
    except Exception as e:
        log_json_entry(
            LogType.ERROR,
            {"message": f"Failed to initialize LCD caption display: {e}"},
            print_message=f"❌ Failed to initialize LCD caption display: {e}",
        )
else:
    debug_print("LCD caption display disabled in config", "INIT")

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

    # Initialize organic left arm controller for servo pins 4 and 5
    debug_print("Initializing organic left arm controller", "INIT")
    try:
        from hand_control.organic_left_arm import OrganicLeftArmController
        import serial

        # Create serial connection to left hand Arduino
        left_arm_serial = serial.Serial(ARDUINO_DEVICES["LEFTHAND"], 9600, timeout=1)
        time.sleep(2)  # Allow Arduino to reset

        # Initialize organic controller
        organic_left_arm = OrganicLeftArmController(serial_connection=left_arm_serial)
        organic_left_arm.enable()

        debug_print("Organic left arm controller started successfully", "INIT")
        log_json_entry(
            LogType.INFO,
            {"message": "Organic left arm controller initialized", "port": ARDUINO_DEVICES["LEFTHAND"]},
            print_message=f"🦾 Organic left arm controller initialized on {ARDUINO_DEVICES['LEFTHAND']}",
        )
    except Exception as e:
        debug_print(f"Failed to initialize organic left arm controller: {e}", "ERROR")
        log_json_entry(
            LogType.ERROR,
            {"message": f"Failed to initialize organic left arm: {e}"},
            print_message=f"❌ Failed to initialize organic left arm: {e}",
        )
        organic_left_arm = None

else:
    debug_print("Hand controller failed to start", "WARN")
    organic_left_arm = None

# Initialize camera reactivity engine
debug_print("Initializing camera reactivity engine", "INIT")
reactivity_engine = CameraReactivityEngine(sensitivity=1.8, smoothing_factor=0.85, pause_threshold=0.20, pause_duration=3.0)
debug_print("Camera reactivity enabled - hand will respond to environmental changes", "INIT")


# Set up image monitor with self-critique callback
def on_drawing_complete(image_path: str):
    """Handle drawing completion with self-critique only (uArm handled via GRBL hook)."""
    captioner.drawing.critique_drawing(image_path)


image_monitor.on_image_complete = on_drawing_complete
image_monitor.start()

# Register GRBL-complete hook to trigger uArm after actual G-code completion
print("🔥 [DEBUG] STARTING HOOK REGISTRATION BLOCK")
try:
    import utils.hooks as _hooks
    from config.config import USE_UARM, UARM_PLAY_AFTER_DRAW, UARM_PLAY_FILE
    print(f"🔥 [DEBUG] Hook setup: USE_UARM={USE_UARM}, UARM_PLAY_AFTER_DRAW={UARM_PLAY_AFTER_DRAW}, UARM_PLAY_FILE={UARM_PLAY_FILE}")
    if USE_UARM and UARM_PLAY_AFTER_DRAW and UARM_PLAY_FILE:
        def _normalize_smooth(path: str) -> str:
            base, ext = os.path.splitext(path)
            while base.lower().endswith('.smooth'):
                base = base[:-7]
            return f"{base}{ext or '.txt'}"

        def _uarm_after_grbl():
            print(f"[DEBUG] _uarm_after_grbl() called - starting papermove sequence")
            def _run():
                try:
                    from grbl.idle_movement_manager import pause_for_drawing, get_manager

                    # Force pause idle movements regardless of CNC state
                    # (since we're in transition period between GRBL completion and uArm execution)
                    manager = get_manager()
                    if manager.process and manager.process.poll() is None:
                        print("[DEBUG] Force pausing idle movements for uArm sequence")
                        manager.pause_for_drawing()
                    else:
                        print("[DEBUG] No idle movements running to pause")

                    target = _normalize_smooth(UARM_PLAY_FILE)
                    app = None
                    try:
                        if 'uarm_teach_app' in globals() and uarm_teach_app:
                            app = uarm_teach_app
                    except Exception:
                        app = None
                    if app is None:
                        from uarm_control.teach_menu import UArmTeachApp
                        app = UArmTeachApp()
                        app.connect()
                    try:
                        app.smoothing_enabled = False
                        app.use_home_before_play = False
                        app.use_home_after_play = False
                    except Exception:
                        pass
                    if os.path.exists(target):
                        app.play_file = target
                    print(f"[uArm] Post-GRBL play: {os.path.basename(getattr(app, 'play_file', target))}")
                    app.play()

                    # Wait for actual movement completion instead of fixed 30s delay
                    if hasattr(app, 'teach') and app.teach:
                        print("[uArm] Waiting for movement completion...")
                        timeout = 60.0  # Maximum wait time
                        start_time = time.time()
                        while (app.teach.is_playing() and
                               (time.time() - start_time) < timeout):
                            time.sleep(0.5)

                        if app.teach.is_playing():
                            print("[uArm] Movement timeout - forcing completion")
                        else:
                            print("[uArm] Movement completed successfully")
                    else:
                        # Fallback to shorter fixed delay if we can't track completion
                        print("[uArm] Using fallback 15s delay")
                        time.sleep(15.0)

                except Exception as e:
                    print(f"[uArm] Post-GRBL play failed: {e}")
                finally:
                    # Brief pause to ensure arm settles before resuming CNC movements
                    try:
                        time.sleep(2.0)
                    except Exception:
                        pass
                    # Note: idle movement resumption is handled by the completion ritual
                    # after CNC state is properly cleared, so we don't need to do it here

            # Run synchronously to ensure GRBL waits for uArm completion before clearing CNC state
            print("[DEBUG] Running uArm sequence synchronously to coordinate with GRBL")
            _run()
            print("[DEBUG] uArm sequence completed, GRBL can now clear CNC state")

        _hooks.on_grbl_drawing_complete = _uarm_after_grbl
        print(f"🔥 [DEBUG] GRBL hook registered successfully - uArm will trigger after drawing completion")
    else:
        print(f"[DEBUG] GRBL hook NOT registered - condition failed")
except Exception as e:
    print(f"[DEBUG] GRBL hook registration failed: {e}")
    import traceback
    traceback.print_exc()

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

    # Start idle CNC movements with restored emotion
    if start_idle_movements(emotion):
        debug_print(f"Idle CNC movements started with emotion: {emotion}", "INIT")
    else:
        debug_print("Failed to start idle CNC movements", "WARN")

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
    )
    # Mark awakening complete to avoid duplicate environmental description
    captioner.mark_awakening_complete()

    # Start idle CNC movements with default emotion
    default_emotion = "calm_observant"
    if start_idle_movements(default_emotion):
        debug_print(f"Idle CNC movements started with default emotion: {default_emotion}", "INIT")
    else:
        debug_print("Failed to start idle CNC movements", "WARN")

debug_print("System initialization complete", "INIT")

best_box = None

# Track last printed caption timestamp to prevent duplicates
last_printed_caption_time = 0.0
last_state_save_time = 0.0


# Redundant functions removed - using existing MoodEngine functionality


def mood_update_thread(mood_frame, timestamp):
    global last_snapshot_time, last_state_save_time, mood_thread_running

    # Set running flag at start
    with mood_thread_lock:
        mood_thread_running = True

    try:
        debug_print("Mood update thread started", "MOOD")
        thread_now = time.time()
        if thread_now - last_snapshot_time >= 10:
            snapshot_path = get_run_image_path(MOOD_SNAPSHOT_FOLDER, f"mood_{int(thread_now)}.jpg")
            cv2.imwrite(snapshot_path, mood_frame)
            debug_print(f"Snapshot saved: {snapshot_path}", "MOOD")

            try:
                # Use existing MoodEngine for emotional state - single source of truth
                thread_emotion = mood_engine.get_emotion_for_hand_controller()
                thread_mood = mood_engine.get_current_mood()

                debug_print(f"Current emotion: {thread_emotion}, mood: {thread_mood:.2f}", "EMOTION")

                # Captioner is now updated in main loop - mood system just reads captions
                debug_print("Captioner updated successfully", "CAPTIONER")

                # Second: Process caption and update physical systems
                if captioner.last_caption:
                    clean_caption = captioner.last_caption
                    if clean_caption.lower().startswith("caption:"):
                        clean_caption = clean_caption[len("caption:") :].strip()

                    # CRITICAL: Process caption through mood analysis (was missing!)
                    mood_engine.analyze_mood(
                        clean_caption,
                        saw_person=best_box is not None,
                        image_path=snapshot_path if 'snapshot_path' in locals() else None,
                        memory_context=captioner.memory_manager if hasattr(captioner, 'memory_manager') else None
                    )
                    debug_print(f"Processed caption through mood analysis: {clean_caption[:100]}...", "MOOD")

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
                    if thread_now - last_state_save_time > 120:  # 2 minutes
                        if _global_state_manager:
                            try:
                                _global_state_manager.save_session_state(captioner, mood_engine)
                                last_state_save_time = thread_now
                            except Exception as e:
                                print(f"[ERROR] Periodic state save failed: {e}")

                    # Update hand controller with new emotional state
                    change_to_emotion(thread_emotion)
                    debug_print(f"Updated hand controller emotion: {thread_emotion}", "HAND")

                    # Update CNC idle movements with new emotional state
                    update_emotion(thread_emotion)
                    debug_print(f"Updated CNC emotion: {thread_emotion}", "CNC")

                    # Third: Update captioner's mood state and pattern data for next cycle
                    captioner.current_mood = thread_mood
                    pattern_data = mood_engine.get_pattern_data()
                    captioner.set_novelty_score(pattern_data["novelty_score"])
                    # Pass recent motifs to captioner for memory integration
                    captioner.current_motifs_from_mood = pattern_data["recent_motifs"]

            except Exception as e:
                debug_print(f"Captioner update failed: {e}", "ERROR")
            last_snapshot_time = thread_now
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

# Real-time camera controls with persistence
import json

CAMERA_SETTINGS_FILE = "camera_settings.json"

default_brightness = CAMERA_BRIGHTNESS if CAMERA_BRIGHTNESS != -1 else 50
default_contrast = CAMERA_CONTRAST if CAMERA_CONTRAST != -1 else 50
default_saturation = CAMERA_SATURATION if CAMERA_SATURATION != -1 else 50
default_sharpness = CAMERA_SHARPNESS if CAMERA_SHARPNESS != -1 else 50

def load_camera_settings():
    """Load camera settings from file, return defaults if file doesn't exist"""
    try:
        with open(CAMERA_SETTINGS_FILE, 'r') as f:
            settings = json.load(f)
            return {
                'brightness': settings.get('brightness', default_brightness),
                'contrast': settings.get('contrast', default_contrast),
                'saturation': settings.get('saturation', default_saturation),
                'sharpness': settings.get('sharpness', default_sharpness)
            }
    except (FileNotFoundError, json.JSONDecodeError):
        return {
            'brightness': default_brightness,
            'contrast': default_contrast,
            'saturation': default_saturation,
            'sharpness': default_sharpness
        }

def save_camera_settings(brightness, contrast, saturation, sharpness):
    """Save current camera settings to file"""
    try:
        settings = {
            'brightness': brightness,
            'contrast': contrast,
            'saturation': saturation,
            'sharpness': sharpness
        }
        with open(CAMERA_SETTINGS_FILE, 'w') as f:
            json.dump(settings, f, indent=2)
        if DEBUG_MODE:
            debug_print("Camera settings saved", "CAMERA")
    except Exception as e:
        if DEBUG_MODE:
            debug_print(f"Failed to save camera settings: {e}", "CAMERA")

# Load saved settings
saved_settings = load_camera_settings()
current_brightness = saved_settings['brightness']
current_contrast = saved_settings['contrast']
current_saturation = saved_settings['saturation']
current_sharpness = saved_settings['sharpness']

def on_brightness_change(val):
    global current_brightness
    current_brightness = val
    try:
        result = cap.set(cv2.CAP_PROP_BRIGHTNESS, val)
        if not result and DEBUG_MODE:
            debug_print(f"Brightness setting to {val} failed (not supported)", "CAMERA")
        save_camera_settings(current_brightness, current_contrast, current_saturation, current_sharpness)
    except Exception as e:
        if DEBUG_MODE:
            debug_print(f"Brightness adjustment error: {e}", "CAMERA")

def on_contrast_change(val):
    global current_contrast
    current_contrast = val
    try:
        result = cap.set(cv2.CAP_PROP_CONTRAST, val)
        if not result and DEBUG_MODE:
            debug_print(f"Contrast setting to {val} failed (not supported)", "CAMERA")
        save_camera_settings(current_brightness, current_contrast, current_saturation, current_sharpness)
    except Exception as e:
        if DEBUG_MODE:
            debug_print(f"Contrast adjustment error: {e}", "CAMERA")

def on_saturation_change(val):
    global current_saturation
    current_saturation = val
    try:
        result = cap.set(cv2.CAP_PROP_SATURATION, val)
        if not result and DEBUG_MODE:
            debug_print(f"Saturation setting to {val} failed (not supported)", "CAMERA")
        save_camera_settings(current_brightness, current_contrast, current_saturation, current_sharpness)
    except Exception as e:
        if DEBUG_MODE:
            debug_print(f"Saturation adjustment error: {e}", "CAMERA")

def on_sharpness_change(val):
    global current_sharpness
    current_sharpness = val
    try:
        result = cap.set(cv2.CAP_PROP_SHARPNESS, val)
        if not result and DEBUG_MODE:
            debug_print(f"Sharpness setting to {val} failed (not supported)", "CAMERA")
        save_camera_settings(current_brightness, current_contrast, current_saturation, current_sharpness)
    except Exception as e:
        if DEBUG_MODE:
            debug_print(f"Sharpness adjustment error: {e}", "CAMERA")

def reset_camera_controls():
    """Reset all camera controls to default values"""
    global current_brightness, current_contrast, current_saturation, current_sharpness
    current_brightness = default_brightness
    current_contrast = default_contrast
    current_saturation = default_saturation
    current_sharpness = default_sharpness

    cv2.setTrackbarPos("Brightness", "mslint camera", default_brightness)
    cv2.setTrackbarPos("Contrast", "mslint camera", default_contrast)
    cv2.setTrackbarPos("Saturation", "mslint camera", default_saturation)
    cv2.setTrackbarPos("Sharpness", "mslint camera", default_sharpness)

    # Apply defaults to camera
    cap.set(cv2.CAP_PROP_BRIGHTNESS, default_brightness)
    cap.set(cv2.CAP_PROP_CONTRAST, default_contrast)
    cap.set(cv2.CAP_PROP_SATURATION, default_saturation)
    cap.set(cv2.CAP_PROP_SHARPNESS, default_sharpness)

    save_camera_settings(default_brightness, default_contrast, default_saturation, default_sharpness)
    if DEBUG_MODE:
        debug_print("Camera controls reset to defaults", "CAMERA")

# Create trackbars for real-time camera adjustment
cv2.namedWindow("mslint camera")
cv2.createTrackbar("Brightness", "mslint camera", current_brightness, 100, on_brightness_change)
cv2.createTrackbar("Contrast", "mslint camera", current_contrast, 100, on_contrast_change)
cv2.createTrackbar("Saturation", "mslint camera", current_saturation, 100, on_saturation_change)
cv2.createTrackbar("Sharpness", "mslint camera", current_sharpness, 100, on_sharpness_change)

debug_print("Camera controls initialized - use trackbars in preview window to adjust in real-time", "INIT")
debug_print("Press 'r' in camera window to reset controls to defaults", "INIT")

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

        # Update person detection state for consciousness context
        if best_box is not None:
            # Convert box to (x1, y1, x2, y2) format
            x1, y1, x2, y2 = best_box
            person_detection.update_face_detection(best_conf, (x1, y1, x2, y2))
        else:
            person_detection.update_face_detection(0.0)

        # Disable YOLO person fallback to rely only on high-confidence face detection
        # This prevents false positives from YOLO detecting "persons" in walls/objects
        # if best_box is None:
        #     labels = DetectionMemory.get_labels()
        #     if "person" in labels:
        #         best_box = [50, 50, 100, 100]  # Dummy box to indicate person present
        #         best_conf = 0.8  # Dummy confidence for YOLO person detection
        #         if int(now) % 5 == 0:  # Every 5 seconds
        #             debug_print("Person detected by YOLO", "PERSON")

        if best_box is not None:
            # Suppress frequent detection messages - only show occasionally
            if int(now) % 5 == 0:  # Every 5 seconds
                source = "Face" if best_conf < 0.7 else "Person"  # Distinguish source
                debug_print(f"{source} detected with confidence: {best_conf:.2f}", "DETECTION")
        else:
            # Only show "no person" occasionally to avoid spam
            if int(now) % 10 == 0:  # Every 10 seconds
                debug_print("No person detected", "DETECTION")

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

        # Attach egocentric orientation and face hint to reactivity data
        reactivity_with_view = dict(reactivity_metrics)
        reactivity_with_view["pan"] = float(pan)
        reactivity_with_view["tilt"] = float(tilt)
        if face_box is not None:
            x1, y1, x2, y2 = face_box
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            reactivity_with_view["face_pos"] = {"x": cx / w, "y": cy / h}

        # Add person consciousness context for rich social awareness
        person_context = person_detection.get_consciousness_context()
        reactivity_with_view["person_consciousness"] = person_context

        # Update captioner with every frame (decoupled from mood system)
        captioner.update(
            frame=frame,
            person_present=best_box is not None,
            mood=mood_engine.get_current_mood() if mood_engine else 0.5,
            reactivity_data=reactivity_with_view,
        )

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
                if DEBUG_MODE and frame_count % 30 == 0:
                    debug_print(f"Sending PAN/TILT: {pan}/{tilt}", "SERVO")
                # Check servo controller health
                if frame_count % 150 == 0:  # Every 5 seconds at 30fps
                    servo_health = "OK" if (servos.ser and servos.ser.is_open) else "DISCONNECTED"
                    debug_print(f"Servo controller health: {servo_health}", "SERVO")
                servos.set_pan(pan)  # type: ignore
                servos.set_tilt(tilt)  # type: ignore
            except Exception as e:
                debug_print(f"Servo command failed: {e}", "ERROR")
                print(f"[SERVO ERROR] Exception details: {type(e).__name__}: {e}")
                # Don't crash the whole system for servo errors
        elif USE_SERVO and not servos and frame_count % 120 == 0:
            debug_print("Servo controller not initialized; skipping PAN/TILT/LUNG sends", "WARN")

        if face_box:
            (x1, y1, x2, y2) = face_box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # === DISPLAY OVERLAYS ===
        debug = f"Mood: {current_mood:.2f} | Lung: {lung_pos} | Pan/Tilt: {pan}/{tilt}"
        cv2.putText(frame, debug, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

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

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            reset_camera_controls()
except KeyboardInterrupt:
    graceful_cleanup()
    sys.exit(0)
