#!/usr/bin/env python3

import argparse
import atexit
import glob
import os
import random
import signal
import subprocess
import sys
import threading
import time


# Before ANY project import can write a timestamp: if the RTC booted us into
# the future, let NTP step the clock BEFORE the run exists, and watch for
# steps after that (a mid-run step froze run 980f6e82 for what would have
# been 51 days). utils/clock_guard is stdlib-only, safe this early.
from utils.clock_guard import guard_clock
from utils.single_instance import refuse_second_machine

refuse_second_machine()
guard_clock()

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
    KINETIC_BUS_ENABLED,
    KINETIC_GANTRY,
    KINETIC_MONITOR_UI,
    MODEL_PATH,
    MOOD_EVALUATION_INTERVAL,
    MOOD_SNAPSHOT_FOLDER,
    PAUSE_DURATION,
    REACTIVITY_PAUSE_COOLDOWN,
    REACTIVITY_PAUSE_DURATION,
    REACTIVITY_PAUSE_THRESHOLD,
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
from grbl.idle_movement_manager import stop_idle_movements  # cleanup-only: kills stray wanderer processes
from image_monitor import ImageMonitor
from mood.mood import MoodEngine
from perception.detection_memory import DetectionMemory
from perception.object_detection import ObjectDetectionThread
from perception.person_detection_state import get_person_detection_state
from safety.aruco_detector import get_aruco_detector
from reactivity.camera_reactive import CameraReactivityEngine
from captioner.frame_buffer import frame_buffer
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

from utils.single_instance import claim_machine_or_exit

claim_machine_or_exit()

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


# Legacy hand-controller import block REMOVED July 28 — the kinetic bus
# (motor_panel/kinetic_bus.py) is the left hand's only driver now.

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
# ComfyUI rides along (Aug 19): spawned detached if its port is silent, so a
# bare boot outside tmux needs no separate launch. It outlives machine.py
# restarts; the drawing path's reachability probe stays the draw-time authority.
try:
    from utils.comfy_launcher import ensure_comfyui_up

    debug_print(f"ComfyUI: {ensure_comfyui_up()}", "INIT")
except Exception as _e:
    debug_print(f"ComfyUI auto-start failed: {_e}", "WARN")

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

            # Send immediate safe initial position to prevent servo jumpiness
            try:
                time.sleep(0.5)  # Allow Arduino to fully initialize
                # Set to neutral position: 90 degrees pan, mid-range tilt
                initial_pan = 90
                initial_tilt = (TILT_MIN + TILT_MAX) // 2
                servos.send(f"P{initial_pan}", key="pan")
                time.sleep(0.1)
                servos.send(f"T{initial_tilt}", key="tilt")
                debug_print(f"Servos initialized to safe position: pan={initial_pan}, tilt={initial_tilt}", "INIT")
            except Exception as e:
                debug_print(f"Initial servo positioning failed: {e}", "INIT")

            # Startup movement sequence disabled to prevent timing conflicts
            debug_print("Servo controller ready with initial positioning", "INIT")

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

                            # DEFERRED to the awakening (July 28): this play used
                            # to fire here at connect, minutes before anything
                            # else moved — now it joins the opening moment
                            _uarm_awakening_play = _uarm_start_play
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

_uarm_awakening_play = _uarm_awakening_play if "_uarm_awakening_play" in dir() else None

# Initialize breathing variables regardless of servo setting
lung_angle = 0.0
breath_speed = 5.0  # Slowed 25% for motor longevity
breath_paused = False
pause_start_time = 0
last_breath_direction = None

last_mood_time = 0
last_seen_time = time.time()
last_time = time.time()

# Person detection now uses unified PersonDetectionState with spatial memory
# (see perception/person_detection_state.py)

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
_caption_monitor_proc = None


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

        # Emergency kinetic bus stop
        try:
            if "kinetic_bus" in globals() and kinetic_bus:
                kinetic_bus.shutdown()
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

        # Terminate caption monitor if running
        try:
            if _caption_monitor_proc and _caption_monitor_proc.poll() is None:
                _caption_monitor_proc.terminate()
        except Exception:
            pass

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

    # Stop kinetic bus
    try:
        if "kinetic_bus" in globals() and kinetic_bus:
            kinetic_bus.shutdown()
            print("[🦾] Kinetic bus shutdown")
    except Exception as e:
        print(f"[ERROR] Error stopping kinetic bus: {e}")

    # Stop uArm controller
    if uarm_controller:
        try:
            debug_print("Disconnecting uArm controller", "CLEANUP")
            uarm_controller.disconnect()
        except Exception as e:
            print(f"[ERROR] Error stopping uArm controller: {e}")
    elif "uarm_teach_app" in globals() and uarm_teach_app:
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

from config.config import OPEN_VOCAB_ENABLED

open_vocab_detector = None
if OPEN_VOCAB_ENABLED:
    from perception.open_vocab_detector import OpenVocabDetectorThread

    open_vocab_detector = OpenVocabDetectorThread()
    open_vocab_detector.start()
    debug_print("Open-vocab detector started (CPU, settle-gated)", "INIT")
    from perception.vocab_promotion import vocab_promoter

    vocab_promoter.attach_detector(open_vocab_detector)
    debug_print("Vocabulary promotion attached (caption nouns -> detector terms)", "INIT")
    from config.config import LABEL_AUDIT_ENABLED

    if LABEL_AUDIT_ENABLED:
        from perception.label_audit import LabelAuditThread

        label_auditor = LabelAuditThread(open_vocab_detector)
        label_auditor.start()
        debug_print("Label audit started (VLM audits detector labels)", "INIT")
else:
    debug_print("Open-vocab detector disabled (OPEN_VOCAB_ENABLED=False)", "INIT")

from config.config import PRESENCE_ADJUDICATION_ENABLED

if PRESENCE_ADJUDICATION_ENABLED:
    from perception.presence_adjudicator import presence_adjudicator

    presence_adjudicator.start()
    debug_print("Presence adjudication started (faceless person-candidates judged by the machine's own eye)", "INIT")

# Start real-time ArUco marker detection (for paper presence)
aruco_detector = get_aruco_detector()
debug_print("ArUco marker detection started", "INIT")

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

# Launch caption monitor in a separate terminal window
_caption_monitor_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "debug", "caption_monitor.py")
_caption_monitor_proc = None
if os.path.exists(_caption_monitor_script):
    try:
        _caption_monitor_proc = subprocess.Popen(
            ["x-terminal-emulator", "-e", sys.executable, _caption_monitor_script],
            start_new_session=True,
        )
        print("[INIT] Caption monitor launched in separate terminal")
    except FileNotFoundError:
        try:
            _caption_monitor_proc = subprocess.Popen(
                ["gnome-terminal", "--", sys.executable, _caption_monitor_script],
                start_new_session=True,
            )
            print("[INIT] Caption monitor launched in separate terminal")
        except FileNotFoundError:
            print("[INIT] Could not find a terminal emulator to launch caption monitor")

debug_print("Initializing mood engine", "INIT")
mood_engine = MoodEngine()
_global_mood_engine = mood_engine
# Ensure llama-server is running (the only backend since July 9 2026 —
# the Ollama fallback was retired with mistral-nemo; the query paths
# auto-restart the server if it dies mid-run)
from utils.llama_server import is_server_running, start_server

if not is_server_running():
    print("[INIT] Starting llama-server...")
    if not start_server():
        print("[INIT] WARNING: llama-server failed to start — captions will retry/restart it per call")
else:
    print("[INIT] llama-server already running")

debug_print("Initializing captioner", "INIT")
captioner = Captioner()
captioner.load_prior_session_caption()  # Load last thought from prior session for awakening
# The drawing drive charges from the continuous mood vector (arousal), same
# injection pattern as the kinetic bus — never the 5-label ladder
captioner.drawing.drive.get_arousal = lambda: mood_engine.mood_vector[1]
_global_captioner = captioner
_global_state_manager = state_manager

# CRITICAL: Register captioner with state_manager so GRBL can access it
# (sys.modules doesn't work from GRBL thread context)
state_manager.captioner = captioner

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

# The lefthand actuators: the kinetic bus is the ONLY driver (recorded
# temperament datasets from the motor panel, markov generation behind the
# mood system). The legacy pair — hand controller autonomous mode +
# organic_left_arm blind wander — was REMOVED July 28.
kinetic_bus = None
if KINETIC_BUS_ENABLED:
    debug_print("Initializing kinetic bus (the left hand's driver)", "INIT")
    try:
        import utils.hooks as _kinetic_hooks
        from motor_panel.kinetic_bus import KineticBus

        _gantry_link = None
        if KINETIC_GANTRY:
            from motor_panel.gantry import GantryLink

            _gantry_link = GantryLink()  # the right arm joins the temperament (acquired at the awakening)
        kinetic_bus = KineticBus(
            # PULL the emotion straight from the mood engine every supervisor
            # tick — the old push plumbing stays as redundancy, but the bus
            # never depends on it (the push path is years of accretion)
            get_emotion=mood_engine.get_emotion_for_hand_controller,
            # arousal (mood vector's 2nd axis) drives the movement sampler —
            # a calm body replays what it knows, an agitated one wanders
            get_arousal=lambda: mood_engine.mood_vector[1],
            on_log=lambda m: debug_print(m, "KINETIC"),
            gantry=_gantry_link,
        )
        # await_homing: the body holds STILL through init; the startup homing
        # choreography is the machine's first gesture and the first
        # temperament blooms as homing completes — the awakening, together
        kinetic_bus.enable(await_homing=True)
        # homing safety: grbl_utils.ensure_homed tucks the left arm clear
        # (and waits out the ramp) before every $H, releases on completion
        _kinetic_hooks.on_grbl_homing_start = kinetic_bus.home_clear
        _kinetic_hooks.on_grbl_homing_done = kinetic_bus.home_release
        # drawing arbitration: pause/resume call sites hand the gantry back
        # and forth between the drawing pipeline and the temperament
        _kinetic_hooks.on_gantry_pause = kinetic_bus.gantry_release
        _kinetic_hooks.on_gantry_resume = kinetic_bus.gantry_acquire
        # paper check: both arms play their recorded get-clear move while
        # the camera inspects the paper, blend back when the check ends
        _kinetic_hooks.on_paper_check_start = kinetic_bus.paper_clear
        _kinetic_hooks.on_paper_check_done = kinetic_bus.paper_release
        if KINETIC_MONITOR_UI:
            try:
                from motor_panel.runtime_monitor import start_runtime_monitor

                start_runtime_monitor(kinetic_bus)
                debug_print("Kinetic runtime monitor window started", "INIT")
            except Exception as e:
                debug_print(f"Runtime monitor unavailable (headless?): {e}", "KINETIC")
        log_json_entry(
            LogType.INFO,
            {"message": "Kinetic bus initialized", "port": ARDUINO_DEVICES["LEFTHAND"]},
            print_message=f"🦾 Kinetic bus initialized on {ARDUINO_DEVICES['LEFTHAND']}",
        )
    except Exception as e:
        debug_print(f"Failed to initialize kinetic bus: {e}", "ERROR")
        kinetic_bus = None
else:
    debug_print("Kinetic bus disabled — left hand will not move", "WARN")

# Initialize camera reactivity engine
debug_print("Initializing camera reactivity engine", "INIT")
reactivity_engine = CameraReactivityEngine(sensitivity=0.2, smoothing_factor=0.95, pause_threshold=0.60, pause_duration=3.0)
debug_print("Camera reactivity enabled - hand will respond to environmental changes", "INIT")


# DRAWING CRITIQUE REMOVED (Aug 5, artist's call): "not useful and
# underutilised... I'd like to redesign that system anyway". It was also the
# most tangled pass in the pipeline — two conflicting critiques of the same
# drawing, one of them invisible to the inventory, and the one whose timeouts
# kept getting stored as the machine's reflection. The completion memory
# survives it (grbl_utils records having drawn, with or without a reflection).
# When it returns it should critique THE PAPER, not the ComfyUI image — judge
# what the pen actually made, not what was intended.
# Set dependencies for paper detection
image_monitor.set_dependencies(cap, servos, captioner)

# Expose shared hardware handles for other modules (e.g., GRBL paper check)
try:
    # Create camera wrapper that provides read_frame() method expected by paper detection
    class CameraWrapper:
        def __init__(self, cv2_camera):
            self.cv2_camera = cv2_camera

        def read_frame(self):
            ret, frame = self.cv2_camera.read()
            return frame if ret else None

        def read(self):
            return self.cv2_camera.read()

    camera_wrapper = CameraWrapper(cap)
    state_manager.set_hardware_refs(camera_wrapper, servos)
    debug_print(f"Camera wrapper shared via state_manager: {cap}", "DEBUG")
except Exception:
    pass
image_monitor.start()

# Register GRBL-complete hook to trigger uArm after actual G-code completion
try:
    import utils.hooks as _hooks
    from config.config import USE_UARM, UARM_PLAY_AFTER_DRAW, UARM_PLAY_FILE

    if USE_UARM and UARM_PLAY_AFTER_DRAW and UARM_PLAY_FILE:

        def _normalize_smooth(path: str) -> str:
            base, ext = os.path.splitext(path)
            while base.lower().endswith(".smooth"):
                base = base[:-7]
            return f"{base}{ext or '.txt'}"

        def _uarm_after_grbl():
            def _run():
                try:
                    from grbl.idle_movement_manager import pause_for_drawing, get_manager

                    # Force pause idle movements regardless of CNC state
                    # (since we're in transition period between GRBL completion and uArm execution)
                    manager = get_manager()
                    if manager.process and manager.process.poll() is None:
                        manager.pause_for_drawing()
                    else:
                        pass

                    target = _normalize_smooth(UARM_PLAY_FILE)
                    app = None
                    try:
                        if "uarm_teach_app" in globals() and uarm_teach_app:
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
                    if hasattr(app, "teach") and app.teach:
                        print("[uArm] Waiting for movement completion...")
                        timeout = 60.0  # Maximum wait time
                        start_time = time.time()
                        while app.teach.is_playing() and (time.time() - start_time) < timeout:
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
            _run()

        _hooks.on_grbl_drawing_complete = _uarm_after_grbl
    else:
        pass
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

    # Send immediate mood update to hand controller with restored state
    debug_print("Sending restored mood to hand controller", "INIT")

    # Direct integration - set emotion based on mood
    emotion = mood_engine.get_emotion_for_hand_controller()
    if kinetic_bus is not None:
        kinetic_bus.set_emotion(emotion)
    debug_print(f"Restored emotion for kinetic bus: {emotion}", "INIT")
    # NOTE: idle CNC / homing no longer starts here (mid-init) — the
    # awakening is staged at the main-loop threshold below, so the homing
    # choreography, the gantry sweep, and the waking senses converge

    # Reset last_caption so remnants from previous session are not printed
    captioner.last_caption = ""
    # Set memory loaded flag BEFORE generating awakening message
    captioner.memory_loaded_from_previous = True

    # Generate awakening message with continuity
    save_time = previous_state["metadata"]["save_time"]
    time_since_last = describe_duration(save_time)
    previous_beliefs = previous_state["captioner"].get("beliefs", {})

    # The awakening is ONE path now (Aug 2): generate_internal_awakening, run as
    # the first caption. This used to also call generate_awakening_message here
    # — a second, older ceremony whose result was only logged, never spoken.
    # Two awakenings coexisted since June; the log recorded the one the machine
    # never said.
    log_json_entry(
        LogType.INFO,
        {"message": "Session restored — awakening runs as the first caption", "continuity": True, "time_since_last": time_since_last},
    )
    # Mark awakening complete to avoid duplicate environmental description
    captioner.mark_awakening_complete()
else:
    # Fresh start
    captioner.memory_loaded_from_previous = False
    log_json_entry(
        LogType.INFO,
        {"message": "Fresh start — awakening runs as the first caption", "continuity": False},
    )
    # Mark awakening complete to avoid duplicate environmental description
    captioner.mark_awakening_complete()

    # (idle CNC wanderer retired July 28 — homing happens at the awakening)

debug_print("System initialization complete", "INIT")

best_box = None
last_face_box = None  # Persisted face box for flicker suppression
last_face_time = 0.0  # When we last had a valid face detection
FACE_PERSIST_DURATION = 0.3  # Hold face box for 300ms after losing detection

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
            # (The per-tick mood_{ts}.jpg snapshot write was removed Aug 12 —
            # it existed only so the retired mood log could record a path
            # nothing read. Caption images are written elsewhere.)
            try:
                # Process caption and update physical systems
                if captioner.last_caption:
                    clean_caption = captioner.last_caption
                    if clean_caption.lower().startswith("caption:"):
                        clean_caption = clean_caption[len("caption:") :].strip()

                    mood_engine.analyze_mood(clean_caption, saw_person=best_box is not None)
                    debug_print(f"Processed caption through mood analysis: {clean_caption[:100]}...", "MOOD")

                    # Sampled AFTER analyze_mood — the push used to run one
                    # 10s tick stale (harmless only because the bus pull wins)
                    thread_emotion = mood_engine.get_emotion_for_hand_controller()
                    thread_mood = mood_engine.get_current_mood()
                    debug_print(f"Current emotion: {thread_emotion}, mood: {thread_mood:.2f}", "EMOTION")

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

                    # Kinetic bus picks the temperament bundle for this emotion
                    if kinetic_bus is not None:
                        kinetic_bus.set_emotion(thread_emotion)
                        debug_print(f"Updated kinetic bus emotion: {thread_emotion}", "HAND")

                    # Update captioner's mood scalar for the next cycle.
                    # (The pattern-engine novelty write is gone — the activation
                    # network is now novelty_score's only writer.)
                    captioner.current_mood = thread_mood

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
        with open(CAMERA_SETTINGS_FILE, "r") as f:
            settings = json.load(f)
            return {
                "brightness": settings.get("brightness", default_brightness),
                "contrast": settings.get("contrast", default_contrast),
                "saturation": settings.get("saturation", default_saturation),
                "sharpness": settings.get("sharpness", default_sharpness),
            }
    except (FileNotFoundError, json.JSONDecodeError):
        return {"brightness": default_brightness, "contrast": default_contrast, "saturation": default_saturation, "sharpness": default_sharpness}


def save_camera_settings(brightness, contrast, saturation, sharpness):
    """Save current camera settings to file"""
    try:
        settings = {"brightness": brightness, "contrast": contrast, "saturation": saturation, "sharpness": sharpness}
        with open(CAMERA_SETTINGS_FILE, "w") as f:
            json.dump(settings, f, indent=2)
        if DEBUG_MODE:
            debug_print("Camera settings saved", "CAMERA")
    except Exception as e:
        if DEBUG_MODE:
            debug_print(f"Failed to save camera settings: {e}", "CAMERA")


# Load saved settings
saved_settings = load_camera_settings()
current_brightness = saved_settings["brightness"]
current_contrast = saved_settings["contrast"]
current_saturation = saved_settings["saturation"]
current_sharpness = saved_settings["sharpness"]


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
cv2.namedWindow("mslint camera", cv2.WINDOW_NORMAL)
cv2.resizeWindow("mslint camera", 1920, 1080)
cv2.createTrackbar("Brightness", "mslint camera", current_brightness, 100, on_brightness_change)
cv2.createTrackbar("Contrast", "mslint camera", current_contrast, 100, on_contrast_change)
cv2.createTrackbar("Saturation", "mslint camera", current_saturation, 100, on_saturation_change)
cv2.createTrackbar("Sharpness", "mslint camera", current_sharpness, 100, on_sharpness_change)

debug_print("Camera controls initialized - use trackbars in preview window to adjust in real-time", "INIT")
debug_print("Press 'r' in camera window to reset controls to defaults", "INIT")

# Freeze watchdog: gaze + lung both go through this loop, so if it stalls the
# whole body freezes. When the heartbeat goes stale, dump every thread's stack
# to the event log folder — the dump shows exactly which call is blocking.
_loop_heartbeat = time.time()
_last_freeze_dump = 0.0


def _freeze_watchdog():
    global _last_freeze_dump
    import faulthandler

    while not shutdown_in_progress:
        time.sleep(5)
        stall = time.time() - _loop_heartbeat
        if stall > 10 and _loop_heartbeat > _last_freeze_dump:
            _last_freeze_dump = time.time()
            dump_path = os.path.join(MOOD_SNAPSHOT_FOLDER, f"freeze_dump_{int(_last_freeze_dump)}.txt")
            try:
                with open(dump_path, "w") as fh:
                    fh.write(f"main loop stalled for {stall:.1f}s\n\n")
                    faulthandler.dump_traceback(file=fh, all_threads=True)
                print(f"[WATCHDOG] Main loop stalled {stall:.1f}s — thread dump: {dump_path}")
            except Exception as e:
                print(f"[WATCHDOG] Stall detected ({stall:.1f}s) but dump failed: {e}")


threading.Thread(target=_freeze_watchdog, daemon=True, name="freeze-watchdog").start()


# THE AWAKENING — one concurrent moment at the threshold, nothing serial:
# machine.py has been silent and still through the whole init (the kinetic
# bus holds the body via await_homing). The awakening thread starts the
# homing choreography + gantry homing + the uArm's opening play WHILE the
# main loop below brings up the camera window, gaze, lung and bulb — so
# everything wakes together and the first temperament blooms when homing
# completes. (Previously each piece fired wherever its init happened to
# sit, minutes apart — uArm first, arm solo, senses, homing last.)
def _awakening():
    try:
        if _uarm_awakening_play is not None:
            threading.Thread(target=_uarm_awakening_play, daemon=True, name="UArmAwakeningPlay").start()
        # Home the gantry and KEEP it — the bus's GantryLink acquires the
        # port (the open resets GRBL), homes it (the tuck choreography
        # fires, simultaneous with $H), and then plays the datasets'
        # recorded x/y through the temperament until a drawing needs the
        # port. No idle-wanderer subprocess anywhere in the path.
        try:
            if kinetic_bus is not None and getattr(kinetic_bus, "gantry", None) is not None:
                kinetic_bus.gantry_acquire()
            else:
                from grbl.grbl_utils import ensure_homed, find_grbl_port

                _ser = find_grbl_port(preferred_port=os.getenv("GRBL_PORT", "/dev/arduino_cnc"))
                if _ser:
                    ensure_homed(_ser, max_retries=3)
                    try:
                        _ser.close()
                    except Exception:
                        pass
                    debug_print("Awakening: gantry homed (direct, in-process)", "INIT")
                else:
                    debug_print("Awakening: no GRBL found — homing skipped", "WARN")
        except Exception as e:
            debug_print(f"Awakening homing failed: {e}", "ERROR")
    except Exception as e:
        debug_print(f"Awakening failed: {e}", "ERROR")


threading.Thread(target=_awakening, daemon=True, name="awakening").start()

try:
    prev_gray = None
    smoothed_pwm = 0
    debug_print("Entering main camera processing loop", "MAIN")
    while True:
        _loop_heartbeat = time.time()
        # Check for shutdown signal
        if shutdown_in_progress:
            print("[SHUTDOWN] Shutdown signal received - breaking main loop")
            break

        debug_print("Reading camera frame", "MAIN")
        ret, frame = cap.read()
        if not ret:
            continue

        # Share full-resolution frame for paper detection (before resize)
        state_manager.update_shared_frame(frame)

        object_detector.set_frame(frame)  # YOLO person detection enabled
        aruco_detector.set_frame(frame)  # Real-time ArUco marker detection
        if open_vocab_detector:
            open_vocab_detector.set_frame(frame)  # zero-shot object naming (settle-gated, CPU)

        # Store full-resolution frame for LLM captioning
        full_res_frame = frame.copy()

        # Frame buffer push moved after detection processing (see below)

        # Keep full resolution for preview (was: frame = cv2.resize(frame, (320, 240)))

        # Force garbage collection periodically to prevent memory accumulation
        frame_count += 1
        if frame_count % 100 == 0:  # Every 100 frames
            import gc

            gc.collect()

        # === CAMERA REACTIVITY PROCESSING ===
        # Downsample frame for reactivity (reduces sensitivity to small changes at 2K)
        reactivity_frame = cv2.resize(frame, (320, 240), interpolation=cv2.INTER_AREA)
        reactivity_metrics = reactivity_engine.process_frame(reactivity_frame)
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

            # (legacy reactivity pause removed July 28 — it drove the old hand
            # controller's Markov loop; the kinetic bus has its own
            # person-reactivity via startle/reach)
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

        # Face detection persistence — suppress single-frame dropouts
        if best_box is not None:
            last_face_box = tuple(best_box)
            last_face_time = now
        elif last_face_box is not None and (now - last_face_time) < FACE_PERSIST_DURATION:
            best_box = last_face_box  # Hold previous face box through brief dropouts
        else:
            last_face_box = None

        # Update person detection state for consciousness context
        if best_box is not None:
            x1, y1, x2, y2 = best_box if isinstance(best_box, tuple) else tuple(best_box)
            person_detection.update_face_detection(best_conf if best_conf > 0 else 0.5, (x1, y1, x2, y2))
        else:
            person_detection.update_face_detection(0.0)

        # YOLO person awareness - broader detection for consciousness context
        # Face detection = gaze tracking (following specific faces)
        # YOLO = general person presence awareness (triggers "aware" gaze state)
        labels = DetectionMemory.get_labels()
        # Body-schema verdict (computed off-thread, cached — an embed here would
        # hitch the servo physics): a faceless "person" matching the reach
        # envelope + gallery is the machine's own arm, and must not enter the
        # person state (gaze would track its own hand, smoothing would draw it).
        person_is_self = False
        if "person" in labels:
            try:
                from perception.body_schema import body_schema

                _verdict, _self_sim = body_schema.cached_person_verdict(max_age=30.0)
                person_is_self = _verdict is True
            except Exception:
                person_is_self = False
        # Effigy veto: a faceless person-shape that has held perfectly still
        # for minutes (the legless floor robot, the sweater doll) is not a
        # person — real people can't do that. A face at its place evicts it.
        person_is_effigy = False
        if "person" in labels and not person_is_self:
            try:
                from perception.effigy_memory import effigy_memory

                _eb = DetectionMemory.get_person_bbox()
                if _eb:
                    _nb = (_eb[0] / w, _eb[1] / h, _eb[2] / w, _eb[3] / h)
                    person_is_effigy = effigy_memory.observe(_nb, face_present=best_box is not None)
            except Exception:
                person_is_effigy = False
        if "person" in labels and not person_is_self and not person_is_effigy:
            raw_bbox = DetectionMemory.get_person_bbox()
            person_detection.update_yolo_detection(True, DetectionMemory.get_person_confidence() or 0.8, bbox=raw_bbox)
        else:
            person_detection.update_yolo_detection(False)

        # Note: best_box is only for FACE tracking, YOLO detection is passed separately

        # Feed frame buffer with detection snapshot for temporal awareness.
        # Servo position lets the buffer flag ego-motion (camera moved vs scene moved).
        try:
            from vision.gaze import physics_state as _gaze_phys

            _cam_pan, _cam_tilt = _gaze_phys.pan, _gaze_phys.tilt
        except Exception:
            _cam_pan, _cam_tilt = None, None

        # Own-body guard: looking well down (>15° below center) puts the
        # machine's own arms in view, and YOLO reads them as a person. A
        # faceless person-hit while tilted down is its own body, not a
        # visitor — without this, phantom arrivals enter the episodic log
        # and (June 12) "the person holding an unpressed pen" — its own
        # arm — was stored as an identity fact.
        own_body_likely = _cam_tilt is not None and _cam_tilt < 75 and best_box is None

        # Person world-angle: camera-compensated position, so person movement is
        # measurable regardless of where the camera points. Pixel diff can't do
        # this — even 1 degree of camera sway shifts every pixel.
        _person_angle = None
        if "person" in labels and _cam_pan is not None and not own_body_likely:
            _pb = DetectionMemory.get_person_bbox()
            if _pb:
                _cx = (_pb[0] + _pb[2]) / 2.0
                _person_angle = _cam_pan + ((_cx / frame.shape[1]) - 0.5) * 60.0  # ~60 deg horizontal FOV

        _face_frac = 0.0
        if best_box is not None:
            try:
                _fx1, _fy1, _fx2, _fy2 = [int(v) for v in best_box]
                _face_frac = max(0, (_fx2 - _fx1)) * max(0, (_fy2 - _fy1)) / float(max(1, w * h))
            except Exception:
                pass
        frame_buffer.push(
            frame,
            detection={
                "face": best_box is not None,
                "face_frac": _face_frac,
                "person": ("person" in labels) and not own_body_likely and not person_is_effigy,
                "person_count": 0 if (own_body_likely or person_is_effigy) else DetectionMemory.get_person_count(),
                "track_id": DetectionMemory.get_best_track_id(),
                "pan": _cam_pan,
                "tilt": _cam_tilt,
                "person_angle": _person_angle,
            },
        )

        if now - last_mood_time > MOOD_EVALUATION_INTERVAL:
            # Check if mood analysis is already running to prevent overlapping threads
            with mood_thread_lock:
                if not mood_thread_running:
                    debug_print(f"Starting mood update thread - interval: {MOOD_EVALUATION_INTERVAL}s", "MOOD")
                    # Use full_res_frame for LLM captioning (2560x1440 instead of 320x240)
                    threading.Thread(target=mood_update_thread, args=(full_res_frame, int(now)), daemon=True).start()
                    last_mood_time = now
                else:
                    debug_print("Mood analysis already running - skipping this interval", "MOOD")

        current_mood = mood_engine.get_current_mood()

        face_box = tuple(best_box) if best_box is not None else None

        # Get person state with SMOOTHED detection (persists through grace period)
        person_state = person_detection.get_person_state()
        person_direction = person_state.get("direction")
        # own_body_likely: arms in view while looking down are not a visitor
        smoothed_person_detected = person_state.get("is_present", False) and not own_body_likely
        # Use smoothed bbox that persists through grace period - prevents tracking dropout
        smoothed_bbox = person_detection.get_smoothed_bbox()

        # Debug: log bbox status when person detected but no bbox
        if smoothed_person_detected and smoothed_bbox is None and random.random() < 0.02:
            debug_print(f"WARN: Person detected but no bbox available", "GAZE")

        # Update gaze with face tracking AND YOLO awareness
        person_present, pan, tilt = update_gaze(
            frame,
            face_box,
            mood_engine.get_emotion_for_hand_controller(),
            yolo_person_detected=smoothed_person_detected,
            person_direction=person_direction,
            person_bbox=smoothed_bbox,
        )

        # Feed servo position to person detection for spatial memory
        person_detection.update_servo_position(pan, tilt)

        # Refresh person state after servo update
        person_state = person_detection.get_person_state()
        person_is_present = person_state["is_present"] and not own_body_likely

        # Switch YOLO to fast mode whenever a person is around, not only once
        # gaze is already tracking — gaze can't lock on quickly from stale
        # idle-cadence boxes. "remembered" keeps the fast cadence through
        # search sweeps so a re-appearing person is re-acquired immediately.
        from vision.gaze import get_gaze_state

        gaze_state_info = get_gaze_state()
        gaze_state_name = gaze_state_info.get("state", "idle") if isinstance(gaze_state_info, dict) else "idle"
        is_tracking_person = gaze_state_name in ("aware", "tracking", "grace") or person_state.get("person_state", "absent") != "absent"
        object_detector.set_tracking_mode(is_tracking_person)

        # Manage gaze search mode based on person detection state
        from vision.gaze import activate_search_mode, deactivate_search_mode, is_search_mode_active

        current_person_state = person_state.get("person_state", "absent")

        # Only activate search mode if person has been "remembered" (not visible) for 3+ seconds
        # Use _last_raw_detection_time (last time YOLO saw someone) not last_detection (arrival time)
        time_since_raw_detection = now - person_detection.get_last_raw_detection_time()
        should_search = current_person_state == "remembered" and time_since_raw_detection > 3.0

        if should_search and not is_search_mode_active():
            # Person lost for 3+ seconds but remembered - activate search to find them
            activate_search_mode(
                last_seen_pan=person_detection.last_seen_servo_pan,
                last_seen_tilt=person_detection.last_seen_servo_tilt,
                zones_visited=person_detection.scan_zones_visited,
            )
        elif current_person_state == "visible" and is_search_mode_active():
            # Person found - stop searching
            deactivate_search_mode()
        elif current_person_state == "absent" and is_search_mode_active():
            # Person confirmed absent - stop searching
            deactivate_search_mode()

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

        # Add person count from YOLO for multi-person awareness
        from perception.detection_memory import DetectionMemory

        reactivity_with_view["person_count"] = DetectionMemory.get_person_count()
        # Face bbox rides with the frame so the captioner can crop a
        # face-context image during eye contact (expressions are unreadable
        # in a wide shot where the face is a hundred pixels)
        reactivity_with_view["face_box"] = face_box

        # Update captioner with every frame (decoupled from mood system)
        # Uses unified PersonDetectionState with spatial memory (visible/remembered/absent)
        captioner.update(
            frame=frame,
            person_present=person_is_present,
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
            person_present=person_is_present,
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
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green = face

        # === SELF VISUALIZATION ===
        if person_is_self:
            _sb = DetectionMemory.get_person_bbox()
            if _sb:
                cv2.rectangle(frame, (_sb[0], _sb[1]), (_sb[2], _sb[3]), (160, 160, 160), 1)  # Gray = own body
                cv2.putText(frame, "SELF", (_sb[0], max(12, _sb[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1)

        # === YOLO PERSON BBOX VISUALIZATION ===
        if smoothed_bbox is not None:
            (px1, py1, px2, py2) = smoothed_bbox
            cv2.rectangle(frame, (px1, py1), (px2, py2), (255, 165, 0), 2)  # Orange = YOLO person
            # Show detection age
            yolo_age = now - person_detection.get_last_raw_detection_time()
            cv2.putText(frame, f"YOLO ({yolo_age:.1f}s)", (px1, py1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 165, 0), 1)

        # === ARUCO MARKER VISUALIZATION ===
        aruco_corners = aruco_detector.get_corners_for_drawing(include_rejected=True)
        for corners, marker_id, is_valid in aruco_corners:
            pts = corners.astype(int)
            color = (0, 255, 255) if is_valid else (0, 0, 255)  # Yellow=valid, Red=rejected
            cv2.polylines(frame, [pts], True, color, 2)
            cx, cy = int(pts[:, 0].mean()), int(pts[:, 1].mean())
            cv2.putText(frame, f"ID:{marker_id}", (cx - 20, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # === OPEN-VOCAB DETECTION VISUALIZATION ===
        if open_vocab_detector:
            for det in open_vocab_detector.get_detections_for_drawing():
                dx1, dy1, dx2, dy2 = det["box"]
                cv2.rectangle(frame, (dx1, dy1), (dx2, dy2), (255, 0, 255), 1)  # Magenta = open-vocab object
                cv2.putText(frame, f"{det['term']} {det['conf']:.2f}", (dx1, max(12, dy1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)

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
