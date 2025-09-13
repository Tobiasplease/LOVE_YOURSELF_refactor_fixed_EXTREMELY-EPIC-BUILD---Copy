LIGHTBULB_SENSITIVITY = 1.5  # Default sensitivity for frame diff to PWM mapping
import os

# All prompts now imported from captioner.prompts

# === SERIAL SETTINGS ===
# === ARDUINO SERIAL PORT CONFIGURATION (Linux) ===
# Each Arduino needs a unique port assignment
SERIAL_PORT = "/dev/arduino_lunggaze"  # Servo controller (PAN/TILT/LUNG) - fixed udev symlink
BAUD_RATE = 9600

# === MODEL PATHS ===
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

# === SERVO SETTINGS ===
USE_SERVO = True
USE_HAND_CONTROLLER = True  # Enable hand controller system
# Natural head movement limits for realistic gaze
PAN_MIN = 65   # Left limit (±25° from center)
PAN_MAX = 115  # Right limit (±25° from center)  
TILT_MIN = 70  # Down limit (±20° from center)
TILT_MAX = 110 # Up limit (±20° from center)
# Legacy values for backwards compatibility
SERVO_MIN = PAN_MIN  # Use PAN_MIN as default
SERVO_MAX = PAN_MAX  # Use PAN_MAX as default
EASING_FACTOR = 0.15  # Slightly faster for more responsive movement

# === SERVO FLIPPING ===
FLIP_X = False  # Pan servo direction is correct
FLIP_Y = True

# === FACE DETECTION ===
CONFIDENCE_THRESHOLD = 0.8  # Higher threshold to distinguish real faces from pareidolia
DEAD_ZONE = 3  # Smaller dead zone for more precise centering

# === IDLE GAZE SETTINGS ===
IDLE_AMPLITUDE_X = 35  # Increased from 10 for more prominent horizontal movement
IDLE_AMPLITUDE_Y = 30  # Increased from 15 for more prominent vertical movement
IDLE_CENTER_X = 90
IDLE_CENTER_Y = 90
FACE_STABLE_TIMEOUT = 3.0  # Time before going idle after losing face
IDLE_SPEED_MIN = 0.15
IDLE_SPEED_MAX = 0.30
IDLE_PAUSE_MIN = 1.5  # Minimum pause between idle movements (more organic)
IDLE_PAUSE_MAX = 6.0  # Maximum pause between idle movements (more frequent)
IDLE_EASING = 0.18  # Easing factor for idle movements (more responsive)
SWEEP_PROBABILITY = 0.6  # Probability of doing a big sweep movement vs small movement


# === BREATHING SETTINGS ===
LUNG_MIN = 60
LUNG_MAX = 110
PAUSE_DURATION = 1.5
LUNG_OFFSET_SCALE = -0.10

# === MOOD SYSTEM ===

OLLAMA_MODEL = "llava:7b-v1.6-mistral-q5_1"

# Tested and Available Models:
# - "llava:7b-v1.6-mistral-q5_1" (good - rich detailed responses, most engaging, needs work)
# - "qwen2.5vl:3b" (experimental - scene-beat structure, more concise but can be repetitive)
# - "qwen2.5vl:7b" (testing - better roleplay than 3b, balanced approach)
#
# To switch models, change OLLAMA_MODEL above or set environment variable:
# set OLLAMA_MODEL=qwen2.5vl:7b

MOOD_SNAPSHOT_FOLDER = os.getenv("MOOD_SNAPSHOT_FOLDER", os.path.join(os.path.dirname(os.path.dirname(__file__)), "event_log"))

# === COMFY STUFF ===

COMFY_OUTPUT_FOLDER = os.getenv("COMFY_OUTPUT_FOLDER", os.path.join(os.path.dirname(os.path.dirname(__file__)), "/home/impostor/ComfyUI/output"))

FLUX_DEV_PATH = os.getenv("FLUX_DEV_PATH", "flux1-dev.sft")
FLUX_GGUF_PATH = os.getenv("FLUX_GGUF_PATH", "flux1-dev-Q4_K_S.gguf")
CONTROLNET_NET_PATH = os.getenv("CONTROLNET_NET_PATH", "flux-dev-controlnet-union-pro-2.safetensors")
COMFY_TEMPLATE_FILE = os.getenv("COMFY_TEMPLATE_FILE", "impostor-template-impostor-bot-svg.json")
COMFY_LORA_PATH = os.getenv("COMFY_LORA_PATH", "impostor-32-balanced-16k.safetensors")
TRIGGER_PROMPT = os.getenv("TRIGGER_PROMPT", "impostor black and white sketch line art ")
# Force single-image generation for stability (do not override via env)
BATCH_SIZE = 1

# === COMFY CONTROLLER SETTINGS ===
COMFY_LORA_STRENGTH = float(os.getenv("COMFY_LORA_STRENGTH", 1.0))
COMFY_CNET_STRENGTH = float(os.getenv("COMFY_CNET_STRENGTH", 0.3))
COMFY_FLUX_GUIDANCE = float(os.getenv("COMFY_FLUX_GUIDANCE", 4.0))
COMFY_LATENT_WIDTH = int(os.getenv("COMFY_LATENT_WIDTH", 1024))
COMFY_LATENT_HEIGHT = int(os.getenv("COMFY_LATENT_HEIGHT", 1024))
COMFY_STEPS = int(os.getenv("COMFY_STEPS", 25))

DRAWING_TIMEOUT = float(
    os.getenv("DRAWING_TIMEOUT", 300.0)
)  # if drawing generation takes longer than this, it will be auto-finished, something is wrong...

# === SVG TO G-CODE SETTINGS ===
# If True, run svg_centerliner on PNGs to create centerline SVGs, then convert to G-code
# If False, convert the latest SVG in output folder to G-code
CENTER_LINE_SVG = True

# === GRBL EXECUTION SETTINGS ===
# If True, actually execute the generated G-code on GRBL hardware
# If False, only generate G-code files without executing them
EXECUTE_GRBL_GCODE = True

# === GRBL WARP TRANSFORM ===
# If True, apply JBE's warp transform to correct robot arm distortion
# If False, use raw coordinates without distortion correction
GRBL_WARP_TRANSFORM = True

# GRBL homing retry configuration
GRBL_HOMING_MAX_RETRIES = 3  # Number of homing attempts before giving up
GRBL_HOMING_TIMEOUT = 120  # Seconds to wait for each homing attempt

# === PEN SERVO (via GRBL spindle PWM) ===
# Scale GRBL $30/$31 to match your servo mapping. Many forks (including Robottini) map S in 0–255.
GRBL_SPINDLE_MAX_S = int(os.getenv("GRBL_SPINDLE_MAX_S", 255))  # -> $30
GRBL_SPINDLE_MIN_S = int(os.getenv("GRBL_SPINDLE_MIN_S", 0))  # -> $31

# Pen up/down S values (relative to $30 scale). Tune for your linkage.
GRBL_PEN_UP_S = int(os.getenv("GRBL_PEN_UP_S", 30))
GRBL_PEN_DOWN_S = int(os.getenv("GRBL_PEN_DOWN_S", 50))

# === GRBL IDLE MOVEMENT SETTINGS ===
# Idle movements happen in far corner away from home (0,0)
# Physical work area constrained to 40x40mm for safe operation
GRBL_IDLE_CENTER = (30, 30)  # Center point for idle movements (far corner of 40x40 area)
GRBL_IDLE_RADIUS_MIN = 5  # Minimum movement radius in mm
GRBL_IDLE_RADIUS_MAX = 8  # Maximum movement radius in mm (reduced for 40x40 area)
GRBL_IDLE_FEED_RATE = 500  # Feed rate for idle movements (mm/min) - very slow and organic
GRBL_IDLE_ZONE = (20, 40, 20, 40)  # Boundary box: (x_min, x_max, y_min, y_max) for 40x40 area
GRBL_IDLE_UPDATE_INTERVAL = 3.0  # Seconds between movement updates - longer pauses

# difference between the below? hmm
MOOD_EVALUATION_INTERVAL = 10  # seconds between mood evaluations
CAPTION_INTERVAL = 10  # seconds between full caption cycles

REASON_INTERVAL = 320  # seconds between reflections (7 minutes)
DRAWING_INTERVAL = 60  # seconds between drawing triggers (debug: ~1 minute)
DRAWING_COOLDOWN = 60  # minimum seconds between drawings (debug)

# === OBJECT DETECTION ===
YOLO_CONFIDENCE_THRESHOLD = 0.3  # Adjustable confidence for YOLOv8
# Use a lightweight model by default (person-only use case)
YOLO_MODEL_PATH = os.getenv(
    "YOLO_MODEL_PATH",
    os.path.join(MODEL_PATH, "yolov8n.pt"),  # nano model for low VRAM use
)

# === CAPTIONER MEMORY CONTROL ===
MOOD_DECAY_RATE = 0.03  # how much mood fades when nothing new happens
NOVELTY_RANDOMNESS = 0.5  # random weight to boost novelty

# === CHANGE DETECTION ===
VISUAL_CHANGE_THRESHOLD = 1.0  # novelty score threshold for triggering change-focused prompts (0.0-1.0) - disabled for now


# === TINYLLAMA SETTINGS ===
MOTIF_MODEL = "tinyllama:latest"
# MOTIF_MODEL = OLLAMA_MODEL

TINYLLAMA_TEMPERATURE = 0.1  # Low temperature for consistent numeric output
TINYLLAMA_TOP_P = 0.8  # Top-p sampling for TinyLlama
TINYLLAMA_NUM_PREDICT = 5  # Very short - just a number like "0.7"
TINYLLAMA_TIMEOUT = 20  # Timeout in seconds for TinyLlama queries

CAMERA_INDEX = 0  # or whichever index your camera uses

# === CAMERA RESOLUTION ===
CAMERA_WIDTH = 1920   # Full HD width for high quality image processing
CAMERA_HEIGHT = 1080  # Full HD height for high quality image processing

# --- Mistral LLM settings ---
# MISTRAL_COOLDOWN_SECS = 1000  # Min seconds between Mistral prompts
MISTRAL_TIMEOUT_SECS = 60  # Max time to wait for Ollama to respond

# === OLLAMA SETTINGS ===
OLLAMA_TIMEOUT_SUMMARY = 60
OLLAMA_TIMEOUT_EVAL = 90
OLLAMA_TIMEOUT_REFLECTION = 120  # Timeout for reflection/reasoning calls
OLLAMA_SHOW_PROGRESS = False  # Show animated progress bar during Ollama API calls

# === OUTPUT SETTINGS ===
# Control which log types are printed to console
# LOG_TYPES_TO_PRINT = ["caption", "reflection", "comfy_prompt", "decision", "mood_update", "new_drawing"]
# To see debug information, add "debug" to LOG_TYPES_TO_PRINT
LOG_TYPES_TO_PRINT = ["caption", "reflection", "decision", "comfy_prompt", "new_drawing"]
CLEAN_LLM_OUTPUT = True  # Print only LLM response text without metadata prefixes
PRINT_CLEAN_CAPTIONS = True  # Suppress verbose runtime messages, show only LLM captions

DEBUG_HAND_CONTROLLER = False  # enable hand controller debug output
DEBUG_EMOTION_CHANGES = False  # suppress detailed emotion switching messages
DEBUG_REACTIVITY_PAUSE = False  # show reactivity pause debug messages
DEBUG_OLLAMA_PROMPTS = False  # enable detailed Ollama debug output with prompt types and errors
OLLAMA_PRINT_FULL_RESPONSE = True  # print full responses in console output (ignores truncation)
NO_HANDS = False

# === REACTIVITY PAUSE SYSTEM ===
REACTIVITY_PAUSE_THRESHOLD = 0.30  # Activity level to trigger pause
REACTIVITY_PAUSE_DURATION = 4.0  # Seconds to pause Markov generation
REACTIVITY_PAUSE_COOLDOWN = 10.0  # Seconds between pause triggers

# === DRAWING MEMORY SETTINGS ===
# Store concise summaries of drawing intents and reflections for future prompts
INCLUDE_DRAWING_HISTORY = True
DRAWING_HISTORY_LIMIT = 3  # how many recent drawing entries to surface in prompts
# === ARDUINO DEVICE CONFIGURATION ===
# Configure each Arduino with its specific Linux serial port
# Use debug/identify_arduinos.py to help identify which device is on which port

# 1. Lightbulb PWM Controller
USE_LIGHTBULB_PWM = True  # Re-enabled with non-blocking controller
LIGHTBULB_SERIAL_PORT = "/dev/arduino_lightbulb"  # Lightbulb controller - fixed udev symlink

# 2. Hand Controller (hardcoded port required)
HAND_CONTROLLER_PORT = "/dev/arduino_lefthand"  # Hand controller (5 micro servos) - fixed udev symlink

# 3. GRBL CNC Controller
GRBL_CNC_PORT = "/dev/arduino_cnc"  # GRBL CNC Arduino (fixed udev symlink)

# 4. uArm Swift Pro Controller
UARM_SWIFT_PORT = "/dev/arduino_uarm"  # uArm Swift Pro Arduino (future connection - fixed udev symlink)

# 5. Additional devices can be added here
# CUSTOM_DEVICE_PORT = "/dev/ttyUSB5"
