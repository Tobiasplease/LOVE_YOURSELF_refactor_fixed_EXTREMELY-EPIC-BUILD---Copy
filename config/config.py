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
PAN_MIN = 45   # Left limit (±45° from center) - expanded range
PAN_MAX = 135  # Right limit (±45° from center) - expanded range
TILT_MIN = 65  # Down limit - matches current working position with lowered mount
TILT_MAX = 150 # Up limit - expanded for upward viewing range
# Legacy values for backwards compatibility
SERVO_MIN = PAN_MIN  # Use PAN_MIN as default
SERVO_MAX = PAN_MAX  # Use PAN_MAX as default
EASING_FACTOR = 0.15  # Slightly faster for more responsive movement

# === SERVO FLIPPING ===
FLIP_X = True   # Test flipping pan direction
FLIP_Y = True

# === FACE DETECTION ===
CONFIDENCE_THRESHOLD = 0.8  # Higher threshold to distinguish real faces from pareidolia
DEAD_ZONE = 1  # Very small dead zone for highly responsive centering

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

# === LLM-DRIVEN GAZE INTENT ===
# Allow the model to influence camera direction via caption content
ENABLE_GAZE_INTENT = True  # Parse captions for directional cues (up/down/left/right)
GAZE_NUDGE_DURATION = 6.0  # How long a gaze nudge lasts before decaying (seconds)


# === BREATHING SETTINGS ===
LUNG_MIN = 60
LUNG_MAX = 110
PAUSE_DURATION = 1.5
LUNG_OFFSET_SCALE = -0.10

# === MOOD SYSTEM ===

OLLAMA_MODEL = "llava:7b-v1.6-mistral-q5_1"

# Active model:
# - "llava:7b-v1.6-mistral-q5_1" (Mistral base - works well with current prompts)
# - "llava-llama3" (Llama 3 base - verbose, prompt leakage issues)
#
# To switch models, change OLLAMA_MODEL above.

# === NARRATIVE/COMPRESSION MODEL ===
# Text-only model for compression, reflection, and narrative tasks
# Uses a storytelling-tuned model for better narrative continuity
COMPRESSION_MODEL = "Tohur/natsumura-storytelling-rp-llama-3.1:8b"
# Alternatives:
# - "mistral:7b-instruct" (4GB, general purpose)
# - "llama3.2:3b" (2GB, lighter)
# - "tinyllama:latest" (637MB, basic - not recommended for narrative)

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
GRBL_PEN_UP_S = int(os.getenv("GRBL_PEN_UP_S", 20))  # Lowered for faster operation
GRBL_PEN_DOWN_S = int(os.getenv("GRBL_PEN_DOWN_S", 52))

# Extra safety to ensure pen is fully UP before any homing ($H)
GRBL_PEN_UP_REPEATS = int(os.getenv("GRBL_PEN_UP_REPEATS", 5))   # How many times to assert M3 S{UP} before homing
GRBL_PEN_UP_DWELL_S = float(os.getenv("GRBL_PEN_UP_DWELL_S", 1.5))  # Dwell seconds after asserting UP before $H

# If True, the pen-up position corresponds to a HIGHER S value; if False, pen-up is a LOWER S value.
# Default assumes up=low (many servo forks use lower PWM as retracted).
GRBL_PEN_UP_IS_HIGH = os.getenv("GRBL_PEN_UP_IS_HIGH", "false").lower() in ("1", "true", "yes")

# Force using the absolute extreme S value for pen-up during homing (extra safety).
# Uses GRBL_SPINDLE_MAX_S when GRBL_PEN_UP_IS_HIGH is True, otherwise GRBL_SPINDLE_MIN_S.
GRBL_FORCE_ABSOLUTE_UP_FOR_HOMING = os.getenv("GRBL_FORCE_ABSOLUTE_UP_FOR_HOMING", "true").lower() in ("1", "true", "yes")

# Use centralized pen-up safety function (disabled by default for conservative rollout)
GRBL_USE_CENTRALIZED_PEN_UP = os.getenv("GRBL_USE_CENTRALIZED_PEN_UP", "false").lower() in ("1", "true", "yes")

# Safety pen up value for homing and critical operations (higher than drawing)
GRBL_SAFETY_PEN_UP = int(os.getenv("GRBL_SAFETY_PEN_UP", 30))  # Higher for safety during homing

# === GRBL IDLE MOVEMENT SETTINGS ===
# Idle movements happen in far corner away from home (0,0)
# Physical work area constrained to 40x40mm for safe operation
GRBL_IDLE_CENTER = (30, 30)  # Center point for idle movements (far corner of 40x40 area)
GRBL_IDLE_RADIUS_MIN = 5  # Minimum movement radius in mm
GRBL_IDLE_RADIUS_MAX = 8  # Maximum movement radius in mm (reduced for 40x40 area)
GRBL_IDLE_FEED_RATE = 500  # Feed rate for idle movements (mm/min) - very slow and organic
GRBL_IDLE_ZONE = (20, 40, 20, 40)  # Boundary box: (x_min, x_max, y_min, y_max) for 40x40 area
GRBL_IDLE_UPDATE_INTERVAL = 3.0  # Seconds between movement updates - longer pauses

# === GRBL G-CODE OPTIMIZATION SETTINGS ===
# Intelligent feed rate and pen lift optimization for better drawing performance

# Master optimization toggles
GRBL_ENABLE_FEED_OPTIMIZATION = os.getenv("GRBL_ENABLE_FEED_OPTIMIZATION", "true").lower() in ("1", "true", "yes")
GRBL_ENABLE_PEN_OPTIMIZATION = os.getenv("GRBL_ENABLE_PEN_OPTIMIZATION", "true").lower() in ("1", "true", "yes")
GRBL_ENABLE_STROKE_FILTERING = os.getenv("GRBL_ENABLE_STROKE_FILTERING", "false").lower() in ("1", "true", "yes")

# === GRBL SEGMENTED EXECUTION ===
# Splits large G-code files into segments to prevent buffer overload
GRBL_ENABLE_SEGMENTED_EXECUTION = os.getenv("GRBL_ENABLE_SEGMENTED_EXECUTION", "true").lower() in ("1", "true", "yes")
GRBL_MAX_SEGMENT_SIZE = int(os.getenv("GRBL_MAX_SEGMENT_SIZE", 150))  # Lines per segment
GRBL_ENABLE_PERSON_DETECTION_PAUSE = os.getenv("GRBL_ENABLE_PERSON_DETECTION_PAUSE", "false").lower() in ("1", "true", "yes")

# === FEED RATE OPTIMIZATION ===
# Speed scaling for different movement types - adjust these to set your preferred overall speed range
GRBL_FEED_RATE_MIN = int(os.getenv("GRBL_FEED_RATE_MIN", 2000))     # Slowest speed for tiny detailed movements (mm/min)
GRBL_FEED_RATE_MAX = int(os.getenv("GRBL_FEED_RATE_MAX", 6000))      # Fastest speed for large sweeping movements (mm/min)
GRBL_BASE_FEED_RATE = int(os.getenv("GRBL_BASE_FEED_RATE", 3500))    # Default/medium speed (mm/min)

# Distance thresholds for feed rate calculation (in mm)
GRBL_SMALL_MOVE_THRESHOLD = float(os.getenv("GRBL_SMALL_MOVE_THRESHOLD", 0.3))   # Below this: use slower speeds (reduced from 1.0mm)
GRBL_LARGE_MOVE_THRESHOLD = float(os.getenv("GRBL_LARGE_MOVE_THRESHOLD", 6.0))  # Above this: use max speed

# === PEN LIFT OPTIMIZATION ===
# Servo values for different pen operations - lower S = more lift (pen higher)
GRBL_NORMAL_PEN_UP = int(os.getenv("GRBL_NORMAL_PEN_UP", 30))         # Drawing pen up value (was 41, now more lift)
GRBL_NORMAL_PEN_DOWN = int(os.getenv("GRBL_NORMAL_PEN_DOWN", GRBL_PEN_DOWN_S))   # Normal pen down value
GRBL_FAST_PEN_UP = int(os.getenv("GRBL_FAST_PEN_UP", 32))       # Cluster pen up (was 43, now more lift)
GRBL_FAST_PEN_DOWN = int(os.getenv("GRBL_FAST_PEN_DOWN", min(60, GRBL_PEN_DOWN_S + 5))) # Fast pen down for clusters

# Cluster detection parameters
GRBL_CLUSTER_DISTANCE_THRESHOLD = float(os.getenv("GRBL_CLUSTER_DISTANCE_THRESHOLD", 5.0))  # Max distance between clustered pen lifts (mm)
GRBL_CLUSTER_SEQUENCE_MIN = int(os.getenv("GRBL_CLUSTER_SEQUENCE_MIN", 3))                  # Minimum pen lifts to consider a cluster

# === EXPERIMENTAL PATH SIMPLIFICATION ===
# WARNING: These are experimental features that may affect drawing quality
# Only enable for testing - disable for production artwork
GRBL_EXPERIMENTAL_SIMPLIFICATION = os.getenv("GRBL_EXPERIMENTAL_SIMPLIFICATION", "true").lower() in ("1", "true", "yes")
GRBL_SIMPLIFICATION_TOLERANCE = float(os.getenv("GRBL_SIMPLIFICATION_TOLERANCE", 0.02))  # Tolerance for path simplification (mm) - smaller = higher quality
GRBL_MERGE_TOLERANCE = float(os.getenv("GRBL_MERGE_TOLERANCE", 0.05))  # Tolerance for line merging (mm)

# === UARM SWIFT PRO SETTINGS ===
USE_UARM = True  # Enable uArm Swift Pro robotic arm integration
UARM_PORT = "/dev/arduino_uarm"  # Fixed udev symlink (matches ARDUINO_DEVICES)
UARM_MOVEMENT_NAMES = {
    1: "pickup",    # Primary pickup motion
    2: "place",     # Primary placement motion
    3: "gesture"    # Gestural expression motion
}
UARM_MOTION_STORAGE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "movement_recordings", "uarm")
UARM_CONNECT_ON_STARTUP = True  # Connect to uArm during system initialization
UARM_HOME_ON_CONNECT = False    # Avoid blocking/slow homing on connect; handled by Teach flow
UARM_DEFAULT_SPEED = 100        # Default movement speed (1-250)

# --- uArm post-drawing playback ---
# If True, after a drawing fully completes AND GRBL has homed, the uArm
# will play a specified Teach movement once, then the system will wait
# (up to 30s) for completion before resuming CNC idle movements.
UARM_PLAY_AFTER_DRAW = True
UARM_PLAY_FILE = os.path.join(
    UARM_MOTION_STORAGE,
    "papermove_20260306_214746.txt",  # Paper movement after GRBL completion
)

# --- uArm play-on-start (connectivity reassurance) ---
UARM_PLAY_ON_START = True
UARM_START_PLAY_FILE = os.path.join(
    UARM_MOTION_STORAGE,
    "startup_20260306_214250.txt",
)

# --- uArm play-on-start (connectivity reassurance) ---
# (reverted) No uArm play-on-start configuration

# difference between the below? hmm
MOOD_EVALUATION_INTERVAL = 10  # seconds between mood evaluations
CAPTION_INTERVAL = 10  # seconds between full caption cycles

# Drawing system intervals
DEBUG_FAST_DRAWING = False # Set to True for rapid drawing testing (1 minute intervals)
REASON_INTERVAL = 320  # seconds between reflections (7 minutes)
DRAWING_INTERVAL = 60 if DEBUG_FAST_DRAWING else 300  # 1 minute debug vs 5 minutes normal (check frequency)
DRAWING_COOLDOWN = 120 if DEBUG_FAST_DRAWING else 720  # 2 minutes debug vs 12 minutes normal
DRAWING_STARTUP_DELAY = 180  # Minimum seconds to wait after startup before first drawing (3 min for full init)

# State-motivated drawing system (when DEBUG_FAST_DRAWING is False)
DRAWING_USE_STATE_MOTIVATION = not DEBUG_FAST_DRAWING  # Enable sophisticated triggering
DRAWING_MIN_INTERVAL = 120 if DEBUG_FAST_DRAWING else 900   # 2 min debug vs 15 min production (max 4/hour)
DRAWING_MAX_INTERVAL = 180 if DEBUG_FAST_DRAWING else 1800  # 3 min debug vs 30 min production (min 2/hour)
DRAWING_BASE_THRESHOLD = 0.72 if DEBUG_FAST_DRAWING else 0.45  # Lowered to allow triggering with modest state values
DRAWING_NOVELTY_WEIGHT = 0.3  # How much novelty influences decision
DRAWING_BOREDOM_WEIGHT = 0.4   # How much boredom influences decision
DRAWING_MOOD_WEIGHT = 0.3      # How much mood influences decision
DRAWING_PERSON_WEIGHT = 0.4    # How much person presence influences decision
DRAWING_PERSON_BONUS = 0.2     # Additional motivation boost when person detected

# === OBJECT DETECTION ===
YOLO_CONFIDENCE_THRESHOLD = 0.55  # Raised to 0.55 to avoid detecting hands/arms as person
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
CAMERA_WIDTH = 1280   # 720p for smooth 30fps live feed
CAMERA_HEIGHT = 720   # LLM snapshots use this resolution

# === CAMERA IMAGE QUALITY ===
CAMERA_SHARPNESS = -1      # Sharpness (0-100, -1 for auto/default)
CAMERA_SATURATION = -1     # Color saturation (-1 for auto/default)
CAMERA_CONTRAST = -1       # Contrast (-1 for auto/default)
CAMERA_BRIGHTNESS = -1     # Brightness (-1 for auto/default)
CAMERA_EXPOSURE = -1       # Exposure (-1 for auto, or manual value)
CAMERA_AUTO_FOCUS = True   # Enable autofocus if available

# --- Mistral LLM settings ---
# MISTRAL_COOLDOWN_SECS = 1000  # Min seconds between Mistral prompts
MISTRAL_TIMEOUT_SECS = 60  # Max time to wait for Ollama to respond

# === OLLAMA SETTINGS ===
OLLAMA_TIMEOUT_SUMMARY = 60
OLLAMA_TIMEOUT_EVAL = 90
OLLAMA_TIMEOUT_REFLECTION = 120  # Timeout for reflection/reasoning calls
OLLAMA_SHOW_PROGRESS = False  # Show animated progress bar during Ollama API calls

# === CAPTIONING TEMPERATURE SETTINGS ===
# Control creativity and expressiveness in different types of responses
CAPTIONER_TEMPERATURE = float(os.getenv("CAPTIONER_TEMPERATURE", 1.0))        # Regular observations (balanced: creative but not flowery)
DRAWING_TEMPERATURE = float(os.getenv("DRAWING_TEMPERATURE", 1.2))            # Drawing prompts (creative but focused)
REFLECTION_TEMPERATURE = float(os.getenv("REFLECTION_TEMPERATURE", 1.1))      # Introspective moments (philosophical)
ENVIRONMENTAL_TEMPERATURE = float(os.getenv("ENVIRONMENTAL_TEMPERATURE", 0.9)) # First observations (slightly more grounded)

# === OUTPUT SETTINGS ===
# Control which log types are printed to console
# LOG_TYPES_TO_PRINT = ["caption", "reflection", "comfy_prompt", "decision", "mood_update", "new_drawing"]
# To see debug information, add "debug" to LOG_TYPES_TO_PRINT
LOG_TYPES_TO_PRINT = ["caption", "reflection", "decision", "comfy_prompt", "new_drawing", "debug"]
CLEAN_LLM_OUTPUT = True  # Print only LLM response text without metadata prefixes
PRINT_CLEAN_CAPTIONS = True  # Suppress verbose runtime messages, show only LLM captions
USE_FOCUSED_PROMPTS = True  # Use streamlined caption prompts (vs verbose structured prompts)
USE_NARRATIVE_PROMPTS = False  # EXPERIMENTAL: Disabled - reverted to original system

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

# === DRAWING ANALYSIS MODE ===
# "natsumura" - Natsumura-driven, identity-aware drawing decisions (experimental, needs debugging)
# "multi_step" - 5-step context-rich analysis with LLaVA (stable)
# "single" - Original single-prompt approach
DRAWING_ANALYSIS_MODE = "multi_step"

# Legacy toggle (for backwards compatibility)
USE_MULTI_STEP_DRAWING_ANALYSIS = DRAWING_ANALYSIS_MODE in ("multi_step",)

# === PAPER DETECTION SAFETY SYSTEM ===
# Prevent drawing on bare surfaces by checking for paper before execution
ENABLE_PAPER_DETECTION = True  # Master toggle for paper detection safety
PAPER_DETECTION_GAZE_PAN = 80  # Pan angle for looking down at drawing area (adjusted further left for better centering)
PAPER_DETECTION_GAZE_TILT = 65  # Tilt angle for looking down at drawing area (low enough to see ArUco marker)
# Reference images used by grbl_utils local heuristic detection
PAPER_PRESENT_REFERENCE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "calibration", "paper_present.jpg")
PAPER_ABSENT_REFERENCE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "calibration", "paper_absent.jpg")
ALLOW_PAPER_DETECTION_OVERRIDE = True  # Allow manual override when paper check fails

# Conservative rollout: only run paper check after GRBL homing when explicitly enabled.
# ArUco detection is fast and reliable - safe to enable for post-home check
ENABLE_POST_HOME_PAPER_CHECK = True
# Early paper check: run ArUco check BEFORE ComfyUI generation to save resources
# This is in addition to the post-home check (double verification)
ENABLE_EARLY_PAPER_CHECK = True
# Soft vs. strict behavior: when False, paper check never blocks drawing
# (it only logs and proceeds). Set to True to enforce blocking on "no paper".
PAPER_CHECK_STRICT_MODE = True
# Max time budget for the post-home paper check (seconds)
PAPER_CHECK_MAX_WAIT_S = 1.0
# Use the same tilt as drawing lock for detection (aligns view)
PAPER_USE_DRAWING_TILT = False  # Use PAPER_DETECTION_GAZE_TILT instead (65° for proper ArUco viewing)
# Use full frame for paper check (not cropped ROI)
PAPER_USE_FULL_FRAME = True

# --- Paper detection debug + tuning ---
# Dump diagnostic images/metrics during post-home paper check
PAPER_DEBUG_DUMP = True
# Disable all local image heuristics during the post-home paper check
# (use LLM-only judgment). When True, skips X/whiteness/reference comparisons
# and relies solely on the LLM single-token decision.
PAPER_DISABLE_LOCAL_HEURISTICS = True
# Whiteness thresholds (0..1)
PAPER_WHITENESS_MIN = 0.25           # Global ROI whiteness guard
PAPER_WHITENESS_WINDOW_MIN = 0.60    # Any 1/9 window whiteness guard (partial paper)
# Correlation decision thresholds
PAPER_CORR_MARGIN = 0.12
PAPER_PRESENT_CORR_MIN = 0.35
PAPER_ABSENT_CORR_MIN = 0.35
# X-detection thresholds
PAPER_X_SCORE_MIN = 0.35              # Min weighted diagonal length score to accept X
PAPER_X_CENTER_TOL_FRAC = 0.20        # Fraction of ROI size for center tolerance
PAPER_X_WHITENESS_OVERRIDE = 0.85     # Only override a strong X if a window whiteness >= this

# --- Optional LLM tie-breaker ---
PAPER_LLM_ENABLED = True             # Use LLM only when local check is inconclusive/NO
PAPER_LLM_TIMEOUT_S = 2.0            # Soft time budget for LLM check (best-effort)
PAPER_LLM_CONFIDENCE_MIN = 0.85      # Require at least this confidence to override
PAPER_LLM_MODE = "always"            # 'tie_break' or 'always'
# === LCD CAPTION DISPLAY ===
USE_CAPTION_DISPLAY = True
CAPTION_DISPLAY_PORT = "/dev/arduino_lcd"

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
