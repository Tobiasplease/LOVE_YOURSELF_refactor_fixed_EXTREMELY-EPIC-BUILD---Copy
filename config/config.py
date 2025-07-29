import os
from .prompt_templates import *  # noqa: F401

# === SERIAL SETTINGS ===
SERIAL_PORT = "COM10"
BAUD_RATE = 9600

# === MODEL PATHS ===
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

# === SERVO SETTINGS ===
USE_SERVO = False
SERVO_MIN = 45
SERVO_MAX = 135
EASING_FACTOR = 0.09

# === SERVO FLIPPING ===
FLIP_X = False
FLIP_Y = True

# === FACE DETECTION ===
CONFIDENCE_THRESHOLD = 0.6
DEAD_ZONE = 15  # Reduced from 30 for more responsive tracking

# === IDLE GAZE SETTINGS ===
IDLE_AMPLITUDE_X = 25  # Reduced from 35 for less dramatic wandering
IDLE_AMPLITUDE_Y = 20  # Reduced from 25
IDLE_CENTER_X = 90
IDLE_CENTER_Y = 90
FACE_STABLE_TIMEOUT = 2.0
IDLE_SPEED_MIN = 0.15
IDLE_SPEED_MAX = 0.30

# === PHYSICS-BASED MOVEMENT ===
PHYSICS_FRICTION = 5.0  # How quickly movement dampens (increased from 3.0)
PHYSICS_SPRING_FORCE = 20.0  # How strongly it moves toward targets (reduced from 35.0)
FACE_LOCK_DURATION = 6.0  # How long to track a face
BLEND_SPEED = 1.5  # How quickly to transition between idle and face tracking (increased from 1.0)


# === BREATHING SETTINGS ===
LUNG_MIN = 60
LUNG_MAX = 110
PAUSE_DURATION = 3.0
LUNG_OFFSET_SCALE = -0.10

# === MOOD SYSTEM ===

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llava:7b-v1.6-mistral-q5_1")

MOOD_SNAPSHOT_FOLDER = os.getenv("MOOD_SNAPSHOT_FOLDER", os.path.join(os.path.dirname(os.path.dirname(__file__)), "event_log"))
COMFY_OUTPUT_FOLDER = os.getenv("COMFY_OUTPUT_FOLDER", os.path.join(os.path.dirname(os.path.dirname(__file__)), "/home/impostor/ComfyUI/output"))
COMFY_TEMPLATE_FILE = os.getenv("COMFY_TEMPLATE_FILE", "impostor-template-impostor-bot-svg.json")
COMFY_LORA_PATH = os.getenv("COMFY_LORA_PATH", "impostor-32-balanced-16k.safetensors")

# === TIMING INTERVALS ===
# Core consciousness timing - all intervals derive from this
CONSCIOUSNESS_CYCLE_INTERVAL = 30  # seconds - primary consciousness processing cycle (increased from 15)

# Derived intervals (automatically calculated)
MOOD_EVALUATION_INTERVAL = CONSCIOUSNESS_CYCLE_INTERVAL  # when mood thread runs
CAPTION_INTERVAL = CONSCIOUSNESS_CYCLE_INTERVAL  # when new captions are generated
MIN_SNAPSHOT_INTERVAL = max(5, CONSCIOUSNESS_CYCLE_INTERVAL // 3)  # minimum time between snapshots

REASON_INTERVAL = 360  # seconds between reflections
DRAWING_INTERVAL = 600  # seconds between drawing triggers
DRAWING_COOLDOWN = 180  # seconds between drawings

# === OBJECT DETECTION ===
YOLO_CONFIDENCE_THRESHOLD = 0.3  # Adjustable confidence for YOLOv8

# === CAPTIONER MEMORY CONTROL ===
MOOD_DECAY_RATE = 0.05  # how much mood fades when nothing new happens
NOVELTY_RANDOMNESS = 0.3  # random weight to boost novelty

CAMERA_INDEX = 0  # or whichever index your camera uses

# --- Mistral LLM settings ---
MISTRAL_COOLDOWN_SECS = 1000  # Min seconds between Mistral prompts
MISTRAL_TIMEOUT_SECS = 60  # Max time to wait for Ollama to respond

# === OLLAMA SETTINGS ===
OLLAMA_TIMEOUT_SUMMARY = 60
OLLAMA_TIMEOUT_EVAL = 90

# === AI MODEL PARAMETERS ===
CONSCIOUSNESS_TEMPERATURE = 0.9  # Higher for more creative, varied consciousness
AWAKENING_TEMPERATURE = 0.7     # Moderate for focused but natural awakening
REFLECTION_TEMPERATURE = 0.8    # High for varied internal reflections

# === OUTPUT SETTINGS ===
CLEAN_CAPTION_OUTPUT = True  # When True, shows only captions in quotes with clean spacing

# === PROMPT TEMPLATES ===
# Imported from config.prompts
