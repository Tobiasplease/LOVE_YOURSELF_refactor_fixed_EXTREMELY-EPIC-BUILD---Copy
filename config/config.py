LIGHTBULB_SENSITIVITY = 2.0  # Default sensitivity for frame diff to PWM mapping
import os

from config.prompt_templates import *  # noqa: F401

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
DEAD_ZONE = 30

# === IDLE GAZE SETTINGS ===
IDLE_AMPLITUDE_X = 20
IDLE_AMPLITUDE_Y = 30
IDLE_CENTER_X = 90
IDLE_CENTER_Y = 90
FACE_STABLE_TIMEOUT = 2.0
IDLE_SPEED_MIN = 0.15
IDLE_SPEED_MAX = 0.30


# === BREATHING SETTINGS ===
LUNG_MIN = 60
LUNG_MAX = 110
PAUSE_DURATION = 3.0
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
COMFY_LORA_STRENGTH = float(os.getenv("COMFY_LORA_STRENGTH", 1.0))
TRIGGER_PROMPT = os.getenv("TRIGGER_PROMPT", "impostor black and white sketch line art ")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 1))

# === SVG TO G-CODE SETTINGS ===
# If True, run svg_centerliner on PNGs to create centerline SVGs, then convert to G-code
# If False, convert the latest SVG in output folder to G-code
CENTER_LINE_SVG = True

# === GRBL EXECUTION SETTINGS ===
# If True, actually execute the generated G-code on GRBL hardware
# If False, only generate G-code files without executing them
EXECUTE_GRBL_GCODE = False

# difference between the below? hmm
MOOD_EVALUATION_INTERVAL = 10  # seconds between mood evaluations
CAPTION_INTERVAL = 10  # seconds between full caption cycles

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

# === OUTPUT SETTINGS ===
# TO SEE ONLY CLEAN CAPTIONS:
# LOG_TYPES_TO_PRINT = ["caption", "reflection", "comfy_prompt", "decision", "mood"]
LOG_TYPES_TO_PRINT = []
PRINT_CLEAN_CAPTIONS = True
DEBUG_HAND_CONTROLLER = False  # enable hand controller debug output
DEBUG_EMOTION_CHANGES = False  # suppress detailed emotion switching messages
DEBUG_REACTIVITY_PAUSE = False  # show reactivity pause debug messages

# === REACTIVITY PAUSE SYSTEM ===
REACTIVITY_PAUSE_THRESHOLD = 0.30  # Activity level to trigger pause
REACTIVITY_PAUSE_DURATION = 4.0  # Seconds to pause Markov generation
REACTIVITY_PAUSE_COOLDOWN = 10.0  # Seconds between pause triggers
USE_LIGHTBULB_PWM = True
LIGHTBULB_SERIAL_PORT = "COM4"  # Update as needed
