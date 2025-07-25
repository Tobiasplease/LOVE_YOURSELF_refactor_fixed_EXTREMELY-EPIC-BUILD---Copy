import os

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

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llava:7b-v1.6-mistral-q5_1")

MOOD_SNAPSHOT_FOLDER = os.getenv("MOOD_SNAPSHOT_FOLDER", os.path.join(os.path.dirname(os.path.dirname(__file__)), "event_log"))
COMFY_OUTPUT_FOLDER = os.getenv("COMFY_OUTPUT_FOLDER", os.path.join(os.path.dirname(os.path.dirname(__file__)), "/home/impostor/ComfyUI/outputs"))
COMFY_TEMPLATE_FILE = os.getenv("COMFY_TEMPLATE_FILE", "impostor-template-impostor-bot.json")
COMFY_LORA_PATH = os.getenv("COMFY_LORA_PATH", "impostor-32-balanced-16k.safetensors")

# difference between the below? hmm
MOOD_EVALUATION_INTERVAL = 10  # seconds between mood evaluations
CAPTION_INTERVAL = 10  # seconds between full caption cycles

REASON_INTERVAL = 360  # seconds between reflections
DRAWING_INTERVAL = 600  # seconds between drawing triggers
DRAWING_COOLDOWN = 180  # seconds between drawings

# === OBJECT DETECTION ===
YOLO_CONFIDENCE_THRESHOLD = 0.3  # Adjustable confidence for YOLOv8

# === CAPTIONER MEMORY CONTROL ===
MOOD_DECAY_RATE = 0.02  # how much mood fades when nothing new happens
NOVELTY_RANDOMNESS = 0.3  # random weight to boost novelty

CAMERA_INDEX = 0  # or whichever index your camera uses

# --- Mistral LLM settings ---
MISTRAL_COOLDOWN_SECS = 1000  # Min seconds between Mistral prompts
MISTRAL_TIMEOUT_SECS = 60  # Max time to wait for Ollama to respond

# === OLLAMA SETTINGS ===
OLLAMA_TIMEOUT_SUMMARY = 60
OLLAMA_TIMEOUT_EVAL = 90

# === RESOURCE MANAGEMENT ===
RESOURCE_QUEUE_ENABLED = True  # Enable resource coordination between Ollama and ComfyUI
MAX_QUEUE_SIZE = 50  # Maximum number of queued resource requests
QUEUE_TIMEOUT = 60.0  # Maximum time to wait for resources (seconds)
