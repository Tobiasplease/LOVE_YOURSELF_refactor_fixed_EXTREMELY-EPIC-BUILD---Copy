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
MOOD_EVALUATION_INTERVAL = 10  # seconds between mood evaluations
MOOD_LOG = "internalvoice.txt"
MOOD_SNAPSHOT_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), "mood_snapshots")
MOOD_SNAPSHOT_DIR = MOOD_SNAPSHOT_FOLDER  # alias to resolve mood.py dependency

SUMMARY_INTERVAL = 60  # seconds between summary evaluations
EVALUATION_INTERVAL = 30  # seconds between detailed evaluations

# === OBJECT DETECTION ===
YOLO_CONFIDENCE_THRESHOLD = 0.3  # Adjustable confidence for YOLOv8

# === CAPTIONING LOGS ===
INTERNAL_VOICE_LOG = MOOD_LOG

# === CAPTIONER MEMORY CONTROL ===
MEMORY_QUEUE_LIMIT = 100  # max short-term memory entries
MOOD_DECAY_RATE = 0.02  # how much mood fades when nothing new happens
NOVELTY_RANDOMNESS = 0.3  # random weight to boost novelty
SNAPSHOT_STORAGE_LIMIT = 100  # number of mood_snapshot images to keep

CAMERA_INDEX = 0  # or whichever index your camera uses

# === LLAVA SETTINGS ===
LLAVA_TIMEOUT_SUMMARY = 60
LLAVA_TIMEOUT_EVAL = 90
