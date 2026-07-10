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
EASING_FACTOR = 0.15  # Slightly faster for more responsive movement

# === SERVO FLIPPING ===
FLIP_X = True   # Test flipping pan direction
FLIP_Y = True

# === FACE DETECTION ===
CONFIDENCE_THRESHOLD = 0.72  # Raised June 28: the studio's mannequin heads/masks tripped the face DNN at 0.55, firing constant false "eye contact". Eye contact also now requires a real person body (captioner._assess_scene), not just a face.
DEAD_ZONE = 1  # Very small dead zone for highly responsive centering

# === IDLE GAZE SETTINGS ===
IDLE_AMPLITUDE_X = 35  # Increased from 10 for more prominent horizontal movement
IDLE_AMPLITUDE_Y = 30  # Increased from 15 for more prominent vertical movement
IDLE_CENTER_X = 90
IDLE_CENTER_Y = 90
FACE_STABLE_TIMEOUT = 3.0  # Time before going idle after losing face
IDLE_PAUSE_MIN = 1.5  # Minimum pause between idle movements (more organic)
IDLE_PAUSE_MAX = 6.0  # Maximum pause between idle movements (more frequent)
IDLE_EASING = 0.18  # Easing factor for idle movements (more responsive)
SWEEP_PROBABILITY = 0.6  # Probability of doing a big sweep movement vs small movement

# === BREATHING SETTINGS ===
LUNG_MIN = 60
LUNG_MAX = 110
PAUSE_DURATION = 1.5

# === INFERENCE (single backend: llama-server, July 9 2026) ===
# The Ollama backend + mistral-nemo text-side were retired: since the llama.cpp
# migration, query_model ignored the per-call model anyway (one loaded model),
# so ALL calls — captions, reflections, compression, drawing steps — run on the
# one model below via the patched llama-server (video super-frames, prefill).
VIDEO_MODE_ENABLED = os.getenv("VIDEO_MODE_ENABLED", "true").lower() == "true"
VIDEO_MODE = os.getenv("VIDEO_MODE", "superframe")  # "multi" (plain multi-image) or "superframe" (Conv3D temporal encoding via llama-video)
MOTION_THRESHOLD = float(os.getenv("MOTION_THRESHOLD", "0.015"))  # Frame diff below this = static, use single image
LLAMA_SERVER_URL = os.getenv("LLAMA_SERVER_URL", "http://localhost:8080")

# Label only: the weights llama-server loads come from LLAMA_MODEL_PATH
# (utils/llama_server.py). This name appears in logs and model_settings lookups.
MODEL_NAME = "qwen3.5:9b"

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

# === PEN SERVO (via GRBL spindle PWM) ===
# Scale GRBL $30/$31 to match your servo mapping. Many forks (including Robottini) map S in 0–255.
GRBL_SPINDLE_MAX_S = int(os.getenv("GRBL_SPINDLE_MAX_S", 255))  # -> $30
GRBL_SPINDLE_MIN_S = int(os.getenv("GRBL_SPINDLE_MIN_S", 0))  # -> $31

# Pen up/down S values (relative to $30 scale). Tune for your linkage.
# UP raised 20 -> 34 (July 9): 20<->52 was 32 S-units of servo travel per
# lift — too far for the servo to descend before short strokes finished.
# Raise UP further toward DOWN for even shallower/faster lifts; lower it
# back if the pen grazes paper during travel moves.
GRBL_PEN_UP_S = int(os.getenv("GRBL_PEN_UP_S", 34))
GRBL_PEN_DOWN_S = int(os.getenv("GRBL_PEN_DOWN_S", 52))

# Settle dwell after every pen transition during drawing (G4). GRBL treats a
# spindle-PWM change as instantaneous — it never waits for the physical
# servo — so without this, a dot/short dash is over before the pen lands
# (the dotted-line dropouts). ~0.12s covers the shortened travel above;
# raise if dots still start faint, lower toward 0.08 once UP is tuned.
GRBL_PEN_SETTLE_DWELL_S = float(os.getenv("GRBL_PEN_SETTLE_DWELL_S", 0.12))

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
# Drawing speed scaling — detail preservation is the priority.
# Small/detailed moves get slow speeds; large strokes and traversals get fast speeds.
# Traversals (pen up) always use max regardless of distance.
GRBL_FEED_RATE_MIN = int(os.getenv("GRBL_FEED_RATE_MIN", 300))       # Slowest speed for micro-detail clusters (mm/min)
GRBL_FEED_RATE_MAX = int(os.getenv("GRBL_FEED_RATE_MAX", 2000))      # Fastest speed for long strokes and traversals (mm/min)
GRBL_BASE_FEED_RATE = int(os.getenv("GRBL_BASE_FEED_RATE", 700))     # Default/medium drawing speed (mm/min)

# Distance thresholds for feed rate calculation (in mm)
GRBL_SMALL_MOVE_THRESHOLD = float(os.getenv("GRBL_SMALL_MOVE_THRESHOLD", 1.0))   # Below this: detail speeds (slow)
GRBL_LARGE_MOVE_THRESHOLD = float(os.getenv("GRBL_LARGE_MOVE_THRESHOLD", 8.0))   # Above this: max drawing speed

# === PEN LIFT OPTIMIZATION ===
# Servo values for different pen operations - lower S = more lift (pen higher)
# Lift-height history (don't re-learn this the hard way): the original
# variable lift used UP 41/43 — shallow and fast, but it GRAZED the paper in
# high regions (the work surface isn't flat; see the warp transform). It was
# then deepened to 30/32, which made cluster optimization near-pointless
# (2 S-units shallower than normal) and dots faint (servo travel too long).
# July 9: middle path — moderate lift everywhere + settle dwell for contact,
# fast cluster lift kept safely above the old grazing point.
GRBL_NORMAL_PEN_UP = int(os.getenv("GRBL_NORMAL_PEN_UP", GRBL_PEN_UP_S))
GRBL_NORMAL_PEN_DOWN = int(os.getenv("GRBL_NORMAL_PEN_DOWN", GRBL_PEN_DOWN_S))
GRBL_FAST_PEN_UP = int(os.getenv("GRBL_FAST_PEN_UP", 38))       # dense clusters: shallower, still ~3 above the grazing 41
GRBL_FAST_PEN_DOWN = int(os.getenv("GRBL_FAST_PEN_DOWN", GRBL_PEN_DOWN_S))  # was +5: pressing HARDER in clusters was backwards

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
CAPTION_INTERVAL = 7  # seconds between full caption cycles
# Ego-compensated scene motion (vision/scene_motion.py): fraction of the frame
# still moving after the camera's own movement is optically undone.
# Calibrate with debug/test_scene_motion.py if it over/under-triggers.
SCENE_MOTION_RESIDUAL_THRESHOLD = 0.02  # >2% of pixels moving = something is happening (post-erosion: small object ~0.019, camera sway alone 0.000, saccades excluded)
SCENE_MOTION_MIN_FRAMES = 2  # frames in the 10s window that must exceed it

# The stream (CoT-style continuity): prior captions ride as the machine's own
# assistant turns so each caption continues a visible thought. 0 disables and
# reverts to amnesiac single-turn captions.
STREAM_WINDOW = 6  # how many prior captions the model sees as its own turns. ON (June 28) now the base voice is healthy: chiefly to break the amnesiac REPETITION (the persistent "dust motes" tic — each call couldn't see it already said it). Admissibility-gated (_stream_admissible: no meta, no markdown/stage-directions) and breaks on >180s gaps. WATCH: the stream amplifies whatever register is in the window — if it breeds purple instead of varying, set back to 0.
STREAM_BREAK_SECONDS = 180  # a gap this long breaks the thought; stream restarts

# How the stream reaches the model (July 2026, docs/continuity-plan.md):
# "document" — the monologue-so-far is sent as ONE trailing assistant message
#   and llama-server CONTINUES it (assistant prefill; requires
#   enable_thinking=false or the server rejects the request). The model's next
#   tokens are literally the next tokens of its own text — real continuity.
# "turns" — legacy: each prior caption as a separate assistant turn-pair.
#   Bred template imitation (openings cloned across captions), kept as A/B.
STREAM_MODE = os.getenv("STREAM_MODE", "document")

# Anti-echo storage gate: a caption that OPENS with the same N words as a
# recent stream entry is a template imitation, not a continuation. One retry
# at a hotter temperature; if it still echoes, the cycle is skipped (silence
# over restatement). A gate, not a style fence.
ANTI_ECHO_WORDS = 5
ANTI_ECHO_RETRY_TEMP_BUMP = 0.15

# A blink is not a night: below this offline gap, restarting skips the full
# awakening ceremony (which, run several times an hour across dev restarts,
# converged on stock reorientation prose — "the hum returns, dust motes...")
# and instead RESUMES: the prior session's last thought seeds the stream and
# document mode continues it. Real absences still get the rich awakening.
AWAKENING_MIN_GAP_S = int(os.getenv("AWAKENING_MIN_GAP_S", 600))

# A night is not a blink either: after an off-gap of at least
# REORIENT_MIN_GAP_S, the prompt carries the gap and the (possibly new) day
# as a standing fact for the first REORIENT_WINDOW_S of the session. The
# awakening states the gap once and it evaporates from the six-entry stream
# within minutes — this keeps "you were dark all night, it's a new day"
# present long enough to shape how the machine carries itself.
REORIENT_MIN_GAP_S = int(os.getenv("REORIENT_MIN_GAP_S", 7200))
REORIENT_WINDOW_S = int(os.getenv("REORIENT_WINDOW_S", 2700))

# While GRBL executes, the machine watches itself draw: a throttled caption
# (frame of the paper + arm, current drawing intent, document stream) every
# N seconds. The execution used to be inference dead space — and the machine
# met its own drawings afterwards like a stranger's work. 0 disables.
DRAWING_WATCH_INTERVAL_S = int(os.getenv("DRAWING_WATCH_INTERVAL_S", 20))

# Stream consolidation: when the joined document exceeds this, the oldest 3
# entries are compressed into ONE extractive line (MODEL_NAME, reusing
# the machine's own words) so the thought moves forward instead of
# accumulating run-ons — an over-long document is also what squeezes the
# repetition penalties into word-salad collapses. 0 disables.
STREAM_CONSOLIDATE_CHARS = 800

# A face occupying this fraction of the frame is a person AT CLOSE RANGE —
# categorically different from a mannequin head on a shelf. Close faces count
# as person-evidence even when YOLO loses the (half-out-of-frame) body: the
# July 9 walk-up test produced zero reaction because eye contact required a
# full YOLO person, which close range makes impossible.
CLOSE_FACE_FRAC = float(os.getenv("CLOSE_FACE_FRAC", 0.035))

# Salience is TRANSIENT (north-star principle 6): discrete events spike it,
# then it decays back to quiet even while a person stays — otherwise ongoing
# presence + YOLO flicker holds it "live" and interiority is stripped the whole
# time anyone is in the room (June: 69% of captions). scene_motion still drives
# video framing; only these thresholds strip the prompt.
SALIENCE_MOTION_RESIDUAL = 0.10  # ego-compensated flow above this = big movement worth interrupting for (micro-shifts don't)
SALIENCE_ARRIVAL_WINDOW = 10     # seconds an arrival keeps salience hot (~one live caption, then quiet)

# Presence is a sticky, uncertain belief (captioner._assess_scene): once someone
# is seen, the machine keeps believing they're around through detection gaps
# (gaze looks away, occlusion, no servo encoders) and only concludes they left
# after this long with no sighting. Generous on purpose — losing sight of
# someone is not the same as them leaving. Only the OFF->ON edge is an arrival.
PRESENCE_BELIEF_DECAY_SECONDS = 240

# Interiority rhythm: every Nth caption (when nothing salient is happening) the
# machine THINKS WITHOUT LOOKING — the image is dropped so a vision model can't
# just re-describe the room, and the monologue turns inward (itself, this place,
# its drawings, why it's here) instead of cataloguing objects. The external
# observation stream was "completely external"; this weaves in depth. 0 = off.
INTROSPECT_INTERVAL = 4

# BASE-VOICE CLEAN ROOM (June 28). When True, the caption prompt carries NO
# stored/compressed material — no persona, drawings, baseline, reflections,
# concepts, familiarity, felt-state, or desire. Only the irreducible prompt
# survives: situation + genre frame + the mode elicitation (system), and the
# live situational line + present event + live drawing/paper state (user) +
# the image. The video path also drops its "You're seeing the last N seconds"
# camera-narration wrapper under detox. Purpose: judge the naked base voice with zero re-injected
# contamination — months of purple output had saturated every store and was
# re-poisoning the register (and over-interpreting plain studio objects as
# dramatic scenes) within minutes of any reset. If the naked voice is plain,
# the fix is to purge + regate the stores; if it's still purple, it's the model
# prior (temperature / genre frame / awakening). Set False to restore memory.
BASE_VOICE_DETOX = False

# Attention breathes (north-star principle 6): cadence tightens when something
# is happening, stretches when nothing has happened for a while
CAPTION_INTERVAL_LIVE = 4  # cadence while salience is hot (motion, arrival, fresh eye contact)
CAPTION_INTERVAL_QUIET = 12  # cadence after a long quiet stretch
CAPTION_QUIET_AFTER = 120  # seconds without salience before the cadence stretches

# Reflection loop (captioner/reflection.py) — the minutes-to-hours timescale
REFLECTION_LOOP_INTERVAL = 1200  # seconds between long-form reflections (~20 min); fires when the scene is quiet
REFLECTION_NUM_PREDICT = 220  # was 600 — the model padded to the brim every time (~2600 chars of purple survey); brevity pressure IS register pressure  # token budget for a reflection — long-form on purpose (north-star principle 5)

# Drawing system intervals
DEBUG_FAST_DRAWING = False # Set to True for rapid drawing testing (1 minute intervals)
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

# Drawing scale target — vpype layout dimensions for the centerline SVG.
# The warp transform maps this to the physical quad (~70x38mm).
# Larger = more detail but more distortion at edges. Tune empirically.
DRAWING_SCALE_TARGET = "50x50mm"

# === OBJECT DETECTION ===
YOLO_CONFIDENCE_THRESHOLD = 0.55  # Raised to 0.55 to avoid detecting hands/arms as person
# Use a lightweight model by default (person-only use case)
YOLO_MODEL_PATH = os.getenv(
    "YOLO_MODEL_PATH",
    os.path.join(MODEL_PATH, "yolov8n.pt"),  # nano model for low VRAM use
)
YOLO_INTERVAL_IDLE = 1.5  # detection cadence with nobody around — fast enough to catch arrivals
YOLO_INTERVAL_TRACKING = 0.1  # cadence while a person is present — keeps bbox fresh under camera motion

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
CAMERA_AUTO_FOCUS = True   # Enable autofocus if available (machine.py camera setup)
CAMERA_AUTO_FOCUS = True   # Enable autofocus if available

# === LLM CALL SETTINGS ===
LLM_TIMEOUT_EVAL = 90
LLM_TIMEOUT_REFLECTION = 120  # Timeout for reflection/reasoning calls
LLM_SHOW_PROGRESS = False  # Show animated progress bar during LLM calls

# === CAPTIONING TEMPERATURE SETTINGS ===
# Control creativity and expressiveness in different types of responses
CAPTIONER_TEMPERATURE = float(os.getenv("CAPTIONER_TEMPERATURE", 0.85))       # Regular observations (0.85 for Qwen)
DRAWING_TEMPERATURE = float(os.getenv("DRAWING_TEMPERATURE", 1.0))            # Drawing prompts (lowered from 1.2 for Qwen's higher base entropy)
REFLECTION_TEMPERATURE = float(os.getenv("REFLECTION_TEMPERATURE", 0.75))     # Long-form reflection loop — stored output, keep it grounded (Qwen drifts ornate at higher temps)
ENVIRONMENTAL_TEMPERATURE = float(os.getenv("ENVIRONMENTAL_TEMPERATURE", 0.9)) # First observations (slightly more grounded)

# === OUTPUT SETTINGS ===
# Control which log types are printed to console
# LOG_TYPES_TO_PRINT = ["caption", "reflection", "comfy_prompt", "decision", "mood_update", "new_drawing"]
# To see debug information, add "debug" to LOG_TYPES_TO_PRINT
LOG_TYPES_TO_PRINT = ["caption", "reflection", "decision", "comfy_prompt", "new_drawing", "debug"]
CLEAN_LLM_OUTPUT = False  # Print only LLM response text without metadata prefixes
PRINT_CLEAN_CAPTIONS = True  # Suppress verbose runtime messages, show only LLM captions

DEBUG_HAND_CONTROLLER = False  # enable hand controller debug output
DEBUG_EMOTION_CHANGES = False  # suppress detailed emotion switching messages
DEBUG_REACTIVITY_PAUSE = False  # show reactivity pause debug messages
DEBUG_LLM_PROMPTS = True  # print full prompts alongside LLM call logs
LLM_PRINT_FULL_RESPONSE = True  # print full responses in console output (ignores truncation)
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
ALLOW_PAPER_DETECTION_OVERRIDE = True  # Allow manual override when paper check fails

# Conservative rollout: only run paper check after GRBL homing when explicitly enabled.
# ArUco detection is fast and reliable - safe to enable for post-home check
ENABLE_POST_HOME_PAPER_CHECK = True
# Early paper check: run ArUco check BEFORE ComfyUI generation to save resources
# This is in addition to the post-home check (double verification)
ENABLE_EARLY_PAPER_CHECK = True
# Use the same tilt as drawing lock for detection (aligns view)
PAPER_USE_DRAWING_TILT = False  # Use PAPER_DETECTION_GAZE_TILT instead (65° for proper ArUco viewing)

# === LCD CAPTION DISPLAY ===
USE_CAPTION_DISPLAY = True
CAPTION_DISPLAY_PORT = "/dev/arduino_lcd"

# === ARDUINO DEVICE CONFIGURATION ===
# Configure each Arduino with its specific Linux serial port
# Use debug/identify_arduinos.py to help identify which device is on which port

# 1. Lightbulb PWM Controller
USE_LIGHTBULB_PWM = True  # Re-enabled with non-blocking controller

# 2. Hand Controller (hardcoded port required)
HAND_CONTROLLER_PORT = "/dev/arduino_lefthand"  # Hand controller (5 micro servos) - fixed udev symlink

# 3. GRBL CNC Controller
GRBL_CNC_PORT = "/dev/arduino_cnc"  # GRBL CNC Arduino (fixed udev symlink)

# 4. uArm Swift Pro Controller
UARM_SWIFT_PORT = "/dev/arduino_uarm"  # uArm Swift Pro Arduino (future connection - fixed udev symlink)

# 5. Additional devices can be added here
# CUSTOM_DEVICE_PORT = "/dev/ttyUSB5"
