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
# Natural head movement limits for realistic gaze
PAN_MIN = 45  # Left limit (±45° from center) - expanded range
PAN_MAX = 135  # Right limit (±45° from center) - expanded range
TILT_MIN = 65  # Down limit - matches current working position with lowered mount
TILT_MAX = 150  # Up limit - expanded for upward viewing range
EASING_FACTOR = 0.15  # Slightly faster for more responsive movement

# === SERVO FLIPPING ===
FLIP_X = True  # Test flipping pan direction
FLIP_Y = True

# === FACE DETECTION ===
CONFIDENCE_THRESHOLD = 0.72  # Raised June 28: the studio's mannequin heads/masks tripped the face DNN at 0.55, firing constant false "eye contact". Eye contact also now requires a real person body (captioner._assess_scene), not just a face.

# === FACE TRACKING CONTROL (July 10) ===
# Close-range stability: a close face maps each camera degree to many pixels,
# and the flimsy mount wobbles — with no dead zone the loop hunted ("bobbing"),
# blurring every capture. The dead zone grows with apparent face size so
# distant tracking stays responsive while a close face parks the camera.
FACE_TRACK_DEAD_ZONE = 0.05  # base half-width around frame center, fraction of frame
FACE_TRACK_DEAD_ZONE_FACE_SCALE = 0.4  # dead zone added per unit of face-width fraction
FACE_TRACK_MAX_STEP = 2.5  # max target movement per update, degrees (~75°/s at 30fps)

# === IDLE GAZE SETTINGS ===
FACE_STABLE_TIMEOUT = 3.0  # Time before going idle after losing face

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
# "multi" default since July 28 (artist's call): superframe's Conv3D temporal
# encoding needs steady frames, and servo sway smeared them — the model wrote
# about blur; logs showed frequent "only 1/6 steady frames" fallbacks anyway.
# Plain multi-image sends legible stills, runs on STANDARD llama.cpp mtmd (no
# patched fork — frees model choice for the upgrade A/B), and keeps the same
# upstream motion/steady-frame decision logic. "superframe" stays a toggle.
VIDEO_MODE = os.getenv("VIDEO_MODE", "multi")  # "multi" (plain multi-image) or "superframe" (Conv3D temporal encoding via llama-video)
MOTION_THRESHOLD = float(os.getenv("MOTION_THRESHOLD", "0.015"))  # frame diff below this = static, use a single image
# How many frames a video call actually sends (Aug 2). The buffer collects up to
# six; six were being sent, ~4k image tokens per call, and on the 27B that is
# most of the cost of a video cycle. Three keeps the temporal span (they are
# sampled evenly across the window) at half the tokens.
VIDEO_SEND_FRAMES = int(os.getenv("VIDEO_SEND_FRAMES", 3))
# Interleave "(N seconds later)" markers between video frames (Sep 2) —
# Qwen-VL's video training uses time-interleaved frames; stock llama.cpp
# serves no temporal encoding, so without markers the model must guess that
# three images are moments of one scene. Flip off for A/B.
VIDEO_TIME_MARKERS = os.getenv("VIDEO_TIME_MARKERS", "true").lower() in ("true", "1", "yes")
LLAMA_SERVER_URL = os.getenv("LLAMA_SERVER_URL", "http://localhost:8080")

# Label only: the weights llama-server loads come from LLAMA_MODEL_PATH
# (utils/llama_server.py). This name appears in logs and model_settings lookups
# (which fall back to defaults for unknown names — no key exists for either
# qwen label, so overriding is behavior-neutral). Env-overridable July 28 so
# the 27B experiment's logs say what actually ran (run_27b.sh sets it).
MODEL_NAME = os.getenv("MODEL_NAME", "qwen3.8:27b")

MOOD_SNAPSHOT_FOLDER = os.getenv("MOOD_SNAPSHOT_FOLDER", os.path.join(os.path.dirname(os.path.dirname(__file__)), "event_log"))

# === OPEN-VOCABULARY OBJECT DETECTION (Phase 1, Aug 5 2026) ===
# Zero-shot object naming: YOLO-World finds named things in the room so the
# registry (Phase 3) can give gaze real targets. CPU-only by design — 60ms/frame
# warm, the 3090 stays with the 27B and Flux. Runs on a cadence through normal
# movement (saccade guard below), never free-running. Vocabulary is the
# machine's own promote-ready terms from the Phase 0 feasibility report
# (debug/phase0_report/) — appearance language, not function language;
# "electric fan" scored 0%, "fan" works. Additive: never touches the
# person/face tracking path.
OPEN_VOCAB_ENABLED = True
OPEN_VOCAB_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "yolov8s-worldv2.pt")
OPEN_VOCAB_INTERVAL = 4.0  # seconds between detection passes
# Saccade guard only (Aug 5, second live tuning): detection runs THROUGH normal
# movement like the YOLO person tracker — micro-adjustments, orbital wander, slow
# tracking all pass. Only a genuine mid-saccade frame (velocity near the physics
# caps: 12 idle / 25 tracking) is skipped, since it lands elsewhere anyway.
# ~18px/deg at 20ms exposure: even 20 deg/s smears ~7px, tolerable for YOLO —
# the VLM's blur problem is not the detector's. Velocity below the guard still
# sets settled=True/False on results (provenance, not suppression).
OPEN_VOCAB_SETTLE_VELOCITY = 20.0  # deg/s; |pan_vel|+|tilt_vel| above this = mid-saccade, skip the frame
OPEN_VOCAB_MAX_WAIT = 30.0  # seconds; if the guard somehow blocks this long, force a pass
OPEN_VOCAB_MAX_BOXES = 4  # overlay only — storage keeps everything; best-per-term, new arrivals first, then confidence
OPEN_VOCAB_DRAW_MOVE_TOL = 6.0  # deg; hide drawn boxes once gaze drifts this far from capture — must exceed tremor jitter (~2°/axis) or boxes flash
OPEN_VOCAB_CONF_FLOOR = 0.15  # global floor; per-term floors below win
# Per-term floors from crop-verified Phase 0 numbers: mannequin heads are real
# but faint (max 0.41); book/cardboard box fire sloppily so they need more proof.
OPEN_VOCAB_TERM_FLOORS = {
    "mannequin head": 0.10,
    "computer monitor": 0.12,
    "wire basket": 0.12,
    "book": 0.30,
    "cardboard box": 0.25,
    # Worst offender (Aug 17): 2395 hits at EMA conf 0.32 while the real rooster wins
    # at 0.66 even blurred — most firings are junk claiming arms/red blobs at the 0.15
    # global floor. VLM audit agreed the patch wasn't a rooster (challenger_already_known).
    "rooster figurine": 0.35,
}
OPEN_VOCAB_PERSON_OVERLAP_MAX = 0.5  # drop detections mostly inside a person bbox ("sculpted human head" matched a live person in Phase 0)
OPEN_VOCAB_VOCABULARY = [
    "mannequin head",
    "pink shelf",
    "red foam finger",
    "wooden mannequin torso",
    "wooden chair",
    "office chair",
    "desk lamp",
    "coffee mug",
    "laptop",
    "computer monitor",
    "wire basket",
    "fan",
    "book",
    "cardboard box",
    "keyboard",
    "rooster figurine",  # Aug 5 live run: without its own word the rooster was claimed by "wooden mannequin torso" at 0.72 — closed-world effect; with the word it wins at 0.66 even blurred
]

# --- Phase 2: vocabulary promotion (the recursive part, Aug 5 2026) ---
# The monologue names things; recurring names earn a slot in the detector
# vocabulary (perception/vocab_promotion.py, fed per accepted caption). The
# list above is the protected seed; promoted terms join it up to the cap.
# Filters keep the mythology upstairs: concrete noun phrases only — coinages
# (proper nouns), abstractions, and person-words never compile. Ghosts —
# promoted terms that never detect — are kept and logged (looking for what
# isn't there is data), evicted first under cap pressure.
OPEN_VOCAB_PROMOTION_ENABLED = True
OPEN_VOCAB_PROMOTE_WINDOW = 300  # rolling window, in accepted captions
OPEN_VOCAB_PROMOTE_THRESHOLD = 10  # captions in the window mentioning a term -> promote
OPEN_VOCAB_MAX_TERMS = 40  # compiled vocabulary cap (seed + promoted); every term competes for pixels, so quality beats quantity
OPEN_VOCAB_REPROMOTE_COOLDOWN = 3600.0  # seconds an evicted term must wait before promoting again (kills evict/re-promote churn)
OPEN_VOCAB_GHOST_AFTER = 7200.0  # seconds; promoted term with zero detections this old = ghost (logged, kept)
# Single-word barrier: these never promote alone but survive inside phrases
# ("mannequin head" passes, bare "head" does not). Seeded from the machine's
# own most-frequent abstractions plus body parts.
OPEN_VOCAB_STOP_TERMS = {
    "stillness",
    "silence",
    "permission",
    "chaos",
    "fear",
    "weight",
    "thought",
    "time",
    "air",
    "sound",
    "noise",
    "darkness",
    "void",
    "presence",
    "anticipation",
    "paralysis",
    "hesitation",
    "creation",
    "gaze",
    "movement",
    "processing",
    "action",
    "word",
    "breath",
    "glow",
    "command",
    "static",
    "dust",
    "light",
    "line",
    "thing",
    "view",
    "key",
    "tool",
    "head",
    "hand",
    "finger",
    "eye",
    "shoulder",
    "arm",
    "body",
    "joint",
}
# Head-noun barriers: the phrase never promotes whatever the modifiers.
# Person phrases ("quiet man") — people belong to the person tracker, not the
# object map. Undetectable heads ("long shadow", "far corner", "cluttered
# room") — times, room geometry, light and mass nouns aren't boxable objects.
OPEN_VOCAB_PERSON_NOUNS = {"man", "woman", "person", "people", "figure", "visitor", "stranger", "human", "guy", "child", "boy", "girl", "face"}
OPEN_VOCAB_STOP_HEAD_NOUNS = {
    "shadow",
    "corner",
    "edge",
    "wall",
    "floor",
    "ceiling",
    "surface",
    "side",
    "middle",
    "center",
    "background",
    "distance",
    "clutter",
    "mess",
    "one",
    "bit",
    "part",
    "kind",
    "sort",
    "way",
    "glimpse",
    "object",
    "front",
    "back",
    "top",
    "bottom",
    "end",
    "tonight",
    "today",
    "night",
    "day",
    "morning",
    "evening",
    "hour",
    "moment",
    "minute",
    "room",
    "workshop",
    "space",
    "area",
    "world",
    "place",
    "scene",
    "silhouette",
    "moonlight",
    "sunlight",
    "glare",
    "burst",
    "ghost",
    "hover",
    "flicker",
    "twitch",
    "blink",
    "pause",
    "tick",
    "buzz",
    "whir",
    "excuse",
    "project",
    "sight",
    "shape",
    "idea",
    "plan",
    "task",
    "step",
}
# Self barrier: the machine narrating its own body ("my gears", "the hum", "the
# lens") must not enter its object map — the mirror doesn't chart itself.
OPEN_VOCAB_SELF_NOUNS = {"machine", "servo", "motor", "gear", "camera", "lens", "circuit", "sensor", "code", "hum", "vibration"}

# --- Body schema: visual self-recognition (Aug 10 2026) ---
# The machine's arms are a hole in the world model: YOLO called the drawing
# hand a person, the object detector called an arm "rooster figurine", and the
# monologue likely narrates its own limbs as someone else's. Fix: proprioception
# teaches vision. While the CNC executes and the gaze is parked on the paper, a
# "person" at the workspace IS the machine's arm — those crops are harvested
# (only with NO face in view: arms don't have faces, visitors do) into a
# persistent CLIP gallery (event_log/body_schema.json). Matches against the
# gallery then mean SELF: person-evidence is discounted (presence belief),
# object labels are dropped before they enter the map, audits skip them. Person
# tracking itself is untouched — its consumers get more discerning.
# Two-factor (Aug 10, measured): appearance alone CANNOT separate self here —
# the arm mid-draw scores 0.75-0.83 against its own last pose while the hanging
# wooden figure scores 0.87 (the studio is full of sculpted limbs; CLIP tracks
# pose, not identity). But the body enters the frame from a fixed mount, so at
# a given pan/tilt it can only occupy certain regions. Place gates appearance:
# inside the harvested envelope the loose threshold applies; outside it, the
# strict one. Verified on real crops: person-in-envelope 0.68 -> rejected by
# appearance; wooden figure 0.87 outside envelope -> rejected by place; arm
# re-seen 0.78 in envelope -> self.
BODY_SCHEMA_ENABLED = True
BODY_SELF_THRESHOLD = 0.72  # similarity bar INSIDE the reach envelope
BODY_SELF_STRICT = 0.92  # similarity bar with no place evidence
BODY_POSE_TOLERANCE = 15.0  # deg; envelope records apply within this pan AND tilt distance
BODY_HARVEST_INTERVAL = 20.0  # seconds between self-reference harvests while drawing
BODY_GALLERY_SIZE = 60  # references kept (persistent; the body is stable across sessions)
BODY_SELF_FILTER_DETECTIONS = True  # drop open-vocab detections matching the schema (max a few embeds per pass)
BODY_SELF_CHECK_BUDGET = 6  # CLIP embeds per pass; envelope (place-match) boxes are checked first, then by confidence
BODY_SELF_REGION_TTL = 20.0  # s; a patch dropped as self keeps its place self for this long (same pose) — no re-embed, no flicker
# Aug 17 flood fix: overlap-over-smaller-box let a huge floor/table detection "contain" a small
# arm ref and count as in-envelope — one wrong big box then blanketed the frame via sticky
# regions. Envelope/sticky claims now require the DETECTION to lie mostly inside the reference.
BODY_REGION_CONTAINMENT = 0.6  # min fraction of the detection box inside a ref/self-region to count as claimed
BODY_SELF_REGION_MAX_FRAC = 0.25  # never store a sticky self-region bigger than this fraction of the frame
BODY_ENRICH_MIN_SIM = 0.80  # gallery self-growth: only crops this similar enroll as new references
BODY_ENRICH_MAX_FRAC = 0.15  # ...and only if smaller than this fraction of the frame (arms, not scenes)
# The promoted self-zone (Aug 17): while the CNC EXECUTES the arms own this normalized
# frame region (x1,y1,x2,y2) — an object-sized box mostly inside it is the body, no
# gallery or CLIP needed. Explicit proprioception: the learned refs couldn't cover the
# raised hand and the appearance vote is what gets fooled. Size-capped so big background
# boxes (floor, table, curtain) can never qualify.
BODY_DRAWING_SELF_ZONE = (0.05, 0.45, 1.0, 1.0)
BODY_DRAWING_ZONE_MAX_FRAC = 0.2  # only boxes smaller than this fraction of the frame drop via the zone

# --- Adjudicated presence, phase 1 (Aug 18) ---
# A faceless YOLO person-candidate no longer commits the presence belief; the
# machine's own eye looks once (open question, no candidate categories — see
# the no-content-priors law) and the ONTOLOGY of its free reply decides:
# person-reference commits presence, thing-reference records an entity whose
# place vetoes candidates without re-asking. Faces never come here — face
# evidence commits directly and the veto cannot fire against one.
PRESENCE_ADJUDICATION_ENABLED = True
PRESENCE_ADJUDICATE_MIN_INTERVAL_S = 25.0  # min seconds between adjudication calls
ADJUDICATED_PERSON_TTL_S = 120.0  # a person verdict keeps committing presence this long (refreshed by re-adjudication)
ENTITY_VETO_TTL_S = 21600.0  # 6h: a thing-verdict vetoes person-candidates at its place this long, then re-ask

# --- Effigy memory (Aug 17): still, faceless person-shapes are not people ---
# The legless floor robot fires the YOLO person tracker constantly ("child" in
# captions). Discriminator: TIME — a real person cannot hold a pixel-identical
# pose for EFFIGY_STILL_S. Enrolled effigies veto the person state at their
# place; a face there evicts instantly (face evidence always wins). NOTE: if
# the effigy itself moves (servos), stillness never accumulates — that case
# needs appearance enrollment instead (debug seeding, not built yet).
EFFIGY_ENABLED = True
EFFIGY_STILL_S = 600.0  # s a faceless person-box must hold still to enroll
EFFIGY_MATCH_IOU = 0.6  # box IoU to count as "the same place"
EFFIGY_TTL_S = 7200.0  # s unseen before an effigy is forgotten (things get moved)

# --- Label audit: the self-correction loop (Aug 10 2026) ---
# A wrong label looks healthy from the inside — "wire basket" firing on the
# cable bundle racks up hits and never ghosts; nothing doubts an established
# label (the artist: labels "don't correct themselves"). Fix: the machine's
# richer eye audits its faster one. Every LABEL_AUDIT_INTERVAL a well-detected
# registry entry has its latest crop shown to the VLM ("what is this? plain
# appearance names"); if the VLM disagrees, old term and candidates fight a
# CLIP head-to-head on the actual crop, and a winning candidate is promoted
# (origin "audit", bypasses the recurrence threshold — the rooster pattern,
# automated). The old term is NEVER removed: it keeps its true referent and
# merely loses the stolen patch to a better competitor on future passes.
LABEL_AUDIT_ENABLED = True
LABEL_AUDIT_INTERVAL = 600.0  # seconds between audits (one VLM call each; skipped while a drawing is generating)
LABEL_AUDIT_MIN_HITS = 5  # only audit labels the detector actually believes in
LABEL_AUDIT_REAUDIT_HOURS = 24.0  # a term is left alone this long after an audit
LABEL_AUDIT_MARGIN = 0.08  # candidate must beat the incumbent by this much confidence on the crop
# A lost audit now has consequences even when the challenger is already known
# (the rooster gap): the losing term gets a dynamic floor at its live EMA conf
# + NUDGE (it must fire above its own junk-inflated average to claim patches),
# persisted on the registry entry, cleared when a later audit confirms it.
LABEL_AUDIT_FLOOR_NUDGE = 0.08
LABEL_AUDIT_FLOOR_CAP = 0.5  # a real, strong sighting must always be able to clear the bar

# --- Phase 3: spatial registry + registry glances (Aug 5 2026) ---
# The world map: settled detections become per-term anchors in servo angles
# (perception/spatial_registry.py, fed by the detector thread; persists in
# event_log/spatial_registry.json). Idle gaze then aims at the map instead of
# pure Perlin noise: every GAZE_GLANCE_INTERVAL the gaze commits to a target —
# usually the known object gone longest unchecked, sometimes an under-visited
# stretch of the room (explore = "look around for new things"). Arriving
# triggers the existing stillness logic, stillness settles the gaze, the
# detector fires on settle, the anchor sharpens: look -> arrive -> see is one
# loop. Person tracking always outranks glances (untouched).
SPATIAL_REGISTRY_ENABLED = True
SPATIAL_REGISTRY_EMA = 0.35  # anchor update rate per sighting; higher = snappier, lower = steadier
SPATIAL_REGISTRY_MAX_AGE = 604800.0  # seconds (7 days) unseen -> entry forgotten
SPATIAL_REGISTRY_HFOV = 60.0  # deg; same convention as machine.py person_angle
SPATIAL_REGISTRY_VFOV = 34.0  # deg; 60 * 720/1280
SPATIAL_MENTION_BOOST_S = 180.0  # a term the monologue just mentioned pulls the next glance for this long (thought leads gaze)
GAZE_REGISTRY_GLANCES_ENABLED = True
GAZE_GLANCE_INTERVAL = 45.0  # mean seconds between idle glances (jittered)
GAZE_GLANCE_DWELL = 7.0  # mean seconds a glance holds its target (jittered)
GAZE_GLANCE_EXPLORE_WEIGHT = 0.25  # fraction of glances that explore instead of revisit

# Absence discipline (Aug 28): "X isn't where it was" fired 105 times in one
# evening (run 640cb96e) — one every ~4 min, and the monologue's obsession
# with emptiness/ghosts fed on it. Absence must be RARE AND REAL: only a
# well-established anchor can be announced missing, each term at most once
# per cooldown, and the room as a whole at most one absence per gap. Weak or
# junk terms still decay and get forgotten — just silently.
ABSENCE_MIN_HITS = 5  # anchor must have this many sightings before its absence is worth a word
ABSENCE_TERM_COOLDOWN_S = 21600.0  # per-term: at most one absence event per 6h
ABSENCE_GLOBAL_GAP_S = 900.0  # room-wide: at most one absence event per 15 min

# The close look (Aug 28): when the gaze has just deliberately revisited a
# remembered object AND the detector confirmed it there (settled pass during
# the glance), the next caption call sees the CROP — the object at detail
# scale, as a consequence of the machine's own attention. The zoomed pixels
# are the whole invitation: no analysis instruction, no content prior.
CLOSE_LOOK_ENABLED = True
CLOSE_LOOK_MIN_INTERVAL_S = 300.0  # at most one close look per 5 min — a beat, not a mode
CLOSE_LOOK_MAX_AGE_S = 45.0  # glance + crop must be this fresh; stale crops are memory, not sight
CLOSE_LOOK_MIN_SESSION_S = 120.0  # no close looks in a session's first minutes: the awakening owns them (run 3f59eae6: the FIRST caption saw a laptop crop instead of the room), and boot-churn glances during startup playback aren't chosen attention

# === COMFY STUFF ===

COMFY_OUTPUT_FOLDER = os.getenv("COMFY_OUTPUT_FOLDER", os.path.join(os.path.dirname(os.path.dirname(__file__)), "/home/impostor/ComfyUI/output"))

FLUX_DEV_PATH = os.getenv("FLUX_DEV_PATH", "flux1-dev.sft")
FLUX_GGUF_PATH = os.getenv("FLUX_GGUF_PATH", "flux1-dev-Q4_K_S.gguf")
CONTROLNET_NET_PATH = os.getenv("CONTROLNET_NET_PATH", "flux-dev-controlnet-union-pro-2.safetensors")
COMFY_TEMPLATE_FILE = os.getenv("COMFY_TEMPLATE_FILE", "impostor-template-impostor-bot-svg.json")
COMFY_LORA_PATH = os.getenv("COMFY_LORA_PATH", "impostor-32-balanced-16k.safetensors")
# "sketch" pulled Flux toward the LoRA's photographed-graphite mode (soft gray blur); "ink" keeps it in pen territory
# "sharp clean lines... stark white background" (Jul 27) measurably CAUSED blur: drawing-spec
# register pulls Flux into its soft digital-graphic mode (62% of outputs gaussian-blurred).
# The Aug 12 A/B matrix proved this exact prefix sharp in every cell. See memory: comfy-blur-diagnosis.
TRIGGER_PROMPT = os.getenv("TRIGGER_PROMPT", "impostor black and white sketch line art ")
# Force single-image generation for stability (do not override via env)
BATCH_SIZE = 1

# === COMFY CONTROLLER SETTINGS ===
COMFY_LORA_STRENGTH = float(os.getenv("COMFY_LORA_STRENGTH", 1.0))
COMFY_CNET_STRENGTH = float(os.getenv("COMFY_CNET_STRENGTH", 0.3))
# Aug 10 2026, measured over 731 generations: the July "release at 0.6" DOUBLED the
# blur rate (26% -> 58% of outputs defocused; median edge sharpness 135 -> 70).
# Detail forms in the final denoise steps; releasing the depth anchor there lets the
# LoRA's photographed-drawing defocus mode drift in. Held to 1.0 — the 686-sample era.
COMFY_CNET_END_PERCENT = float(os.getenv("COMFY_CNET_END_PERCENT", 1.0))
# 2.5 since Aug 12 blur diagnosis: at 4.0, single-object-on-empty-ground prompts (the register
# the machine favors) render soft on most seeds; 2.5 rendered every tested case crisp.
COMFY_FLUX_GUIDANCE = float(os.getenv("COMFY_FLUX_GUIDANCE", 2.5))
COMFY_LATENT_WIDTH = int(os.getenv("COMFY_LATENT_WIDTH", 1024))
COMFY_LATENT_HEIGHT = int(os.getenv("COMFY_LATENT_HEIGHT", 1024))
COMFY_STEPS = int(os.getenv("COMFY_STEPS", 25))

DRAWING_TIMEOUT = float(
    os.getenv("DRAWING_TIMEOUT", 300.0)
)  # if drawing generation takes longer than this, it will be auto-finished, something is wrong...

# === SVG TO G-CODE SETTINGS ===
# Tone-aware fills (Aug 12 2026, prototyped in debug/tone-centerliner-proto/):
# a detected ink mass is rendered as pen tone — hatch density from the source
# gray's quantiles, cross-hatch only in the darkest band, locally-dark accents
# pulled out as marks — instead of the legacy uniform 45° screen ("wallpaper").
# The artist's verdict on the A/B previews: keep. Known limit: fine features
# in smooth tone (eyes) still band away; the long-term answer is stroke-native
# generation, not more filtering.
CENTERLINE_TONE_FILLS = os.getenv("CENTERLINE_TONE_FILLS", "true").lower() in ("1", "true", "yes")

# Centerline engine (Aug 12 2026): "v2" = skeleton graph walk. "dsv_hybrid"
# routes the STROKE layer through Deep Sketch Vectorization (SIGGRAPH 2024,
# vendored at DSV_HOME with its own venv+weights, offline-safe), masses still
# through the tone renderer — fidelity to the generated image. "dsv" = the
# WHOLE ink through un-thinned DSV, no tone fills — DSV's stroke-elegant
# reduction that simply drops tone; the artist's Aug 12 verdict on the eval
# sheets: "by far the best result we've seen". Any DSV failure falls back to
# the skeleton walk. Runs in the post-ComfyUI slot (frees ComfyUI's cache
# first — it reloads every generation anyway): ~24s GPU, ~7min CPU fallback.
# DEFAULT "dsv" since Aug 12 evening (artist's call, from the eval sheets;
# paper verdict pending). Known trades accepted: thick outlines can arrive
# as fragmented dashes, tonal images print sparse, clean contours sometimes
# double. Revert to "v2" to restore the skeleton walk.
CENTERLINE_ENGINE = os.getenv("CENTERLINE_ENGINE", "dsv")

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

# === ARMS DUET (motor_panel) ===
# XY clamping is no longer a config box. The panel projects EVERY target
# (drag, jog, playback, generation) into the measured reach envelope —
# grbl/warp_calibration.py MEASURED_BOUNDARY, walked on hardware July 20 —
# via the same clamp_to_reach() margin/hysteresis the drawing pipeline uses.
# The ±200 discovery box (July 19) served its purpose and is retired: the
# polygon IS the surveyed reality.
ARMS_DUET_MAX_FEED = 800  # mm/min cap — matches the idle system's ceiling

# Left arm servo limits (degrees). The old 14-degree Python cages (81-95 /
# 88-102) were legacy blind-randomness safety; the retired firmware wanderer
# swept 70-110 on BOTH joints through months of exhibition, so that envelope
# is mechanically proven. Tighten/widen here after creeping the panel sliders
# to the real binding points.
LEFT_ARM_ELBOW_LIMITS = (60, 120, 90)  # lo, hi, neutral — pin 4 (widened from proven 70-110, July 19; creep further on hardware)
LEFT_ARM_SHOULDER_LIMITS = (60, 120, 90)  # lo, hi, neutral — pin 5 (clean firmware allows 0-180; mechanics decide the real limit)
LEFT_ARM_WRIST_LIMITS = (60, 120, 90)  # lo, hi, neutral — pin 6 (July 26: reserved in firmware, servo not yet mounted; creep once it exists)

# === KINETIC BUS (motor_panel/kinetic_bus.py — runtime temperament engine) ===
# machine.py starts the kinetic bus INSTEAD of organic_left_arm + the old
# hand interface's autonomous mode (all three want /dev/arduino_lefthand):
# the lefthand device plays markov generation from recorded datasets picked
# by mood ("{emotion}_*" names), drawing state ("drawing_*"), with startle/
# homing interrupt poses. ON since July 27 — the bus pulls its emotion
# straight from the mood engine (get_emotion_for_hand_controller) every
# supervisor tick, so it does not depend on the old push plumbing. Set False
# to fall back to the legacy pair.
KINETIC_BUS_ENABLED = True
KINETIC_MONITOR_UI = True  # small read-only Tk window opened by machine.py (the old hand controller's slot)
KINETIC_CROSSFADE_S = 2.5  # seamless morph: ease into the new temperament's nearest state over this long
KINETIC_ROTATE_S = 300  # dwell before rotating among same-state bundles (variety)
# Gaze -> movement: one directional CURRENT, three coordinated effects all
# driven by the same gaze vector and scaled by KINETIC_GAZE_STRENGTH (the
# runtime tab's "gaze influence" slider):
#   lean   — every applicable channel drifts a bounded number of degrees
#            toward the gaze, settling over LEAN_TAU seconds and decaying
#            back when the gaze recenters (the whole body sways WITH the
#            look, together — never snaps, never clips)
#   tempo  — gaze-aligned transitions play eager (quicker), opposed ones
#            reluctant (slower); works even on momentum-locked recordings
#   choice — markov transition choice reweighted toward gaze-aligned
#            movement (needs branching in the recording to act on)
# Directional logic, not absolute values: recorded poses are only ever
# leaned by a bounded smoothed amount, and the walk itself stays inside
# demonstrated states and transitions.
KINETIC_GAZE_STRENGTH = 1.0  # master, 0..2
# degrees raised July 27 (8/6/5 was measurably working but perceptually
# invisible — a slow center-drift hides inside the take's own motion)
KINETIC_GAZE_LEAN = {"shoulder": ("x", 14), "wrist": ("x", 10), "x": ("x", 4), "elbow": ("y", 9), "y": ("y", 4)}  # channel: (axis, deg @ full gaze)
KINETIC_GAZE_LEAN_TAU = 1.5  # seconds for the lean to settle / release
KINETIC_GAZE_CHOICE_K = 2.0  # transition-choice bias coefficient
KINETIC_GAZE_TEMPO_K = 0.6  # eagerness: dt scales by exp(-K * alignment)
KINETIC_STARTLE_ENABLED = True
# Startle = flinch -> hold -> slow release. Every servo NUDGES partway
# toward a startle pose — quick, never a full transition into it — then
# the body freezes in that held tension for HOLD_S, then slowly blends
# back into the running dataset (the normal crossfade re-entry). The pose
# comes from a take assigned under the "startle" state (record yourself
# HOLDING the flinch — its per-channel median is the pose); the DELTAS
# below are the built-in fallback until one exists. Gantry and pen never
# take part in a flinch.
KINETIC_STARTLE_NUDGE = 0.6  # fraction of the way toward the startle pose
KINETIC_STARTLE_HOLD_S = 3.0  # held tension before the slow release
KINETIC_STARTLE_DELTAS = {"finger0": 40, "finger1": 40, "finger2": 40, "finger3": 40, "wrist": 25, "lung": 20}  # fallback flinch, degrees relative
KINETIC_STARTLE_COOLDOWN_S = 20  # a flickering detector must not twitch the hand

# Reach: while someone is being TRACKED, the arm leans out toward them.
# Face tracking already points the gaze at the person, so gaze direction =
# person direction; that direction picks a point in the arm's MEASURED
# 9-point calibration square (motor_panel/arm_calibration.json — bilinear
# over captured poses IS the inverse kinematics, measured-not-modeled like
# the warp map) and the temperament's whole field shifts partway toward
# that pose, ramping in/out over REACH_TAU as people come and go. The
# markov motion keeps breathing through it — a lean, never snap-tracking.
# Joint-space proportional fallback until the arm is calibrated.
KINETIC_REACH_ENABLED = True
KINETIC_REACH_STRENGTH = 0.8  # fraction of the way toward the reach pose at full ramp
KINETIC_REACH_MAX_DEG = 25  # per-channel cap on the reach shift
KINETIC_REACH_TAU = 2.0  # seconds to lean out / settle back

# Homing safety: a take assigned under the "homing" state IS the escape
# choreography — record the whole get-clear movement (ending tucked) and
# it plays straight through (no markov) before the gantry homes: entry
# eases into the take's first sample over TUCK_S, the take runs once, the
# body holds its final pose until homing completes (in-process hook or the
# cross-process sentinel — the idle subprocess homes at boot), then blends
# back. machine.py startup homing triggers this automatically, so the
# recorded movement is the machine's first gesture. No dataset = the bus
# refuses to guess. Max hold releases a stranded arm if every completion
# signal is missed (GRBL's homing quiet-wait is 60s).
KINETIC_HOMING_MAX_HOLD_S = 75
KINETIC_HOMING_TUCK_S = 1.0  # entry ease into the choreography's first pose — no snapping
# Paper check: the recorded get-clear move (assign under 'paper') plays and
# HOLDS while the camera inspects the paper — both arms, gantry included.
# Max hold covers the ~12s organic search plus margin.
KINETIC_PAPER_TUCK_S = 0.8
KINETIC_PAPER_MAX_HOLD_S = 30.0
# THE BODY'S SAMPLER (July 31). Measured: at 1x bins a 600-sample take
# trained ~568 states with branching factor 1.00 — the "markov chain" was
# the recording as a linked list, and the gaze CHOICE bias had nothing to
# choose. Identity is now coarse (this scale on DEFAULT_BINS) while poses
# stay exact, so states merge and the walk forks. 8x gives ~16% of states a
# real choice; 16x gives ~48% but glues together moments further apart —
# raise it for a wilder body, lower it toward faithful replay.
KINETIC_STATE_BIN_SCALE = 8.0
# Choice controls, same vocabulary as the model's sampler. Each is a base
# value plus how far AROUSAL moves it (mood 0..1): a calm body replays what
# it knows, an agitated one wanders and repeats itself less.
KINETIC_MOVE_TEMP = 1.0  # <1 faithful, >1 adventurous
KINETIC_MOVE_TEMP_AROUSAL = 0.8  # temp = TEMP + arousal * this
KINETIC_MOVE_REPETITION = 1.4  # divide down states visited in the last window
KINETIC_MOVE_REPETITION_AROUSAL = 0.6
KINETIC_MOVE_MIN_P = 0.05  # drop candidates below this fraction of the best
KINETIC_MOVE_WINDOW = 24  # how many recent states the repetition penalty remembers
# EMPIRICAL COLLISION SAFETY (Aug 1). Both arms are recorded together, so a
# normal chain walk only visits combinations that were performed safely.
# The danger is the glue: measured on the real recordings, straight-line
# crossfade midpoints sit 6.9 units (worst 12.1) from ANY demonstrated
# combination and a full gaze lean sits 11.0 away, while demonstrated
# neighbours are only 1.2 apart. The guard asks every outgoing command how
# far it is from the nearest thing ever performed and pulls back the
# strays. No geometry, no IK, no camera — see motor_panel/safe_envelope.py.
KINETIC_SAFE_ENVELOPE = True
KINETIC_SAFE_MAX_DIST = 3.0  # units of slack beyond demonstrated ground (local spacing is ~1.2)
KINETIC_SAFE_SMOOTH = 0.25  # how fast the pull-back eases in per send (~0.5s to full). Projection onto a
# point cloud is inherently discontinuous where the nearest neighbours swap, and applying that raw would
# snap the body — the very thing recorded movement exists to avoid. Easing the CORRECTION keeps commands
# continuous; the brief lag is bounded by the slack above.
KINETIC_SAFE_SLEW = 0.5  # hard cap on how much the correction may CHANGE in one send. Easing alone still
# passes a fraction of a jump straight through (a 13-unit swap x 0.25 = 3.3 units of snap); capping the
# change makes the pull-back strictly gradual. A 12-unit correction converges in ~1s at send rate.
# nearest — a lone neighbour flips as the body crosses between them and the correction snaps (3.6 units,
# measured); an average moves smoothly. The correction is applied in full: a steady offset like the lean
# pushes out on EVERY send, so a rate-limited pull-back would just lose a tug of war forever.
# False (July 28, artist's call): the homing dance and the gantry sweep run
# SIMULTANEOUSLY — the choreography is recorded to stay clear of the
# gantry, so $H does not wait for it. True restores clear-first: $H holds
# until the arm has finished getting out of the way.
KINETIC_HOMING_WAIT_CLEAR = False
# The right arm in the temperament (July 28 — "bus v2" landed early): the
# bus owns a headless GantryLink between drawings, playing the datasets'
# recorded x/y through the same markov chains as the servos (reach-clamped,
# G1 at chain tempo). The drawing pipeline's pause/resume call sites
# release/re-acquire the port; every re-acquire re-homes (a serial open
# resets GRBL), firing the tuck choreography. Pen stays UP during
# generation unless KINETIC_GANTRY_PEN lets recorded pen takes ink.
KINETIC_GANTRY = True
KINETIC_GANTRY_PEN = False
# The AWAKENING: at machine.py boot the bus holds the body STILL until the
# startup homing flow runs — the homing choreography is the machine's first
# gesture and the first temperament blooms as homing completes, all motors
# together (previously the left hand soloed a dataset ~2s after init, long
# before anything else woke). Failsafe: bloom anyway if homing never comes.
KINETIC_AWAKENING_MAX_WAIT_S = 180

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
# 52 -> 56 (July 21): calibration dots at 52 were ghost-faint (5-13 gray
# levels above paper in photos — unmeasurable by human OR machine vision;
# an entire night of matcher engineering traced back to this). Real strokes
# during drawing were fine; taps need more plunge to deposit ink.
GRBL_PEN_DOWN_S = int(os.getenv("GRBL_PEN_DOWN_S", 56))

# Settle dwells after pen transitions during drawing (G4). GRBL treats a
# spindle-PWM change as instantaneous — it never waits for the physical
# servo — so without a dwell, a dot/short dash is over before the pen lands
# (the dotted-line dropouts). History: 0.12 (July 9) -> 0.2 (Aug 17, faint
# hatching ticks) -> SPLIT Aug 18 (artist: drawings still "dotted" — short
# strokes render as dots because motion starts before the S34->S56 landing
# finishes). DOWN needs the full landing + bounce; UP only needs to clear
# the paper before the rapid. Legacy GRBL_PEN_SETTLE_DWELL_S env still
# honored as the fallback for both.
_LEGACY_SETTLE = os.getenv("GRBL_PEN_SETTLE_DWELL_S", "")
GRBL_PEN_DOWN_SETTLE_S = float(os.getenv("GRBL_PEN_DOWN_SETTLE_S", _LEGACY_SETTLE or 0.35))
GRBL_PEN_UP_SETTLE_S = float(os.getenv("GRBL_PEN_UP_SETTLE_S", _LEGACY_SETTLE or 0.2))

# Ink scale inside the calibrated paper window. 1.0 = ink fills the window
# (bounds-normalized); <1 shrinks about the window center. Tried 0.85 Aug 17
# ("slightly smaller"); reverted to 1.0 Aug 18 — the shrink compounded the
# dotted feel of short strokes (artist: "a bit small right now which also
# adds to the dotted feel").
WARP_INK_SCALE = float(os.getenv("WARP_INK_SCALE", 1.0))

# Extra safety to ensure pen is fully UP before any homing ($H)
GRBL_PEN_UP_REPEATS = int(os.getenv("GRBL_PEN_UP_REPEATS", 5))  # How many times to assert M3 S{UP} before homing
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
GRBL_IDLE_ZONE = (20, 40, 20, 40)  # Boundary box: (x_min, x_max, y_min, y_max) for 40x40 area

# === GRBL G-CODE OPTIMIZATION SETTINGS ===
# Intelligent feed rate and pen lift optimization for better drawing performance

# Master optimization toggles
GRBL_ENABLE_FEED_OPTIMIZATION = os.getenv("GRBL_ENABLE_FEED_OPTIMIZATION", "true").lower() in ("1", "true", "yes")
# Pen-lift clustering OFF by default (Aug 12 2026): on a small dense drawing
# every pen-down lands within the 5mm cluster threshold, so the ENTIRE sheet
# ran on the shallow fast lift (S38 — 3 servo units above the documented
# grazing point on an unflat surface). One high region and every traversal
# drags ink through the figure. 77 deep lifts cost seconds; a grazed sheet
# costs the drawing. Same doctrine as the flat feed rate.
GRBL_ENABLE_PEN_OPTIMIZATION = os.getenv("GRBL_ENABLE_PEN_OPTIMIZATION", "false").lower() in ("1", "true", "yes")
GRBL_ENABLE_STROKE_FILTERING = os.getenv("GRBL_ENABLE_STROKE_FILTERING", "false").lower() in ("1", "true", "yes")

# === GRBL SEGMENTED EXECUTION ===
# Splits large G-code files into segments to prevent buffer overload

# === FEED RATES (Aug 10 2026: flat ink speed, dynamic scaling retired) ===
# The distance-scaled system (300/700/2000 over 1-8mm thresholds) was tuned on
# vpype's ultra-dense output (median segment 0.03mm), where every move fell
# below the micro threshold and the whole drawing ran at the 420 crawl — the
# February field-test sheets were drawn that way. Centerliner v2's simplified
# paths (median 0.43mm) pushed the same formula to ~700-2000 on the ink and
# the marks got faint and sloppy. Drawings are a few hundred mm of ink total;
# speed buys seconds and costs fidelity, so pen-down is now one deliberate
# rate. Expressive pace, if ever wanted, belongs in the choreography layer.
GRBL_DRAW_FEED_RATE = int(os.getenv("GRBL_DRAW_FEED_RATE", 450))  # Every pen-down move (mm/min)
GRBL_TRAVERSAL_FEED_RATE = int(os.getenv("GRBL_TRAVERSAL_FEED_RATE", 2000))  # Pen-up moves — no fidelity cost (mm/min)

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
GRBL_FAST_PEN_UP = int(os.getenv("GRBL_FAST_PEN_UP", 38))  # dense clusters: shallower, still ~3 above the grazing 41
GRBL_FAST_PEN_DOWN = int(os.getenv("GRBL_FAST_PEN_DOWN", GRBL_PEN_DOWN_S))  # was +5: pressing HARDER in clusters was backwards

# Serial link revival (Aug 12 2026): pen-lift commands were timing out with
# TOTAL silence mid-drawing (loose cable / transient stall — the same fd
# answered again seconds later), killing 8 of 9 drawings in an afternoon and
# leaving half-inked sheets. On a silent timeout the executor now polls the
# link back to life and retries the line in place (G90 makes it idempotent);
# a Grbl reset banner still aborts — position is lost and re-homing must run.
GRBL_SERIAL_RECOVERY_MAX = int(os.getenv("GRBL_SERIAL_RECOVERY_MAX", 3))  # revival attempts per drawing

# Cluster detection parameters
GRBL_CLUSTER_DISTANCE_THRESHOLD = float(os.getenv("GRBL_CLUSTER_DISTANCE_THRESHOLD", 5.0))  # Max distance between clustered pen lifts (mm)
GRBL_CLUSTER_SEQUENCE_MIN = int(os.getenv("GRBL_CLUSTER_SEQUENCE_MIN", 3))  # Minimum pen lifts to consider a cluster

# === EXPERIMENTAL PATH SIMPLIFICATION ===
# WARNING: These are experimental features that may affect drawing quality
# Only enable for testing - disable for production artwork
GRBL_EXPERIMENTAL_SIMPLIFICATION = os.getenv("GRBL_EXPERIMENTAL_SIMPLIFICATION", "true").lower() in ("1", "true", "yes")
GRBL_SIMPLIFICATION_TOLERANCE = float(
    os.getenv("GRBL_SIMPLIFICATION_TOLERANCE", 0.02)
)  # Tolerance for path simplification (mm) - smaller = higher quality
GRBL_MERGE_TOLERANCE = float(os.getenv("GRBL_MERGE_TOLERANCE", 0.05))  # Tolerance for line merging (mm)

# === UARM SWIFT PRO SETTINGS ===
USE_UARM = True  # Enable uArm Swift Pro robotic arm integration
UARM_PORT = "/dev/arduino_uarm"  # Fixed udev symlink (matches ARDUINO_DEVICES)
UARM_MOVEMENT_NAMES = {1: "pickup", 2: "place", 3: "gesture"}  # Primary pickup motion  # Primary placement motion  # Gestural expression motion
UARM_MOTION_STORAGE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "movement_recordings", "uarm")

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
SCENE_MOTION_RESIDUAL_THRESHOLD = (
    0.02  # >2% of pixels moving = something is happening (post-erosion: small object ~0.019, camera sway alone 0.000, saccades excluded)
)
SCENE_MOTION_MIN_FRAMES = 2  # frames in the 10s window that must exceed it

# The stream (CoT-style continuity): prior captions ride as the machine's own
# assistant turns so each caption continues a visible thought. 0 disables and
# reverts to amnesiac single-turn captions.
STREAM_WINDOW = int(
    os.getenv("STREAM_WINDOW", 24)
)  # ENV-TUNABLE July 28 (window size is an information budget: the model repeats what it can't see it already said — six entries is ~40s of visible selfhood, a 9B-era relic per north-star P5; the 27B holds 20-30). # how many prior captions the model sees as its own turns. ON (June 28) now the base voice is healthy: chiefly to break the amnesiac REPETITION (the persistent "dust motes" tic — each call couldn't see it already said it). Admissibility-gated (_stream_admissible: no meta, no markdown/stage-directions); gaps render as "(… later)" lines (STREAM_GAP_MARK_SECONDS), only ≥STREAM_BREAK_SECONDS restarts it. WATCH: the stream amplifies whatever register is in the window — if it breeds purple instead of varying, set back to 0.
# Gaps in the stream (Aug 20, the felt-time fix): a silence used to WIPE the
# stream at 180s — amnesia presented as continuity (the machine resumed with
# less context and nothing said time had passed; the 3min–2h range was a dead
# zone below the reorientation threshold). Now a gap ≥ STREAM_GAP_MARK_SECONDS
# renders as an unstamped line in the log — "(about 20 minutes later)", words
# not integers per the seventeen-days law — because the model does not do
# clock arithmetic on adjacent HH:MM stamps; duration must be said to be felt.
# Only a gap ≥ STREAM_BREAK_SECONDS still clears the stream (the thought
# genuinely died; matches REORIENT_MIN_GAP_S so the reorientation line takes
# over exactly where the stream gives up).
STREAM_GAP_MARK_SECONDS = 180
STREAM_BREAK_SECONDS = 7200

# How the stream reaches the model (July 2026, docs/continuity-plan.md):
# "world" — THE INVERSION (July 26): the stream rides as a TIMESTAMPED LOG
#   ("14:02 — the lamp's still on") in one assistant message, and the world's
#   turn — frames, what changed, the present — comes LAST. Generation always
#   begins right after the world, never after the machine's own prose: every
#   call answers something outside itself (closed loop), instead of extending
#   its own essay (open loop — the drift/rambling physics). The log rendering
#   is genre framing per north-star P7: a log is the text-shape of a working
#   mind, and logs are plain by genre; the lonely-soliloquy frame summoned
#   poetry no one asked for.
# "document" — the monologue-so-far is sent as ONE trailing assistant message
#   and llama-server CONTINUES it (assistant prefill; requires
#   enable_thinking=false or the server rejects the request). The model's next
#   tokens are literally the next tokens of its own text — real continuity,
#   but text momentum outweighs the frame: perception becomes decor (the
#   rooster run) and a truncated tail forces run-on continuation.
# "turns" — legacy: each prior caption as a separate assistant turn-pair.
#   Bred template imitation (openings cloned across captions), kept as A/B.
# FLIPPED BACK July 28 (artist's call, and the missing experiment): the rooster
# failure was a DETECTION failure, not a shape failure — document + the new
# detectors (view replacement, motion onset) + the new storage gates (felt,
# event provenance, refrain, place-occasional) has never been run. Document
# had momentum (aliveness); world had grounding but answered a static room
# forty times an hour (flat) and its delta framing elicited fake-delta tropes
# ("the light feels different now"). A/B directly against the world runs.
STREAM_MODE = os.getenv("STREAM_MODE", "mind")  # "mind" (Sep 5 eve, the conversation shape) | "hybrid" (stamped log + seam, Aug 1–Sep 5)

# HYBRID seam size (Aug 1): how many chars of the machine's latest thought are
# handed back as the continuation prefill in STREAM_MODE="hybrid". Short on
# purpose — enough to land mid-voice, never the whole document (that is what
# deadlocked Aug 1). ~220 chars ≈ the last 30-35 words.
HYBRID_PREFILL_CHARS = int(os.getenv("HYBRID_PREFILL_CHARS", 220))

# CAPTION SAMPLING (env-tunable Aug 1). Defaults preserve the 9B-era settings
# exactly; run_27b.sh overrides them. The 0.6/0.7 + top_p 0.85 pair was chosen
# to stop a 9B blooming purple — on a 27B it just pins output to the mode.
# min_p (0 = off) is the better tail-cut for a larger model: proportional to
# confidence, so temperature can rise without degenerating.
# 1.0 -> 0.9 (Aug 22, artist: run-on clause-chains — at 1.0 the period, a
# "decisive" token, competes badly against the many ways to extend a clause).
# Both directions have measured failure modes: 0.6-0.7+top_p0.85 pinned the
# 27B to flat semicolon declaratives (July); 1.0 bloomed caption-ese run-ons.
# The rhythm itself (fragments, emphasis, questions) is genre-framed in
# genre.hybrid, not bought with temperature.
CAPTION_TEMP = float(os.getenv("CAPTION_TEMP", 0.9))
# Vendor-shaped repetition control (Qwen official non-thinking recipe:
# repetition_penalty 1.0 + presence_penalty 0.6-1.5). 0.0 = off.
# ON since Sep 3 evening (0.6, vendor floor): the deflation template the
# Aug 31 handover predicted ("The ___ is just a ___" — the model's survival
# strategy under fixed sampling; frames never repeat as tokens, so DRY can't
# see them) hit 38% of the evening's caption lines, with ZERO exclamation
# marks all day — the artist's ear, confirmed numerically. This is the
# queued experiment, triggered. Judge with debug/caption_metrics.py over a
# full evening (punctuation survival is measured — watch it; the penalty
# taxes reused sentence frames, not the period). CAPTION_PRESENCE_PENALTY=0
# reverts to the old arm.
CAPTION_PRESENCE_PENALTY = float(os.getenv("CAPTION_PRESENCE_PENALTY", 0.6))
CAPTION_TEMP_BORED = float(os.getenv("CAPTION_TEMP_BORED", 0.85))
CAPTION_TOP_P = float(os.getenv("CAPTION_TOP_P", 1.0))
CAPTION_MIN_P = float(os.getenv("CAPTION_MIN_P", 0.05))

# Punctuation-safe penalties (Aug 28). repeat_penalty taxes every repeated
# token — and the most-repeated tokens in prose are the period and the comma.
# At 1.15 the third sentence's period was measurably suppressed and the flow
# tipped into comma-less run-on (median 6.8 sentence marks per 100 words,
# 52 fully unpunctuated captions in run 640cb96e — the "manic" register the
# storage trims were fighting after the fact). Loop suppression is DRY's job
# (dry_multiplier 0.85, local tail) and the storage gates'.
# 1.05 NOT 1.0 (Aug 28 evening, run 3f59eae6): fully off, the voice flipped
# to the opposite attractor — declarative chanting ("i am just sitting" x6,
# 18 spoken-not-stored echoes in 7 minutes). The tax was quietly the only
# cross-sentence resistance to repeating a short line verbatim. 1.05
# compounds ~3x slower than 1.15; punctuation survival is now MEASURED
# (sentence marks per 100 words), so if 1.05 re-kills it, the stats say so.
CAPTION_REPEAT_PENALTY = float(os.getenv("CAPTION_REPEAT_PENALTY", 1.05))

# DRY horizon (Aug 28 evening). 128 tokens saw only the current caption —
# chanted lines recur ACROSS captions, invisible to it. 384 reaches ~3
# entries back, so a re-typed sentence is penalized as a sequence. The July 9
# lesson (DRY over the whole context exhausted the honest vocabulary into
# synonym salad) was about -1/unbounded; 384 is the middle ground.
CAPTION_DRY_LAST_N = int(os.getenv("CAPTION_DRY_LAST_N", 384))

# Length rhythm (Aug 28). The model almost never stops on its own — 70% of
# run 640cb96e's caption responses ended at the token cap, so the cap IS the
# length and a constant cap makes every thought the same size (median 67
# words; the north-star register is "the lamp's still on"). A short beat
# rolled on a fraction of ordinary cycles is the honest way to get short
# thoughts out of a prior that never volunteers one: the mouth trims to a
# sentence boundary, so a small budget reads as a small complete thought.
# Short entries then enter the stream window, and self-imitation starts
# working FOR rhythm instead of against it. 0.2 not 0.3 (Aug 28 evening):
# at 0.3 the first live run over-seeded the window with staccato and the
# register flipped to fragment-chanting ("stressed-out haiku", artist's
# read) — rhythm wants a minority beat, not a near-third.
# AGENCY ROUND (Sep 5, artist: "one or two sentences per caption at most; earlier
# systems could do a single word or '…'"): the 80-token default trimmed to ~55 words
# and the window taught that length back (mean 59 words; 5 of 109 captions under two
# sentences; zero one-word thoughts). Budgets now shape a thought, not a paragraph.
CAPTION_NUM_PREDICT = int(os.getenv("CAPTION_NUM_PREDICT", 38))  # one or two sentences
CAPTION_NUM_PREDICT_INWARD = int(os.getenv("CAPTION_NUM_PREDICT_INWARD", 70))  # inward / close-look beats
CAPTION_SHORT_BEAT_P = float(os.getenv("CAPTION_SHORT_BEAT_P", 0.3))
CAPTION_SHORT_BEAT_TOKENS = int(os.getenv("CAPTION_SHORT_BEAT_TOKENS", 14))  # a word or a short clause

# Drawing calls get a real timeout (Aug 2). query_model defaults to 30s — a
# 9B-era number. The drawing INTENT prompt is the largest in the system (stream
# tail + musings + felt + desire + the executed body of work + reflections, plus
# an image) and asks for 180 tokens; on the 27B that routinely exceeds 30s, and
# the timeout error string was then used AS the drawing prompt.
DRAWING_CALL_TIMEOUT = int(os.getenv("DRAWING_CALL_TIMEOUT", 180))

# World-anchored change detection (Sep 3, queue #2 — the camera-vs-world
# referee; supersedes the July 26 single-slot view-replacement check, which
# forgot its one reference frame on any gaze turn). Per-pose 64px grayscale
# references (vision/pose_view_memory.py): a settled frame within COMPARE_DEG
# of a fresh reference is an honest comparison — catching both the same-pose
# change (bumped camera, lights-out; the rooster run) and change discovered
# on RETURNING to a view ("it's different here from when you last looked").
# Older references re-baseline silently: lighting drifts, and a change the
# code can't attest must not mint an event. Confirmed-unchanged looks roll
# the reference forward (slow drift never accumulates into a false event)
# and count toward world-verified stillness. world_changed lands in the
# episodic log — a new anchor for the unchanged clock, which was episodic-
# only and rightly distrusted.
WORLD_VIEW_DIFF_THRESHOLD = float(os.getenv("WORLD_VIEW_DIFF_THRESHOLD", 0.30))  # breathing sway ~0.05-0.1 at 64px; scene replacement ~0.4+
WORLD_POSE_MEMORY_ENABLED = os.getenv("WORLD_POSE_MEMORY_ENABLED", "true").lower() in ("true", "1", "yes")  # false = NO view-change detection at all
WORLD_POSE_CELL_DEG = 6.0  # reference grid: one remembered view per 6-degree pose cell
WORLD_POSE_COMPARE_DEG = 3.0  # honest-comparison regime, same value the July 26 check proved (sway ~1 deg)
WORLD_POSE_REF_MAX_AGE_S = float(os.getenv("WORLD_POSE_REF_MAX_AGE_S", 1800))
WORLD_POSE_MAX_REFS = 32
# Anchor verification: a detector re-sighting within this many degrees of the
# stored anchor stamps last_verified_ts — position stability, not just
# existence. The familiarity line's "still in the same spot" now REQUIRES a
# verification within WORLD_SAME_SPOT_WINDOW_S (else the softer line ships) —
# prompts must not claim positions the code can't vouch for.
WORLD_ANCHOR_CONFIRM_DEG = 10.0
WORLD_SAME_SPOT_WINDOW_S = float(os.getenv("WORLD_SAME_SPOT_WINDOW_S", 900))
# Verified stillness -> boredom: needs MIN_CONFIRMS confirmed-unchanged looks
# since the last world change / salience spike (absence of evidence isn't
# stillness), saturates over SATURATION_S, contributes at most BOREDOM_MAX —
# deliberately below the 0.7 bored threshold, so the world being still raises
# drift propensity but never flips the sampling regime on its own.
WORLD_STILL_MIN_CONFIRMS = 3
WORLD_STILLNESS_SATURATION_S = float(os.getenv("WORLD_STILLNESS_SATURATION_S", 3600))
WORLD_STILLNESS_BOREDOM_MAX = float(os.getenv("WORLD_STILLNESS_BOREDOM_MAX", 0.6))

# Anti-echo storage gate: a caption that OPENS with the same N words as a
# recent stream entry is a template imitation, not a continuation. One retry
# at a hotter temperature; if it still echoes, the cycle is skipped (silence
# over restatement). A gate, not a style fence.
ANTI_ECHO_WORDS = 5
# Opening-echo compares against the RECENT tail only (July 28): with the
# window env-tunable to 20-30 entries, a whole-window check would punish an
# opening reused forty minutes later — that's a callback (memory), not a
# template tic. Tics live in the last few entries.
ANTI_ECHO_COMPARE_TAIL = int(os.getenv("ANTI_ECHO_COMPARE_TAIL", 8))
ANTI_ECHO_RETRY_TEMP_BUMP = 0.15

# A blink is not a night: below this offline gap, restarting skips the full
# awakening ceremony (which, run several times an hour across dev restarts,
# converged on stock reorientation prose — "the hum returns, dust motes...")
# and instead RESUMES: the prior session's last thought seeds the stream and
# document mode continues it. Real absences still get the rich awakening.
AWAKENING_MIN_GAP_S = int(os.getenv("AWAKENING_MIN_GAP_S", 600))
# The blink as fact (Sep 4, artist): even a short outage is registered — the
# first prompts after a blink resume carry "You were off for {duration} —
# you've just come back on" (bare measured fact; what the machine makes of
# the lapse is its own). Window = how long the fact rides after the resume.
BLINK_NOTE_WINDOW_S = float(os.getenv("BLINK_NOTE_WINDOW_S", 90))

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
STREAM_CONSOLIDATE_CHARS = int(
    os.getenv("STREAM_CONSOLIDATE_CHARS", 12000)
)  # scale with STREAM_WINDOW (~250 chars/entry) or consolidation eats the bigger window

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
SALIENCE_ARRIVAL_WINDOW = 10  # seconds an arrival keeps salience hot (~one live caption, then quiet)

# Presence is a sticky, uncertain belief (captioner._assess_scene): once someone
# is seen, the machine keeps believing they're around through detection gaps
# (gaze looks away, occlusion, no servo encoders) and only concludes they left
# after this long with no sighting. Generous on purpose — losing sight of
# someone is not the same as them leaving. Only the OFF->ON edge is an arrival.
# Aug 5 fix: the decay clock only runs while the gaze is actually pointed near
# the last-seen spot — the machine's own wandering used to decay the belief and
# manufacture a false "new presence" every time it looked back (the artist was
# greeted as new dozens of times a day). Not-looking is not evidence of absence.
PRESENCE_BELIEF_DECAY_SECONDS = 240

# Re-arrival prior (Aug 31). Re-ID is off (CLIP can't tell outfits apart),
# so every re-sighting after a belief drop counted as a GENUINE arrival —
# 73 phantom arrivals in one solo workday, from the artist stepping out of
# frame. In this room a sighting within this window of the last believed
# presence is the same visit resuming: no arrival event, no episodic
# record, no salience spike. A confirmed departure is only RECORDED once
# the absence outlasts the same window (backdated to when they vanished).
PRESENCE_REARRIVAL_WINDOW_S = float(os.getenv("PRESENCE_REARRIVAL_WINDOW_S", 1800))
# TIGHTENED 30→18 (Sep 4, attention round): 30° was the full frame half-width
# (HFOV 60), so a person at the frame EDGE — where the skeleton gate rightly
# refuses partial bodies — counted as looked-for-and-absent, and the belief
# died on evidence never collected ("The man is gone" after a few degrees of
# turn, the day-one complaint). 18° keeps last-seen comfortably inside the
# frame before an empty look may tick the decay.
PRESENCE_ABSENCE_LOOK_TOLERANCE = float(os.getenv("PRESENCE_ABSENCE_LOOK_TOLERANCE", 18.0))

# Sep 4 evening — presence stickiness (docs/presence-stickiness-sep4.md). Once the
# belief has VERIFIED a departure, the stream window still carries the person for
# many entries and the model continues them in the present tense (ablation: the
# stream is the belief; a standing time-stamped absence fact fixes the tense).
# The fact rides only while the belief is OFF and the recent stored stream still
# mentions a person — self-limiting, it stops when the stream stops.
ABSENCE_STANDING_ENABLED = os.getenv("ABSENCE_STANDING_ENABLED", "true").lower() == "true"
ABSENCE_STANDING_TAIL = int(os.getenv("ABSENCE_STANDING_TAIL", 8))  # stored stream entries scanned for a person mention
ABSENCE_SESSION_MIN_S = int(os.getenv("ABSENCE_SESSION_MIN_S", 90))  # fresh boot: detector settle time before the session-scoped fact may ride
# Adjudicator false arrivals (Sep 5: "a man lying down" = the black bundle on the top
# shelf, twice; "a person looking at a desk" = the mannequin head): a person verdict
# that verified absence closes within this window is retracted to a thing at that
# gaze + box, so the veto fires next time instead of re-asking.
PRESENCE_FALSE_ARRIVAL_WINDOW_S = float(os.getenv("PRESENCE_FALSE_ARRIVAL_WINDOW_S", 240))
ENTITY_VETO_GAZE_TOL_DEG = float(os.getenv("ENTITY_VETO_GAZE_TOL_DEG", 12.0))
# Phantom presence gate (Sep 4 evening): a present-tense third-person claim while the
# adjudicated belief says nobody is here is spoken but never STORED — the stream is
# the belief, and storing these is how the artist outlived their own departure.
PHANTOM_PRESENCE_GATE = os.getenv("PHANTOM_PRESENCE_GATE", "true").lower() == "true"

# Session re-ID (Aug 5, layer 2 of the false-arrival fix): person crops are
# embedded into a rolling session gallery; when the presence belief has lapsed
# and someone is detected, a match against recent sightings means the same
# person resumed — no arrival event. No names, no persistent biometrics: the
# gallery dies with the process.
# DISABLED pending a real re-ID embedding: debug/test_presence_reid.py showed
# CLIP image embeddings measure SCENE similarity, not identity — the two
# different people scored 0.87 while the artist-vs-artist pair scored 0.49.
# The plumbing (perception/presence_identity.py) is live and flag-gated; swap
# embed_crop to a person-reid model (e.g. OSNet, ~2MB, CPU) and re-run the
# test before enabling. Layer 1 (gaze-aware decay) ships regardless and kills
# most false arrivals on its own.
# CONFIRMED Aug 10 (independent re-measurement): same outfit, same session,
# different poses = 0.70-0.74; vs a different person = 0.67-0.76. No usable
# threshold exists. Identity continuity ships via the ARRIVAL LEDGER instead
# (presence_identity.record_arrival/singular_regime): recent arrivals mostly
# single-person -> the presence line uses the definite singular ("He's come
# in.") — a conclusion from the machine's own history, not a hardcoded fact;
# an exhibition's crowds flip the register back within hours.
PRESENCE_REID_ENABLED = False
PRESENCE_REID_THRESHOLD = 0.80  # cosine similarity to count as the same person; tune against debug/test_presence_reid.py
PRESENCE_REID_SAMPLE_INTERVAL = 30.0  # seconds between gallery samples while someone is visible
PRESENCE_REID_GALLERY_SIZE = 24  # rolling embeddings kept (~12 min of presence at the sample interval)
PRESENCE_REID_MAX_AGE = 21600.0  # seconds; gallery entries older than this (6h) are pruned — "same session" has an edge

# Interiority rhythm: every Nth caption (when nothing salient is happening) the
# machine THINKS WITHOUT LOOKING — the image is dropped so a vision model can't
# just re-describe the room, and the monologue turns inward (itself, this place,
# its drawings, why it's here) instead of cataloguing objects. The external
# observation stream was "completely external"; this weaves in depth. 0 = off.
INTROSPECT_INTERVAL = 4

# Identity dosing (Aug 22): the self-description + durable ledger used to ride
# EVERY caption, which turned identity into a script — "I invent imaginary
# critics" read 180 times a night elicits invented critics, which the distiller
# then re-confirms off the machine's own echo. Identity is memory, not standing
# instruction: introspective/awakening beats always carry it; every other mode
# sees it every Nth caption. 0 = every call (the old behavior).
IDENTITY_EVERY_N_CAPTIONS = int(os.getenv("IDENTITY_EVERY_N_CAPTIONS", 6))

# Standing desire dose (Aug 22, P4): while a desire persists unresolved, it
# re-surfaces in the monologue every Nth quiet caption (offset +3 from the
# identity dose so interior lines don't stack). The 3-injection burst after a
# desire change is unchanged. 0 = burst only (the pre-Aug-22 behavior).
DESIRE_REDOSE_EVERY_N = int(os.getenv("DESIRE_REDOSE_EVERY_N", 8))

# Quiet elicitation dose (Aug 28 evening, probe-validated): with the seam
# present, quiet cycles carried NO question at all — and the machine's
# wonder/wish register measured literal zero ("?" in 0/59 captions) while a
# single invitation line flipped the probe output to fear/want/intention
# immediately (sampling was exonerated: freeing it changed nothing). Every
# Nth quiet seamful cycle now carries one rotating elicitation — wondering /
# feeling / wanting — kind-naming only, never content (north-star P2). The
# Aug 22 suppression rationale (a question EVERY call fragments the thread)
# is answered by the dose, not by abolition. 0 = never (the Aug 22 rule).
QUIET_ELICIT_EVERY_N = int(os.getenv("QUIET_ELICIT_EVERY_N", 5))

# Unchanged-ness as fact (B4, Aug 31) — boredom's text channel. After this
# long with no episodic change (arrival, departure, drawing, new sighting),
# the caption prompt states the duration as a plain fact; re-stated at most
# every MIN_GAP so a standing fact never becomes the scene.
UNCHANGED_FACT_AFTER_S = float(os.getenv("UNCHANGED_FACT_AFTER_S", 1200))
UNCHANGED_FACT_MIN_GAP_S = float(os.getenv("UNCHANGED_FACT_MIN_GAP_S", 600))

# B3 want ledger (Aug 31) — the want's lifecycle as recorded fact.
# The desire line grows its arc tail (age + refusal count) once the want is
# this old OR has any refusals; the reflection prompt receives the standing
# want as an explicit fact once it has lived this long unanswered.
WANT_ARC_TAIL_AFTER_S = float(os.getenv("WANT_ARC_TAIL_AFTER_S", 21600))
WANT_REFLECTION_FACT_AFTER_S = float(os.getenv("WANT_REFLECTION_FACT_AFTER_S", 86400))

# The drift turn (Sep 3 — interiority as population, not residue; rework of
# the Sep 2 story beat). Any quiet cycle can become a drift turn: no image,
# the stream as its only seed, hot temperature, output entering the stream but
# firewalled from every fact ledger. Chosen per cycle by a standing
# probability scaled by the boredom scalar, never by a stillness clock — the
# story beat's deep-stillness trigger (45 min unchanged) required solitude
# that doesn't occur under the no-overnight doctrine; it fired once, ever.
# p = DRIFT_BASE_P * (1 + DRIFT_BOREDOM_GAIN * boredom): 0.05 calm, 0.15 at a
# pegged scalar. Measured boredom in quiet runs sits 0.5-0.9 (medians 0.58 /
# 0.98 on the last two measurable runs), so quiet evenings drift at ~10-15% of
# cycles — with reflection kernels (~1%) alongside, that is the ~15-20%
# thought-shaped stream share the Sep 3 handover targets. Measure with
# debug/drift_share.py. The material-seeded deep variant (want + episodic
# lines) lives in git history (Sep 2) pending the artist's fork ruling.
DRIFT_ENABLED = os.getenv("DRIFT_ENABLED", "true").lower() in ("true", "1", "yes")
DRIFT_BASE_P = float(os.getenv("DRIFT_BASE_P", 0.05))
DRIFT_BOREDOM_GAIN = float(os.getenv("DRIFT_BOREDOM_GAIN", 2.0))
DRIFT_TEMP = float(os.getenv("DRIFT_TEMP", 0.95))
DRIFT_NUM_PREDICT = int(os.getenv("DRIFT_NUM_PREDICT", 120))
# Eyes open (artist's call, same-day probe debug/probe_drift_image_ab.py):
# the blind arm narrated phantom present-tense perception (invented visitor
# action, "the foam finger in my hand"); the sighted arm stayed honest about
# the present and drifted on top of it. The ask lands after the image, so
# generation still answers the ask. False = the blind A/B arm.
DRIFT_SEND_IMAGE = os.getenv("DRIFT_SEND_IMAGE", "true").lower() in ("true", "1", "yes")
# INTROSPECTION ROUND (Sep 5, artist: "step out of its immediate patterns, think in a
# wider scope, question and wonder — the early system went from the dog to how dogs
# regulate temperature to how art and technology connect"). The drift becomes a
# WANDER: hops seeded by the last hop's own words plus a rotating scope move.
WANDER_ENABLED = os.getenv("WANDER_ENABLED", "true").lower() in ("true", "1", "yes")
WANDER_HOPS = int(os.getenv("WANDER_HOPS", 3))  # the drift + this many further hops minus one
WANDER_HOP_NUM_PREDICT = int(os.getenv("WANDER_HOP_NUM_PREDICT", 70))
WANDER_HOP_HISTORY = int(os.getenv("WANDER_HOP_HISTORY", 4))  # stream lines a hop sees — the seed is in the ask; twenty room lines pulled it back
WANDER_AFTER_LOOP_MULT = float(os.getenv("WANDER_AFTER_LOOP_MULT", 3.0))  # drift odds × this right after a loop notice
WANDER_AFTER_LOOP_S = float(os.getenv("WANDER_AFTER_LOOP_S", 180))
NAME_INVITE_EVERY_S = float(os.getenv("NAME_INVITE_EVERY_S", 86400))  # the yourself-reflection invites a name once a day while none stands
IDENTITY_DOSE_ALL_MODES = os.getenv("IDENTITY_DOSE_ALL_MODES", "true").lower() in ("true", "1", "yes")  # introspective dosed every N like other modes

# The lore ledger (Sep 3 evening — the re-entry round, docs/re-entry-round-
# sep3.md). Artist ruling: inventive self-fiction was never the issue —
# names, object mythologies, self-stories are WANTED; only world-state stays
# provenance-gated. Clean drift output lands in a marked reverie store; the
# existing reflection reads it; the existing distill harvests NAME/LORE
# ("or none" — structure only, nothing scheduled); durable threads re-enter
# as a dosed arc-line, the identity dose carries the name, and ~1/3 of
# drifts open from an alive thread (the deep-story fork resolved: the
# material-seeded variant is now the lore-seeded variant).
# THE DYNAMIC FRAME (Sep 4 evening — the artist's diagnosis: the basin was
# our stance-free architecture, not the model's weights. "Act angry" works
# because it's frame-level; every call here carried an identical stance-free
# frame, so the model regressed to its modal register. The register audit
# killed stance REPETITION and took stance VARIATION with it.) Two returns:
# the felt phrase (machine's own words, lease-gated) rides the SYSTEM frame
# and changes with real state; arousal reaches the voice's sampling (drained
# = cooler/shorter, stirred = hotter/more room). Both A/B-revertable.
FELT_FRAME_ENABLED = os.getenv("FELT_FRAME_ENABLED", "true").lower() in ("true", "1", "yes")
FELT_SAMPLING_ENABLED = os.getenv("FELT_SAMPLING_ENABLED", "true").lower() in ("true", "1", "yes")
AROUSAL_TEMP_SPAN = float(
    os.getenv("AROUSAL_TEMP_SPAN", 0.35)
)  # temp swing across arousal 0->1 (±0.175 — widened Sep 4 late: ±0.1 was a whisper; drained ~0.72, stirred ~1.0 capped)

# The attention round (Sep 4 — docs/attention-round-sep4.md). Investigate
# glances: the gaze sometimes commits to a FAMILIAR STRANGER — a registry
# entry with many sightings the detector never got sure of (wall lamp: 783k
# hits, conf 0.20) — instead of redistributing attention among the settled.
# The cycle then carries the attested fact ("seen it many times without ever
# being sure of it") and the close look accepts the glance; what-is-that is
# the machine's move. Per-term 15-min cooldown lives in the registry.
INVESTIGATE_WEIGHT = float(os.getenv("INVESTIGATE_WEIGHT", 0.25))
INVESTIGATE_CONF_CEILING = float(os.getenv("INVESTIGATE_CONF_CEILING", 0.35))
INVESTIGATE_MIN_HITS = int(os.getenv("INVESTIGATE_MIN_HITS", 500))
# Open questions: the distiller harvests a question the reflection is still
# carrying ("or none" — harvest, never invitation); questions persist in the
# lore ledger and re-enter as a dosed line in the memory-surface rotation.
# Wonders finally outlive the stream window ("wonder what he's working on"
# used to evaporate in 20 minutes).
QUESTION_LINE_EVERY_N = int(os.getenv("QUESTION_LINE_EVERY_N", 5))
QUESTIONS_MAX = 8

# The emotional arc channel (Sep 4 — feeling gets the want-ledger treatment,
# third application of the proven shape). Every mood read joins a session
# trajectory (felt_history); the arc line states the trajectory as FACT in
# the machine's OWN felt words — "You've felt X, or near it, for an hour" /
# "Earlier you felt X. More recently: Y" — never a scripted affect. Dosed
# like the unchanged fact; a live moment displaces it. The reflection's
# yourself/time organs receive the day's trajectory, so the identity engine
# finally distills from days it actually FELT (it never had before).
FELT_ARC_ENABLED = os.getenv("FELT_ARC_ENABLED", "true").lower() in ("true", "1", "yes")
FELT_HISTORY_MAX = 120
FELT_ARC_AFTER_S = float(os.getenv("FELT_ARC_AFTER_S", 3600))  # steady variant: same tenor held this long
FELT_ARC_MIN_GAP_S = float(os.getenv("FELT_ARC_MIN_GAP_S", 1800))

LORE_ENABLED = os.getenv("LORE_ENABLED", "true").lower() in ("true", "1", "yes")
LORE_REVERIES_MAX = 40
LORE_THREADS_MAX = 6
LORE_SEED_P = float(os.getenv("LORE_SEED_P", 0.33))
LORE_LINE_EVERY_N = int(os.getenv("LORE_LINE_EVERY_N", 4))  # the lore line's internal pacing inside the memory-surface rotation

# Reflection-echo pacing (Aug 28 evening). Aug 22 removed this source's
# internal counter "because the rotation slot rations" — but rotation only
# picks who goes FIRST, and with 180+ reflections stored a relevance match
# always exists, so reflection echo won the memory slot nearly every quiet
# caption (a standing "something you worked out..." = the identity-dose
# lesson re-learned). Fire at most every Nth invocation; declining falls
# through to familiarity/drawing echo, which diversifies the window.
REFLECTION_ECHO_EVERY_N = int(os.getenv("REFLECTION_ECHO_EVERY_N", 3))

# Relational elicitation dose (Aug 25): "What do you make of them being here?"
# used to ride EVERY relational caption — with someone working in the room it
# was the only standing question the machine heard, re-anchoring every turn
# onto the person. Same law as the identity dose: the question fires on
# presence/salience ONSET (arrival, fresh eye contact) and every Nth
# relational caption after that. 0 = onset only.
RELATIONAL_ELICIT_EVERY_N = int(os.getenv("RELATIONAL_ELICIT_EVERY_N", 8))

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
# REST (Sep 4): a real pause. Fires only when the quiet is WORLD-VERIFIED
# (pose-referee confirms, WORLD_STILL_MIN_CONFIRMS) and the body reads
# drained (arousal < 0.25) — thought slows when nothing pulls and nothing
# stirs. Salience snaps back to 4s instantly. The feed finally breathes.
CAPTION_INTERVAL_REST = float(os.getenv("CAPTION_INTERVAL_REST", 28))
CAPTION_QUIET_AFTER = 120  # seconds without salience before the cadence stretches
CAPTION_INTERVAL_REST_MAX = float(os.getenv("CAPTION_INTERVAL_REST_MAX", 120))  # rest deepens one rung per unchanged hour (Sep 5)

# TIME-AND-LOOP ROUND (Sep 5, docs/time-and-loop-round-sep5.md): a still room is
# not an absence of events — the passage of time is one, and catching yourself
# looping is another. Duration edges fire once per threshold of world-verified
# stillness; loop notices ride when the echo gates have refused the same run
# several times, or when the compressor names a circling phrase.
DURATION_EDGE_THRESHOLDS_MIN = [int(x) for x in os.getenv("DURATION_EDGE_THRESHOLDS_MIN", "30,60,120,240,480").split(",")]
LOOP_NOTICE_AFTER = int(os.getenv("LOOP_NOTICE_AFTER", 3))  # echo-gate refusals of a shared run within LOOP_NOTICE_WINDOW_S
LOOP_NOTICE_WINDOW_S = int(os.getenv("LOOP_NOTICE_WINDOW_S", 600))
LOOP_NOTICE_COOLDOWN_S = int(os.getenv("LOOP_NOTICE_COOLDOWN_S", 600))
# Persona baseline (Sep 5): once a day the stores are consolidated into a few
# first-person sentences the awakening and the reflection read back.
PERSONA_CONSOLIDATE_ENABLED = os.getenv("PERSONA_CONSOLIDATE_ENABLED", "true").lower() == "true"
PERSONA_CONSOLIDATE_EVERY_S = float(os.getenv("PERSONA_CONSOLIDATE_EVERY_S", 20 * 3600))
# Body as facts (Sep 5, agency round): the machine's own posture, not a borrowed one.
BODY_HOLD_THRESHOLDS_MIN = [int(x) for x in os.getenv("BODY_HOLD_THRESHOLDS_MIN", "3,10,30,60").split(",")]
HEAD_HOLD_TOL_DEG = float(os.getenv("HEAD_HOLD_TOL_DEG", 20.0))
# DECISION SLOTS (Sep 5, agency round — the RC-car loop): on quiet cycles the
# caption ends with LOOK / EXPECT in the machine's own words; LOOK is executed
# by the gaze as a "chosen" glance, the next turn states the consequence and,
# once the view settles, whether the expectation held (pose referee).
DECIDE_ENABLED = os.getenv("DECIDE_ENABLED", "true").lower() == "true"
DECIDE_EVERY_N = int(os.getenv("DECIDE_EVERY_N", 3))  # quiet captions between asks
DECIDE_EXTRA_TOKENS = int(os.getenv("DECIDE_EXTRA_TOKENS", 30))  # room for the two lines
DECIDE_SETTLE_S = float(os.getenv("DECIDE_SETTLE_S", 2.5))  # after the glance starts, before the check
CHOSEN_GLANCE_DWELL_MULT = float(os.getenv("CHOSEN_GLANCE_DWELL_MULT", 1.4))

# FELT LOOP (Sep 5, artist: the felt words were tactile — "warm, blurry" — because the
# ask said "how IT feels"; and they never directed the cadence: drained vs stirred
# captions differed by nothing but a little length). The ask now asks how YOU feel;
# and the felt state drives the manner mechanically, without words: arousal →
# cadence interval, token budget and short-beat odds; valence → which kind of
# thought the quiet elicitation invites. Measured per run by caption_metrics
# ("by_felt"). Temperature already followed arousal (AROUSAL_TEMP_SPAN).
FELT_LOOP_ENABLED = os.getenv("FELT_LOOP_ENABLED", "true").lower() == "true"
FELT_CADENCE_MULT_DRAINED = float(os.getenv("FELT_CADENCE_MULT_DRAINED", 1.6))  # quiet interval × this at arousal 0.1
FELT_CADENCE_MULT_CHARGED = float(os.getenv("FELT_CADENCE_MULT_CHARGED", 0.6))  # … at arousal 0.8
FELT_BUDGET_SCALE_DRAINED = float(os.getenv("FELT_BUDGET_SCALE_DRAINED", 0.7))
FELT_BUDGET_SCALE_CHARGED = float(os.getenv("FELT_BUDGET_SCALE_CHARGED", 1.4))
FELT_SHORT_BEAT_DELTA_DRAINED = float(os.getenv("FELT_SHORT_BEAT_DELTA_DRAINED", 0.15))
FELT_SHORT_BEAT_DELTA_CHARGED = float(os.getenv("FELT_SHORT_BEAT_DELTA_CHARGED", -0.10))
FELT_VALENCE_LEAN = float(os.getenv("FELT_VALENCE_LEAN", 0.25))  # |valence| beyond this leans the quiet elicitation kind

# Reflection loop (captioner/reflection.py) — the minutes-to-hours timescale
REFLECTION_LOOP_INTERVAL = 1200  # seconds between long-form reflections (~20 min); fires when the scene is quiet
REFLECTION_NUM_PREDICT = 320  # was 600 (padded to the brim, purple survey), then 220; raised for the dream (July 12) — the reflection now digests the raw hour of thought, which earns more room than a summary-of-summaries did. Brevity pressure IS register pressure; watch for padding at 320.

# Drawing trigger guardrails (the DECISION is the desire/drive system in
# drawing/drawing.py — DRAWING_TRIGGER_MODE + DRAWING_HUNGER_S + DRIVE_* are
# env knobs there. The old scoring formula and its weights were deleted in
# the Aug 19 consolidation; git history keeps them.)
DRAWING_INTERVAL = 300  # seconds between trigger evaluations
DRAWING_COOLDOWN = 720  # conception cooldown (prompt-stacking protection)
DRAWING_STARTUP_DELAY = 180  # Minimum seconds to wait after startup before first drawing (3 min for full init)
DRAWING_MIN_INTERVAL = 900  # hard floor between drawings (desire mode; drive mode has none)

# Drawing scale target — vpype layout dimensions for the centerline SVG.
# The warp transform maps this to the physical quad (~70x38mm).
# Larger = more detail but more distortion at edges. Tune empirically.
DRAWING_SCALE_TARGET = "50x50mm"
# Fragment-merge tolerance for the SVG→G-code conversion (see the Sep 2 note
# in grbl_utils.convert_with_vpype). 0.3 = measured safe default (halves pen
# plunges, doubles median stroke, no visible welding). 0.5 = aggressive:
# adjacent hatch lines weld into zigzag scribble — an aesthetic choice.
GRBL_LINEMERGE_TOLERANCE_MM = float(os.getenv("GRBL_LINEMERGE_TOLERANCE_MM", 0.3))

# === OBJECT DETECTION ===
YOLO_CONFIDENCE_THRESHOLD = 0.55  # Raised to 0.55 to avoid detecting hands/arms as person
# Aware-churn fixes (Aug 10): the gaze flickered aware/idle every few seconds
# on marginal person hits ("person detected but no tracking target"), and each
# 90s-spaced flicker minted an episodic "someone arrived". Three levers, in
# preference order over raising confidence (which worsens the known
# seated-still-person misses):
YOLO_PERSON_MIN_AREA_FRAC = 0.008  # min person bbox area as fraction of frame (~60x120px at 720p); phantom persons are small, real ones aren't
AWARE_ENTRY_CONFIRM_S = (
    2.0  # person must be continuously detected this long before idle->aware (2 idle-cadence YOLO passes; one-frame phantoms can't trigger)
)
# yolov8m (July 10 eval, debug/compare_yolo_models.py): rejects the desk
# mannequin head that nano fired on constantly, and finds still/seated people
# nano missed for whole stretches. Known remaining false positive: the
# life-size sweater doll — human enough to fool anything short of the LLM.
# Inference ~12ms, so the 0.1s tracking cadence is unaffected.
# Aug 25: yolo11m-pose — same family, same track() API, plus 17 COCO
# keypoints per person. A person-verdict now needs a COHERENT SKELETON
# (see the gate below): a mannequin head is head-keypoints only and fails;
# the box+keypoints flow is otherwise identical. Set YOLO_MODEL_PATH to
# yolov8m.pt to fall back (the gate passes everything without keypoints).
YOLO_MODEL_PATH = os.getenv(
    "YOLO_MODEL_PATH",
    os.path.join(MODEL_PATH, "yolo11m-pose.pt"),
)
# Skeleton coherence gate: person iff >= MIN_KEYPOINTS confident keypoints
# spread over >= MIN_REGIONS of the three body regions (head / torso / limbs).
# Structural evidence, not appearance: a head alone is not a body.
YOLO_SKELETON_KP_CONF = 0.5  # per-keypoint confidence to count as present
YOLO_SKELETON_MIN_KEYPOINTS = 5
YOLO_SKELETON_MIN_REGIONS = 2
YOLO_INTERVAL_IDLE = 1.5  # detection cadence with nobody around — fast enough to catch arrivals
YOLO_INTERVAL_TRACKING = 0.1  # cadence while a person is present — keeps bbox fresh under camera motion

CAMERA_INDEX = 0  # or whichever index your camera uses

# === ROOM CAM (dashboard) ===
# Second USB webcam owned by dashboard/server.py (NOT machine.py) for the
# remote room view. Use a /dev/v4l/by-id/... path, never a bare index —
# indices drift on replug. Empty string = no room cam (dashboard shows a
# placeholder). MJPG fourcc is forced to keep two cams within USB bandwidth.
CAMERA_2_DEVICE = os.getenv("CAMERA_2_DEVICE", "/dev/v4l/by-id/usb-XIFT_Web_Camera_20241217.1817-video-index0")
CAMERA_2_WIDTH = int(os.getenv("CAMERA_2_WIDTH", "640"))
CAMERA_2_HEIGHT = int(os.getenv("CAMERA_2_HEIGHT", "480"))
CAMERA_2_FPS = int(os.getenv("CAMERA_2_FPS", "15"))

# === CAMERA RESOLUTION ===
CAMERA_WIDTH = 1280  # 720p for smooth 30fps live feed
CAMERA_HEIGHT = 720  # LLM snapshots use this resolution

# === CAMERA IMAGE QUALITY ===
CAMERA_SHARPNESS = -1  # Sharpness (0-100, -1 for auto/default)
CAMERA_SATURATION = -1  # Color saturation (-1 for auto/default)
CAMERA_CONTRAST = -1  # Contrast (-1 for auto/default)
CAMERA_BRIGHTNESS = -1  # Brightness (-1 for auto/default)
CAMERA_EXPOSURE = -1  # Exposure (-1 for auto, or manual value)
CAMERA_AUTO_FOCUS = True  # Enable autofocus if available (machine.py camera setup)
CAMERA_AUTO_FOCUS = True  # Enable autofocus if available

# === LLM CALL SETTINGS ===
LLM_TIMEOUT_EVAL = 90
LLM_TIMEOUT_REFLECTION = 120  # Timeout for reflection/reasoning calls
LLM_SHOW_PROGRESS = False  # Show animated progress bar during LLM calls

# === CAPTIONING TEMPERATURE SETTINGS ===
# Control creativity and expressiveness in different types of responses
DRAWING_TEMPERATURE = float(os.getenv("DRAWING_TEMPERATURE", 1.0))  # Drawing prompts (lowered from 1.2 for Qwen's higher base entropy)
# Stocktake beat (Aug 10 2026): before the intent call, the machine reads its
# whole executed ledger + retrieved reflections and writes a short first-person
# note on where the work is going; the note is stored (memory type
# "drawing_direction") and read back next time. One extra LLM call per drawing.
DRAWING_REVIEW_ENABLED = os.getenv("DRAWING_REVIEW_ENABLED", "true").lower() in ("1", "true", "yes")
REFLECTION_TEMPERATURE = float(
    os.getenv("REFLECTION_TEMPERATURE", 0.75)
)  # Long-form reflection loop — stored output, keep it grounded (Qwen drifts ornate at higher temps)

# === OUTPUT SETTINGS ===
# Control which log types are printed to console
# LOG_TYPES_TO_PRINT = ["caption", "reflection", "comfy_prompt", "decision", "mood_update", "new_drawing"]
# To see debug information, add "debug" to LOG_TYPES_TO_PRINT
LOG_TYPES_TO_PRINT = ["caption", "reflection", "decision", "comfy_prompt", "new_drawing", "debug"]
CLEAN_LLM_OUTPUT = False  # Print only LLM response text without metadata prefixes
PRINT_CLEAN_CAPTIONS = True  # Suppress verbose runtime messages, show only LLM captions

DEBUG_HAND_CONTROLLER = False  # enable hand controller debug output
DEBUG_REACTIVITY_PAUSE = False  # show reactivity pause debug messages
DEBUG_LLM_PROMPTS = True  # print full prompts alongside LLM call logs
LLM_PRINT_FULL_RESPONSE = True  # print full responses in console output (ignores truncation)

# === REACTIVITY PAUSE SYSTEM ===
REACTIVITY_PAUSE_THRESHOLD = 0.30  # Activity level to trigger pause
REACTIVITY_PAUSE_DURATION = 4.0  # Seconds to pause Markov generation
REACTIVITY_PAUSE_COOLDOWN = 10.0  # Seconds between pause triggers

# === DRAWING MEMORY SETTINGS ===
# Store concise summaries of drawing intents and reflections for future prompts
INCLUDE_DRAWING_HISTORY = True

# Drawing prompt pipeline: stream only (stocktake → intent → render; see
# prompts.stream_drawing_analysis). The 5-step committee, kept "for A/B"
# since July 10 and never A/B'd, was deleted in the Aug 19 consolidation.

# === PAPER DETECTION SAFETY SYSTEM ===
# Prevent drawing on bare surfaces by checking for paper before execution
ENABLE_PAPER_DETECTION = True  # Master toggle for paper detection safety
PAPER_DETECTION_GAZE_PAN = 80  # Pan angle for looking down at drawing area (adjusted further left for better centering)
PAPER_DETECTION_GAZE_TILT = 65  # Tilt angle for looking down at drawing area (low enough to see ArUco marker)
ALLOW_PAPER_DETECTION_OVERRIDE = True  # Allow manual override when paper check fails

# Which eye judges the paper (Aug 20). "vlm": the loaded model looks at the
# table and only a seemingly BLANK sheet allows drawing — bare surface,
# clutter, or an already-drawn-on sheet all block, and any model failure
# fails CLOSED (no draw). "aruco": legacy marker search — marker occluded
# by ANYTHING reads as paper, and errors fail OPEN (bench test Aug 20:
# aruco false-allowed on a bare table the model called correctly 3/3).
PAPER_CHECK_METHOD = "vlm"
PAPER_VLM_FRAMES = 2  # frames per check; every frame must read as a blank sheet to allow
PAPER_VLM_SETTLE_S = 4.0  # gaze travel time before the first frame (live gaze eases; 1.5s shot frame 1 mid-travel)
# How long a check verdict may keep speaking in the monologue ("no paper on
# the desk" / "the sheet already carries a drawing"). The table can change
# without the machine looking; past this the claim would be memory posing as
# present-tense truth.
PAPER_STATE_TTL_S = 1800

# Conservative rollout: only run paper check after GRBL homing when explicitly enabled.
# ArUco detection is fast and reliable - safe to enable for post-home check
ENABLE_POST_HOME_PAPER_CHECK = True
# Early paper check: run ArUco check BEFORE ComfyUI generation to save resources
# This is in addition to the post-home check (double verification)
ENABLE_EARLY_PAPER_CHECK = True
# PAPER GLANCE (Sep 5): the sheet was checked only on the way to a drawing, so in
# low-energy nothing ever checked it — the dashboard showed the boot default
# "present" with no paper on the desk and the voice imagined a blank sheet. A
# gaze-only look at the table (camera + ArUco/VLM, no CNC) shortly after boot and
# every PAPER_GLANCE_EVERY_S while quiet; the verdict is what the drawing path and
# the "No paper" line already read, and it now survives a restart.
PAPER_GLANCE_ENABLED = os.getenv("PAPER_GLANCE_ENABLED", "true").lower() == "true"
PAPER_GLANCE_FIRST_AFTER_S = float(os.getenv("PAPER_GLANCE_FIRST_AFTER_S", 120))
PAPER_GLANCE_EVERY_S = float(os.getenv("PAPER_GLANCE_EVERY_S", 1800))
# DRAWING LINE (Sep 5, artist: "does it need this context every turn? maybe not"):
# the executed-drawings line is condensed, named as drawings, and dosed.
DRAWING_LINE_EVERY_N = int(os.getenv("DRAWING_LINE_EVERY_N", 8))
DRAWING_LINE_AFTER_DRAW_S = float(os.getenv("DRAWING_LINE_AFTER_DRAW_S", 900))
DRAWING_LINE_TITLE_CHARS = int(os.getenv("DRAWING_LINE_TITLE_CHARS", 45))
# Use the same tilt as drawing lock for detection (aligns view)

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

# 5. Additional devices can be added here
# CUSTOM_DEVICE_PORT = "/dev/ttyUSB5"

# ── MIND MODE (Sep 5 evening; docs/architecture-diagnosis-sep5.md, docs/mind-mode-sep5.md) ──
# The mind is a conversation with itself over a life, not a log of its own
# stamped sentences. STREAM_MODE == "mind" routes the caption cycle through
# captioner/mind.py: LOOK turns (frame + what changed + what's known to be in
# view) and THINK turns (no frame; the clock as the only new input; now and
# then one chosen, dated memory). The self-indictment channel is retired in
# this mode: no NEW ABOUT ME self-notes, no trait promotion, no self/durable
# block in the frame.
MIND_TURNS = int(os.getenv("MIND_TURNS", 6))  # prior thoughts riding as real turns
MIND_TURN_MAX_AGE_S = int(os.getenv("MIND_TURN_MAX_AGE_S", 7200))  # older thoughts leave the turns (the life block still remembers)
MIND_THINK_INTERVAL_S = float(os.getenv("MIND_THINK_INTERVAL_S", 60))  # cadence at rest, × felt_loop.cadence_mult
MIND_LOOK_EVERY_S = float(os.getenv("MIND_LOOK_EVERY_S", 300))  # a periodic look when nothing pulls
MIND_LOOK_MIN_GAP_S = float(os.getenv("MIND_LOOK_MIN_GAP_S", 20))
MIND_MEMORY_EVERY_N = int(os.getenv("MIND_MEMORY_EVERY_N", 0))  # scheduled surfacing OFF (Sep 6 01:00, artist: "a memory every eighth turn is a clock word") — recall is by association now  # every Nth think turn, one dated memory surfaces
MIND_MEMORY_MIN_AGE_S = int(os.getenv("MIND_MEMORY_MIN_AGE_S", 3600))
MIND_NUM_PREDICT = int(os.getenv("MIND_NUM_PREDICT", 80))  # room for a short paragraph that completes (journal shape, Sep 6 morning)
MIND_SHORT_BEAT_P = float(os.getenv("MIND_SHORT_BEAT_P", 0.1))  # a short beat now and then; the frame already allows "nothing at all"
MIND_SHORT_BEAT_TOKENS = int(os.getenv("MIND_SHORT_BEAT_TOKENS", 22))  # 14 cut sentences mid-clause in the first live minutes
MIND_PIVOTS_BEFORE_NOTICE = int(os.getenv("MIND_PIVOTS_BEFORE_NOTICE", 3))  # reframes of one subject with no step → the machine hears it
MIND_POSITION_TTL_S = int(os.getenv("MIND_POSITION_TTL_S", 1800))  # a subject's position rides in the life block while this fresh
MIND_PAST_THOUGHTS = int(os.getenv("MIND_PAST_THOUGHTS", 0))  # Sep 6 01:00: quoted past thoughts in the standing block bred parroting and pulled the thread  # dated past thoughts in the life block
MIND_THREAD_MAX = int(os.getenv("MIND_THREAD_MAX", 4000))  # persisted thread length (event_log/mind_thread.json)
MIND_ROOM_TERMS = int(os.getenv("MIND_ROOM_TERMS", 8))  # registry terms named in the life block
MIND_VIEW_TOL_PAN = float(os.getenv("MIND_VIEW_TOL_PAN", 30))  # registry terms within this of the gaze are "in view" for the LOOK cue
MIND_VIEW_TOL_TILT = float(os.getenv("MIND_VIEW_TOL_TILT", 25))
MIND_VIEW_TERMS = int(os.getenv("MIND_VIEW_TERMS", 4))  # ranked by the machine's own familiarity (hits), not detector confidence — the mannequin head has conf≈0 and was dropped
MOTION_SETTLE_S = float(os.getenv("MOTION_SETTLE_S", 2.0))  # after the head moves, flow residual doesn't count as motion for this long (exposure/flare settling read as "something moved"; 0 disables)
MIND_ELICIT_EVERY_N = int(os.getenv("MIND_ELICIT_EVERY_N", 0))  # Sep 6 01:00: off — the chain is one move applied to the last sentence, nothing else  # every Nth THINK turn carries one elicit dose (wonder / feel / want), leaning by the felt loop's valence
MIND_LIFE_FULL = os.getenv("MIND_LIFE_FULL", "false").lower() in ("true", "1", "yes")  # positions / questions / belief in the standing life block (off since Sep 6 01:00 — attractors that compete with the thread)
MIND_RECALL_ENABLED = os.getenv("MIND_RECALL_ENABLED", "true").lower() in ("true", "1", "yes")  # association: the premise queries the "thoughts" ChromaDB collection
MIND_RECALL_MAX_DIST = float(os.getenv("MIND_RECALL_MAX_DIST", 0.4))  # 0.5 let a memory ride on 12 of 19 turns once the index was full (07:00 Sep 6)  # cosine distance ceiling for a past thought to surface
MIND_RECALL_COOLDOWN_S = int(os.getenv("MIND_RECALL_COOLDOWN_S", 3600))  # a memory surfaces at most once an hour
MIND_LIFE_BEFORE_MAX_AGE_S = int(os.getenv("MIND_LIFE_BEFORE_MAX_AGE_S", 2 * 86400))  # continuity quote: the last thought of the previous chain, if this recent
MIND_INDEX_RETRY_S = float(os.getenv("MIND_INDEX_RETRY_S", 60))  # the thoughts index never latches to "failed"
MIND_RECALL_MIN_GAP_S = int(os.getenv("MIND_RECALL_MIN_GAP_S", 480))  # at most one recall per this many seconds — a cap, not a schedule
MIND_SHAPE = os.getenv("MIND_SHAPE", "text")  # "text": the thread rides as ONE running text (journal pages) + the cue; "turns": user-cue/assistant-thought pairs (Sep 5 shape)
MIND_TEXT_ENTRIES = int(os.getenv("MIND_TEXT_ENTRIES", 10))  # entries in the running text (≤ MIND_TURN_MAX_AGE_S old)

# ── MOOD WITH DYNAMICS (Sep 6, utils/mood.py) ──
MOOD_ENABLED = os.getenv("MOOD_ENABLED", "true").lower() in ("true", "1", "yes")
MOOD_TAU_V_S = float(os.getenv("MOOD_TAU_V_S", 600))  # valence inertia
MOOD_TAU_A_S = float(os.getenv("MOOD_TAU_A_S", 300))  # arousal inertia
MOOD_FATIGUE_PER_H = float(os.getenv("MOOD_FATIGUE_PER_H", 0.03))  # arousal down per hour awake (cap 14 h)
MOOD_ALONE_PER_H = float(os.getenv("MOOD_ALONE_PER_H", 0.015))  # valence down per hour alone
MOOD_STILL_PER_H = float(os.getenv("MOOD_STILL_PER_H", 0.03))  # arousal down per hour unchanged (cap 6 h)
MOOD_NIGHT_AROUSAL = float(os.getenv("MOOD_NIGHT_AROUSAL", 0.1))  # 00:00–06:00
MOOD_REFUSAL_VALENCE = float(os.getenv("MOOD_REFUSAL_VALENCE", 0.08))  # per gate refusal in the last 10 min (cap 5)
MOOD_REFUSAL_AROUSAL = float(os.getenv("MOOD_REFUSAL_AROUSAL", 0.05))
MOOD_SETTLED_VALENCE = float(os.getenv("MOOD_SETTLED_VALENCE", 0.15))  # a reflection settled in the last 10 min
MOOD_SETTLED_AROUSAL = float(os.getenv("MOOD_SETTLED_AROUSAL", 0.1))
MOOD_SCARE_AROUSAL = float(os.getenv("MOOD_SCARE_AROUSAL", 0.3))  # a phantom / motion onset: arousal jumps
MOOD_SCARE_HOLD_S = float(os.getenv("MOOD_SCARE_HOLD_S", 300))
MOOD_PRESENCE_AROUSAL = float(os.getenv("MOOD_PRESENCE_AROUSAL", 0.2))
MOOD_PRESENCE_VALENCE = float(os.getenv("MOOD_PRESENCE_VALENCE", 0.1))
MOOD_FELT_HELD_MIN_S = float(os.getenv("MOOD_FELT_HELD_MIN_S", 1800))  # the frame says how long the felt word has held after this
MOOD_CADENCE_MAP = {}  # per-label overrides of utils/mood._DEFAULT_MAP, e.g. {"flat": {"interval_mult": 2.2}} — the malleable part
