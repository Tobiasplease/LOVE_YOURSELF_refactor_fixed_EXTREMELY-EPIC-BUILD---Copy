import random
import time
import math
import threading
import numpy as np

from config.config import (
    DEAD_ZONE,
    EASING_FACTOR,
    FACE_STABLE_TIMEOUT,
    FLIP_X,
    FLIP_Y,
    IDLE_AMPLITUDE_X,
    IDLE_AMPLITUDE_Y,
    IDLE_CENTER_X,
    IDLE_CENTER_Y,
    IDLE_EASING,
    IDLE_PAUSE_MAX,
    IDLE_PAUSE_MIN,
    PAN_MIN,
    PAN_MAX,
    TILT_MIN,
    TILT_MAX,
    SWEEP_PROBABILITY,
)

# === VELOCITY LIMITING CONSTANTS ===
# Face tracking - fast response for immediate tracking
FACE_PAN_VELOCITY = 15.0   # Maximum degrees per update during face tracking
FACE_TILT_VELOCITY = 12.0  # Maximum degrees per update during face tracking
FACE_VELOCITY_SMOOTHING = 0.0  # No velocity smoothing for face tracking (immediate response)

# General movement - higher limits for responsive tracking
MAX_PAN_VELOCITY = 15.0   # Maximum degrees per update for pan servo
MAX_TILT_VELOCITY = 12.0  # Maximum degrees per update for tilt servo
VELOCITY_SMOOTHING = 0.3  # Less smoothing = faster response (0.0 = instant, 1.0 = never change)

# Dynamic pause timing for idle movement
IDLE_PAUSE_LONG = 6.0  # Occasional longer contemplative pauses
IDLE_LONG_PAUSE_CHANCE = 0.10  # 10% chance for a longer pause
IDLE_STILLNESS_MIN = 1.0  # Minimum stillness duration (seconds)
IDLE_STILLNESS_MAX = 3.0  # Maximum stillness duration (seconds)
IDLE_STILLNESS_CHANCE = 0.35  # 35% of movements end in stillness — more alive

# Override dead zone for precise face tracking
PRECISE_DEAD_ZONE = 0.2  # Ultra-small dead zone for highly responsive face tracking (overrides config DEAD_ZONE)

# === SIMPLIFIED STATE MANAGEMENT ===
servo_x = 90
servo_y = 90
target_x = 90
target_y = 90
last_seen_time = time.time() - 10  # Start as if we've been idle for 10 seconds
state = "idle"
idle_next_move_time = 0  # Trigger immediate movement
idle_in_stillness = False  # Whether we're in a stillness period (no target updates)
idle_stillness_until = 0  # Timestamp when stillness ends
startup_sequence_active = False  # Flag to prevent conflicts during startup
drawing_sequence_active = False  # Flag to prevent conflicts during CNC drawing

# === AWARE STATE BREAK SYSTEM ===
# After tracking someone for a while, look away naturally
aware_break_until = 0.0  # Timestamp when break ends (0 = not in break)
aware_break_target_pan = 90  # Where to look during break (away from person)
aware_break_target_tilt = 110
drawing_target_x = 90  # Target position for drawing mode
drawing_target_y = 90  # Target position for drawing mode
drawing_transition_active = False  # Flag for smooth transition to/from drawing position

# === ORGANIC MOVEMENT DECOUPLING ===
# Independent timing and curves for pan/tilt
pan_offset_time = random.uniform(0, 2.0)  # Random phase offset for pan
tilt_offset_time = random.uniform(0, 2.0)  # Random phase offset for tilt
pan_micro_target = 90  # Intermediate target for curved movement
tilt_micro_target = 90  # Intermediate target for curved movement
pan_easing_variance = 1.0  # Dynamic easing multiplier for pan
tilt_easing_variance = 1.0  # Dynamic easing multiplier for tilt

# === PERLIN NOISE FOR ORGANIC MOVEMENT ===
pan_noise_offset = random.uniform(0, 1000)  # Random seed for pan noise
tilt_noise_offset = random.uniform(0, 1000)  # Random seed for tilt noise
pan_frequency = 0.08  # Pan movement frequency (slower = more contemplative)
tilt_frequency = 0.06  # Tilt movement frequency (different from pan)

# === INDEPENDENT SERVO CONTROLLERS ===
pan_velocity = 0.0  # Current pan movement velocity
tilt_velocity = 0.0  # Current tilt movement velocity
pan_target_time = 0.0  # When pan reaches target
tilt_target_time = 0.0  # When tilt reaches target
last_state_change = 0.0  # Track state transitions for clean handoff
aware_minimum_dwell = 5.0  # Minimum seconds to stay in aware state before returning to idle

# === VELOCITY LIMITING ===
last_pan_velocity = 0.0  # Previous pan velocity for smoothing
last_tilt_velocity = 0.0  # Previous tilt velocity for smoothing

# === THREAD SAFETY ===
_gaze_lock = threading.Lock()


# === PHYSICS-BASED GAZE SIMULATION ===
class GazePhysicsState:
    """Spring-damper physics simulation for organic gaze movement."""

    def __init__(self):
        self.pan = 90.0
        self.tilt = 90.0
        self.pan_velocity = 0.0
        self.tilt_velocity = 0.0
        self.pan_target = 90.0
        self.tilt_target = 90.0
        self.mass = 1.0
        self.spring_constant = 8.0
        self.damping = 5.0
        self.tremor_amplitude = 0.2
        self.orbital_phase = 0.0
        self.orbital_strength = 0.2
        self.last_update_time = time.time()
        self.current_emotion = "calm_observant"
        self.param_blend_factor = 0.0

    def blend_params(self, target_params: dict, blend_rate: float = 0.05):
        """Smoothly blend physics parameters to avoid jarring transitions."""
        self.mass += (target_params["mass"] - self.mass) * blend_rate
        self.spring_constant += (target_params["spring_constant"] - self.spring_constant) * blend_rate
        self.damping += (target_params["damping"] - self.damping) * blend_rate
        self.tremor_amplitude += (target_params["tremor_amplitude"] - self.tremor_amplitude) * blend_rate
        self.orbital_strength += (target_params["orbital_strength"] - self.orbital_strength) * blend_rate


physics_state = GazePhysicsState()

PHYSICS_PATTERNS = {
    # Movement RANGE varies by emotional state - energized moves more, calm moves less
    # mass/spring/damping create the emotional feel: snappy vs languid
    # tremor/orbital scaled down for calmer states = less overall movement when idle
    "energized_engaged": {
        "mass": 0.3,           # Light - quick acceleration
        "spring_constant": 18.0,  # Strong pull - responsive
        "damping": 1.8,        # Low damping - bouncy, alive
        "tremor_amplitude": 1.4,  # High tremor - jittery life
        "orbital_strength": 1.3,  # Full wandering force
    },
    "alert_curious": {
        "mass": 0.45,          # Light-medium
        "spring_constant": 14.0,  # Good responsiveness
        "damping": 2.5,        # Some damping
        "tremor_amplitude": 1.1,  # Good tremor
        "orbital_strength": 1.2,  # Slightly less wandering
    },
    "calm_observant": {
        "mass": 0.6,           # Medium-light - responsive movement
        "spring_constant": 11.0,  # Moderate-strong pull
        "damping": 3.5,        # Less damping — more drift and life
        "tremor_amplitude": 0.9,  # Noticeable tremor — alive
        "orbital_strength": 1.3,  # Full wandering force
    },
    "quiet_detached": {
        "mass": 0.8,           # Medium - still moves
        "spring_constant": 7.0,   # Moderate pull
        "damping": 4.0,        # Moderate damping
        "tremor_amplitude": 0.6,  # Visible tremor
        "orbital_strength": 1.0,  # Still wanders
    },
    "withdrawn_distant": {
        "mass": 1.2,           # Heavy but not dead
        "spring_constant": 5.0,   # Weak-moderate pull
        "damping": 5.0,        # Damped but not frozen
        "tremor_amplitude": 0.4,  # Subtle tremor
        "orbital_strength": 0.7,  # Reduced but present wandering
    },
}

TRACKING_PHYSICS = {
    "mass": 0.1,             # Very light for immediate response
    "spring_constant": 45.0,  # Strong pull toward face target
    "damping": 2.5,          # Low damping — fast, responsive tracking
    "tremor_amplitude": 0.0,
    "orbital_strength": 0.0,
}


def update_physics_step(dt: float, is_tracking: bool = False, is_still: bool = False) -> tuple:
    """
    Core spring-damper physics update with orbital wandering and tremor.

    Physics model:
        F_spring = -k * (position - target)
        F_damping = -c * velocity
        F_total = F_spring + F_damping + F_orbital
        acceleration = F_total / mass
        velocity += acceleration * dt
        position += velocity * dt

    Args:
        dt: Time delta since last update
        is_tracking: True during face/person tracking (no tremor/orbital)
        is_still: True during stillness periods (dampen velocity, no tremor/orbital)

    Returns:
        (pan, tilt) as floats
    """
    global physics_state

    dt = min(dt, 0.1)
    if dt <= 0:
        return (physics_state.pan, physics_state.tilt)

    ps = physics_state
    now = time.time()

    for axis in ["pan", "tilt"]:
        pos = ps.pan if axis == "pan" else ps.tilt
        vel = ps.pan_velocity if axis == "pan" else ps.tilt_velocity
        target = ps.pan_target if axis == "pan" else ps.tilt_target
        bounds = (PAN_MIN, PAN_MAX) if axis == "pan" else (TILT_MIN, TILT_MAX)

        f_spring = -ps.spring_constant * (pos - target)
        f_damping = -ps.damping * vel

        f_orbital = 0.0
        # No orbital force during tracking or stillness
        if not is_tracking and not is_still and ps.orbital_strength > 0.01:
            orbit_freq = 0.28 if axis == "pan" else 0.22
            phase_offset = 0.0 if axis == "pan" else math.pi / 2
            f_orbital = ps.orbital_strength * 15.0 * math.sin(now * orbit_freq * 2 * math.pi + phase_offset)
            f_orbital += ps.orbital_strength * 8.0 * math.sin(now * orbit_freq * 1.6 * 2 * math.pi + phase_offset + 0.8)
            f_orbital += ps.orbital_strength * 4.0 * math.sin(now * orbit_freq * 2.4 * 2 * math.pi + phase_offset + 1.5)

        f_total = f_spring + f_damping + f_orbital
        acceleration = f_total / ps.mass

        # During stillness, apply extra damping to settle faster
        if is_still:
            vel *= 0.85  # Decay velocity faster during stillness

        vel += acceleration * dt
        max_vel = 25.0 if is_tracking else 12.0
        vel = max(-max_vel, min(max_vel, vel))
        pos += vel * dt

        tremor = 0.0
        # No tremor during tracking or stillness
        if not is_tracking and not is_still and ps.tremor_amplitude > 0.01:
            base_freq = 3.2 if axis == "pan" else 3.8
            tremor = ps.tremor_amplitude * 0.6 * math.sin(now * base_freq * 2 * math.pi)
            tremor += ps.tremor_amplitude * 0.35 * math.sin(now * base_freq * 1.5 * 2 * math.pi + 0.5)
            tremor += ps.tremor_amplitude * 0.25 * math.sin(now * base_freq * 2.1 * 2 * math.pi + 1.2)
            tremor += ps.tremor_amplitude * 0.4 * math.sin(now * 0.8 * 2 * math.pi + 2.3)
        pos += tremor

        pos = max(bounds[0], min(bounds[1], pos))

        if axis == "pan":
            ps.pan = pos
            ps.pan_velocity = vel
        else:
            ps.tilt = pos
            ps.tilt_velocity = vel

    ps.last_update_time = now
    return (ps.pan, ps.tilt)


# === LLM-DRIVEN GAZE NUDGE (MODIFIER MODE) ===
# Allows the model to influence idle gaze via caption content
# Acts as a decaying bias added to organic Perlin movement - preserves natural sway
pan_nudge = 0.0  # Current pan nudge offset (degrees)
tilt_nudge = 0.0  # Current tilt nudge offset (degrees)
nudge_start_time = 0.0  # When the nudge was applied
nudge_duration = 5.0  # How long nudge lasts before decaying (seconds)

# === SEARCHING STATE ===
# Goal-directed searching when person is "remembered" or LLM has interest points
searching_active = False
searching_target_pan = 90.0  # Current search target
searching_target_tilt = 90.0
searching_zones_to_visit = []  # Queue of zones to scan
searching_last_known_pan = None  # Where person was last seen
searching_last_known_tilt = None
searching_interest_points = []  # LLM-driven points of interest [(pan, tilt, priority, expiry_time), ...]
searching_current_goal = None  # "zone_scan", "last_known", "interest_point"
searching_goal_start_time = 0.0
searching_goal_dwell_time = 2.0  # How long to dwell at each search point

# === LLM-DIRECTED ZONE SYSTEM ===
# The LLM decides which zone to look at, organic movement happens within that zone

GAZE_ZONES_PAN = {
    "left": (105, PAN_MAX),      # 105-135° (left side - servo inverted)
    "ahead": (75, 105),          # 75-105° (center)
    "right": (PAN_MIN, 75),      # 45-75° (right side - servo inverted)
}

GAZE_ZONES_TILT = {
    "up": (125, TILT_MAX),       # 125-150° (looking up - servo inverted)
    "level": (95, 125),          # 95-125° (level/center)
    "down": (TILT_MIN, 95),      # 65-95° (looking down - servo inverted)
}

# Current LLM-directed zone
llm_target_zone_pan = "ahead"
llm_target_zone_tilt = "level"
llm_zone_active = False  # Whether LLM is actively directing gaze
llm_zone_set_time = 0.0  # When LLM zone was last set
llm_zone_timeout = 45.0  # LLM zones expire after 45 seconds, return to free wandering

# Tracking angle awareness
tracking_person_position = "center"  # left/center/right
tracking_person_last_x = None
tracking_person_movement = "stationary"


def set_llm_zone(pan_zone: str, tilt_zone: str = None):
    """Set the LLM-directed gaze zone. Camera will wander within this zone."""
    global llm_target_zone_pan, llm_target_zone_tilt, llm_zone_active, llm_zone_set_time
    global drawing_sequence_active, state

    # Don't change zones during drawing execution - gaze is locked to drawing surface
    if drawing_sequence_active:
        return

    # Don't change zones during tracking/aware - hardware tracking overrides LLM
    # LLM gaze commands only work during idle
    if state in ("tracking", "grace", "aware"):
        return

    if pan_zone == "person":
        # Special case: LLM wants to track the person (let tracking take over)
        llm_zone_active = False
        print(f"[👁️] LLM: Look at person (tracking mode)")
        return

    if pan_zone == "stay":
        # Keep current zone, don't change (but refresh timeout)
        llm_zone_set_time = time.time()
        print(f"[👁️] LLM: Stay in current zone ({llm_target_zone_pan}/{llm_target_zone_tilt})")
        return

    if pan_zone in GAZE_ZONES_PAN:
        llm_target_zone_pan = pan_zone
        llm_zone_active = True
        llm_zone_set_time = time.time()

    if tilt_zone and tilt_zone in GAZE_ZONES_TILT:
        llm_target_zone_tilt = tilt_zone
        llm_zone_set_time = time.time()
    elif pan_zone in ("up", "down", "level"):
        # If user passed tilt direction as pan_zone
        llm_target_zone_tilt = pan_zone
        llm_zone_active = True
        llm_zone_set_time = time.time()
    elif tilt_zone is None:
        llm_target_zone_tilt = "level"  # Default to level if not specified

    # Show the actual angle target for debugging
    pan_min, pan_max = GAZE_ZONES_PAN.get(llm_target_zone_pan, (75, 105))
    tilt_min, tilt_max = GAZE_ZONES_TILT.get(llm_target_zone_tilt, (95, 125))
    pan_center = (pan_min + pan_max) / 2
    tilt_center = (tilt_min + tilt_max) / 2
    print(f"[👁️] LLM zone: {llm_target_zone_pan}/{llm_target_zone_tilt} → target pan={pan_center:.0f}° tilt={tilt_center:.0f}°")


def update_tracking_awareness(face_box, frame_width: int):
    """Track where the person is relative to frame center."""
    global tracking_person_position, tracking_person_last_x, tracking_person_movement

    if face_box is None:
        return

    x1, y1, x2, y2 = face_box
    face_center_x = (x1 + x2) / 2
    frame_center_x = frame_width / 2

    # Relative position
    offset_ratio = (face_center_x - frame_center_x) / frame_center_x
    if offset_ratio < -0.2:
        tracking_person_position = "left"
    elif offset_ratio > 0.2:
        tracking_person_position = "right"
    else:
        tracking_person_position = "center"

    # Movement detection
    if tracking_person_last_x is not None:
        delta = face_center_x - tracking_person_last_x
        if delta > 20:
            tracking_person_movement = "moved_right"
        elif delta < -20:
            tracking_person_movement = "moved_left"
        else:
            tracking_person_movement = "stationary"
    tracking_person_last_x = face_center_x


def get_current_nudge() -> tuple:
    """Get current nudge values with decay applied."""
    global pan_nudge, tilt_nudge, nudge_start_time, nudge_duration

    if nudge_duration <= 0:
        return (0.0, 0.0)

    elapsed = time.time() - nudge_start_time
    if elapsed >= nudge_duration:
        return (0.0, 0.0)

    # Smooth decay using cosine curve (fast at start, slow at end)
    decay = 0.5 * (1 + math.cos(math.pi * elapsed / nudge_duration))
    return (pan_nudge * decay, tilt_nudge * decay)


def nudge_toward_concept(pan_zone: str = None, tilt_zone: str = None):
    """Nudge gaze toward a concept's known spatial location.

    Called when a concept with spatial metadata enters the machine's attention.
    Uses the existing nudge system — a decaying bias on idle movement, not a
    hard override. Tracking and drawing modes are unaffected.
    """
    global pan_nudge, tilt_nudge, nudge_start_time, nudge_duration
    global state, drawing_sequence_active

    # Don't nudge during tracking or drawing — only during idle
    if state in ("tracking", "grace", "aware") or drawing_sequence_active:
        return

    # Map zone names to nudge offsets (degrees from center)
    pan_offsets = {"left": 15.0, "right": -15.0, "ahead": 0.0}
    tilt_offsets = {"up": 12.0, "down": -12.0}

    new_pan = pan_offsets.get(pan_zone, 0.0)
    new_tilt = tilt_offsets.get(tilt_zone, 0.0)

    if new_pan == 0.0 and new_tilt == 0.0:
        return

    pan_nudge = new_pan
    tilt_nudge = new_tilt
    nudge_start_time = time.time()
    nudge_duration = 6.0

    direction = pan_zone or ""
    if tilt_zone:
        direction = f"{direction} {tilt_zone}".strip()
    print(f"[👁️] Nudge toward concept: {direction} (pan={new_pan:+.0f}° tilt={new_tilt:+.0f}°)")


# === SEARCHING STATE FUNCTIONS ===

def activate_search_mode(last_seen_pan: float = None, last_seen_tilt: float = None, zones_visited: set = None):
    """
    Activate searching state when person is "remembered" but not visible.
    Camera will systematically scan unvisited zones and check last known location.
    """
    global searching_active, searching_last_known_pan, searching_last_known_tilt
    global searching_zones_to_visit, searching_current_goal, searching_goal_start_time

    searching_active = True
    searching_last_known_pan = last_seen_pan
    searching_last_known_tilt = last_seen_tilt

    # Build queue of unvisited zones (0-5, each 30 degrees)
    # But ONLY include zones that are at least partially within servo range
    all_zones = set()
    for zone in range(6):
        zone_start = zone * 30
        zone_end = zone_start + 30
        # Include zone if it overlaps with servo range [PAN_MIN, PAN_MAX]
        if zone_end > PAN_MIN and zone_start < PAN_MAX:
            all_zones.add(zone)

    visited = zones_visited or set()
    unvisited = sorted(all_zones - visited)

    # Prioritize zones near last known location
    if last_seen_pan is not None:
        last_zone = int(clamp(last_seen_pan, PAN_MIN, PAN_MAX) // 30)
        # Sort by distance from last known zone
        unvisited.sort(key=lambda z: abs(z - last_zone))

    searching_zones_to_visit = unvisited
    searching_current_goal = None
    searching_goal_start_time = 0.0

    print(f"[🔍] Search mode activated - {len(unvisited)} zones to scan, last seen at pan={last_seen_pan}")


def deactivate_search_mode():
    """Deactivate searching state (person found or confirmed absent)."""
    global searching_active, searching_current_goal, searching_interest_points

    if searching_active:
        print("[🔍] Search mode deactivated")
    searching_active = False
    searching_current_goal = None
    # Keep interest points - they can persist beyond search mode


def get_search_target() -> tuple:
    """
    Get the next target for searching behavior.
    Returns (pan, tilt, goal_type) or (None, None, None) if no targets.

    Priority: interest_points > last_known_location (briefly) > zone_scan
    """
    global searching_zones_to_visit, searching_interest_points
    global searching_last_known_pan, searching_last_known_tilt
    global searching_goal_start_time, searching_current_goal

    now = time.time()

    # Clean expired interest points
    searching_interest_points = [
        p for p in searching_interest_points if p[3] > now
    ]

    # Priority 1: High-priority interest points (>0.7)
    high_priority = [p for p in searching_interest_points if p[2] > 0.7]
    if high_priority:
        point = high_priority[0]
        return (point[0], point[1], "interest_point")

    # Priority 2: Last known location - but only for first 4 seconds of search
    # After that, move on to zone scanning even if we haven't "reached" it
    if searching_last_known_pan is not None:
        # Time-based fallthrough: don't stay stuck on last_known forever
        time_on_last_known = now - searching_goal_start_time if searching_current_goal == "last_known" else 0
        if time_on_last_known < 4.0:
            clamped_pan = clamp(searching_last_known_pan, PAN_MIN, PAN_MAX)
            clamped_tilt = clamp(searching_last_known_tilt or 90, TILT_MIN, TILT_MAX)
            return (clamped_pan, clamped_tilt, "last_known")
        else:
            # Timed out - clear last_known and move to zone scanning
            print(f"[🔍] Last known location timed out after {time_on_last_known:.1f}s - moving to zone scan")
            searching_last_known_pan = None
            searching_last_known_tilt = None

    # Priority 3: Unvisited zones
    if searching_zones_to_visit:
        zone = searching_zones_to_visit[0]
        zone_pan = zone * 30 + 15  # Center of zone
        # CRITICAL: Clamp to actual servo limits so we don't get stuck trying to reach unreachable positions
        zone_pan = clamp(zone_pan, PAN_MIN, PAN_MAX)
        zone_tilt = (TILT_MIN + TILT_MAX) / 2  # Middle height
        return (zone_pan, zone_tilt, "zone_scan")

    # Priority 4: Lower-priority interest points
    if searching_interest_points:
        point = searching_interest_points[0]
        return (point[0], point[1], "interest_point")

    return (None, None, None)


def update_search_progress(current_pan: float, current_tilt: float, person_found: bool = False):
    """
    Update search progress based on current position.
    Call this from the main gaze update loop.
    """
    global searching_zones_to_visit, searching_current_goal, searching_goal_start_time
    global searching_last_known_pan, searching_interest_points

    if not searching_active:
        return

    now = time.time()

    # If person was found, clear the search
    if person_found:
        deactivate_search_mode()
        return

    # Check if we've reached current goal
    target_pan, target_tilt, goal_type = get_search_target()
    if target_pan is None:
        return

    # Track when we start targeting a new goal type
    if searching_current_goal != goal_type:
        searching_current_goal = goal_type
        searching_goal_start_time = now

    pan_diff = abs(current_pan - target_pan)
    tilt_diff = abs(current_tilt - (target_tilt or current_tilt))
    # More lenient at_target check - physics + tremor makes exact convergence hard
    at_target = pan_diff < 15 and tilt_diff < 15

    # Calculate time spent on current goal
    time_on_goal = now - searching_goal_start_time

    # Mark target as visited if: at target for dwell_time OR stuck too long (timeout)
    max_time_per_target = 6.0  # Don't spend more than 6 seconds on any target
    should_advance = False

    if at_target and time_on_goal >= searching_goal_dwell_time:
        should_advance = True
    elif time_on_goal >= max_time_per_target:
        # Timeout - move on even if we didn't reach target
        print(f"[🔍] Target timeout ({goal_type}) after {time_on_goal:.1f}s - moving to next")
        should_advance = True

    if should_advance:
        if goal_type == "zone_scan" and searching_zones_to_visit:
            visited_zone = searching_zones_to_visit.pop(0)
            print(f"[🔍] Zone {visited_zone} scanned ({len(searching_zones_to_visit)} remaining)")
        elif goal_type == "last_known":
            searching_last_known_pan = None
            searching_last_known_tilt = None
            print("[🔍] Last known location checked - person not there")
        elif goal_type == "interest_point" and searching_interest_points:
            searching_interest_points.pop(0)
            print(f"[🔍] Interest point visited ({len(searching_interest_points)} remaining)")

        # Reset for next target
        searching_current_goal = None
        searching_goal_start_time = now  # Reset timer for next target


def is_search_mode_active() -> bool:
    """Check if search mode is currently active."""
    return searching_active


def has_search_targets() -> bool:
    """Check if there are any targets to search for."""
    target_pan, _, _ = get_search_target()
    return target_pan is not None


def get_gaze_description() -> str:
    """
    Get human-readable description of current gaze position.
    Returns description like "looking down-left" or "looking straight ahead".
    """
    global servo_x, servo_y

    # Use actual config centers, not hardcoded 90
    pan_center = (PAN_MIN + PAN_MAX) / 2  # 90
    tilt_center = (TILT_MIN + TILT_MAX) / 2  # 107.5

    pan_offset = servo_x - pan_center  # Negative = left, Positive = right
    # TILT_MIN (65) = DOWN at paper, TILT_MAX (150) = UP
    # So: lower servo value = looking down, higher = looking up
    tilt_offset = servo_y - tilt_center  # Negative = down, Positive = up

    # Lower thresholds to match actual movement range
    h_threshold = 8  # degrees from center to count as left/right
    v_threshold = 8  # degrees from center to count as up/down

    # Determine horizontal direction
    if pan_offset < -h_threshold:
        h_dir = "left"
    elif pan_offset > h_threshold:
        h_dir = "right"
    else:
        h_dir = ""

    # Determine vertical direction (corrected for hardware)
    if tilt_offset < -v_threshold:
        v_dir = "down"  # Lower servo = looking down at paper
    elif tilt_offset > v_threshold:
        v_dir = "up"  # Higher servo = looking up
    else:
        v_dir = ""

    # Combine into description
    if v_dir and h_dir:
        return f"{v_dir}-{h_dir}"  # e.g., "down-left"
    elif v_dir:
        return v_dir  # e.g., "up"
    elif h_dir:
        return h_dir  # e.g., "right"
    else:
        return "ahead"  # centered


def get_gaze_state() -> dict:
    """Get current gaze state for prompt injection."""
    global servo_x, servo_y, state

    direction = get_gaze_description()
    return {
        "pan": int(servo_x),
        "tilt": int(servo_y),
        "direction": direction,
        "state": state,  # "idle", "tracking", "grace", "aware"
    }


def get_gaze_narrative() -> str:
    """Return simple first-person gaze direction.

    Returns direct statements like 'Looking up.' to match first-person prompts.
    Clear and unambiguous for the vision model.
    """
    global servo_x, servo_y, state

    pan_center = (PAN_MIN + PAN_MAX) / 2
    tilt_center = (TILT_MIN + TILT_MAX) / 2

    pan_offset = servo_x - pan_center
    tilt_offset = servo_y - tilt_center

    # Determine direction components
    v_dir = None
    if tilt_offset < -8:
        v_dir = "down"
    elif tilt_offset > 8:
        v_dir = "up"

    h_dir = None
    if pan_offset < -8:
        h_dir = "left"
    elif pan_offset > 8:
        h_dir = "right"

    # Build simple direction
    if v_dir and h_dir:
        direction = f"{v_dir}-{h_dir}"
    elif v_dir:
        direction = v_dir
    elif h_dir:
        direction = h_dir
    else:
        direction = "ahead"

    # Simple first-person statement
    if state == "tracking":
        return f"Looking {direction}. Someone here."
    elif state == "aware":
        return f"Looking {direction}. Someone nearby."
    elif state == "searching":
        return f"Looking {direction}. Searching."
    else:
        return f"Looking {direction}."


def clamp(val, min_val, max_val):
    return max(min_val, min(max_val, val))


def perlin_noise_1d(x, octaves=3, persistence=0.5):
    """Simple 1D Perlin noise implementation for organic movement"""
    total = 0.0
    frequency = 1.0
    amplitude = 1.0
    max_value = 0.0

    for _ in range(octaves):
        total += amplitude * (math.sin(x * frequency) * 0.5 + 0.5)
        max_value += amplitude
        amplitude *= persistence
        frequency *= 2.0

    return total / max_value


def update_organic_movement(now):
    """Generate organic movement targets using Perlin noise within the LLM-directed zone."""
    global pan_micro_target, tilt_micro_target, pan_easing_variance, tilt_easing_variance
    global pan_noise_offset, tilt_noise_offset, pan_frequency, tilt_frequency
    global llm_target_zone_pan, llm_target_zone_tilt, llm_zone_active

    # Generate independent Perlin noise for pan and tilt
    pan_noise_time = (now * pan_frequency) + pan_noise_offset
    tilt_noise_time = (now * tilt_frequency) + tilt_noise_offset

    pan_noise = perlin_noise_1d(pan_noise_time, octaves=3, persistence=0.5)
    tilt_noise = perlin_noise_1d(tilt_noise_time, octaves=1, persistence=0.3)  # Reduced for less jitter

    if llm_zone_active:
        # LLM is directing gaze - stay within the specified zone
        pan_min, pan_max = GAZE_ZONES_PAN.get(llm_target_zone_pan, (75, 105))
        tilt_min, tilt_max = GAZE_ZONES_TILT.get(llm_target_zone_tilt, (95, 125))

        # Zone center
        pan_center = (pan_min + pan_max) / 2
        tilt_center = (tilt_min + tilt_max) / 2

        # Organic variance bounded to stay in zone (80% of zone width)
        pan_range = (pan_max - pan_min) * 0.8
        tilt_range = (tilt_max - tilt_min) * 0.8

        pan_micro_target = pan_center + (pan_noise - 0.5) * pan_range
        tilt_micro_target = tilt_center + (tilt_noise - 0.5) * tilt_range

        # Clamp to zone bounds
        pan_micro_target = clamp(pan_micro_target, pan_min, pan_max)
        tilt_micro_target = clamp(tilt_micro_target, tilt_min, tilt_max)
    else:
        # Full contemplative range — use most of the servo range
        pan_range = (PAN_MAX - PAN_MIN) * 0.85
        tilt_range = (TILT_MAX - TILT_MIN) * 0.55

        pan_center = (PAN_MIN + PAN_MAX) / 2
        tilt_center = (TILT_MIN + TILT_MAX) / 2 + 5  # Slight downward bias

        pan_micro_target = pan_center + (pan_noise - 0.5) * pan_range
        tilt_micro_target = tilt_center + (tilt_noise - 0.5) * tilt_range

        # Ensure within overall bounds
        pan_micro_target = clamp(pan_micro_target, PAN_MIN, PAN_MAX)
        tilt_micro_target = clamp(tilt_micro_target, TILT_MIN, TILT_MAX)

    # Create organic easing variance
    pan_easing_variance = 0.8 + 0.4 * perlin_noise_1d(pan_noise_time * 0.5)
    tilt_easing_variance = 0.7 + 0.3 * perlin_noise_1d(tilt_noise_time * 0.3)




def update_gaze(frame, face_box, current_emotion_state="calm_observant", yolo_person_detected=False, person_direction=None, person_bbox=None):
    """
    Update gaze position based on face detection, YOLO detection, and emotional state.
    Uses spring-damper physics for organic movement with emotional modulation.

    Args:
        frame: Camera frame
        face_box: Face bounding box if detected, None otherwise
        current_emotion_state: Emotional state for movement modulation
        yolo_person_detected: True if YOLO detected a person (even without face)
        person_direction: Direction of last known person ("to my left", "to my right", "ahead", None)
        person_bbox: YOLO person bounding box (x1, y1, x2, y2) for tracking in aware state
    """
    global servo_x, servo_y, target_x, target_y, last_seen_time, state, idle_next_move_time
    global startup_sequence_active, drawing_sequence_active, last_state_change
    global drawing_target_x, drawing_target_y, drawing_transition_active
    global physics_state, llm_zone_active, llm_zone_set_time
    global aware_break_until, aware_break_target_pan, aware_break_target_tilt

    now = time.time()
    dt = now - physics_state.last_update_time

    if startup_sequence_active:
        return False, int(servo_x + 0.5), int(servo_y + 0.5)

    # Handle drawing sequence with smooth transitions
    if drawing_sequence_active:
        dx = abs(servo_x - drawing_target_x)
        dy = abs(servo_y - drawing_target_y)

        drawing_easing = 0.08
        servo_x = velocity_limited_step(servo_x, drawing_target_x, drawing_easing, FACE_PAN_VELOCITY * 0.6, "pan")
        servo_y = velocity_limited_step(servo_y, drawing_target_y, drawing_easing, FACE_TILT_VELOCITY * 0.6, "tilt")

        # Keep physics state synced during drawing
        physics_state.pan = servo_x
        physics_state.tilt = servo_y
        physics_state.pan_velocity = 0.0
        physics_state.tilt_velocity = 0.0
        physics_state.last_update_time = now

        if dx < 1.0 and dy < 1.0:
            drawing_transition_active = False

        return False, int(servo_x + 0.5), int(servo_y + 0.5)

    elif drawing_transition_active:
        # Transitioning out of drawing mode - sync physics state and resume
        drawing_transition_active = False
        physics_state.pan = servo_x
        physics_state.tilt = servo_y
        physics_state.pan_velocity = 0.0
        physics_state.tilt_velocity = 0.0
        print("[👁️] Drawing transition complete - resuming normal gaze")

    h, w = frame.shape[:2]
    person_present = face_box is not None

    previous_state = state

    # === CLEAR STATE MACHINE WITH CLEAN TRANSITIONS ===
    if person_present:
        if state != "tracking":
            last_state_change = now
            # Coming from aware/idle — kill residual velocity for clean handoff to face tracking
            physics_state.pan_velocity = 0.0
            physics_state.tilt_velocity = 0.0

        state = "tracking"
        last_seen_time = now
        update_tracking_awareness(face_box, w)

        (startX, startY, endX, endY) = face_box
        face_center_x = (startX + endX) // 2
        face_center_y = (startY + endY) // 2

        if FLIP_X:
            face_center_x = w - face_center_x
        if FLIP_Y:
            face_center_y = h - face_center_y

        face_x_norm = face_center_x / w
        face_y_norm = face_center_y / h

        target_x = PAN_MIN + (PAN_MAX - PAN_MIN) * face_x_norm
        target_y = TILT_MIN + (TILT_MAX - TILT_MIN) * face_y_norm

        # Physics tracking: set target, snap immediately to tracking params
        physics_state.pan_target = target_x
        physics_state.tilt_target = target_y
        physics_state.blend_params(TRACKING_PHYSICS, blend_rate=0.9)  # Near-instant snap to face tracking physics

        pan, tilt = update_physics_step(dt, is_tracking=True)
        servo_x = pan
        servo_y = tilt

    elif state == "tracking" and now - last_seen_time < FACE_STABLE_TIMEOUT:
        # Grace period - hold position using physics (maintains smooth momentum)
        state = "grace"
        physics_state.blend_params(TRACKING_PHYSICS, blend_rate=0.2)  # Slightly faster during grace
        pan, tilt = update_physics_step(dt, is_tracking=True)
        servo_x = pan
        servo_y = tilt

    elif state in ["tracking", "grace"] and now - last_seen_time >= FACE_STABLE_TIMEOUT:
        # Transition from tracking - check if YOLO still sees person
        if yolo_person_detected:
            # Person present but no face - enter aware state
            if state != "aware":
                last_state_change = now
                llm_zone_active = False  # Clear LLM zone - real person overrides LLM's imagined directions
                print("[👁️] Entering 'aware' state - person detected but no face")
            state = "aware"
        else:
            # Clean transition to idle
            if state != "idle":
                last_state_change = now
            state = "idle"
            # Dynamic pause timing - shorter base pauses with occasional longer ones
            if random.random() < IDLE_LONG_PAUSE_CHANCE:
                pause_duration = random.uniform(IDLE_PAUSE_LONG * 0.8, IDLE_PAUSE_LONG * 1.2)
            else:
                pause_duration = random.uniform(IDLE_PAUSE_MIN, IDLE_PAUSE_MAX)
            idle_next_move_time = now + pause_duration

    elif state == "aware":
        # AWARE STATE: Person detected by YOLO but no face to track
        # Tracks person bounding box with softer physics than face tracking

        time_in_aware = now - last_state_change

        # TIMEOUT: Don't stay locked on person forever - take breaks to look natural
        AWARE_MAX_DURATION = 30.0  # Max seconds before taking a break
        AWARE_BREAK_DURATION = 5.0  # How long to look away

        if not yolo_person_detected and time_in_aware > aware_minimum_dwell:
            last_state_change = now
            print(f"[👁️] Person no longer detected after {time_in_aware:.1f}s - returning to idle")
            state = "idle"
            # Record event for episodic timeline
            try:
                from utils.episodic_log import episodic_log
                episodic_log.record("person_left", "they're gone now")
            except Exception:
                pass
        elif time_in_aware > AWARE_MAX_DURATION:
            # Natural falloff - look AWAY from where the person was
            # Pick a direction opposite to current gaze
            current_pan = physics_state.pan
            if current_pan < 90:
                # Looking left, so look right
                aware_break_target_pan = random.uniform(100, 120)
            else:
                # Looking right, so look left
                aware_break_target_pan = random.uniform(60, 80)
            # Vary tilt too
            aware_break_target_tilt = random.uniform(95, 120)
            aware_break_until = now + AWARE_BREAK_DURATION

            last_state_change = now
            print(f"[👁️] Aware timeout - looking away (pan={aware_break_target_pan:.0f}°) for {AWARE_BREAK_DURATION}s")
            state = "idle"  # Go to idle but with break target active
        elif yolo_person_detected or time_in_aware <= aware_minimum_dwell:
            update_organic_movement(now)

            # If we have a person bbox, track its center (like face tracking but softer)
            if person_bbox is not None:
                bbox_x1, bbox_y1, bbox_x2, bbox_y2 = person_bbox
                person_center_x = (bbox_x1 + bbox_x2) // 2
                # Target upper portion of body (where head likely is)
                person_center_y = bbox_y1 + (bbox_y2 - bbox_y1) // 4
                # Debug: log tracking target occasionally
                if random.random() < 0.02:
                    print(f"[👁️ AWARE] Tracking bbox center=({person_center_x}, {person_center_y}) → target pan/tilt")

                if FLIP_X:
                    person_center_x = w - person_center_x
                if FLIP_Y:
                    person_center_y = h - person_center_y

                person_x_norm = person_center_x / w
                person_y_norm = person_center_y / h

                # Calculate raw target from person position
                person_target_x = PAN_MIN + (PAN_MAX - PAN_MIN) * person_x_norm
                person_target_y = TILT_MIN + (TILT_MAX - TILT_MIN) * person_y_norm

                # INCREMENTAL UPDATES: Cap how far target can move per step to prevent overshoot
                MAX_TARGET_STEP = 6.0  # Max degrees per update — responsive enough to follow a moving person
                current_pan = physics_state.pan_target if physics_state.pan_target else physics_state.pan
                current_tilt = physics_state.tilt_target if physics_state.tilt_target else physics_state.tilt
                pan_diff = person_target_x - current_pan
                tilt_diff = person_target_y - current_tilt
                # Clamp the step size
                pan_diff = max(-MAX_TARGET_STEP, min(MAX_TARGET_STEP, pan_diff))
                tilt_diff = max(-MAX_TARGET_STEP, min(MAX_TARGET_STEP, tilt_diff))
                person_target_x = current_pan + pan_diff
                person_target_y = current_tilt + tilt_diff

                # Add subtle micro-movement overlay
                micro_sway_pan = (pan_micro_target - 90) * 0.1
                micro_sway_tilt = (tilt_micro_target - 107.5) * 0.08
                pan_scaled = person_target_x + micro_sway_pan
                tilt_scaled = person_target_y + micro_sway_tilt

            elif llm_zone_active:
                pan_min, pan_max = GAZE_ZONES_PAN.get(llm_target_zone_pan, (75, 105))
                tilt_min, tilt_max = GAZE_ZONES_TILT.get(llm_target_zone_tilt, (95, 125))
                pan_center = (pan_min + pan_max) / 2
                tilt_center = (tilt_min + tilt_max) / 2
                micro_sway_pan = (pan_micro_target - 90) * 0.15
                micro_sway_tilt = (tilt_micro_target - 107.5) * 0.1
                pan_scaled = pan_center + micro_sway_pan
                tilt_scaled = tilt_center + micro_sway_tilt
            else:
                # No bbox available - just subtle sway in place
                if random.random() < 0.01:
                    print(f"[👁️ AWARE] No bbox - person detected but no tracking target")
                micro_sway_pan = (pan_micro_target - 90) * 0.08
                micro_sway_tilt = (tilt_micro_target - 107.5) * 0.05
                pan_scaled = physics_state.pan + micro_sway_pan * 0.3
                tilt_scaled = physics_state.tilt + micro_sway_tilt * 0.3

            pan_scaled = clamp(pan_scaled, PAN_MIN, PAN_MAX)
            tilt_scaled = clamp(tilt_scaled, TILT_MIN, TILT_MAX)

            # Aware state with person bbox: faster tracking with higher update frequency
            aware_params = PHYSICS_PATTERNS.get(current_emotion_state, PHYSICS_PATTERNS["calm_observant"]).copy()
            if person_bbox is not None:
                # Tracking person bbox — responsive, follows body movement
                aware_params["mass"] *= 0.7           # Lighter — quicker to respond
                aware_params["spring_constant"] *= 1.2  # Stronger pull toward target
                aware_params["damping"] *= 1.1        # Light damping — smooth but not sluggish
                aware_params["tremor_amplitude"] *= 0.15  # Minimal tremor while tracking
                aware_params["orbital_strength"] *= 0.1  # Almost no wandering while tracking
            else:
                # No bbox - use normal subdued parameters
                aware_params["mass"] *= 1.2
                aware_params["spring_constant"] *= 0.8
                aware_params["tremor_amplitude"] *= 0.7
                aware_params["orbital_strength"] *= 0.8
            physics_state.blend_params(aware_params, blend_rate=0.25)  # Faster blend

            physics_state.pan_target = pan_scaled
            physics_state.tilt_target = tilt_scaled
            pan, tilt = update_physics_step(dt, is_tracking=(person_bbox is not None))
            servo_x = pan
            servo_y = tilt

    elif state == "idle" or state == "searching":
        # Check if we're in a break from aware state (looking away intentionally)
        in_aware_break = aware_break_until > now

        # Check if YOLO detected someone - transition to aware (unless in break)
        if yolo_person_detected and not in_aware_break:
            last_state_change = now
            llm_zone_active = False  # Clear LLM zone - real person overrides LLM's imagined directions
            print("[👁️] Person detected while idle - entering 'aware' state")
            state = "aware"
            deactivate_search_mode()
            # Record event for episodic timeline
            try:
                from utils.episodic_log import episodic_log
                episodic_log.record("person_arrived", "someone arrived")
            except Exception:
                pass
            # Movement will be handled by aware state on next frame

        # During break: look at the break target (away from person)
        elif in_aware_break:
            # Gently move toward break target
            physics_state.pan_target = aware_break_target_pan
            physics_state.tilt_target = aware_break_target_tilt
            idle_params = PHYSICS_PATTERNS.get(current_emotion_state, PHYSICS_PATTERNS["calm_observant"]).copy()
            physics_state.blend_params(idle_params, blend_rate=0.1)
            pan, tilt = update_physics_step(dt, is_tracking=False)
            servo_x = pan
            servo_y = tilt

        # === SEARCHING STATE: Goal-directed movement with physics ===
        elif searching_active and has_search_targets():
            state = "searching"
            target_pan, target_tilt, goal_type = get_search_target()

            if target_pan is not None:
                update_search_progress(servo_x, servo_y, person_found=False)

                # Add subtle organic variation to avoid robotic movement
                time_jitter = math.sin(now * 0.3) * 3
                target_with_jitter_pan = clamp(target_pan + time_jitter, PAN_MIN, PAN_MAX)
                target_with_jitter_tilt = clamp(target_tilt + math.cos(now * 0.25) * 2, TILT_MIN, TILT_MAX)

                # Searching: deliberate but still alive physics
                # Keep orbital/tremor similar to idle so it doesn't feel "stuck"
                search_params = {
                    "mass": 0.8,
                    "spring_constant": 7.0,
                    "damping": 4.0,
                    "tremor_amplitude": 0.8,  # Still alive while searching
                    "orbital_strength": 1.0,  # Still wanders while searching
                }
                physics_state.blend_params(search_params, blend_rate=0.3)

                physics_state.pan_target = target_with_jitter_pan
                physics_state.tilt_target = target_with_jitter_tilt
                pan, tilt = update_physics_step(dt, is_tracking=False)
                servo_x = pan
                servo_y = tilt

                if random.random() < 0.02:
                    print(f"[🔍] Searching: {goal_type} → pan={target_pan:.0f}° (current={servo_x:.0f}°)")
            else:
                deactivate_search_mode()
                state = "idle"

        else:
            # Regular idle - deactivate any lingering search state
            global idle_in_stillness, idle_stillness_until
            if state == "searching":
                deactivate_search_mode()
            state = "idle"

            # Check LLM zone timeout - if stuck in same zone too long, encourage exploration
            if llm_zone_active and (now - llm_zone_set_time) > llm_zone_timeout:
                print(f"[👁️] LLM zone '{llm_target_zone_pan}/{llm_target_zone_tilt}' timed out after {llm_zone_timeout:.0f}s → exploring freely")
                llm_zone_active = False

            # === STILLNESS LOGIC: Pause during stillness periods ===
            if idle_in_stillness and now < idle_stillness_until:
                # In stillness - hold current position with damped physics
                # No target updates, no tremor/orbital, just settle toward current target
                physics_params = PHYSICS_PATTERNS.get(current_emotion_state, PHYSICS_PATTERNS["calm_observant"])
                physics_state.blend_params(physics_params, blend_rate=0.2)
                pan, tilt = update_physics_step(dt, is_tracking=False, is_still=True)
                servo_x = pan
                servo_y = tilt
            else:
                # Stillness ended or not in stillness - update targets
                if idle_in_stillness:
                    idle_in_stillness = False  # Exit stillness

                # Perlin noise frequency scaling based on emotional state
                FREQUENCY_SCALE = {
                    "energized_engaged": 2.5,
                    "alert_curious": 2.0,
                    "calm_observant": 1.6,
                    "quiet_detached": 1.3,
                    "withdrawn_distant": 1.0,
                }
                freq_scale = FREQUENCY_SCALE.get(current_emotion_state, 1.2)

                global pan_frequency, tilt_frequency
                base_pan_freq = 0.38  # Faster base for more alive movement
                base_tilt_freq = 0.22  # Faster tilt — still slower than pan for natural feel
                pan_frequency = base_pan_freq * freq_scale
                tilt_frequency = base_tilt_freq * freq_scale

                update_organic_movement(now)

                # Determine target position from Perlin noise or LLM zone
                if llm_zone_active:
                    pan_scaled = pan_micro_target
                    tilt_scaled = tilt_micro_target
                else:
                    pan_center = (PAN_MIN + PAN_MAX) / 2
                    tilt_center = (TILT_MIN + TILT_MAX) / 2
                    pan_scaled = pan_center + (pan_micro_target - pan_center) * 1.0
                    tilt_scaled = tilt_center + (tilt_micro_target - tilt_center) * 1.0

                    # Apply LLM-driven gaze nudge as modifier
                    nudge_pan, nudge_tilt = get_current_nudge()
                    pan_scaled += nudge_pan * 0.6
                    tilt_scaled += nudge_tilt * 0.6

                pan_scaled = clamp(pan_scaled, PAN_MIN, PAN_MAX)
                tilt_scaled = clamp(tilt_scaled, TILT_MIN, TILT_MAX)

                # Check if we should enter stillness after reaching target
                dist_to_target = abs(servo_x - pan_scaled) + abs(servo_y - tilt_scaled)
                if dist_to_target < 6.0 and random.random() < IDLE_STILLNESS_CHANCE:
                    # Close to target - enter stillness
                    idle_in_stillness = True
                    idle_stillness_until = now + random.uniform(IDLE_STILLNESS_MIN, IDLE_STILLNESS_MAX)

                # Physics-based idle movement - blend to emotional parameters
                physics_params = PHYSICS_PATTERNS.get(current_emotion_state, PHYSICS_PATTERNS["calm_observant"])
                physics_state.blend_params(physics_params, blend_rate=0.35)

                physics_state.pan_target = pan_scaled
                physics_state.tilt_target = tilt_scaled
                pan, tilt = update_physics_step(dt, is_tracking=False)
                servo_x = pan
                servo_y = tilt

    # Keep decimal precision for smoother movement - only round at final output
    return person_present, int(servo_x + 0.5), int(servo_y + 0.5)


def velocity_limited_step(current, target, factor, max_velocity, axis="pan", smoothing=None):
    """Velocity-limited smooth movement to prevent hard lock-ins and protect servos"""
    global last_pan_velocity, last_tilt_velocity

    # Use default smoothing if not specified
    if smoothing is None:
        smoothing = VELOCITY_SMOOTHING

    diff = target - current

    # Calculate desired step based on easing factor
    desired_step = diff * factor

    # Limit step size to maximum velocity
    if abs(desired_step) > max_velocity:
        step = max_velocity if desired_step > 0 else -max_velocity
    else:
        step = desired_step

    # Smooth velocity changes to prevent jerky acceleration
    if axis == "pan":
        smoothed_step = last_pan_velocity * smoothing + step * (1.0 - smoothing)
        last_pan_velocity = smoothed_step
    else:  # tilt
        smoothed_step = last_tilt_velocity * smoothing + step * (1.0 - smoothing)
        last_tilt_velocity = smoothed_step

    # Apply final velocity limiting after smoothing
    if abs(smoothed_step) > max_velocity:
        smoothed_step = max_velocity if smoothed_step > 0 else -max_velocity

    # Prevent tiny oscillations by stopping when close enough
    if abs(diff) < 0.1:
        return target

    return current + smoothed_step


def set_drawing_mode(active: bool, drawing_pan: int = 90, drawing_tilt: int = None):
    """Control drawing sequence mode with smooth transitions (thread-safe)."""
    global drawing_sequence_active, drawing_target_x, drawing_target_y, drawing_transition_active
    from config.config import TILT_MIN

    with _gaze_lock:
        if active:
            drawing_tilt = drawing_tilt or (TILT_MIN + 2)
            drawing_target_x = drawing_pan
            drawing_target_y = drawing_tilt
            drawing_transition_active = True
            drawing_sequence_active = True
            print(f"[👁️] Smoothly transitioning to drawing position: pan={drawing_pan}°, tilt={drawing_tilt}°")
        else:
            drawing_sequence_active = False
            drawing_transition_active = False
            print("[👁️] Gaze drawing lock released - resuming normal operation immediately")


# === PAPER SEARCH MODE ===
# Organic searching movement for ArUco paper detection
_paper_search_active = False
_paper_search_center_pan = 90.0
_paper_search_center_tilt = 75.0
_paper_search_range_pan = 20.0
_paper_search_range_tilt = 10.0
_paper_search_start_time = 0.0


def set_paper_search_mode(active: bool, center_pan: float = 90.0, center_tilt: float = 75.0,
                          range_pan: float = 20.0, range_tilt: float = 10.0):
    """
    Activate paper search mode for ArUco detection with organic movement.

    When active, gaze moves organically within the specified range around the center,
    allowing for marker detection at various angles. Uses same mechanism as drawing
    mode but with continuously updating targets.

    Args:
        active: Enable/disable search mode
        center_pan: Center pan angle for search area
        center_tilt: Center tilt angle for search area
        range_pan: Search range for pan (±degrees from center)
        range_tilt: Search range for tilt (±degrees from center)
    """
    global _paper_search_active, _paper_search_center_pan, _paper_search_center_tilt
    global _paper_search_range_pan, _paper_search_range_tilt, _paper_search_start_time
    global drawing_sequence_active, drawing_target_x, drawing_target_y, drawing_transition_active

    with _gaze_lock:
        if active:
            _paper_search_active = True
            _paper_search_center_pan = center_pan
            _paper_search_center_tilt = center_tilt
            _paper_search_range_pan = range_pan
            _paper_search_range_tilt = range_tilt
            _paper_search_start_time = time.time()

            # Use drawing mode mechanism but we'll update target continuously
            drawing_target_x = center_pan
            drawing_target_y = center_tilt
            drawing_transition_active = True
            drawing_sequence_active = True
            print(f"[📄🔍] Paper search mode activated: center=({center_pan}°, {center_tilt}°), range=±({range_pan}°, {range_tilt}°)")
        else:
            _paper_search_active = False
            drawing_sequence_active = False
            drawing_transition_active = False
            print("[📄🔍] Paper search mode deactivated")


def update_paper_search_target():
    """
    Update the search target position with organic movement pattern.
    Call this periodically during paper search to create smooth, searching motion.

    Returns:
        (pan, tilt): Current target position, or (None, None) if search not active
    """
    global drawing_target_x, drawing_target_y

    if not _paper_search_active:
        return (None, None)

    now = time.time()
    elapsed = now - _paper_search_start_time

    # Multi-frequency organic pattern for natural searching motion
    # Slow, leisurely sweep - contemplative searching
    primary_freq_pan = 0.08  # Very slow primary sweep (~12 second cycle)
    primary_freq_tilt = 0.06  # Even slower tilt (~16 second cycle)
    secondary_freq_pan = 0.18  # Gentle secondary motion
    secondary_freq_tilt = 0.14

    # Calculate organic offsets using layered sine waves
    pan_offset = (
        0.65 * math.sin(elapsed * primary_freq_pan * 2 * math.pi) +
        0.25 * math.sin(elapsed * secondary_freq_pan * 2 * math.pi + 0.7) +
        0.10 * math.sin(elapsed * 0.28 * 2 * math.pi + 1.3)
    )

    tilt_offset = (
        0.65 * math.sin(elapsed * primary_freq_tilt * 2 * math.pi + math.pi / 3) +
        0.25 * math.sin(elapsed * secondary_freq_tilt * 2 * math.pi + 0.9) +
        0.10 * math.sin(elapsed * 0.22 * 2 * math.pi + 2.1)
    )

    # Apply offset within search range
    target_pan = _paper_search_center_pan + pan_offset * _paper_search_range_pan
    target_tilt = _paper_search_center_tilt + tilt_offset * _paper_search_range_tilt

    # Clamp to servo limits
    target_pan = clamp(target_pan, PAN_MIN, PAN_MAX)
    target_tilt = clamp(target_tilt, TILT_MIN, TILT_MAX)

    # Update drawing targets (used by main gaze loop when drawing_sequence_active)
    with _gaze_lock:
        drawing_target_x = target_pan
        drawing_target_y = target_tilt

    return (target_pan, target_tilt)


def is_paper_search_active() -> bool:
    """Check if paper search mode is currently active."""
    return _paper_search_active
