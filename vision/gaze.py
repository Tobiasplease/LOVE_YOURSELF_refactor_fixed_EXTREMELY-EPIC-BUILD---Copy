import random
import time
import math
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
# Face tracking - reduced for smoother movement
FACE_PAN_VELOCITY = 8.0   # Maximum degrees per update during face tracking (smoother)
FACE_TILT_VELOCITY = 7.0  # Maximum degrees per update during face tracking (smoother)
FACE_VELOCITY_SMOOTHING = 0.0  # No velocity smoothing for face tracking (immediate response)

# General movement - higher limits for idle movement
MAX_PAN_VELOCITY = 8.0   # Maximum degrees per update for pan servo (prevents hard lock-ins)
MAX_TILT_VELOCITY = 6.0  # Maximum degrees per update for tilt servo (prevents hard lock-ins)
VELOCITY_SMOOTHING = 0.7 # How much to smooth velocity changes (0.0 = instant, 1.0 = never change)

# Dynamic pause timing for idle movement
IDLE_PAUSE_LONG = 8.0  # Occasional longer contemplative pauses
IDLE_LONG_PAUSE_CHANCE = 0.15  # 15% chance for a longer pause

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
startup_sequence_active = False  # Flag to prevent conflicts during startup
drawing_sequence_active = False  # Flag to prevent conflicts during CNC drawing
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

# === VELOCITY LIMITING ===
last_pan_velocity = 0.0  # Previous pan velocity for smoothing
last_tilt_velocity = 0.0  # Previous tilt velocity for smoothing

# === LLM-DRIVEN GAZE NUDGE (MODIFIER MODE) ===
# Allows the model to influence idle gaze via caption content
# Acts as a decaying bias added to organic Perlin movement - preserves natural sway
pan_nudge = 0.0  # Current pan nudge offset (degrees)
tilt_nudge = 0.0  # Current tilt nudge offset (degrees)
nudge_start_time = 0.0  # When the nudge was applied
nudge_duration = 5.0  # How long nudge lasts before decaying (seconds)


def apply_gaze_nudge(pan_delta: float, tilt_delta: float, duration: float = 5.0):
    """
    Apply a directional nudge as a modifier to idle gaze movement.
    The nudge adds a decaying bias to organic Perlin movement, creating
    a gentle drift in the specified direction while preserving natural sway.

    Args:
        pan_delta: Degrees to bias pan (-left, +right)
        tilt_delta: Degrees to bias tilt (+up, -down) - matches servo config
        duration: How long the bias lasts before decaying
    """
    global pan_nudge, tilt_nudge, nudge_start_time, nudge_duration

    pan_nudge = pan_delta
    tilt_nudge = tilt_delta
    nudge_start_time = time.time()
    nudge_duration = duration

    if abs(pan_delta) > 0.1 or abs(tilt_delta) > 0.1:
        direction = []
        # Positive tilt delta = UP (higher servo), Negative = DOWN (lower servo)
        if tilt_delta > 1:
            direction.append("up")
        elif tilt_delta < -1:
            direction.append("down")
        if pan_delta < -1:
            direction.append("left")
        elif pan_delta > 1:
            direction.append("right")
        print(f"[👁️] Gaze nudge: {' '.join(direction)} (pan={pan_delta:+.1f}°, tilt={tilt_delta:+.1f}°)")


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
        "state": state,  # "idle", "tracking", "grace"
    }


def get_gaze_narrative() -> str:
    """Return roleplay-style gaze description based on servo position.

    Returns immersive narrative like '*Your gaze drifts upward.*' rather than
    dry coordinate descriptions. Uses actual servo config for correct direction.
    """
    global servo_x, servo_y, state

    # Use actual servo center from config (not hardcoded 90)
    pan_center = (PAN_MIN + PAN_MAX) / 2  # 90
    tilt_center = (TILT_MIN + TILT_MAX) / 2  # 107.5

    pan_offset = servo_x - pan_center  # Negative = left, Positive = right
    tilt_offset = servo_y - tilt_center  # Positive = up (higher servo), Negative = down

    # State-aware narrative - distinguish tracking from idle
    if state == "tracking":
        return "*Your eyes are fixed on someone.*"

    # Pure directional narrative with embodied language
    # Lower thresholds to trigger even with subtle/tired movement (±8 degrees)
    # TILT_MIN (65) = looking DOWN at paper, TILT_MAX (150) = looking UP
    # So: servo_y < center → negative offset → looking DOWN
    v_dir = None
    if tilt_offset < -8:  # Looking down (lower servo value = paper)
        v_dir = "down at the desk"
    elif tilt_offset > 8:  # Looking up (higher servo value)
        v_dir = "upward"

    h_dir = None
    if pan_offset < -8:  # Looking left
        h_dir = "to your left"
    elif pan_offset > 8:  # Looking right
        h_dir = "to your right"

    # Combine with more embodied phrasing
    if v_dir and h_dir:
        return f"*Your gaze rests {v_dir}, {h_dir}.*"
    elif v_dir:
        return f"*Your gaze drifts {v_dir}.*"
    elif h_dir:
        return f"*You're looking {h_dir}.*"
    else:
        return "*Your gaze settles forward.*"


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


def bezier_curve(t, p0, p1, p2):
    """Quadratic Bézier curve for smooth movement paths"""
    return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t ** 2 * p2


def update_organic_movement(now):
    """Generate organic movement targets using Perlin noise for natural idle patterns"""
    global pan_micro_target, tilt_micro_target, pan_easing_variance, tilt_easing_variance
    global pan_noise_offset, tilt_noise_offset, pan_frequency, tilt_frequency

    # Generate independent Perlin noise for pan and tilt
    pan_noise_time = (now * pan_frequency) + pan_noise_offset
    tilt_noise_time = (now * tilt_frequency) + tilt_noise_offset

    pan_noise = perlin_noise_1d(pan_noise_time, octaves=3, persistence=0.6)
    tilt_noise = perlin_noise_1d(tilt_noise_time, octaves=2, persistence=0.4)

    # Convert noise to movement range
    pan_range = (PAN_MAX - PAN_MIN) * 0.6  # Use 60% of full range for contemplative movement
    tilt_range = (TILT_MAX - TILT_MIN) * 0.4  # Use 40% of full range for subtle vertical movement

    # Apply noise to center position with natural bias
    pan_center = (PAN_MIN + PAN_MAX) / 2
    tilt_center = (TILT_MIN + TILT_MAX) / 2 + 5  # Slight downward bias for natural head position

    pan_micro_target = pan_center + (pan_noise - 0.5) * pan_range
    tilt_micro_target = tilt_center + (tilt_noise - 0.5) * tilt_range

    # Ensure within bounds
    pan_micro_target = clamp(pan_micro_target, PAN_MIN, PAN_MAX)
    tilt_micro_target = clamp(tilt_micro_target, TILT_MIN, TILT_MAX)

    # Create organic easing variance
    pan_easing_variance = 0.8 + 0.4 * perlin_noise_1d(pan_noise_time * 0.5)
    tilt_easing_variance = 0.7 + 0.3 * perlin_noise_1d(tilt_noise_time * 0.3)




def update_gaze(frame, face_box, current_emotion_state="calm_observant"):
    global servo_x, servo_y, target_x, target_y, last_seen_time, state, idle_next_move_time
    global startup_sequence_active, drawing_sequence_active, last_state_change
    global drawing_target_x, drawing_target_y, drawing_transition_active

    # Skip gaze updates during startup sequence to prevent conflicts
    if startup_sequence_active:
        return False, int(servo_x + 0.5), int(servo_y + 0.5)

    # Handle drawing sequence with smooth transitions
    if drawing_sequence_active:
        # Actively locked - smooth transition to drawing position
        dx = abs(servo_x - drawing_target_x)
        dy = abs(servo_y - drawing_target_y)

        # Use slower easing for smooth drawing transitions
        drawing_easing = 0.08  # Slower for very smooth movement
        servo_x = velocity_limited_step(servo_x, drawing_target_x, drawing_easing, FACE_PAN_VELOCITY * 0.6, "pan")
        servo_y = velocity_limited_step(servo_y, drawing_target_y, drawing_easing, FACE_TILT_VELOCITY * 0.6, "tilt")

        # Stop transition when close enough
        if dx < 1.0 and dy < 1.0:
            drawing_transition_active = False

        return False, int(servo_x + 0.5), int(servo_y + 0.5)

    elif drawing_transition_active:
        # Transitioning out of drawing mode - just clear the flag and resume normal operation
        drawing_transition_active = False
        print("[👁️] Drawing transition complete - resuming normal gaze")

    h, w = frame.shape[:2]
    person_present = face_box is not None
    now = time.time()

    # Track state changes for clean transitions
    previous_state = state

    # === CLEAR STATE MACHINE WITH CLEAN TRANSITIONS ===
    if person_present:
        # Clean transition from idle to tracking
        if state != "tracking":
            last_state_change = now

        state = "tracking"
        last_seen_time = now

        # Direct position mapping for responsive tracking
        (startX, startY, endX, endY) = face_box
        face_center_x = (startX + endX) // 2
        face_center_y = (startY + endY) // 2

        if FLIP_X:
            face_center_x = w - face_center_x
        if FLIP_Y:
            face_center_y = h - face_center_y

        # Map face position directly to servo range
        face_x_norm = face_center_x / w  # 0.0 to 1.0
        face_y_norm = face_center_y / h  # 0.0 to 1.0

        # Direct mapping to face position for natural eye contact
        target_x = PAN_MIN + (PAN_MAX - PAN_MIN) * face_x_norm
        target_y = TILT_MIN + (TILT_MAX - TILT_MIN) * face_y_norm

        # Apply very small dead zone and optimized face tracking velocity limits
        dx = abs(target_x - servo_x)
        dy = abs(target_y - servo_y)

        # Face tracking with dedicated velocity limits and reduced smoothing for responsiveness
        if dx > PRECISE_DEAD_ZONE:
            servo_x = velocity_limited_step(servo_x, target_x, EASING_FACTOR, FACE_PAN_VELOCITY, "pan", FACE_VELOCITY_SMOOTHING)
        if dy > PRECISE_DEAD_ZONE:
            servo_y = velocity_limited_step(servo_y, target_y, EASING_FACTOR, FACE_TILT_VELOCITY, "tilt", FACE_VELOCITY_SMOOTHING)

    elif state == "tracking" and now - last_seen_time < FACE_STABLE_TIMEOUT:
        # Grace period - hold position
        state = "grace"

    elif state in ["tracking", "grace"] and now - last_seen_time >= FACE_STABLE_TIMEOUT:
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

    elif state == "idle":
        # Organic idle behavior using Perlin noise with emotional modulation

        # 5-State Emotional Gaze Patterns - more dynamic and faster
        gaze_patterns = {
            "energized_engaged": {
                "movement_scale": 1.6,  # Large, expressive movements
                "frequency_scale": 2.0,  # Much faster frequency changes
                "easing_scale": 1.5,  # Very responsive movement
            },
            "alert_curious": {
                "movement_scale": 1.3,  # Quick, darting movements
                "frequency_scale": 1.6,  # Frequent changes
                "easing_scale": 1.3,  # More responsive movement
            },
            "calm_observant": {
                "movement_scale": 1.0,  # Smooth, but more dynamic
                "frequency_scale": 1.2,  # Slightly faster than before
                "easing_scale": 1.1,  # Bit more responsive
            },
            "quiet_detached": {
                "movement_scale": 0.7,  # Small, but not tiny movements
                "frequency_scale": 0.8,  # Slower but not too slow
                "easing_scale": 0.9,  # Still responsive
            },
            "withdrawn_distant": {
                "movement_scale": 0.5,  # Small, listless
                "frequency_scale": 0.6,  # Slower changes
                "easing_scale": 0.7,  # Slower, disengaged
            },
        }

        # Get current emotional pattern (fallback to calm)
        pattern = gaze_patterns.get(current_emotion_state, gaze_patterns["calm_observant"])

        # Apply emotional scaling to noise frequencies - faster base frequencies for more dynamic movement
        global pan_frequency, tilt_frequency
        base_pan_freq = 0.12  # Increased from 0.08 for more dynamic movement
        base_tilt_freq = 0.10  # Increased from 0.06 for more dynamic movement
        pan_frequency = base_pan_freq * pattern["frequency_scale"]
        tilt_frequency = base_tilt_freq * pattern["frequency_scale"]

        # Generate organic movement targets using Perlin noise
        update_organic_movement(now)

        # Scale movement based on emotional state
        pan_center = (PAN_MIN + PAN_MAX) / 2
        tilt_center = (TILT_MIN + TILT_MAX) / 2

        # Apply emotional scaling to movement range
        pan_scaled = pan_center + (pan_micro_target - pan_center) * pattern["movement_scale"]
        tilt_scaled = tilt_center + (tilt_micro_target - tilt_center) * pattern["movement_scale"]

        # Apply LLM-driven gaze nudge as MODIFIER (gentle drift, not override)
        # Nudge adds a decaying bias to organic movement - preserves natural sway
        nudge_pan, nudge_tilt = get_current_nudge()

        # Add nudge as directional bias while keeping organic Perlin sway
        # Lower blend = more subtle influence, higher = stronger pull
        nudge_blend = 0.6  # 60% of nudge value added to organic movement
        pan_scaled += nudge_pan * nudge_blend
        tilt_scaled += nudge_tilt * nudge_blend

        # Ensure within bounds
        pan_scaled = clamp(pan_scaled, PAN_MIN, PAN_MAX)
        tilt_scaled = clamp(tilt_scaled, TILT_MIN, TILT_MAX)

        # Independent pan/tilt movement with emotional easing and organic variance
        emotional_easing = IDLE_EASING * pattern["easing_scale"]
        pan_easing = emotional_easing * pan_easing_variance
        tilt_easing = emotional_easing * tilt_easing_variance

        # Use velocity limiting for idle movement
        pan_velocity = MAX_PAN_VELOCITY * 0.8
        tilt_velocity = MAX_TILT_VELOCITY * 0.8
        servo_x = velocity_limited_step(servo_x, pan_scaled, pan_easing, pan_velocity, "pan")
        servo_y = velocity_limited_step(servo_y, tilt_scaled, tilt_easing, tilt_velocity, "tilt")

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


def smooth_step(current, target, factor):
    """Legacy smooth step function for non-critical movement"""
    diff = target - current
    step = diff * factor

    if abs(diff) < 0.1:
        return target

    return current + step


def set_drawing_mode(active: bool, drawing_pan: int = 90, drawing_tilt: int = None):
    """Control drawing sequence mode with smooth transitions"""
    global drawing_sequence_active, drawing_target_x, drawing_target_y, drawing_transition_active
    from config.config import TILT_MIN

    if active:
        # Set targets and begin smooth transition to drawing position
        drawing_tilt = drawing_tilt or (TILT_MIN + 2)  # Lowest safe position
        drawing_target_x = drawing_pan
        drawing_target_y = drawing_tilt
        drawing_transition_active = True
        drawing_sequence_active = True
        print(f"[👁️] Smoothly transitioning to drawing position: pan={drawing_pan}°, tilt={drawing_tilt}°")
    else:
        # Release drawing lock immediately - no transition delay
        drawing_sequence_active = False
        drawing_transition_active = False
        print("[👁️] Gaze drawing lock released - resuming normal operation immediately")


def startup_movement_sequence(servos, duration=5.0):
    """Perform single figure-8 startup sequence to establish presence
    
    Args:
        servos: ServoController instance
        duration: Total duration of the sequence in seconds
    """
    global startup_sequence_active, servo_x, servo_y, target_x, target_y
    
    startup_sequence_active = True  # Block normal gaze updates
    print("🌟 Performing startup movement sequence...")
    
    # Much finer interpolation for ultra-smooth movement
    steps = 100  # Many more steps for smoothness
    center_x, center_y = 90, 90
    amplitude_x = 18  # Horizontal amplitude within natural limits
    amplitude_y = 12  # Vertical amplitude within natural limits
    
    step_duration = duration / steps
    
    # Track current positions for smooth interpolation
    current_pan = 90.0
    current_tilt = 90.0
    
    for i in range(steps):
        # Single figure-8 parametric equations
        t = (i / steps) * 2 * math.pi  # One complete cycle
        
        # Calculate target figure-8 positions
        target_x_local = center_x + amplitude_x * math.sin(t)
        target_y_local = center_y + amplitude_y * math.sin(2 * t)
        
        # Constrain to natural limits
        target_x_local = max(PAN_MIN, min(PAN_MAX, target_x_local))
        target_y_local = max(TILT_MIN, min(TILT_MAX, target_y_local))
        
        # Smooth interpolation toward target (smaller steps)
        easing = 0.3  # Slower easing for smoother motion
        current_pan += (target_x_local - current_pan) * easing
        current_tilt += (target_y_local - current_tilt) * easing
        
        # Update global state to prevent conflicts
        servo_x = current_pan
        servo_y = current_tilt
        target_x = current_pan
        target_y = current_tilt
        
        # Send commands
        servos.set_pan(int(current_pan + 0.5))
        time.sleep(0.01)  # Very small delay
        servos.set_tilt(int(current_tilt + 0.5))
        
        time.sleep(step_duration - 0.01 if step_duration > 0.01 else 0.01)
    
    # Final return to center
    servos.set_pan(90)
    time.sleep(0.05)
    servos.set_tilt(90)
    time.sleep(0.3)
    
    # Update global state and release control
    servo_x = servo_y = target_x = target_y = 90
    startup_sequence_active = False
    
    print("✅ Startup sequence complete")
