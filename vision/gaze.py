import random
import time
import math

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

# === ORGANIC MOVEMENT DECOUPLING ===
# Independent timing and curves for pan/tilt
pan_offset_time = random.uniform(0, 2.0)  # Random phase offset for pan
tilt_offset_time = random.uniform(0, 2.0)  # Random phase offset for tilt
pan_micro_target = 90  # Intermediate target for curved movement
tilt_micro_target = 90  # Intermediate target for curved movement
pan_easing_variance = 1.0  # Dynamic easing multiplier for pan
tilt_easing_variance = 1.0  # Dynamic easing multiplier for tilt


def clamp(val, min_val, max_val):
    return max(min_val, min(max_val, val))


def update_organic_variance(now):
    """Update pan/tilt easing variance for organic movement"""
    global pan_easing_variance, tilt_easing_variance, pan_offset_time, tilt_offset_time

    # Create organic variance with different frequencies for pan/tilt
    pan_wave = math.sin((now + pan_offset_time) * 0.3) * 0.3 + 1.0  # 0.7 to 1.3 multiplier
    tilt_wave = math.cos((now + tilt_offset_time) * 0.4) * 0.2 + 1.0  # 0.8 to 1.2 multiplier

    pan_easing_variance = max(0.5, min(1.5, pan_wave))
    tilt_easing_variance = max(0.5, min(1.5, tilt_wave))


def create_micro_targets(target_x, target_y, now):
    """Create intermediate curved movement targets for organic motion"""
    global pan_micro_target, tilt_micro_target

    # Add slight curved paths using offset sine waves
    curve_intensity = 2.0  # Maximum deviation in degrees

    # Pan gets horizontal curves, tilt gets vertical curves
    pan_curve = math.sin((now + pan_offset_time) * 0.8) * curve_intensity
    tilt_curve = math.cos((now + tilt_offset_time) * 0.6) * curve_intensity

    # Update micro targets with curves
    pan_micro_target = target_x + pan_curve
    tilt_micro_target = target_y + tilt_curve

    # Keep within bounds
    pan_micro_target = clamp(pan_micro_target, PAN_MIN, PAN_MAX)
    tilt_micro_target = clamp(tilt_micro_target, TILT_MIN, TILT_MAX)


def update_gaze(frame, face_box, current_emotion_state="calm_observant"):
    global servo_x, servo_y, target_x, target_y, last_seen_time, state, idle_next_move_time, startup_sequence_active, drawing_sequence_active
    
    # Skip gaze updates during startup sequence to prevent conflicts
    if startup_sequence_active:
        return False, int(servo_x + 0.5), int(servo_y + 0.5)
    
    # Skip gaze updates during drawing sequence - maintain drawing position
    if drawing_sequence_active:
        return False, int(servo_x + 0.5), int(servo_y + 0.5)

    h, w = frame.shape[:2]
    person_present = face_box is not None
    now = time.time()

    # === CLEAR STATE MACHINE ===
    if person_present:
        state = "tracking"
        last_seen_time = now

        # Direct position mapping (not incremental!)
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

        # Natural head movement mapping
        target_x = PAN_MIN + (PAN_MAX - PAN_MIN) * face_x_norm
        target_y = TILT_MIN + (TILT_MAX - TILT_MIN) * face_y_norm

        # Update organic movement patterns
        update_organic_variance(now)
        create_micro_targets(target_x, target_y, now)

        # Apply dead zone only for small movements - but use decoupled easing
        dx = abs(pan_micro_target - servo_x)
        dy = abs(tilt_micro_target - servo_y)

        # Independent pan/tilt movement with organic variance
        if dx > DEAD_ZONE:
            pan_easing = EASING_FACTOR * pan_easing_variance
            servo_x = smooth_step(servo_x, pan_micro_target, pan_easing)
        if dy > DEAD_ZONE:
            tilt_easing = EASING_FACTOR * tilt_easing_variance
            servo_y = smooth_step(servo_y, tilt_micro_target, tilt_easing)

    elif state == "tracking" and now - last_seen_time < FACE_STABLE_TIMEOUT:
        # Grace period - hold position
        state = "grace"

    elif state in ["tracking", "grace"] and now - last_seen_time >= FACE_STABLE_TIMEOUT:
        # Transition to idle
        state = "idle"
        idle_next_move_time = now + random.uniform(IDLE_PAUSE_MIN, IDLE_PAUSE_MAX)

    elif state == "idle":
        # Dynamic idle behavior with 5-state emotional modulation

        # 5-State Emotional Gaze Patterns (define outside to ensure scope)
        gaze_patterns = {
            "energized_engaged": {
                "amplitude_scale": 1.6,  # Large, expressive movements
                "sweep_prob": 0.9,  # Lots of dramatic sweeps
                "pause_scale": 0.3,  # Very short pauses (hyperactive)
                "easing_scale": 1.4,  # Faster movement
            },
            "alert_curious": {
                "amplitude_scale": 1.3,  # Quick, darting movements
                "sweep_prob": 0.7,  # Frequent scanning sweeps
                "pause_scale": 0.6,  # Quick pauses for attention
                "easing_scale": 1.2,  # Responsive movement
            },
            "calm_observant": {
                "amplitude_scale": 1.0,  # Smooth, contemplative
                "sweep_prob": 0.5,  # Balanced movement
                "pause_scale": 1.0,  # Normal contemplative pauses
                "easing_scale": 1.0,  # Steady movement
            },
            "quiet_detached": {
                "amplitude_scale": 0.5,  # Small, hesitant movements
                "sweep_prob": 0.2,  # Mostly local, minimal
                "pause_scale": 2.2,  # Long hesitant pauses
                "easing_scale": 0.8,  # Slower, uncertain
            },
            "withdrawn_distant": {
                "amplitude_scale": 0.3,  # Very small, listless
                "sweep_prob": 0.1,  # Almost no sweeps
                "pause_scale": 3.0,  # Very long pauses
                "easing_scale": 0.6,  # Slow, disengaged
            },
        }

        # Get current emotional pattern (fallback to calm)
        pattern = gaze_patterns.get(current_emotion_state, gaze_patterns["calm_observant"])

        if now >= idle_next_move_time:

            # Apply emotional amplitude scaling with natural head limits
            emotion_amp_x = int(IDLE_AMPLITUDE_X * pattern["amplitude_scale"])
            emotion_amp_y = int(IDLE_AMPLITUDE_Y * pattern["amplitude_scale"])

            # Decide between small local movement or big sweep
            if random.random() < pattern["sweep_prob"]:
                # Big sweeping movement with natural head constraints
                if random.choice([True, False]):
                    # Horizontal sweep within natural pan range
                    target_x = random.choice([PAN_MIN + 5, PAN_MAX - 5])
                    target_y = clamp(IDLE_CENTER_Y + random.randint(-emotion_amp_y // 2, emotion_amp_y // 2), TILT_MIN, TILT_MAX)
                else:
                    # Vertical sweep within natural tilt range  
                    target_y = random.choice([TILT_MIN + 5, TILT_MAX - 5])
                    target_x = clamp(IDLE_CENTER_X + random.randint(-emotion_amp_x // 2, emotion_amp_x // 2), PAN_MIN, PAN_MAX)

                # Emotionally-scaled pause after big movements
                base_pause = random.uniform(IDLE_PAUSE_MAX * 1.5, IDLE_PAUSE_MAX * 2.5)
                idle_next_move_time = now + base_pause * pattern["pause_scale"]
            else:
                # Smaller local movements with natural head constraints
                jitter_x = random.randint(-emotion_amp_x, emotion_amp_x)
                jitter_y = random.randint(-emotion_amp_y, emotion_amp_y)
                target_x = clamp(IDLE_CENTER_X + jitter_x, PAN_MIN, PAN_MAX)
                target_y = clamp(IDLE_CENTER_Y + jitter_y, TILT_MIN, TILT_MAX)

                # Emotionally-scaled pause for small movements
                base_pause = random.uniform(IDLE_PAUSE_MIN, IDLE_PAUSE_MAX)
                idle_next_move_time = now + base_pause * pattern["pause_scale"]

        # Update organic movement patterns for idle state
        update_organic_variance(now)
        create_micro_targets(target_x, target_y, now)

        # Movement toward targets with emotional easing and organic decoupling
        emotional_easing = IDLE_EASING * pattern["easing_scale"]

        # Apply independent pan/tilt easing with organic variance
        pan_easing = emotional_easing * pan_easing_variance
        tilt_easing = emotional_easing * tilt_easing_variance

        servo_x = smooth_step(servo_x, pan_micro_target, pan_easing)
        servo_y = smooth_step(servo_y, tilt_micro_target, tilt_easing)

    # Keep decimal precision for smoother movement - only round at final output
    return person_present, int(servo_x + 0.5), int(servo_y + 0.5)


def smooth_step(current, target, factor):
    """Smooth exponential easing for silky servo movement"""
    diff = target - current
    # Use smaller steps for very smooth movement
    step = diff * factor

    # Prevent tiny oscillations by stopping when close enough
    if abs(diff) < 0.1:
        return target

    return current + step


def set_drawing_mode(active: bool, drawing_pan: int = 90, drawing_tilt: int = None):
    """Control drawing sequence mode to lock gaze during CNC drawing"""
    global drawing_sequence_active, servo_x, servo_y, target_x, target_y
    from config.config import TILT_MIN
    
    drawing_sequence_active = active
    
    if active:
        # Lock gaze to drawing position
        drawing_tilt = drawing_tilt or (TILT_MIN + 2)  # Lowest safe position
        servo_x = drawing_pan
        servo_y = drawing_tilt
        target_x = drawing_pan  
        target_y = drawing_tilt
        print(f"[👁️] Gaze locked for drawing: pan={drawing_pan}°, tilt={drawing_tilt}°")
    else:
        # Release drawing lock - gaze will return to normal operation
        print("[👁️] Gaze drawing lock released")


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
