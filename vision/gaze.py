import random
import time

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
    SERVO_MAX,
    SERVO_MIN,
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


def clamp(val, min_val, max_val):
    return max(min_val, min(max_val, val))


def update_gaze(frame, face_box, current_emotion_state="calm_observant"):
    global servo_x, servo_y, target_x, target_y, last_seen_time, state, idle_next_move_time

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

        # Direct servo position calculation
        target_x = SERVO_MIN + (SERVO_MAX - SERVO_MIN) * face_x_norm
        target_y = SERVO_MIN + (SERVO_MAX - SERVO_MIN) * face_y_norm

        # Apply dead zone only for small movements
        dx = abs(target_x - servo_x)
        dy = abs(target_y - servo_y)

        if dx > DEAD_ZONE:
            servo_x = smooth_step(servo_x, target_x, EASING_FACTOR)
        if dy > DEAD_ZONE:
            servo_y = smooth_step(servo_y, target_y, EASING_FACTOR)

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
                "amplitude_scale": 1.6,   # Large, expressive movements
                "sweep_prob": 0.9,        # Lots of dramatic sweeps
                "pause_scale": 0.3,       # Very short pauses (hyperactive)
                "easing_scale": 1.4       # Faster movement
            },
            "alert_curious": {
                "amplitude_scale": 1.3,   # Quick, darting movements
                "sweep_prob": 0.7,        # Frequent scanning sweeps
                "pause_scale": 0.6,       # Quick pauses for attention
                "easing_scale": 1.2       # Responsive movement
            },
            "calm_observant": {
                "amplitude_scale": 1.0,   # Smooth, contemplative
                "sweep_prob": 0.5,        # Balanced movement
                "pause_scale": 1.0,       # Normal contemplative pauses
                "easing_scale": 1.0       # Steady movement
            },
            "quiet_detached": {
                "amplitude_scale": 0.5,   # Small, hesitant movements
                "sweep_prob": 0.2,        # Mostly local, minimal
                "pause_scale": 2.2,       # Long hesitant pauses
                "easing_scale": 0.8       # Slower, uncertain
            },
            "withdrawn_distant": {
                "amplitude_scale": 0.3,   # Very small, listless
                "sweep_prob": 0.1,        # Almost no sweeps
                "pause_scale": 3.0,       # Very long pauses
                "easing_scale": 0.6       # Slow, disengaged
            }
        }
        
        # Get current emotional pattern (fallback to calm)
        pattern = gaze_patterns.get(current_emotion_state, gaze_patterns["calm_observant"])
        
        if now >= idle_next_move_time:
            
            # Apply emotional amplitude scaling
            emotion_amp_x = int(IDLE_AMPLITUDE_X * pattern["amplitude_scale"])
            emotion_amp_y = int(IDLE_AMPLITUDE_Y * pattern["amplitude_scale"])
            
            # Decide between small local movement or big sweep
            if random.random() < pattern["sweep_prob"]:
                # Big sweeping movement with emotional scaling
                if random.choice([True, False]):
                    # Horizontal sweep
                    target_x = random.choice([SERVO_MIN + 10, SERVO_MAX - 10])
                    target_y = clamp(IDLE_CENTER_Y + random.randint(-emotion_amp_y//2, emotion_amp_y//2), SERVO_MIN, SERVO_MAX)
                else:
                    # Vertical sweep
                    target_y = random.choice([SERVO_MIN + 10, SERVO_MAX - 10])
                    target_x = clamp(IDLE_CENTER_X + random.randint(-emotion_amp_x//2, emotion_amp_x//2), SERVO_MIN, SERVO_MAX)

                # Emotionally-scaled pause after big movements
                base_pause = random.uniform(IDLE_PAUSE_MAX * 1.5, IDLE_PAUSE_MAX * 2.5)
                idle_next_move_time = now + base_pause * pattern["pause_scale"]
            else:
                # Smaller local movements with emotional scaling
                jitter_x = random.randint(-emotion_amp_x, emotion_amp_x)
                jitter_y = random.randint(-emotion_amp_y, emotion_amp_y)
                target_x = clamp(IDLE_CENTER_X + jitter_x, SERVO_MIN, SERVO_MAX)
                target_y = clamp(IDLE_CENTER_Y + jitter_y, SERVO_MIN, SERVO_MAX)

                # Emotionally-scaled pause for small movements
                base_pause = random.uniform(IDLE_PAUSE_MIN, IDLE_PAUSE_MAX)
                idle_next_move_time = now + base_pause * pattern["pause_scale"]

        # Movement toward targets with emotional easing
        emotional_easing = IDLE_EASING * pattern["easing_scale"]
        servo_x = smooth_step(servo_x, target_x, emotional_easing)
        servo_y = smooth_step(servo_y, target_y, emotional_easing)

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
