import time
import random
from config.config import (
    SERVO_MIN,
    SERVO_MAX,
    FLIP_X,
    FLIP_Y,
)

# === SIMPLIFIED STATE MANAGEMENT ===
servo_x = 90
servo_y = 90
target_x = 90
target_y = 90
last_seen_time = time.time() - 10  # Start as if we've been idle for 10 seconds
state = "idle"
idle_next_move_time = 0  # Trigger immediate movement

# === CONFIGURABLE PARAMETERS ===
FACE_TIMEOUT = 3.0   # Wait longer before going idle
TRACK_EASING = 0.15  # Smooth but responsive tracking
DEAD_ZONE = 5        # Reasonable threshold
IDLE_CENTER_X = 90
IDLE_CENTER_Y = 90
IDLE_RANGE = 35      # Much larger sweeping movements  
IDLE_PAUSE_MIN = 3.0   # Shorter pause between moves
IDLE_PAUSE_MAX = 12.0  # Still good pausing
IDLE_EASING = 0.12   # Faster than original but not too aggressive
SWEEP_PROBABILITY = 0.6  # 60% chance of big sweeping movement


def clamp(val, min_val, max_val):
    return max(min_val, min(max_val, val))


def update_gaze(frame, face_box, current_mood=0.0):
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
            servo_x = smooth_step(servo_x, target_x, TRACK_EASING)
        if dy > DEAD_ZONE:
            servo_y = smooth_step(servo_y, target_y, TRACK_EASING)

    elif state == "tracking" and now - last_seen_time < FACE_TIMEOUT:
        # Grace period - hold position
        state = "grace"

    elif state in ["tracking", "grace"] and now - last_seen_time >= FACE_TIMEOUT:
        # Transition to idle
        state = "idle"
        idle_next_move_time = now + random.uniform(IDLE_PAUSE_MIN, IDLE_PAUSE_MAX)

    elif state == "idle":
        # Dynamic idle behavior with sweeping movements
        if now >= idle_next_move_time:
            # Decide between small local movement or big sweep
            if random.random() < SWEEP_PROBABILITY:
                # Big sweeping movement across the full range
                if random.choice([True, False]):
                    # Horizontal sweep
                    target_x = random.choice([SERVO_MIN + 10, SERVO_MAX - 10])
                    target_y = clamp(IDLE_CENTER_Y + random.randint(-20, 20), SERVO_MIN, SERVO_MAX)
                else:
                    # Vertical sweep  
                    target_y = random.choice([SERVO_MIN + 10, SERVO_MAX - 10])
                    target_x = clamp(IDLE_CENTER_X + random.randint(-20, 20), SERVO_MIN, SERVO_MAX)
                    
                # Longer pause after big movements to "observe" and complete movement
                idle_next_move_time = now + random.uniform(IDLE_PAUSE_MAX * 1.5, IDLE_PAUSE_MAX * 2.5)
            else:
                # Smaller local movements around center
                jitter_x = random.randint(-IDLE_RANGE, IDLE_RANGE)
                jitter_y = random.randint(-IDLE_RANGE, IDLE_RANGE)
                target_x = clamp(IDLE_CENTER_X + jitter_x, SERVO_MIN, SERVO_MAX)
                target_y = clamp(IDLE_CENTER_Y + jitter_y, SERVO_MIN, SERVO_MAX)
                
                # Shorter pause for small movements
                idle_next_move_time = now + random.uniform(IDLE_PAUSE_MIN, IDLE_PAUSE_MAX)

        # Movement toward targets with good easing
        servo_x = smooth_step(servo_x, target_x, IDLE_EASING)
        servo_y = smooth_step(servo_y, target_y, IDLE_EASING)

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
