import time
import random
import math
from config.config import (
    SERVO_MIN, SERVO_MAX,
    FLIP_X, FLIP_Y, DEAD_ZONE,
    IDLE_CENTER_X, IDLE_CENTER_Y,
    IDLE_SPEED_MIN, IDLE_SPEED_MAX,
)

servo_x = 90
servo_y = 90
target_x = 90
target_y = 90

# Independent idle targets and timers
idle_target_x = IDLE_CENTER_X
idle_target_y = IDLE_CENTER_Y
idle_hold_until_x = 0
idle_hold_until_y = 0
idle_speed_x = 0.1
idle_speed_y = 0.1

last_seen_time = time.time()
face_lock_start = 0

# === CONFIGURABLE PARAMETERS ===
FACE_LOCK_TIMEOUT = 6.0
TRACK_EASING = 0.5
IDLE_PAUSE_MIN = 0.5
IDLE_PAUSE_MAX = 5.0
IDLE_JITTER = 40
SYNC_PROBABILITY = 0.1  # 10% chance that both axes move together

def clamp(val, min_val, max_val):
    return max(min_val, min(max_val, val))

def ease_in_out(current, target, t):
    # t in 0.0–1.0 range; smooth cosine-based easing
    t = clamp(t, 0.0, 1.0)
    eased = (1 - math.cos(t * math.pi)) / 2
    return current + (target - current) * eased

def update_gaze(frame, face_box, current_mood=0.0):
    global servo_x, servo_y, target_x, target_y
    global idle_target_x, idle_target_y
    global idle_hold_until_x, idle_hold_until_y
    global idle_speed_x, idle_speed_y
    global last_seen_time, face_lock_start

    h, w = frame.shape[:2]
    person_present = face_box is not None
    now = time.time()

    # === FACE TRACKING ===
    if person_present:
        (startX, startY, endX, endY) = face_box
        face_center_x = (startX + endX) // 2
        face_center_y = (startY + endY) // 2
        if FLIP_X:
            face_center_x = w - face_center_x
        if FLIP_Y:
            face_center_y = h - face_center_y

        dx = face_center_x - (w // 2)
        dy = face_center_y - (h // 2)

        face_movement = abs(dx) + abs(dy)
        if face_movement > 15:
            face_lock_start = now

        if now - face_lock_start < FACE_LOCK_TIMEOUT:
            if abs(dx) > DEAD_ZONE:
                target_x = clamp(target_x + dx * 0.05, SERVO_MIN, SERVO_MAX)
            if abs(dy) > DEAD_ZONE:
                target_y = clamp(target_y + dy * 0.05, SERVO_MIN, SERVO_MAX)

            servo_x = smooth_step(servo_x, target_x, TRACK_EASING)
            servo_y = smooth_step(servo_y, target_y, TRACK_EASING)
            last_seen_time = now
        else:
            person_present = False  # break away

    # === IDLE WANDERING GAZE ===
    if not person_present and now - last_seen_time > FACE_LOCK_TIMEOUT:

        sync_axes = random.random() < SYNC_PROBABILITY

        if now > idle_hold_until_x or sync_axes:
            jitter_x = random.randint(-IDLE_JITTER, IDLE_JITTER)
            idle_target_x = clamp(IDLE_CENTER_X + jitter_x, SERVO_MIN, SERVO_MAX)
            idle_hold_until_x = now + random.uniform(IDLE_PAUSE_MIN, IDLE_PAUSE_MAX)
            idle_speed_x = random.uniform(IDLE_SPEED_MIN, IDLE_SPEED_MAX)

        if now > idle_hold_until_y or sync_axes:
            jitter_y = random.randint(-IDLE_JITTER, IDLE_JITTER)
            idle_target_y = clamp(IDLE_CENTER_Y + jitter_y, SERVO_MIN, SERVO_MAX)
            idle_hold_until_y = now + random.uniform(IDLE_PAUSE_MIN, IDLE_PAUSE_MAX)
            idle_speed_y = random.uniform(IDLE_SPEED_MIN, IDLE_SPEED_MAX)

        # Eased wandering movement
        servo_x = ease_in_out(servo_x, idle_target_x, idle_speed_x)
        servo_y = ease_in_out(servo_y, idle_target_y, idle_speed_y)

    return person_present, round(servo_x), round(servo_y)

def smooth_step(current, target, factor):
    return current + (target - current) * factor
