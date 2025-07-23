import math
import time
import random
from config.config import LUNG_MIN, LUNG_MAX, PAUSE_DURATION, EASING_FACTOR

lung_eased = 90.0  # ✨ persistent easing memory

MIN_LUNG_SPEED = 1.0  # fast cycle (high mood)
MAX_LUNG_SPEED = 12.0  # slow cycle (low mood)

# ✨ Breathing mode state
breath_mode = "BIRTH_WAKE"
mode_timer = time.time() + 6  # Birth mode lasts ~6s

def update_lung_position(current_mood, person_present, delta, lung_angle, breath_speed,
                         breath_paused, last_breath_direction, pause_start_time, pause_duration,
                         servo_controller=None):
    global lung_eased, breath_mode, mode_timer

    now = time.time()

    # === Normalize mood to 0..1 range for scaling ===
    mood_clamped = max(0.0, min(1.0, (current_mood + 1.0) / 2.0))

    # === Determine base breathing speed from mood ===
    mood_factor = 1.0 - mood_clamped  # high mood = 0, low mood = 1
    base_speed = MIN_LUNG_SPEED + (MAX_LUNG_SPEED - MIN_LUNG_SPEED) * mood_factor

    # === Breathing mode transitions ===
    if now > mode_timer:
        if breath_mode == "BIRTH_WAKE":
            breath_mode = "NORMAL"
            mode_timer = now + random.uniform(3, 5)
        elif abs(current_mood) > 0.7 and random.random() < 0.2:
            breath_mode = "FAST_BURST"
            mode_timer = now + random.uniform(2, 4)
        elif random.random() < 0.05:
            breath_mode = "SLOW_SIGH"
            mode_timer = now + random.uniform(4, 6)
        else:
            breath_mode = "NORMAL"
            mode_timer = now + random.uniform(3, 5)

    # === Apply breathing mode multiplier ===
    if breath_mode == "FAST_BURST":
        target_breath_speed = max(0.5, base_speed * 0.25)
    elif breath_mode == "SLOW_SIGH":
        target_breath_speed = base_speed * 2.2
    elif breath_mode == "BIRTH_WAKE":
        target_breath_speed = 0.4  # ✨ Extra fast startup
    else:
        target_breath_speed = base_speed

    effective_speed = breath_speed
    breath_speed = breath_speed * 0.85 + target_breath_speed * 0.15

    angular_speed = 2 * math.pi / effective_speed

    # === Breathing phase movement ===
    breath_phase = math.sin(lung_angle)

    # ✨ Mood-scaled dynamic pause modulation ===
    pause_mood_scale = 1.5 - mood_clamped  # low mood = longer pause
    mode_modifier = {
        "FAST_BURST": 0.2,
        "SLOW_SIGH": 1.6,
        "BIRTH_WAKE": 0.1,
        "NORMAL": 1.0
    }.get(breath_mode, 1.0)

    dynamic_pause = pause_duration * pause_mood_scale * mode_modifier

    if not breath_paused:
        if breath_phase > 0.98 and last_breath_direction != 'up':
            breath_paused = True
            pause_start_time = time.time()
            last_breath_direction = 'up'
        elif breath_phase < -0.98 and last_breath_direction != 'down':
            breath_paused = True
            pause_start_time = time.time()
            last_breath_direction = 'down'
        else:
            lung_angle += angular_speed * delta
    else:
        if time.time() - pause_start_time > dynamic_pause:
            breath_paused = False

    # === Offset and amplitude modulation ===
    offset = 0.2 * current_mood

    amplitude_multiplier = 1.0 + (0.8 * (1.0 - mood_clamped))

    if breath_mode == "FAST_BURST":
        amplitude_multiplier *= 1.3
    elif breath_mode == "SLOW_SIGH":
        amplitude_multiplier *= 1.4
    elif breath_mode == "BIRTH_WAKE":
        amplitude_multiplier *= 0.4

    raw_lung = (math.sin(lung_angle + offset) * amplitude_multiplier + 1) / 2
    raw_lung = max(0.0, min(1.0, raw_lung))

    target_lung = raw_lung * (LUNG_MAX - LUNG_MIN) + LUNG_MIN

    # ✨ Restore config-based easing instead of dynamic
    lung_eased = lung_eased * (1 - EASING_FACTOR) + target_lung * EASING_FACTOR

    if servo_controller:
        servo_controller.set_lung_position(int(lung_eased), force=True)

    return int(lung_eased), lung_angle, breath_speed, breath_paused, last_breath_direction, pause_start_time
