import math
import random
import time

from config.config import EASING_FACTOR, LUNG_MAX, LUNG_MIN

lung_eased = 90.0  # SPARKLE persistent easing memory

MIN_LUNG_SPEED = 1.0  # fast cycle (high mood)
MAX_LUNG_SPEED = 12.0  # slow cycle (low mood)

# SPARKLE Breathing mode state
breath_mode = "BIRTH_WAKE"
mode_timer = time.time() + 6  # Birth mode lasts ~6s


def update_lung_position(
    current_emotion_state,  # Now takes 5-state emotion instead of scalar mood
    person_present,
    delta,
    lung_angle,
    breath_speed,
    breath_paused,
    last_breath_direction,
    pause_start_time,
    pause_duration,
    servo_controller=None,
):
    global lung_eased, breath_mode, mode_timer

    now = time.time()

    # === 5-State Emotional Breathing Patterns ===
    breathing_patterns = {
        "energized_engaged": {
            "base_speed": 0.5,  # Fast, excited breathing
            "amplitude": 1.4,  # Deep breaths
            "pause_scale": 0.2,  # Very short pauses
            "special_modes": ["FAST_BURST", "EXCITED_PANTING"],
        },
        "alert_curious": {
            "base_speed": 0.8,  # Quick, attentive breathing
            "amplitude": 1.1,  # Slightly deeper
            "pause_scale": 0.4,  # Quick pauses for attention
            "special_modes": ["FAST_BURST", "ALERT_HOLD"],
        },
        "calm_observant": {
            "base_speed": 2.5,  # Slow, meditative breathing
            "amplitude": 1.0,  # Normal depth
            "pause_scale": 0.7,  # Contemplative pauses
            "special_modes": ["SLOW_SIGH", "NORMAL"],
        },
        "quiet_detached": {
            "base_speed": 3.5,  # Shallow, minimal breathing
            "amplitude": 0.6,  # Shallow breaths
            "pause_scale": 1.0,  # Moderate hesitant pauses
            "special_modes": ["SHALLOW", "NORMAL"],
        },
        "withdrawn_distant": {
            "base_speed": 5.0,  # Very slow, listless breathing
            "amplitude": 0.8,  # Somewhat shallow
            "pause_scale": 1.5,  # Longer pauses but not excessive
            "special_modes": ["SLOW_SIGH", "DEPRESSED_HOLD"],
        },
    }

    # Get current emotional pattern (fallback to calm if unknown)
    pattern = breathing_patterns.get(current_emotion_state, breathing_patterns["calm_observant"])
    base_speed = pattern["base_speed"]

    # === Breathing mode transitions based on emotion ===
    if now > mode_timer:
        if breath_mode == "BIRTH_WAKE":
            breath_mode = "NORMAL"
            mode_timer = now + random.uniform(3, 5)
        elif random.random() < 0.15:  # 15% chance of special breathing
            breath_mode = random.choice(pattern["special_modes"])
            mode_timer = now + random.uniform(2, 6)
        else:
            breath_mode = "NORMAL"
            mode_timer = now + random.uniform(3, 8)

    # === Apply breathing mode multiplier ===
    mode_multipliers = {
        "FAST_BURST": 0.25,
        "EXCITED_PANTING": 0.15,
        "ALERT_HOLD": 1.0,
        "SLOW_SIGH": 2.2,
        "SHALLOW": 1.5,
        "DEPRESSED_HOLD": 3.0,
        "BIRTH_WAKE": 0.4,
        "NORMAL": 1.0,
    }

    multiplier = mode_multipliers.get(breath_mode, 1.0)
    target_breath_speed = max(0.3, base_speed * multiplier)

    effective_speed = breath_speed
    breath_speed = breath_speed * 0.85 + target_breath_speed * 0.15

    angular_speed = 2 * math.pi / effective_speed

    # === Breathing phase movement ===
    breath_phase = math.sin(lung_angle)

    # === Emotion-based pause modulation ===
    pause_scale = pattern["pause_scale"]
    mode_pause_modifier = {
        "FAST_BURST": 0.2,
        "EXCITED_PANTING": 0.1,
        "ALERT_HOLD": 0.8,
        "SLOW_SIGH": 1.8,
        "SHALLOW": 1.4,
        "DEPRESSED_HOLD": 2.5,
        "BIRTH_WAKE": 0.1,
        "NORMAL": 1.0,
    }.get(breath_mode, 1.0)

    dynamic_pause = pause_duration * pause_scale * mode_pause_modifier

    if not breath_paused:
        if breath_phase > 0.98 and last_breath_direction != "up":
            breath_paused = True
            pause_start_time = time.time()
            last_breath_direction = "up"
        elif breath_phase < -0.98 and last_breath_direction != "down":
            breath_paused = True
            pause_start_time = time.time()
            last_breath_direction = "down"
        else:
            lung_angle += angular_speed * delta
    else:
        if time.time() - pause_start_time > dynamic_pause:
            breath_paused = False

    # === Emotion-based amplitude modulation ===
    base_amplitude = pattern["amplitude"]

    mode_amplitude_modifier = {
        "FAST_BURST": 1.3,
        "EXCITED_PANTING": 1.1,
        "ALERT_HOLD": 1.0,
        "SLOW_SIGH": 1.4,
        "SHALLOW": 0.7,
        "DEPRESSED_HOLD": 0.9,
        "BIRTH_WAKE": 0.4,
        "NORMAL": 1.0,
    }.get(breath_mode, 1.0)

    amplitude_multiplier = base_amplitude * mode_amplitude_modifier

    # Small phase offset for natural variation
    offset = 0.1
    raw_lung = (math.sin(lung_angle + offset) * amplitude_multiplier + 1) / 2
    raw_lung = max(0.0, min(1.0, raw_lung))

    target_lung = raw_lung * (LUNG_MAX - LUNG_MIN) + LUNG_MIN

    # SPARKLE Restore config-based easing instead of dynamic
    lung_eased = lung_eased * (1 - EASING_FACTOR) + target_lung * EASING_FACTOR

    if servo_controller:
        try:
            servo_controller.set_lung_position(int(lung_eased), force=True)
        except Exception as e:
            print(f"[ERROR] Lung servo command failed: {e}")
            # Don't crash the whole system for servo errors

    return int(lung_eased), lung_angle, breath_speed, breath_paused, last_breath_direction, pause_start_time
