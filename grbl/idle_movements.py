"""
GRBL Idle Movement Controller
Generates smooth, organic movements for CNC when not drawing
Movements concentrated in far corner away from home position
"""

import math
import random
import time
from typing import Optional, Tuple

try:
    from config.config import (
        GRBL_IDLE_CENTER,
        GRBL_IDLE_FEED_RATE,
        GRBL_IDLE_RADIUS_MAX,
        GRBL_IDLE_RADIUS_MIN,
        GRBL_IDLE_UPDATE_INTERVAL,
        GRBL_IDLE_ZONE,
    )
except ImportError:
    # Safe fallback defaults matching your calibrated system
    GRBL_IDLE_CENTER = (30, 30)
    GRBL_IDLE_RADIUS_MIN = 5
    GRBL_IDLE_RADIUS_MAX = 8
    GRBL_IDLE_FEED_RATE = 500
    GRBL_IDLE_ZONE = (20, 40, 20, 40)
    GRBL_IDLE_UPDATE_INTERVAL = 3.0


class IdleMovementController:
    """Generate organic idle movements using Lissajous curves"""

    def __init__(
        self,
        center: Tuple[float, float] = GRBL_IDLE_CENTER,
        radius_min: float = GRBL_IDLE_RADIUS_MIN,
        radius_max: float = GRBL_IDLE_RADIUS_MAX,
        feed_rate: int = GRBL_IDLE_FEED_RATE,
        boundary: Tuple[float, float, float, float] = GRBL_IDLE_ZONE,
    ):
        self.center_x, self.center_y = center
        self.radius_min = radius_min
        self.radius_max = radius_max
        self.feed_rate = feed_rate
        self.boundary = boundary  # (x_min, x_max, y_min, y_max)

        # Movement pattern parameters
        self.time_offset = 0.0
        self.radius = radius_min
        self.radius_target = radius_min

        # Lissajous curve parameters
        self.freq_x = 2.0  # X frequency multiplier
        self.freq_y = 3.0  # Y frequency multiplier
        self.phase_shift = math.pi / 4  # Phase difference

        # Variation parameters for organic feel
        self.freq_variation = 0.0
        self.phase_variation = 0.0

        # Cycle management for longer movement sequences
        self.current_cycle_duration = 0
        self.cycle_start_time = 0
        self.movements_in_cycle = 0

        # Emotional state (can be set externally)
        self.emotion_state = "calm_observant"

    def set_emotion_state(self, emotion: str):
        """Adjust movement parameters based on emotional state"""
        self.emotion_state = emotion

        emotion_params = {
            "energized_engaged": {
                "radius": self.radius_max,
                "freq_x": 0.8,
                "freq_y": 1.2,
                "feed_rate": 800,
                "variation": 0.4,
                "speed_variation": 0.3,
            },
            "alert_curious": {
                "radius": self.radius_max * 0.8,
                "freq_x": 0.5,
                "freq_y": 0.8,
                "feed_rate": 650,
                "variation": 0.3,
                "speed_variation": 0.25,
            },
            "calm_observant": {
                "radius": (self.radius_min + self.radius_max) / 2,
                "freq_x": 0.4,
                "freq_y": 0.6,
                "feed_rate": 500,
                "variation": 0.2,
                "speed_variation": 0.2,
            },
            "quiet_detached": {
                "radius": self.radius_min * 1.5,
                "freq_x": 0.25,
                "freq_y": 0.4,
                "feed_rate": 350,
                "variation": 0.15,
                "speed_variation": 0.15,
            },
            "withdrawn_distant": {
                "radius": self.radius_min,
                "freq_x": 0.15,
                "freq_y": 0.25,
                "feed_rate": 250,
                "variation": 0.1,
                "speed_variation": 0.1,
            },
        }

        params = emotion_params.get(emotion, emotion_params["calm_observant"])
        self.radius_target = params["radius"]
        self.freq_x = params["freq_x"]
        self.freq_y = params["freq_y"]
        self.feed_rate = params["feed_rate"]
        variation_amount = params["variation"]
        self.speed_variation = params["speed_variation"]

        # Add more randomness for organic feel
        self.freq_variation = random.uniform(-variation_amount, variation_amount)
        self.phase_variation = random.uniform(-variation_amount, variation_amount)

        # Random frequency drift for less predictable patterns
        self.freq_drift_x = random.uniform(-0.1, 0.1)
        self.freq_drift_y = random.uniform(-0.1, 0.1)

        # Start new movement cycle with random duration
        self.start_new_cycle()

    def generate_position(self, time_t: Optional[float] = None) -> Tuple[float, float]:
        """Generate next position using Lissajous curves"""
        if time_t is None:
            time_t = time.time() * 0.5  # Slow down time for smoother motion

        # Smoothly transition radius
        self.radius += (self.radius_target - self.radius) * 0.1

        # Add breathing effect to radius
        breathing = 1.0 + 0.2 * math.sin(time_t * 0.3)
        current_radius = self.radius * breathing

        # Calculate Lissajous curve position
        x = self.center_x + current_radius * math.sin((self.freq_x + self.freq_variation) * time_t)
        y = self.center_y + current_radius * math.sin((self.freq_y + self.freq_variation) * time_t + self.phase_shift + self.phase_variation)

        # Enforce boundaries
        x = max(self.boundary[0], min(self.boundary[1], x))
        y = max(self.boundary[2], min(self.boundary[3], y))

        return x, y

    def generate_smooth_path(self, duration: float = 10.0, steps: int = 100) -> list:
        """Generate a smooth path over specified duration"""
        path = []
        start_time = self.time_offset
        time_step = duration / steps

        for i in range(steps):
            t = start_time + i * time_step
            x, y = self.generate_position(t)
            path.append((x, y, self.feed_rate))

        self.time_offset = start_time + duration
        return path

    def get_gcode_command(self, x: float, y: float, feed_rate: Optional[int] = None) -> str:
        """Convert position to G-code command with speed variation"""
        if feed_rate is None:
            # Add random speed variation for more organic movement
            speed_mult = 1.0 + random.uniform(-self.speed_variation, self.speed_variation)
            feed_rate = int(self.feed_rate * speed_mult)
            feed_rate = max(100, min(2000, feed_rate))  # Clamp to reasonable range
        return f"G1 X{x:.3f} Y{y:.3f} F{feed_rate}"

    def start_new_cycle(self):
        """Start a new movement cycle with random duration"""
        import time

        # Cycle durations: mostly short, occasionally much longer
        cycle_type = random.choice(
            ["short", "medium", "long", "extended"]  # 50% - 30-90 seconds  # 30% - 2-4 minutes  # 15% - 5-8 minutes  # 5% - 10-20 minutes
        )

        if cycle_type == "short":
            self.current_cycle_duration = random.uniform(30, 90)
        elif cycle_type == "medium":
            self.current_cycle_duration = random.uniform(120, 240)
        elif cycle_type == "long":
            self.current_cycle_duration = random.uniform(300, 480)
        else:  # extended
            self.current_cycle_duration = random.uniform(600, 1200)

        self.cycle_start_time = time.time()
        self.movements_in_cycle = 0
        print(f"[🌊] Starting new {cycle_type} cycle ({self.current_cycle_duration/60:.1f} minutes)")

    def should_start_new_cycle(self) -> bool:
        """Check if current cycle is complete"""
        import time

        return (time.time() - self.cycle_start_time) > self.current_cycle_duration

    def get_pen_height_command(self) -> str:
        """Generate subtle pen height variation command"""
        # Pen heights: S10-S25 = safe up range, S50 = down (never use S50)
        # Create subtle breathing-like height variations well above paper
        base_height = 15  # Much higher base position
        height_variation = random.choice(
            [
                0,  # 40% - stay at base height (S15)
                2,  # 25% - slightly lower (S17)
                4,  # 20% - bit more lower (S19)
                6,  # 10% - noticeably lower but still safe (S21)
                8,  # 5% - lowest safe height (S23)
            ]
        )

        height = base_height + height_variation
        return f"M3 S{height}"

    def generate_micro_movement(self, current_x: float, current_y: float) -> Tuple[float, float]:
        """Generate tiny movements for contemplative moments instead of stopping"""
        # Very small movements (1-3mm) around current position
        micro_radius = random.uniform(0.5, 2.5)
        angle = random.uniform(0, 2 * math.pi)

        # Add tiny shift with boundary checking
        new_x = current_x + micro_radius * math.cos(angle)
        new_y = current_y + micro_radius * math.sin(angle)

        # Keep within boundaries
        new_x = max(self.boundary[0], min(self.boundary[1], new_x))
        new_y = max(self.boundary[2], min(self.boundary[3], new_y))

        return new_x, new_y

    def generate_contemplation_sequence(self, start_x: float, start_y: float, duration_seconds: float) -> list:
        """Generate a sequence of micro-movements for contemplative moments"""
        sequence = []
        num_micros = random.randint(3, 8)  # 3-8 tiny movements

        current_x, current_y = start_x, start_y

        for i in range(num_micros):
            # Generate next micro position
            current_x, current_y = self.generate_micro_movement(current_x, current_y)

            # Very slow feed rate for contemplative micro-movements
            micro_feed_rate = random.randint(150, 300)  # Very slow and gentle

            sequence.append((current_x, current_y, micro_feed_rate))

        return sequence

    def reset(self):
        """Reset movement controller to initial state"""
        self.time_offset = 0.0
        self.radius = self.radius_min
        self.radius_target = self.radius_min
        self.freq_variation = 0.0
        self.phase_variation = 0.0
        self.movements_in_cycle = 0
        self.start_new_cycle()
