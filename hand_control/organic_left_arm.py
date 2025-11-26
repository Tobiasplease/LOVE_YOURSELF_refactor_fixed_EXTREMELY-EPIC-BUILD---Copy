#!/usr/bin/env python3
"""
Organic Left Arm Movement Controller
===================================

Handles organic, life-like movement for left arm servos with:
- Long random pauses (3-20+ seconds)
- Variable movement speeds
- Natural position changes
- Breathing-like rhythm
"""

import random
import time
import threading
from typing import Optional, Dict, Any

try:
    from mood.mood import MoodEngine
except ImportError:
    MoodEngine = None


class OrganicLeftArmController:
    """Controls organic movement patterns for left arm servos."""

    def __init__(self, serial_connection=None, mood_engine=None):
        self.serial_connection = serial_connection
        self.mood_engine = mood_engine
        self.running = False
        self.enabled = False
        self.thread: Optional[threading.Thread] = None
        self.current_emotional_state = "neutral"

        # Movement parameters - individualized per servo
        self.servo_count = 2

        # Pin 4: 81°-95° (center 88°), Pin 5: 88°-102° (center 95°) - pin 4 shifted 7° lower
        self.center_positions = [88, 95]  # [pin4, pin5]
        self.min_positions = [81, 88]     # [pin4, pin5]
        self.max_positions = [95, 102]   # [pin4, pin5]

        # Current state for each servo
        self.current_positions = self.center_positions.copy()
        # Initialize with different random targets to start movement
        self.target_positions = []
        for i in range(self.servo_count):
            target = random.randint(self.min_positions[i], self.max_positions[i])
            self.target_positions.append(target)
            print(f"🎯 Servo {i} initial target: {target}° (current: {self.center_positions[i]}°)")
        self.last_move_time = [time.time()] * self.servo_count
        self.last_speed_change = [time.time()] * self.servo_count

        # Coordinated movement state
        self.movement_mode = "independent"  # "independent", "coordinated", "sweep", "retract"
        self.mode_change_time = time.time()
        self.mode_duration = random.uniform(5.0, 15.0)  # Time to stay in current mode

        # Pause system
        self.in_pause = [False] * self.servo_count
        self.pause_start = [0.0] * self.servo_count
        self.pause_duration = [0.0] * self.servo_count

        # Movement speeds (seconds between moves)
        self.current_speeds = [1.0] * self.servo_count  # Start at 1 second intervals

        # Organic movement constants - more reactive while still smooth
        self.min_speed = 0.02  # Fastest: 20ms between moves (very responsive)
        self.max_speed = 0.15  # Slowest: 150ms between moves (responsive but smooth)
        self.speed_change_interval = 8.0   # Change speed every 8 seconds (more dynamic)


        # Base pause constants - adjusted by emotional state (less frequent pauses)
        self.base_pause_chance = 0.03  # Reduced from 0.05 - less frequent pauses
        self.base_min_pause = 8.0      # Slightly shorter minimum pause
        self.base_max_pause = 25.0     # Slightly shorter maximum pause
        self.base_long_pause_chance = 0.3  # Reduced from 0.4 - fewer long pauses
        self.base_long_pause_duration = 45.0  # Shorter long pauses

        # Current emotional modifiers (updated by mood system)
        self.pause_chance = self.base_pause_chance
        self.min_pause_duration = self.base_min_pause
        self.max_pause_duration = self.base_max_pause
        self.long_pause_chance = self.base_long_pause_chance
        self.long_pause_duration = self.base_long_pause_duration

        # Emotional state profiles (more pronounced but still natural)
        self.emotional_profiles = {
            "excited": {
                "pause_multiplier": 0.5,  # Much shorter pauses
                "speed_multiplier": 0.6,  # Faster movement
                "range_multiplier": 1.2   # Slightly larger range
            },
            "happy": {
                "pause_multiplier": 0.7,  # Shorter pauses
                "speed_multiplier": 0.8,  # Faster movement
                "range_multiplier": 1.1
            },
            "neutral": {
                "pause_multiplier": 1.0,
                "speed_multiplier": 1.0,
                "range_multiplier": 1.0
            },
            "sad": {
                "pause_multiplier": 1.5,  # Longer pauses
                "speed_multiplier": 1.4,  # Slower movement
                "range_multiplier": 0.8   # Smaller range
            },
            "angry": {
                "pause_multiplier": 0.4,  # Very short pauses
                "speed_multiplier": 0.5,  # Much faster movement
                "range_multiplier": 1.3   # Larger range within limits
            }
        }

        print("🤖 Organic Left Arm Controller initialized with mood integration")

    def set_serial_connection(self, serial_connection):
        """Update the serial connection."""
        self.serial_connection = serial_connection

    def set_mood_engine(self, mood_engine):
        """Update the mood engine reference."""
        self.mood_engine = mood_engine

    def update_emotional_state(self, emotion_state: str, mood_value: float = 0.0):
        """Receive mood updates from the mood system (called externally)."""
        if emotion_state != self.current_emotional_state:
            self.current_emotional_state = emotion_state
            self._apply_emotional_modifiers()
            print(f"🌊 Left arm adapting to {emotion_state} mood")

    def _apply_emotional_modifiers(self):
        """Apply very subtle emotional modifiers to movement parameters."""
        profile = self.emotional_profiles.get(self.current_emotional_state, self.emotional_profiles["neutral"])

        # Apply VERY subtle modifications
        self.pause_chance = self.base_pause_chance
        self.min_pause_duration = self.base_min_pause * profile["pause_multiplier"]
        self.max_pause_duration = self.base_max_pause * profile["pause_multiplier"]
        self.long_pause_duration = self.base_long_pause_duration * profile["pause_multiplier"]

        # Adjust movement speeds slightly
        speed_mod = profile["speed_multiplier"]
        self.min_speed = 0.05 * speed_mod
        self.max_speed = 0.3 * speed_mod

        # Adjust range very subtly for each servo individually
        range_mod = profile["range_multiplier"]

        # Pin 4: base range 81°-95° (14°), Pin 5: base range 88°-102° (14°)
        base_ranges = [14, 14]  # Both servos have 14° base ranges

        for i in range(self.servo_count):
            adjusted_range = max(5, min(15, int(base_ranges[i] * range_mod)))  # 5°-15° range
            half_range = adjusted_range // 2

            # Pin 4: centered around 88°, bounded 81°-95°; Pin 5: centered around 95°, bounded 88°-102°
            if i == 0:  # Pin 4 (elbow) - shifted 7° lower
                self.min_positions[i] = max(81, self.center_positions[i] - half_range)
                self.max_positions[i] = min(95, self.center_positions[i] + half_range)
            else:  # Pin 5 - original range
                self.min_positions[i] = max(88, self.center_positions[i] - half_range)
                self.max_positions[i] = min(102, self.center_positions[i] + half_range)

    def enable(self):
        """Enable organic movement."""
        if self.enabled:
            return

        self.enabled = True

        # Start movement thread (no Arduino enable command needed - servos auto-attach when moved)
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._movement_loop, daemon=True)
            self.thread.start()
            print("🌊 Organic left arm movement ENABLED")

    def disable(self):
        """Disable organic movement."""
        if not self.enabled:
            return

        self.enabled = False
        print("🛑 Organic left arm movement DISABLED")

    def pause_for_drawing(self):
        """Temporarily pause organic movement for CNC drawing."""
        if self.enabled:
            self.enabled = False
            self.paused_for_drawing = True
            print("⏸️ Organic left arm PAUSED for drawing")

    def resume_after_drawing(self):
        """Resume organic movement after CNC drawing."""
        if hasattr(self, 'paused_for_drawing') and self.paused_for_drawing:
            self.enabled = True
            self.paused_for_drawing = False
            print("▶️ Organic left arm RESUMED after drawing")

    def shutdown(self):
        """Shutdown the controller."""
        self.disable()
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        print("🧹 Organic left arm controller shutdown")

    def _movement_loop(self):
        """Main movement loop running in background thread."""
        # Initialize with random pauses
        current_time = time.time()
        for i in range(self.servo_count):
            self._start_initial_pause(i, current_time)
            self._change_servo_speed(i)

        print("🌊 Organic movement loop started")

        while self.running:
            try:
                current_time = time.time()

                if self.enabled:
                    self._update_organic_movement(current_time)

                # Sleep for a short interval
                time.sleep(0.1)  # 100ms update rate

            except Exception as e:
                print(f"⚠️ Error in organic movement loop: {e}")
                time.sleep(1.0)

        print("🛑 Organic movement loop stopped")

    def _update_organic_movement(self, current_time: float):
        """Update organic movement for all servos with intelligent patterns."""
        # Check if it's time to change movement mode
        if current_time - self.mode_change_time >= self.mode_duration:
            self._change_movement_mode(current_time)

        # Handle paused servos first
        for i in range(self.servo_count):
            if self.in_pause[i]:
                remaining_pause = self.pause_duration[i] - (current_time - self.pause_start[i])
                if remaining_pause <= 0:
                    # End pause
                    self.in_pause[i] = False
                    self.last_move_time[i] = current_time
                    # print(f"🌊 Servo {i} resuming after {self.pause_duration[i]:.1f}s pause")
                    self._change_servo_speed(i)
                else:
                    continue  # Still pausing, skip movement

        # Execute movement based on current mode
        if self.movement_mode == "coordinated":
            self._update_coordinated_movement(current_time)
        elif self.movement_mode == "sweep":
            self._update_sweep_movement(current_time)
        elif self.movement_mode == "retract":
            self._update_retract_movement(current_time)
        else:  # independent
            self._update_independent_movement(current_time)

    def _move_servo(self, servo_index: int, current_time: float):
        """Move a single servo organically."""
        # Move toward target (smaller steps for much smoother movement)
        step_size = 1  # Single degree steps for smoothest movement
        if self.current_positions[servo_index] < self.target_positions[servo_index]:
            self.current_positions[servo_index] = min(self.target_positions[servo_index],
                                                    self.current_positions[servo_index] + step_size)
        elif self.current_positions[servo_index] > self.target_positions[servo_index]:
            self.current_positions[servo_index] = max(self.target_positions[servo_index],
                                                    self.current_positions[servo_index] - step_size)

        # If reached target, pick new target
        if abs(self.current_positions[servo_index] - self.target_positions[servo_index]) <= 1:
            # Pick new random target using individual servo range
            min_pos = self.min_positions[servo_index]
            max_pos = self.max_positions[servo_index]
            range_size = max_pos - min_pos
            new_target = min_pos + random.randint(0, range_size)

            # Avoid staying at same position - require minimal movement for brief cycles
            attempts = 0
            while abs(new_target - self.current_positions[servo_index]) < 3 and attempts < 10:
                new_target = min_pos + random.randint(0, range_size)
                attempts += 1

            self.target_positions[servo_index] = new_target

            # print(f"🎯 Servo {servo_index} new target: {new_target} "
            #       f"(current: {self.current_positions[servo_index]}, "
            #       f"speed: {self.current_speeds[servo_index]:.1f}s)")

        # Send position to Arduino (convert to actual servo positions for pins 4,5)
        self._send_servo_positions()

        self.last_move_time[servo_index] = current_time

        # Random pause chance
        if random.random() < self.pause_chance:
            self._start_pause(servo_index, current_time)

    def _send_servo_positions(self):
        """Send current servo positions to Arduino via direct servo commands."""
        if not self.serial_connection:
            print("🔌 No serial connection - skipping servo commands")
            return

        try:
            # Send direct servo position commands for pins 4 and 5
            # Format: "SERVO,pin,angle"
            for i, position in enumerate(self.current_positions):
                pin = 4 + i  # Pins 4 and 5 for left arm servos
                command = f"SERVO,{pin},{int(position)}\n"
                # print(f"📤 Sending: {command.strip()}")  # Suppressed debug output
                self.serial_connection.write(command.encode())
                self.serial_connection.flush()

        except Exception as e:
            print(f"⚠️ Error sending servo positions: {e}")


    def _start_pause(self, servo_index: int, current_time: float):
        """Start a pause for a servo."""
        self.in_pause[servo_index] = True
        self.pause_start[servo_index] = current_time

        # Random pause duration
        pause_range = self.max_pause_duration - self.min_pause_duration
        pause_duration = self.min_pause_duration + random.random() * pause_range

        # 15% chance for longer pause
        if random.random() < self.long_pause_chance:
            pause_duration = self.long_pause_duration

        self.pause_duration[servo_index] = pause_duration

        # print(f"⏸️ Servo {servo_index} starting {pause_duration:.1f}s pause")

    def _start_initial_pause(self, servo_index: int, current_time: float):
        """Start initial pause for a servo."""
        self.in_pause[servo_index] = True
        self.pause_start[servo_index] = current_time
        self.pause_duration[servo_index] = random.uniform(0.0, 8.0)  # 0-8 second initial pause

    def _change_servo_speed(self, servo_index: int):
        """Change the movement speed for a servo."""
        speed_range = self.max_speed - self.min_speed
        self.current_speeds[servo_index] = self.min_speed + random.random() * speed_range

        # print(f"🏃 Servo {servo_index} speed changed to {self.current_speeds[servo_index]:.1f}s")

    def get_status(self) -> Dict[str, Any]:
        """Get current status."""
        return {
            "enabled": self.enabled,
            "running": self.running,
            "current_positions": self.current_positions.copy(),
            "target_positions": self.target_positions.copy(),
            "in_pause": self.in_pause.copy(),
            "current_speeds": [f"{s:.1f}s" for s in self.current_speeds],
            "pause_remaining": [
                max(0, self.pause_duration[i] - (time.time() - self.pause_start[i]))
                if self.in_pause[i] else 0
                for i in range(self.servo_count)
            ]
        }

    def _change_movement_mode(self, current_time: float):
        """Change to a new movement mode."""
        modes = ["independent", "coordinated", "sweep", "retract"]
        weights = [0.25, 0.45, 0.25, 0.05]  # coordinated movement is now most common

        self.movement_mode = random.choices(modes, weights=weights)[0]
        self.mode_change_time = current_time
        self.mode_duration = random.uniform(5.0, 15.0)

        # print(f"🎭 Movement mode: {self.movement_mode} for {self.mode_duration:.1f}s")

    def _update_independent_movement(self, current_time: float):
        """Update independent movement for each servo."""
        for i in range(self.servo_count):
            if not self.in_pause[i]:
                # Change speed periodically
                if current_time - self.last_speed_change[i] >= self.speed_change_interval:
                    self._change_servo_speed(i)
                    self.last_speed_change[i] = current_time

                # Check if it's time to move
                if current_time - self.last_move_time[i] >= self.current_speeds[i]:
                    self._move_servo(i, current_time)

    def _update_coordinated_movement(self, current_time: float):
        """Update coordinated movement - servos move together."""
        # Use the fastest servo's timing to coordinate
        min_speed = min(self.current_speeds)

        for i in range(self.servo_count):
            if not self.in_pause[i]:
                # All servos move at the same rate when coordinated
                if current_time - self.last_move_time[i] >= min_speed:
                    self._move_servo(i, current_time)

    def _update_sweep_movement(self, current_time: float):
        """Update sweep movement - smooth coordinated sweeps."""
        # Both servos move in the same direction for sweeping motions
        for i in range(self.servo_count):
            if not self.in_pause[i]:
                if current_time - self.last_move_time[i] >= self.current_speeds[i]:
                    # In sweep mode, make targets more coordinated
                    current_pos = self.current_positions[i]
                    target_pos = self.target_positions[i]

                    # If at target, pick a new target in same general direction
                    if abs(current_pos - target_pos) <= 1:
                        direction = 1 if current_pos < self.center_positions[i] else -1
                        if random.random() < 0.3:  # 30% chance to reverse
                            direction *= -1

                        # Make a sweep target
                        if direction > 0:
                            new_target = random.randint(current_pos + 3, self.max_positions[i])
                        else:
                            new_target = random.randint(self.min_positions[i], current_pos - 3)

                        self.target_positions[i] = max(self.min_positions[i], min(self.max_positions[i], new_target))
                        # print(f"🌊 Servo {i} sweep target: {self.target_positions[i]}°")

                    self._move_servo(i, current_time)

    def _update_retract_movement(self, current_time: float):
        """Update retract movement - quick return to center."""
        for i in range(self.servo_count):
            if not self.in_pause[i]:
                # Fast movement back toward center
                if current_time - self.last_move_time[i] >= 0.1:  # Very fast for retraction
                    # Set target to center position
                    self.target_positions[i] = self.center_positions[i]
                    self._move_servo(i, current_time)

                    # If close to center, end retract mode early
                    if abs(self.current_positions[i] - self.center_positions[i]) <= 2:
                        # print(f"✨ Servo {i} retracted to center")
                        # Switch to a new random target after retraction
                        self.target_positions[i] = random.randint(self.min_positions[i], self.max_positions[i])