"""
Standalone Hand Expression Controller
====================================
Extracted from the main servo_control system for standalone operation.
Provides the exact same API as the original HandExpressionController.
"""

import os
import time
from typing import Optional

import serial


class HandExpressionController:
    """Standalone hand expression controller with exact same API as original."""

    def __init__(self, port: str = None, baudrate: int = 9600, clean_output: bool = True):
        # Set clean_output first (used in _detect_hand_controller_port)
        self.clean_output = clean_output

        # Auto-detect port if not specified
        if port is None:
            port = self._detect_hand_controller_port()

        self.port = port
        self.baudrate = baudrate
        self.serial_connection = None
        self.manual_override = False

        # Command throttling system (matches original)
        self.last_command_time = 0.0
        self.min_command_interval = 0.05  # 20Hz max
        self.last_sent_positions = {}
        self.position_change_threshold = 3.0

        if self.port:
            self._init_serial()
        else:
            if not self.clean_output:
                print("WARNING: No hand controller port detected - running in simulation mode")

    def _detect_hand_controller_port(self) -> Optional[str]:
        """Return fixed udev symlink for hand controller."""
        # Use fixed udev symlink
        port = "/dev/arduino_lefthand"
        if not self.clean_output:
            print(f"Using fixed hand controller port: {port}")
        return port

        # Note: Removed environment variable checks - using fixed udev symlink
        try:
            import os as os_mod
            import sys

            # Add parent directory to path
            parent_dir = os_mod.path.dirname(os_mod.path.dirname(os_mod.path.abspath(__file__)))
            if parent_dir not in sys.path:
                sys.path.insert(0, parent_dir)

            from config.config import HAND_CONTROLLER_PORT

            if not self.clean_output:
                print(f"Using hand controller port from config: {HAND_CONTROLLER_PORT}")
            return HAND_CONTROLLER_PORT
        except ImportError:
            if not self.clean_output:
                print("WARNING: Could not import config - no port available")
            return None

    def _init_serial(self):
        """Initialize serial connection to Arduino hand controller."""
        try:
            self.serial_connection = serial.Serial(self.port, self.baudrate, timeout=1)
            time.sleep(2)  # Arduino boot time
            if not self.clean_output:
                print(f"SUCCESS Connected to hand controller on {self.port} at {self.baudrate} baud")
            # Send test command to verify connection
            test_command = "HAND,90,90,90,90\n"
            self.serial_connection.write(test_command.encode())
            if not self.clean_output:
                print(f"📤 Test command sent: {test_command.strip()}")
        except Exception as e:
            if not self.clean_output:
                print(f"ERROR Failed to connect to {self.port}: {e}")
            self.serial_connection = None

    def set_hand_positions(self, positions: list):
        """
        Set hand positions with proper throttling.
        positions: list of 4 angles [index, middle, ring, pinky] (0-180 degrees)
        """
        if len(positions) != 4:
            raise ValueError("Must provide exactly 4 positions for 4 fingers")

        if not self.serial_connection:
            return

        current_time = time.time()

        # Rate limiting: Don't send more than 20 commands per second
        if current_time - self.last_command_time < self.min_command_interval:
            return

        # Convert to finger dictionary for position change detection
        finger_positions = {}
        for i, angle in enumerate(positions):
            # Clamp to Arduino range (40-130°)
            arduino_min = 40
            arduino_max = 130
            arduino_center = 85
            arduino_range = 90

            # Convert from 0-180° system to Arduino's 40-130° system
            offset_from_center = angle - 90.0
            arduino_offset = (offset_from_center / 90.0) * (arduino_range / 2.0)
            arduino_position = arduino_center + arduino_offset
            arduino_position = max(arduino_min, min(arduino_max, arduino_position))

            finger_positions[f"finger{i}"] = int(arduino_position)

        # Position change detection: Only send if positions changed significantly
        if self.last_sent_positions:
            position_changed = False
            for finger_name, new_pos in finger_positions.items():
                old_pos = self.last_sent_positions.get(finger_name, 0)
                if abs(new_pos - old_pos) > self.position_change_threshold:
                    position_changed = True
                    break

            if not position_changed:
                return  # Skip sending - positions haven't changed enough

        try:
            # Send command in format expected by Arduino: "HAND,f0,f1,f2,f3\n"
            pos_list = [finger_positions.get(f"finger{i}", 85) for i in range(4)]
            command = f"HAND,{','.join(map(str, pos_list))}\n"
            self.serial_connection.write(command.encode())

            # Update tracking variables
            self.last_command_time = current_time
            self.last_sent_positions = finger_positions.copy()

            # Debug output occasionally
            if not hasattr(self, "_debug_count"):
                self._debug_count = 0
            self._debug_count += 1
            if self._debug_count % 20 == 0:  # Every 20 commands
                print(f"📤 SERIAL: {command.strip()}")

        except Exception as e:
            if not self.clean_output:
                print(f"ERROR Serial write error: {e}")

    def enable_manual_override(self):
        """Enable manual override mode."""
        self.manual_override = True
        if not self.clean_output:
            print("🎮 Manual override ENABLED")

    def disable_manual_override(self):
        """Disable manual override mode."""
        self.manual_override = False
        if not self.clean_output:
            print("🤖 Manual override DISABLED")

    def enable_left_arm_movement(self):
        """Enable autonomous left arm movement."""
        if self.serial_connection:
            try:
                command = "LEFT_ARM_ENABLE\n"
                self.serial_connection.write(command.encode())
                self.serial_connection.flush()
                if not self.clean_output:
                    print("🤖 Left arm movement ENABLED")
            except Exception as e:
                if not self.clean_output:
                    print(f"ERROR Failed to enable left arm movement: {e}")

    def disable_left_arm_movement(self):
        """Disable autonomous left arm movement."""
        if self.serial_connection:
            try:
                command = "LEFT_ARM_DISABLE\n"
                self.serial_connection.write(command.encode())
                self.serial_connection.flush()
                if not self.clean_output:
                    print("🤖 Left arm movement DISABLED")
            except Exception as e:
                if not self.clean_output:
                    print(f"ERROR Failed to disable left arm movement: {e}")

    def cleanup(self):
        """Clean shutdown of hand controller."""
        if self.serial_connection:
            try:
                self.serial_connection.close()
                if not self.clean_output:
                    print("🔌 Serial connection closed")
            except Exception as e:
                if not self.clean_output:
                    print(f"ERROR Error closing serial: {e}")
