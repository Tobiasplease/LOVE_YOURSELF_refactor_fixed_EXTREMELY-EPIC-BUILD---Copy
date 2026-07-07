"""
Hand Expression Controller (Consolidated)
==========================================
8-servo controller using HAND8 protocol.
Adapted from standalone repo for Linux udev symlinks.
"""

import time
from typing import Optional

import serial


class HandExpressionController:
    """8-servo hand expression controller using HAND8 serial protocol."""

    def __init__(self, port: str = None, baudrate: int = 9600, clean_output: bool = True, min_angle: int = 0, max_angle: int = 180):
        self.clean_output = clean_output
        self.min_angle = min_angle
        self.max_angle = max_angle

        if port is None:
            port = self._detect_port()

        self.port = port
        self.baudrate = baudrate
        self.serial_connection = None
        self.manual_override = False

        # Command throttling (proven values from both repos)
        self.last_command_time = 0.0
        self.min_command_interval = 0.05  # 20Hz max
        self.last_sent_positions = {}
        self.position_change_threshold = 3.0

        if self.port:
            self._init_serial()
        else:
            if not self.clean_output:
                print("WARNING: No hand controller port detected - running in simulation mode")

    def _detect_port(self) -> Optional[str]:
        """Detect hand controller port via udev symlinks (Linux) or fallback."""
        import os

        # Linux udev symlinks (primary)
        candidates = ["/dev/arduino_lefthand", "/dev/arduino_righthand", "/dev/arduino_hand"]
        for candidate in candidates:
            if os.path.exists(candidate):
                if not self.clean_output:
                    print(f"Using hand controller port: {candidate}")
                return candidate

        # Fallback: try config
        try:
            import sys

            parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if parent_dir not in sys.path:
                sys.path.insert(0, parent_dir)
            from config.config import HAND_CONTROLLER_PORT

            if not self.clean_output:
                print(f"Using hand controller port from config: {HAND_CONTROLLER_PORT}")
            return HAND_CONTROLLER_PORT
        except ImportError:
            pass

        if not self.clean_output:
            print("WARNING: No hand controller port found")
        return None

    def _init_serial(self):
        """Initialize serial connection to Arduino."""
        try:
            self.serial_connection = serial.Serial(self.port, self.baudrate, timeout=1)
            time.sleep(2)  # Arduino boot time
            if not self.clean_output:
                print(f"Connected to hand controller on {self.port} at {self.baudrate} baud")
            # Send test command (8 servos at center)
            test_command = "HAND8,90,90,90,90,90,90,90,90\n"
            self.serial_connection.write(test_command.encode())
        except Exception as e:
            if not self.clean_output:
                print(f"Failed to connect to {self.port}: {e}")
            self.serial_connection = None

    def set_hand_positions(self, positions: list):
        """
        Set servo positions with throttling.
        positions: list of 8 angles (0-180 degrees)
        Pads with 90 if fewer than 8 values provided.
        """
        # Pad to 8 if needed
        while len(positions) < 8:
            positions.append(90)

        if not self.serial_connection:
            return

        current_time = time.time()
        if current_time - self.last_command_time < self.min_command_interval:
            return

        # Map to hardware range and check for changes
        mapped_positions = {}
        for i, angle in enumerate(positions[:8]):
            arduino_position = int((angle / 180.0) * (self.max_angle - self.min_angle) + self.min_angle)
            arduino_position = max(self.min_angle, min(self.max_angle, arduino_position))
            mapped_positions[f"servo{i}"] = arduino_position

        # Skip if no significant change
        if self.last_sent_positions:
            changed = False
            for name, new_pos in mapped_positions.items():
                old_pos = self.last_sent_positions.get(name, 0)
                if abs(new_pos - old_pos) > self.position_change_threshold:
                    changed = True
                    break
            if not changed:
                return

        try:
            pos_list = [mapped_positions.get(f"servo{i}", 90) for i in range(8)]
            command = f"HAND8,{','.join(map(str, pos_list))}\n"
            self.serial_connection.write(command.encode())

            self.last_command_time = current_time
            self.last_sent_positions = mapped_positions.copy()

            if not hasattr(self, "_debug_count"):
                self._debug_count = 0
            self._debug_count += 1
            if self._debug_count % 20 == 0:
                print(f"SERIAL: {command.strip()}")

        except Exception as e:
            if not self.clean_output:
                print(f"Serial write error: {e}")

    def enable_manual_override(self):
        self.manual_override = True

    def disable_manual_override(self):
        self.manual_override = False

    def cleanup(self):
        """Clean shutdown."""
        if self.serial_connection:
            try:
                self.serial_connection.close()
            except Exception:
                pass
