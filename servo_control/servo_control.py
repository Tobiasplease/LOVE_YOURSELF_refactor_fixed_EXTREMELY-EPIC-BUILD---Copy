import serial
import time

ANGLE_THRESHOLD = 1  # Reduce to 1 degree for much smoother motion


class ServoController:
    def __init__(self, port="COM3", baudrate=9600):
        try:
            self.ser = serial.Serial(port, baudrate, timeout=1)
            self.serial = self.ser  # For optional external use
            self.serial.setDTR(False)  # type: ignore
            time.sleep(1)
            self.serial.setDTR(True)  # type: ignore
            time.sleep(2)
            print(f"[ServoController] Connected on {port} at {baudrate} baud.")
        except serial.SerialException as e:
            print(f"[ERROR] Could not connect to {port}: {e}")
            self.ser = None
            self.serial = None

        self.last_sent = {}

    def send(self, message: str, key=None):
        if not self.ser or not self.ser.is_open:
            return
        if key and self.last_sent.get(key) == message:
            return
        full = message.strip() + "\n"
        self.ser.write(full.encode("utf-8"))
        if key:
            self.last_sent[key] = message

    def set_pan(self, angle: int):
        if self._should_send("pan", angle):
            self.send(f"PAN:{angle}", key="pan")

    def set_tilt(self, angle: int):
        if self._should_send("tilt", angle):
            self.send(f"TILT:{angle}", key="tilt")

    def set_lung(self, mode: str):
        self.send(f"LUNG:{mode}", key="lung_mode")

    def set_lung_position(self, angle: int, force=False):
        if force or self._should_send("lung_angle", angle):
            self.send(f"LUNG:{angle}", key="lung_angle")

    def set_hand_finger(self, finger_index: int, angle: int):
        """Set individual finger position (0-3 for fingers, angle 40-130)."""
        if self._should_send(f"finger{finger_index}", angle):
            self.send(f"FINGER{finger_index}:{angle}", key=f"finger{finger_index}")
    
    def set_hand_gesture(self, finger_angles: dict):
        """Set all finger positions from dict: {"finger0": angle, "finger1": angle, ...}"""
        for finger_key, angle in finger_angles.items():
            if finger_key.startswith("finger"):
                finger_index = int(finger_key[-1])  # Extract number from "finger0", "finger1", etc.
                self.set_hand_finger(finger_index, angle)
    
    def enable_hand_tapping(self, enable: bool = True):
        """Enable/disable the tapping behavior on the tap finger."""
        command = "TAP:ON" if enable else "TAP:OFF"
        self.send(command, key="tap_mode")

    def _should_send(self, key: str, new_angle: int) -> bool:
        last_msg = self.last_sent.get(key)
        if not last_msg:
            return True
        try:
            last_angle = int(last_msg.split(":")[1])
            return abs(last_angle - new_angle) >= ANGLE_THRESHOLD
        except (IndexError, ValueError):
            return True