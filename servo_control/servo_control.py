import threading
import time

import serial

# Smooth movement settings for natural head motion
ANGLE_THRESHOLD = 0.3  # Even smaller threshold for ultra-smooth movement
MIN_COMMAND_INTERVAL = 0.01  # 10ms between commands (100Hz) for ultra-silky smooth motion


class ServoController:
    def __init__(self, port="COM3", baudrate=9600):
        self.port = port
        self.baudrate = baudrate
        self.ser = None
        self.serial = None
        self.last_sent = {}
        self.last_send_time = 0
        self.send_lock = threading.Lock()
        self.last_error_log = 0.0  # Throttle error logs
        self._reconnect_at = 0.0  # earliest next reconnect attempt
        self._connect(verbose=True)

    def _connect(self, verbose=False) -> bool:
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=1)
            self.serial = self.ser  # For optional external use
            # Skip DTR manipulation - causes issues with lint arduinoserial sketch
            time.sleep(0.5)  # Brief delay for Arduino initialization
            print(f"[ServoController] Connected on {self.port} at {self.baudrate} baud.")
            self.last_sent = {}  # a re-enumerated Arduino rebooted — resend everything fresh
            return True
        except serial.SerialException as e:
            if verbose:
                print(f"[ERROR] Could not connect to {self.port}: {e}")
            self.ser = None
            self.serial = None
            return False

    def _drop_connection(self):
        try:
            if self.ser:
                self.ser.close()
        except Exception:
            pass
        self.ser = None
        self.serial = None

    def send(self, message: str, key=None):
        if not self.ser or not self.ser.is_open:
            # AUTO-RECONNECT (throttled): the lunggaze board re-enumerates on
            # USB hiccups — the udev symlink vanishes and returns. The old
            # behavior disabled the port PERMANENTLY on the first I/O error,
            # which is why gaze/lung "simply stopped" until a restart.
            now = time.time()
            if now < self._reconnect_at:
                return
            self._reconnect_at = now + 3.0
            if not self._connect():
                return
        if key and self.last_sent.get(key) == message:
            return

        with self.send_lock:
            # Rate limiting: enforce minimum interval between commands
            now = time.time()
            time_since_last = now - self.last_send_time
            if time_since_last < MIN_COMMAND_INTERVAL:
                time.sleep(MIN_COMMAND_INTERVAL - time_since_last)

            try:
                full = message.strip() + "\n"
                self.ser.write(full.encode("utf-8"))
                self.ser.flush()  # Ensure command is sent immediately
                self.last_send_time = time.time()

                if key:
                    self.last_sent[key] = message

            except Exception as e:
                now = time.time()
                if now - self.last_error_log >= 2.0:
                    print(f"[ERROR] Servo send failed: {e} — will retry connection")
                    self.last_error_log = now
                if isinstance(e, serial.SerialException) or "i/o error" in str(e).lower() or "input/output error" in str(e).lower():
                    # drop the stale handle; the next send attempts reconnect
                    self._drop_connection()
                    self._reconnect_at = now + 3.0
                else:
                    try:
                        self.ser.reset_output_buffer()
                    except Exception:
                        pass

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

    def _should_send(self, key: str, new_angle: int) -> bool:
        last_msg = self.last_sent.get(key)
        if not last_msg:
            return True
        try:
            last_angle = int(last_msg.split(":")[1])
            return abs(last_angle - new_angle) >= ANGLE_THRESHOLD
        except (IndexError, ValueError):
            return True
