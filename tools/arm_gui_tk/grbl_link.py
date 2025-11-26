import threading
import time
from dataclasses import dataclass
from typing import Optional, Tuple

from grbl.grbl_utils import (
    DEFAULT_FEED_RATE,
    PEN_DOWN_CMD,
    PEN_UP_CMD,
    get_status,
    ensure_homed,
    find_grbl_port,
    send_cmd,
    setup_basic_grbl,
    wait_until_idle,
)


@dataclass
class GrblState:
    connected: bool = False
    homed: bool = False
    feed_rate: int = DEFAULT_FEED_RATE
    last_error: Optional[str] = None
    port: Optional[str] = None


class GrblLink:
    """Minimal GRBL link for live control from the Tk GUI.

    - Connects to GRBL, homes, sets up units and feed.
    - Provides pen up/down and throttled move commands.
    """

    def __init__(self):
        self.ser = None
        self.state = GrblState()
        self._lock = threading.Lock()
        self._last_move_ts = 0.0
        self._move_min_interval = 0.05  # 20 Hz max command rate
        self.max_speed_mm_s = 5.0       # hard speed limit (safety)
        self.max_step_mm = 0.8          # max step per command (safety)

    def connect_and_home(self, preferred_port: Optional[str] = None, max_homing_retries: int = -1) -> bool:
        try:
            # Connect
            self.ser = find_grbl_port(preferred_port=preferred_port, continuous_retry=False)
            self.state.connected = True
            # IMPORTANT: Home/clear alarm BEFORE any setup commands.
            # Some GRBLs reject modal/setup commands while in Alarm state.
            ensure_homed(self.ser, max_retries=max_homing_retries)
            # Basic setup (units, plane, feed) after homing
            setup_basic_grbl(self.ser, feed_rate=self.state.feed_rate, use_absolute_positioning=True)
            self.state.homed = True
            return True
        except Exception as e:
            self.state.last_error = str(e)
            self.state.connected = False
            self.state.homed = False
            try:
                if self.ser:
                    self.ser.close()
            except Exception:
                pass
            self.ser = None
            return False

    def disconnect(self):
        with self._lock:
            try:
                if self.ser:
                    send_cmd(self.ser, PEN_UP_CMD, wait_ok=False)
                    wait_until_idle(self.ser, 2.0)
            except Exception:
                pass
            try:
                if self.ser:
                    self.ser.close()
            except Exception:
                pass
            finally:
                self.ser = None
                self.state = GrblState()

    def set_feed(self, feed_rate: int):
        self.state.feed_rate = int(feed_rate)
        if self.ser:
            send_cmd(self.ser, f"F{int(feed_rate)}")

    def pen_up(self):
        if self.ser:
            send_cmd(self.ser, PEN_UP_CMD, wait_ok=False)

    def pen_down(self):
        if self.ser:
            send_cmd(self.ser, PEN_DOWN_CMD, wait_ok=False)

    def move_xy(self, x: float, y: float, rapid: bool = False):
        """Throttled move to XY in mm."""
        if not self.ser:
            return
        now = time.time()
        if now - self._last_move_ts < self._move_min_interval:
            return
        self._last_move_ts = now
        mode = "G0" if rapid else "G1"
        cmd = f"{mode} X{float(x):.3f} Y{float(y):.3f}"
        try:
            # Non-blocking OK wait to keep UI responsive
            send_cmd(self.ser, cmd, wait_ok=False)
        except Exception as e:
            self.state.last_error = str(e)

    def _allowed_step(self) -> float:
        now = time.time()
        dt = max(1e-3, now - self._last_move_ts)
        speed_step = self.max_speed_mm_s * dt
        return max(0.05, min(self.max_step_mm, speed_step))

    def set_relative_mode(self, relative: bool = True):
        if not self.ser:
            return
        try:
            send_cmd(self.ser, "G91" if relative else "G90", wait_ok=False)
        except Exception as e:
            self.state.last_error = str(e)

    def move_delta(self, dx: float, dy: float, rapid: bool = False):
        """Issue a relative move with hard step/speed limits.
        Caller should pre-clamp dx,dy; this also throttles by time.
        """
        if not self.ser:
            return
        now = time.time()
        if now - self._last_move_ts < self._move_min_interval:
            return
        self._last_move_ts = now
        # Enforce feed cap by re-sending F (GRBL caches it)
        try:
            send_cmd(self.ser, f"F{min(self.state.feed_rate, 800)}", wait_ok=False)
        except Exception:
            pass
        cmd = ("G0" if rapid else "G1") + f" X{dx:.3f} Y{dy:.3f}"
        try:
            send_cmd(self.ser, cmd, wait_ok=False)
        except Exception as e:
            self.state.last_error = str(e)

    def move_xy_safe(self, target_x: float, target_y: float, max_step: float = 1.5, rapid: bool = False):
        """Move toward target in small clamped steps based on current machine pos.
        Reduces risk of large unintended swings after home/sync.
        """
        if not self.ser:
            return
        cur = self.get_machine_position()
        if not cur:
            # No position -> do nothing
            return
        cx, cy = cur
        dx = target_x - cx
        dy = target_y - cy
        dist = (dx * dx + dy * dy) ** 0.5
        if dist <= 1e-6:
            return
        # Clamp step by both configured max_step and speed budget
        allow = min(max_step, self._allowed_step())
        if dist > allow and dist > 0:
            scale = allow / dist
            dx *= scale
            dy *= scale
        # Ensure relative mode
        self.set_relative_mode(True)
        self.move_delta(dx, dy, rapid=rapid)

    def set_work_origin_here(self):
        """Set current position as work origin (G54) using G10 L20 P1 X0 Y0 Z0."""
        if not self.ser:
            return
        try:
            send_cmd(self.ser, "G90", wait_ok=False)
            send_cmd(self.ser, "G54", wait_ok=False)
            send_cmd(self.ser, "G10 L20 P1 X0 Y0 Z0", wait_ok=False)
        except Exception as e:
            self.state.last_error = str(e)

    def set_origin_offset(self, x: float, y: float, z: float = 0.0):
        """Apply an origin offset via G92 to match existing pipeline mapping."""
        if not self.ser:
            return
        try:
            send_cmd(self.ser, f"G92 X{float(x)} Y{float(y)} Z{float(z)} ; Set origin offset", wait_ok=False)
        except Exception as e:
            self.state.last_error = str(e)

    def get_machine_position(self) -> Tuple[float, float] | None:
        """Parse MPos or WPos from GRBL status line. Returns (x,y) in mm or None."""
        if not self.ser:
            return None
        try:
            s = get_status(self.ser)
            # Example: <Idle|MPos:0.000,0.000,0.000|FS:0,0>
            # Prefer WPos if present; else MPos
            x = y = None
            if "WPos:" in s:
                part = s.split("WPos:", 1)[1]
            elif "MPos:" in s:
                part = s.split("MPos:", 1)[1]
            else:
                return None
            coords = part.split("|", 1)[0]
            nums = coords.split(",")
            if len(nums) >= 2:
                x = float(nums[0])
                y = float(nums[1])
                return (x, y)
        except Exception:
            return None
        return None
