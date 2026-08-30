"""
GRBL Utility Functions
Shared functions for GRBL communication and control
"""

import subprocess
import threading
import time
import math

import serial
from serial.tools import list_ports

from event_logging.event_logger import log_json_entry
from event_logging.log_type import LogType

try:
    from .warp_transform import warp_transform_line, find_xy_bounds_from_lines
except ImportError:
    from warp_transform import warp_transform_line, find_xy_bounds_from_lines

# Import pen servo configuration
try:
    from config.config import (
        GRBL_PEN_DOWN_S,
        GRBL_PEN_UP_S,
        GRBL_SPINDLE_MAX_S,
        GRBL_SPINDLE_MIN_S,
        GRBL_WARP_TRANSFORM,
        GRBL_PEN_UP_REPEATS,
        GRBL_PEN_UP_DWELL_S,
        GRBL_USE_CENTRALIZED_PEN_UP,
        GRBL_FORCE_ABSOLUTE_UP_FOR_HOMING,
        GRBL_PEN_UP_IS_HIGH,
        GRBL_PEN_DOWN_SETTLE_S,
        GRBL_PEN_UP_SETTLE_S,
    )
except Exception:
    GRBL_PEN_UP_S, GRBL_PEN_DOWN_S, GRBL_SPINDLE_MAX_S, GRBL_SPINDLE_MIN_S = 30, 50, 255, 0
    GRBL_WARP_TRANSFORM = True
    GRBL_PEN_UP_REPEATS = 5
    GRBL_PEN_UP_DWELL_S = 1.5
    GRBL_USE_CENTRALIZED_PEN_UP = False
    GRBL_FORCE_ABSOLUTE_UP_FOR_HOMING = True
    GRBL_PEN_UP_IS_HIGH = False
    GRBL_PEN_DOWN_SETTLE_S = 0.35
    GRBL_PEN_UP_SETTLE_S = 0.2

# Default configuration
DEFAULT_BAUD = 115200
DEFAULT_STATUS_POLL = 0.1
DEFAULT_HOME_TIMEOUT = 120  # seconds
DEFAULT_MOVE_TIMEOUT = 15  # seconds
DEFAULT_CMD_TIMEOUT = 5.0  # seconds
DEFAULT_FEED_RATE = 24000  # Doubled from 12000 for faster drawing execution

PEN_DOWN_CMD = f"M3 S{GRBL_PEN_DOWN_S} ; PEN DOWN"  # Command to lower pen
PEN_UP_CMD = f"M3 S{GRBL_PEN_UP_S} ; PEN UP"  # Command to raise pen


class DrawingLightbulbFluctuation:
    """Smooth lightbulb fluctuation during drawing operations."""

    def __init__(self, min_brightness=40, max_brightness=200, period_seconds=3.0):
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness
        self.period_seconds = period_seconds
        self.running = False
        self.thread = None
        self.lightbulb_controller = None

    def start_fluctuation(self):
        """Start the smooth brightness fluctuation."""
        if self.running:
            return

        # Try to get lightbulb controller instance
        try:
            from utils.state_manager import state_manager

            if hasattr(state_manager, "lightbulb") and state_manager.lightbulb:
                self.lightbulb_controller = state_manager.lightbulb
            else:
                # Try to import and create if not available
                from servo_control.lightbulb_controller_nonblocking import NonBlockingLightbulbController
                from config.config import USE_LIGHTBULB_PWM

                if USE_LIGHTBULB_PWM:
                    self.lightbulb_controller = NonBlockingLightbulbController("/dev/arduino_lightbulb", debug=False)
        except Exception as e:
            print(f"[⚠️] Could not get lightbulb controller for drawing fluctuation: {e}")
            return

        if not self.lightbulb_controller:
            return

        self.running = True
        self.thread = threading.Thread(target=self._fluctuation_loop, daemon=True)
        self.thread.start()
        print(
            f"[💡] Started smooth lightbulb fluctuation during drawing (range: {self.min_brightness}-{self.max_brightness}, period: {self.period_seconds}s)"
        )

    def stop_fluctuation(self):
        """Stop the brightness fluctuation."""
        if not self.running:
            return

        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        print("[💡] Stopped lightbulb fluctuation")

    def _fluctuation_loop(self):
        """Main fluctuation loop running in background thread."""
        start_time = time.time()

        while self.running:
            try:
                elapsed = time.time() - start_time
                # Create smooth sine wave oscillation
                phase = (elapsed % self.period_seconds) / self.period_seconds * 2 * math.pi
                brightness_factor = (math.sin(phase) + 1) / 2  # Normalize to 0-1

                # Map to brightness range
                brightness = int(self.min_brightness + (self.max_brightness - self.min_brightness) * brightness_factor)

                # Send brightness command
                if self.lightbulb_controller:
                    self.lightbulb_controller.set_frame_diff_brightness(brightness)

                time.sleep(0.05)  # 20Hz update rate for smooth transitions

            except Exception as e:
                print(f"[⚠️] Error in lightbulb fluctuation: {e}")
                break


def find_grbl_port(baud=DEFAULT_BAUD, timeout=0.5, preferred_port=None, continuous_retry=False):
    """Find and connect to GRBL controller port with optional continuous retry"""

    # Try preferred port first if specified
    if preferred_port:
        try:
            log_json_entry(
                LogType.GRBL,
                {"message": "Testing preferred GRBL port", "action": "preferred_port_test", "port": preferred_port, "baud": baud},
                print_message=f"[🔌] Testing preferred port {preferred_port}...",
            )
            ser = serial.Serial(preferred_port, baud, timeout=timeout, exclusive=True)
            time.sleep(2.0)
            ser.reset_input_buffer()
            ser.write(b"?")
            ser.flush()
            line = ser.readline().decode(errors="ignore").strip()
            if line.startswith("<") or "Grbl" in line:
                log_json_entry(
                    LogType.GRBL,
                    {
                        "message": "Preferred GRBL port found",
                        "action": "preferred_port_found",
                        "port": preferred_port,
                        "response": line,
                        "baud": baud,
                    },
                    print_message=f"[✅] {preferred_port} responds as GRBL: {line}",
                )
                return ser
            ser.close()
        except Exception as e:
            log_json_entry(
                LogType.GRBL,
                {
                    "message": "Preferred port test failed, falling back to auto-discovery",
                    "action": "preferred_port_failed",
                    "port": preferred_port,
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
                print_message=f"[⚠️] Preferred port {preferred_port} failed, trying auto-discovery...",
            )

    # Ports to exclude from GRBL scanning (other Arduino devices)
    excluded_ports = {
        "/dev/arduino_lunggaze",  # Servo controller - DO NOT SCAN
        "/dev/arduino_lightbulb",  # Lightbulb controller
        "/dev/arduino_lefthand",  # Hand gesture controller
        "/dev/arduino_uarm",  # uArm controller
    }

    ports = list(list_ports.comports())
    if not ports:
        raise RuntimeError("No serial ports found")

    # Filter out excluded ports to prevent interference
    filtered_ports = [p for p in ports if p.device not in excluded_ports]

    log_json_entry(
        LogType.GRBL,
        {
            "message": "Available serial ports for GRBL scanning",
            "action": "port_discovery",
            "all_ports": [p.device for p in ports],
            "excluded_ports": list(excluded_ports),
            "scannable_ports": [p.device for p in filtered_ports],
            "port_count": len(filtered_ports),
        },
        print_message=f"[🔌] Scannable ports for GRBL: {', '.join(p.device for p in filtered_ports)}",
    )

    for p in filtered_ports:
        try:
            log_json_entry(
                LogType.GRBL,
                {"message": "Testing port for GRBL", "action": "port_test", "port": p.device, "baud": baud},
                print_message=f"[🔌] Testing {p.device}...",
            )
            ser = serial.Serial(p.device, baud, timeout=timeout, exclusive=True)
            time.sleep(2.0)
            ser.reset_input_buffer()
            ser.write(b"?")
            ser.flush()
            line = ser.readline().decode(errors="ignore").strip()
            if line.startswith("<") or "Grbl" in line:
                log_json_entry(
                    LogType.GRBL,
                    {"message": "GRBL port found", "action": "port_found", "port": p.device, "response": line, "baud": baud},
                    print_message=f"[✅] {p.device} responds as GRBL: {line}",
                )
                return ser
            ser.close()
        except Exception as e:
            log_json_entry(
                LogType.GRBL,
                {"message": "Port test failed", "action": "port_test_failed", "port": p.device, "error": str(e), "error_type": type(e).__name__},
                print_message=f"[❌] {p.device} failed ({e})",
            )

    if continuous_retry:
        log_json_entry(
            LogType.GRBL,
            {"message": "No GRBL port found, retrying in 5s (continuous mode)", "action": "port_retry"},
            print_message="[⏳] No GRBL port found, retrying in 5s...",
        )
        time.sleep(5)
        return find_grbl_port(baud, timeout, preferred_port, continuous_retry)
    else:
        raise RuntimeError("No GRBL port found")


def read_until_ok_or_error(ser, timeout=DEFAULT_CMD_TIMEOUT):
    """Read serial until OK or error response"""
    start = time.time()
    log = []
    last = None

    while time.time() - start < timeout:
        line = ser.readline().decode(errors="ignore").strip()
        if line:
            log.append(line)
            lower_case_line = line.lower()
            if lower_case_line == "ok" or lower_case_line.startswith("error"):
                last = line
                break

    return last, log


def send_cmd(ser, cmd, wait_ok=True, timeout=DEFAULT_CMD_TIMEOUT):
    """Send command to GRBL and optionally wait for OK"""
    # Only print GRBL commands if verbose mode is enabled
    try:
        from config.config import PRINT_CLEAN_CAPTIONS

        if not PRINT_CLEAN_CAPTIONS:
            print(f"[📤] {cmd}")
    except ImportError:
        pass  # Default to quiet if config unavailable
    ser.write((cmd + "\n").encode())
    ser.flush()

    if not wait_ok:
        return []

    last, log = read_until_ok_or_error(ser, timeout)
    if last is None:
        raise TimeoutError(f"Timeout on {cmd}, response={log}")
    if last.lower().startswith("error"):
        raise RuntimeError(f"GRBL error on {cmd}, response={log}")

    return log


def get_status(ser):
    """Get current GRBL status"""
    ser.write(b"?")
    ser.flush()
    return ser.readline().decode(errors="ignore").strip()


def _revive_link(ser, grace=20.0):
    """After a silent line timeout (response=[]): poll the SAME fd until GRBL
    answers a status query. The Aug 12 pen-up dropouts showed the link going
    quiet for 15s+ mid-drawing and then answering again seconds later — a
    transient stall, not a dead port (pen safety and homing succeeded on the
    same fd right after). Returns:
      "alive" — status answered, no reset banner: position intact, safe to
                retry the failed line (G90 absolute makes the retry idempotent)
      "reset" — a Grbl banner appeared: the controller rebooted, position is
                LOST, the drawing must abort so pen-safety + homing run
      "dead"  — nothing within grace, or the fd itself errors
    """
    deadline = time.time() + grace
    saw_banner = False
    while time.time() < deadline:
        try:
            ser.reset_input_buffer()
            ser.write(b"?")
            ser.flush()
        except Exception:
            return "dead"
        window = time.time() + 1.0
        while time.time() < window:
            try:
                raw = ser.readline().decode(errors="ignore").strip()
            except Exception:
                return "dead"
            if not raw:
                break
            if "Grbl" in raw:
                saw_banner = True
            if raw.startswith("<"):
                return "reset" if saw_banner else "alive"
        time.sleep(0.5)
    return "dead"


def parse_state(status_line):
    """Parse state from GRBL status line"""
    if not status_line.startswith("<"):
        return ""

    body = status_line[1:-1] if status_line.endswith(">") else status_line[1:]
    for sep in ("|", ","):
        idx = body.find(sep)
        if idx != -1:
            return body[:idx]

    return body


def wait_until_idle(ser, max_wait, poll_interval=DEFAULT_STATUS_POLL):
    """Wait until GRBL is in Idle state"""
    start = time.time()

    while time.time() - start < max_wait:
        status = get_status(ser)
        state = parse_state(status)
        if state == "Idle":
            return
        time.sleep(poll_interval)

    raise TimeoutError("Did not become Idle within timeout")


def ensure_homed(ser, home_timeout=DEFAULT_HOME_TIMEOUT, max_retries=-1):
    """Ensure GRBL is homed and setup coordinate system with continuous retry logic"""

    # PREP: Clear ALARM first and set minimal spindle config so pen-up will be accepted
    try:
        status = get_status(ser)
        if parse_state(status) == "Alarm":
            log_json_entry(
                LogType.GRBL,
                {"message": "Clearing alarm state before safety pen-up", "action": "clear_alarm_pre_safety", "command": "$X", "status": status},
                print_message="[⚠️] Clearing alarm state before pen-up",
            )
            send_cmd(ser, "$X", wait_ok=True)
            time.sleep(0.4)
        # Ensure laser mode is OFF and spindle scale is known before pen-up
        try:
            send_cmd(ser, "$32=0", wait_ok=True)
            send_cmd(ser, f"$30={GRBL_SPINDLE_MAX_S}", wait_ok=True)
            send_cmd(ser, f"$31={GRBL_SPINDLE_MIN_S}", wait_ok=True)
            time.sleep(0.2)
        except Exception:
            pass
        # CRITICAL SAFETY: Centralized pen-up sequence before homing
        log_json_entry(
            LogType.GRBL,
            {"message": "STARTUP SAFETY: Using centralized pen-up before homing", "action": "startup_centralized_pen_safety"},
            print_message="[🚨 CRITICAL] Centralized pen-up sequence before homing...",
        )

        # Use centralized pen-up function for consistent, conflict-free operation
        try:
            ensure_pen_up_critical_safety(ser, context="startup_before_homing", use_repeats=True)
        except Exception as e:
            log_json_entry(
                LogType.GRBL,
                {
                    "message": f"CRITICAL: Centralized startup pen-up failed, using emergency fallback",
                    "action": "startup_pen_emergency_fallback",
                    "error": str(e),
                },
                print_message=f"[❌ CRITICAL] Centralized pen-up failed, emergency fallback: {e}",
            )
            # Emergency fallback: replicate the original logic exactly
            try:
                from config.config import GRBL_FORCE_ABSOLUTE_UP_FOR_HOMING, GRBL_PEN_UP_IS_HIGH

                if GRBL_FORCE_ABSOLUTE_UP_FOR_HOMING:
                    sup = GRBL_SPINDLE_MAX_S if GRBL_PEN_UP_IS_HIGH else GRBL_SPINDLE_MIN_S
                    send_cmd(ser, f"M3 S{sup}", wait_ok=False)
                    time.sleep(0.3)
            except Exception:
                pass
            for _ in range(int(GRBL_PEN_UP_REPEATS)):
                try:
                    send_cmd(ser, PEN_UP_CMD, wait_ok=False)
                    time.sleep(0.25)
                except Exception:
                    pass
            try:
                send_cmd(ser, f"G4 P{GRBL_PEN_UP_DWELL_S}", wait_ok=False)
                time.sleep(float(GRBL_PEN_UP_DWELL_S))
            except Exception:
                pass
    except Exception as e:
        print(f"[⚠️] Pen safety sequence error (continuing anyway): {e}")

    attempt = 0
    while max_retries == -1 or attempt < max_retries:
        try:
            # Re-send pen up before each homing attempt (already lifted but ensure it stays up)
            if GRBL_USE_CENTRALIZED_PEN_UP:
                # Use centralized pen-up safety function
                try:
                    ensure_pen_up_critical_safety(ser, context=f"pre_homing_attempt_{attempt+1}", use_repeats=False)
                except Exception as e:
                    log_json_entry(
                        LogType.GRBL,
                        {"message": f"Centralized pen-up failed, falling back to legacy", "action": "centralized_fallback", "error": str(e)},
                        print_message=f"[⚠️] Centralized pen-up failed, using legacy method: {e}",
                    )
                    # Fallback to original method if centralized fails
                    try:
                        send_cmd(ser, PEN_UP_CMD, wait_ok=False)
                        time.sleep(0.2)
                    except Exception:
                        pass
            else:
                # Original legacy method (unchanged)
                try:
                    send_cmd(ser, PEN_UP_CMD, wait_ok=False)
                    time.sleep(0.2)
                except Exception:
                    pass
            retry_msg = "∞" if max_retries == -1 else str(max_retries)
            log_json_entry(
                LogType.GRBL,
                {
                    "message": "Starting homing attempt",
                    "action": "homing_attempt_start",
                    "attempt": attempt + 1,
                    "max_retries": max_retries,
                    "timeout": home_timeout,
                },
                print_message=f"[🏠] Homing attempt {attempt + 1}/{retry_msg}...",
            )

            # Clear any existing alarm state again (belt-and-suspenders)
            status = get_status(ser)
            if parse_state(status) == "Alarm":
                log_json_entry(
                    LogType.GRBL,
                    {"message": "Clearing alarm state", "action": "clear_alarm", "command": "$X", "status": status, "attempt": attempt + 1},
                    print_message="[⚠️] Clearing alarm state with $X",
                )
                send_cmd(ser, "$X", wait_ok=True)
                time.sleep(0.5)
                try:
                    send_cmd(ser, PEN_UP_CMD, wait_ok=False)
                    time.sleep(0.25)
                except Exception:
                    pass

            # Soft reset before homing to ensure clean state
            if attempt > 0:  # Only on retries
                log_json_entry(
                    LogType.GRBL,
                    {"message": "Performing soft reset before retry", "action": "soft_reset", "attempt": attempt + 1},
                    print_message="[🔄] Performing soft reset before retry...",
                )
                ser.write(b"\x18")  # Send Ctrl-X (soft reset)
                time.sleep(2.0)  # Wait for reset to complete
                ser.reset_input_buffer()

            # Kinetic bus: the left arm must be CLEAR before the sweep. Two
            # gates, whichever applies: in-process hooks (panel, manual
            # tools) start the choreography here and wait it out; the
            # cross-process ARM_CLEAR sentinel (runtime: the parent started
            # the choreography as it spawned us, our preamble ran alongside)
            # holds $H only for whatever remains of the clearing time.
            try:
                from utils import hooks as _kin_hooks

                if _kin_hooks.on_grbl_homing_start:
                    _tuck_wait = float(_kin_hooks.on_grbl_homing_start() or 0.0)
                    if _tuck_wait > 0:
                        log_json_entry(
                            LogType.GRBL,
                            {"message": "Waiting for left arm tuck before homing", "action": "homing_tuck_wait", "seconds": _tuck_wait},
                            print_message=f"[🦾] Left arm tucking clear — homing in {_tuck_wait:.1f}s",
                        )
                        time.sleep(_tuck_wait)
                else:
                    try:
                        with open(_kin_hooks.ARM_CLEAR_SENTINEL) as _f:
                            _clear_at = float(_f.read().strip())
                        _remaining = min(max(0.0, _clear_at - time.time()), 60.0)
                        if _remaining > 0:
                            log_json_entry(
                                LogType.GRBL,
                                {"message": "Waiting for left arm clear (sentinel)", "action": "homing_arm_clear_wait", "seconds": _remaining},
                                print_message=f"[🦾] $H holds {_remaining:.1f}s for the left arm to clear",
                            )
                            time.sleep(_remaining)
                    except (OSError, ValueError):
                        pass  # no sentinel — nothing choreographing, home freely
            except Exception:
                pass

            log_json_entry(
                LogType.GRBL,
                {"message": "Running homing cycle", "action": "homing_start", "command": "$H", "timeout": home_timeout, "attempt": attempt + 1},
                print_message="[🏠] Running homing cycle ($H)...",
            )
            send_cmd(ser, "$H", wait_ok=False)

            # Wait for homing to complete
            start = time.time()
            while time.time() - start < home_timeout:
                status = get_status(ser)
                state = parse_state(status)

                # Debug: Log what state GRBL is actually reporting
                if time.time() - start > 1.0:  # After initial startup
                    log_json_entry(
                        LogType.GRBL,
                        {"message": f"Homing status check", "raw_status": status, "parsed_state": state, "elapsed": time.time() - start},
                        print_message=f"[🔍] GRBL state: '{state}' (raw: {status})",
                    )

                if state == "Idle" or state == "Home":
                    # Homing successful - setup coordinate system
                    log_json_entry(
                        LogType.GRBL,
                        {
                            "message": "Homing complete",
                            "action": "homing_complete",
                            "final_status": status,
                            "duration": time.time() - start,
                            "attempt": attempt + 1,
                        },
                        print_message="[✅] Homing complete",
                    )

                    # Kinetic bus: release the tucked left arm — it blends
                    # back into its running dataset. The sentinel makes this
                    # work ACROSS PROCESSES: the idle subprocess homes, the
                    # bus lives in machine.py and watches the file's mtime.
                    try:
                        from utils import hooks as _kin_hooks

                        # sentinel FIRST — a crashing hook must not eat the
                        # cross-process release (both release paths fire)
                        with open(_kin_hooks.HOMING_SENTINEL, "w") as _hf:
                            _hf.write(str(time.time()))
                        if _kin_hooks.on_grbl_homing_done:
                            try:
                                _kin_hooks.on_grbl_homing_done()
                            except Exception as _he:
                                print(f"[WARN] homing-done hook failed: {_he}")
                    except Exception:
                        pass

                    # Ensure pen is up after homing completes (double-assert with dwell for servo to catch)
                    try:
                        send_cmd(ser, PEN_UP_CMD, wait_ok=False)
                        send_cmd(ser, "G4 P0.2", wait_ok=False)  # short dwell
                        send_cmd(ser, PEN_UP_CMD, wait_ok=False)
                    except Exception:
                        pass

                    # Setup coordinate system
                    send_cmd(ser, "G54")
                    wait_until_idle(ser, DEFAULT_CMD_TIMEOUT)
                    send_cmd(ser, "G10 L20 P1 X0 Y0 Z0")
                    wait_until_idle(ser, DEFAULT_CMD_TIMEOUT)

                    log_json_entry(
                        LogType.GRBL,
                        {
                            "message": "Work coordinate system setup complete",
                            "action": "coordinate_system_setup",
                            "coordinate_system": "G54",
                            "origin": "0,0,0",
                            "successful_attempt": attempt + 1,
                        },
                        print_message="[📍] Work coordinate system G54 set to 0,0,0 at home position",
                    )
                    return  # Success!

                if state == "Alarm":
                    error_msg = f"Homing failed with alarm: {status}"
                    log_json_entry(
                        LogType.ERROR,
                        {"message": error_msg, "action": "homing_alarm", "status": status, "attempt": attempt + 1, "component": "grbl"},
                        print_message=f"[❌] {error_msg}",
                    )
                    break  # Break inner loop to try again

                time.sleep(DEFAULT_STATUS_POLL)
            else:
                # Timeout occurred
                error_msg = f"Homing timeout after {home_timeout}s"
                log_json_entry(
                    LogType.ERROR,
                    {"message": error_msg, "action": "homing_timeout", "timeout": home_timeout, "attempt": attempt + 1, "component": "grbl"},
                    print_message=f"[❌] {error_msg}",
                )

        except Exception as e:
            log_json_entry(
                LogType.ERROR,
                {
                    "message": f"Homing attempt failed with exception: {e}",
                    "action": "homing_exception",
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "attempt": attempt + 1,
                    "component": "grbl",
                },
                print_message=f"[❌] Homing attempt {attempt + 1} failed: {e}",
            )

        # If we get here, this attempt failed
        attempt += 1

        # For infinite retry, never give up
        if max_retries == -1:
            retry_delay = min(30, 2 + (attempt * 2))  # Cap at 30s delay
            log_json_entry(
                LogType.GRBL,
                {
                    "message": f"Homing attempt {attempt} failed, retrying in {retry_delay}s (continuous retry mode)",
                    "action": "homing_retry_delay",
                    "delay": retry_delay,
                    "attempt": attempt,
                },
                print_message=f"[⏳] Homing failed, retrying in {retry_delay}s... (continuous retry mode)",
            )
            time.sleep(retry_delay)
            continue

        # For limited retries, check if we should continue
        if attempt < max_retries:
            retry_delay = 2 + attempt  # Increasing delay: 2s, 3s, 4s
            log_json_entry(
                LogType.GRBL,
                {
                    "message": f"Homing attempt {attempt} failed, retrying in {retry_delay}s",
                    "action": "homing_retry_delay",
                    "delay": retry_delay,
                    "attempt": attempt,
                },
                print_message=f"[⏳] Homing failed, retrying in {retry_delay}s...",
            )
            time.sleep(retry_delay)
        else:
            # All limited attempts failed
            error_msg = f"All {max_retries} homing attempts failed"
            log_json_entry(
                LogType.ERROR,
                {"message": error_msg, "action": "homing_all_attempts_failed", "max_retries": max_retries, "component": "grbl"},
                print_message=f"[❌] {error_msg}",
            )
            raise RuntimeError(error_msg)


def convert_with_vpype(svg_file, output_file, scale_to=None):
    """Convert SVG to G-code using vpype with vpype-gcode plugin"""
    try:
        cmd = ["vpype", "read", svg_file]

        # Scale drawing to fit target dimensions
        if scale_to:
            cmd.extend(["layout", "--fit-to-margins", "1cm", scale_to])

        cmd.extend(
            ["linemerge", "--tolerance", "0.1mm", "linesimplify", "--tolerance", "0.05mm", "linesort", "gwrite", "--profile", "gcodemm", output_file]
        )

        log_json_entry(
            LogType.GRBL,
            {
                "message": "Running vpype with gcode plugin",
                "action": "vpype_conversion",
                "command": " ".join(cmd),
                "input_file": svg_file,
                "output_file": output_file,
                "scale": scale_to,
            },
            print_message=f"[🔧] Running vpype with gcode plugin: {' '.join(cmd)}",
        )
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        log_json_entry(
            LogType.GRBL,
            {
                "message": "vpype gcode conversion successful",
                "action": "vpype_success",
                "input_file": svg_file,
                "output_file": output_file,
                "result": str(result),
            },
            print_message="[✅] vpype gcode conversion successful",
        )
        return True

    except subprocess.CalledProcessError as e:
        log_json_entry(
            LogType.ERROR,
            {"message": "vpype gcode conversion failed", "component": "grbl", "error": e.stderr, "input_file": svg_file, "output_file": output_file},
            print_message=f"[❌] vpype gcode conversion failed: {e.stderr}",
        )
    except FileNotFoundError:
        log_json_entry(
            LogType.ERROR,
            {"message": "vpype or vpype-gcode not installed", "component": "grbl", "input_file": svg_file, "output_file": output_file},
            print_message="[❌] vpype or vpype-gcode not installed",
        )
        return False


def convert_gcode_to_servo_format(input_gcode, output_gcode):
    """Convert vpype-generated G-code to servo format with optional optimization"""
    try:
        with open(input_gcode, "r") as f:
            lines = f.readlines()

        # Apply G-code optimization if enabled
        try:
            from config.config import GRBL_ENABLE_FEED_OPTIMIZATION, GRBL_ENABLE_PEN_OPTIMIZATION

            # Only use optimizer if at least one optimization is enabled
            if GRBL_ENABLE_FEED_OPTIMIZATION or GRBL_ENABLE_PEN_OPTIMIZATION:
                try:
                    from .gcode_optimizer import create_optimizer_from_config
                except ImportError:
                    from gcode_optimizer import create_optimizer_from_config
                optimizer = create_optimizer_from_config()

                # Convert lines to strings for processing
                line_strings = [line.rstrip() for line in lines]

                # First pass: convert to servo format
                servo_lines = []
                pen_down = False

                for line in line_strings:
                    line = line.strip()

                    # Skip vpype headers and comments
                    if line.startswith(";") or line.startswith("%") or not line:
                        continue

                    # Handle movement commands (check G01 first to avoid G0 matching G01)
                    if line.startswith("G01") or (line.startswith("G1 ") and " " in line and not line.startswith("G17")):
                        # Linear move - pen should be down
                        if not pen_down:
                            servo_lines.append(f"{PEN_DOWN_CMD}")
                            # GRBL treats spindle-PWM changes as instantaneous —
                            # it never waits for the physical servo. Without a
                            # settle dwell, a dot/short dash is over before the
                            # pen lands (the dotted-line dropouts, July 9).
                            # Split dwells Aug 18: DOWN gets the full landing
                            # (S34->S56 travel + bounce) so short strokes start
                            # with the tip already on paper instead of inking
                            # only their tail end as a dot.
                            if GRBL_PEN_DOWN_SETTLE_S > 0:
                                servo_lines.append(f"G4 P{GRBL_PEN_DOWN_SETTLE_S} ; pen settle (down)")
                            pen_down = True
                        servo_lines.append(line)
                    elif line.startswith("G00") or (line.startswith("G0") and " " in line):
                        # Rapid move - pen should be up
                        if pen_down:
                            servo_lines.append(f"{PEN_UP_CMD}")
                            # settle before the rapid too, or the still-low pen
                            # drags a tail out of the stroke
                            if GRBL_PEN_UP_SETTLE_S > 0:
                                servo_lines.append(f"G4 P{GRBL_PEN_UP_SETTLE_S} ; pen settle (up)")
                            pen_down = False
                        servo_lines.append(line)
                    else:
                        # Pass through other commands (G17, G20, G21, G90, etc.)
                        servo_lines.append(line)

                # Ensure pen is up at the end for safety
                servo_lines.append(f"{PEN_UP_CMD}")

                # Second pass: apply optimization
                optimized_lines = optimizer.optimize_gcode(servo_lines)

                # Write optimized G-code
                with open(output_gcode, "w") as f:
                    for line in optimized_lines:
                        f.write(line + "\n")

            else:
                # Skip optimization if both are disabled
                raise ImportError("G-code optimization disabled in configuration")

        except ImportError:
            # Fallback to original method if optimization not available
            log_json_entry(
                LogType.GRBL,
                {"message": "G-code optimization not available, using original method", "action": "optimization_fallback"},
                print_message="[⚠️] G-code optimization not available",
            )

            with open(output_gcode, "w") as f:
                pen_down = False
                for line in lines:
                    line = line.strip()

                    # Skip vpype headers and comments
                    if line.startswith(";") or line.startswith("%") or not line:
                        continue

                    # Handle movement commands (check G01 first to avoid G0 matching G01)
                    if line.startswith("G01") or (line.startswith("G1 ") and " " in line and not line.startswith("G17")):
                        # Linear move - pen should be down
                        if not pen_down:
                            f.write(f"{PEN_DOWN_CMD}\n")
                            pen_down = True
                        f.write(f"{line}\n")
                    elif line.startswith("G00") or (line.startswith("G0") and " " in line):
                        # Rapid move - pen should be up
                        if pen_down:
                            f.write(f"{PEN_UP_CMD}\n")
                            pen_down = False
                        f.write(f"{line}\n")
                    else:
                        # Pass through other commands (G17, G20, G21, G90, etc.)
                        f.write(f"{line}\n")

                # Ensure pen is up at the end of the file for safety
                try:
                    f.write(f"\n{PEN_UP_CMD}\n")
                except Exception:
                    pass

        log_json_entry(
            LogType.GRBL,
            {
                "message": "G-code converted to servo format",
                "action": "servo_conversion_success",
                "input_file": input_gcode,
                "output_file": output_gcode,
            },
            print_message=f"[✅] G-code converted to servo format: {output_gcode}",
        )
        return True

    except Exception as e:
        log_json_entry(
            LogType.ERROR,
            {
                "message": "Servo formatting failed",
                "component": "grbl",
                "error": str(e),
                "error_type": type(e).__name__,
                "input_file": input_gcode,
                "output_file": output_gcode,
            },
            print_message=f"[❌] Servo formatting failed: {e}",
        )
        return False


def setup_basic_grbl(ser, feed_rate=DEFAULT_FEED_RATE, use_absolute_positioning=False):
    """Setup basic GRBL configuration"""
    # Ensure laser mode is OFF so GRBL doesn't auto-zero spindle PWM on rapids (which would drop pen signal)
    try:
        send_cmd(ser, "$32=0")  # disable laser mode
        time.sleep(0.1)
    except Exception:
        pass
    # Align spindle scaling to servo expectations
    try:
        send_cmd(ser, f"$30={GRBL_SPINDLE_MAX_S}")
        send_cmd(ser, f"$31={GRBL_SPINDLE_MIN_S}")
        time.sleep(0.1)
    except Exception:
        pass
    send_cmd(ser, "G21")  # mm units
    wait_until_idle(ser, DEFAULT_CMD_TIMEOUT)

    if use_absolute_positioning:
        send_cmd(ser, "G90")  # absolute positioning
        wait_until_idle(ser, DEFAULT_CMD_TIMEOUT)

    send_cmd(ser, "G17")  # XY-plane
    wait_until_idle(ser, DEFAULT_CMD_TIMEOUT)
    send_cmd(ser, "G54")  # coordinate system
    wait_until_idle(ser, DEFAULT_CMD_TIMEOUT)
    send_cmd(ser, f"F{feed_rate}")  # set feed rate
    wait_until_idle(ser, DEFAULT_CMD_TIMEOUT)


def set_work_origin_and_offset(ser, origin, origin_offset, move_timeout=DEFAULT_MOVE_TIMEOUT):
    """Move to and set work origin offset"""
    if origin[0] != 0 or origin[1] != 0:
        log_json_entry(
            LogType.GRBL,
            {"message": "Moving to work origin", "action": "move_to_origin", "origin_x": origin[0], "origin_y": origin[1]},
            print_message=f"[📍] Moving to work origin: X{origin[0]} Y{origin[1]}",
        )
        send_cmd(ser, f"G0 X{origin[0]} Y{origin[1]}", timeout=move_timeout)
        wait_until_idle(ser, move_timeout)
        send_cmd(ser, "G55")
        wait_until_idle(ser, DEFAULT_CMD_TIMEOUT)
        send_cmd(ser, "G10 L20 P2 X0 Y0 Z0")
        wait_until_idle(ser, DEFAULT_CMD_TIMEOUT)
        log_json_entry(
            LogType.GRBL,
            {"message": "Work origin set", "action": "origin_set", "coordinate_system": "G55", "origin": origin},
            print_message="[📍] Work origin set in G55 coordinate system",
        )

    if origin_offset != (0, 0, 0):
        # difference from G55? G10? L20?
        send_cmd(ser, f"G92 X{origin_offset[0]} Y{origin_offset[1]} Z{origin_offset[2]} ; Set origin offset")


def pen_control(ser, pen_down=True, pen_down_cmd=PEN_DOWN_CMD, pen_up_cmd=PEN_UP_CMD):
    """Control pen up/down using servo commands"""
    if pen_down:
        send_cmd(ser, pen_down_cmd)
    else:
        send_cmd(ser, pen_up_cmd)
    wait_until_idle(ser, DEFAULT_CMD_TIMEOUT)


def ensure_pen_up_critical_safety(ser, context="general", use_repeats=True):
    """
    Centralized pen-up safety function with consistent timing and extensive logging.

    Args:
        ser: Serial connection to GRBL
        context: String describing where this was called from (for debugging)
        use_repeats: If True, uses GRBL_PEN_UP_REPEATS; if False, single command

    This function consolidates all pen-up safety logic to prevent servo signal conflicts.
    Uses existing config values to maintain backward compatibility.
    """
    start_time = time.time()

    log_json_entry(
        LogType.GRBL,
        {
            "message": f"CENTRALIZED PEN SAFETY: Starting pen-up sequence",
            "action": "centralized_pen_up_start",
            "context": context,
            "use_repeats": use_repeats,
            "config_repeats": GRBL_PEN_UP_REPEATS,
            "config_dwell_s": GRBL_PEN_UP_DWELL_S,
            "config_pen_up_s": GRBL_PEN_UP_S,
        },
        print_message=f"[🛡️ SAFETY] Centralized pen-up sequence ({context})",
    )

    try:
        # Optional absolute extreme UP first (maintains existing behavior)
        if GRBL_FORCE_ABSOLUTE_UP_FOR_HOMING:
            extreme_up_s = GRBL_SPINDLE_MAX_S if GRBL_PEN_UP_IS_HIGH else GRBL_SPINDLE_MIN_S
            extreme_cmd = f"M3 S{extreme_up_s}"
            log_json_entry(
                LogType.GRBL,
                {"message": "Sending absolute extreme UP", "action": "extreme_up", "command": extreme_cmd, "context": context},
            )
            send_cmd(ser, extreme_cmd, wait_ok=False)
            time.sleep(0.3)  # Match existing timing

        # Main pen-up sequence
        repeat_count = int(GRBL_PEN_UP_REPEATS) if use_repeats else 1
        for i in range(repeat_count):
            try:
                log_json_entry(
                    LogType.GRBL,
                    {"message": f"Pen-up command {i+1}/{repeat_count}", "action": "pen_up_command", "command": PEN_UP_CMD, "context": context},
                )
                send_cmd(ser, PEN_UP_CMD, wait_ok=False)
                time.sleep(0.25)  # Consistent timing between commands
            except Exception as cmd_error:
                log_json_entry(
                    LogType.GRBL,
                    {"message": f"Pen-up command {i+1} failed", "action": "pen_up_error", "error": str(cmd_error), "context": context},
                    print_message=f"[⚠️] Pen-up command {i+1} failed: {cmd_error}",
                )

        # Final dwell for servo to settle (maintains existing behavior)
        if use_repeats and GRBL_PEN_UP_DWELL_S > 0:
            dwell_cmd = f"G4 P{GRBL_PEN_UP_DWELL_S}"
            log_json_entry(
                LogType.GRBL,
                {"message": "Final dwell for servo settling", "action": "pen_up_dwell", "command": dwell_cmd, "context": context},
            )
            send_cmd(ser, dwell_cmd, wait_ok=False)
            time.sleep(float(GRBL_PEN_UP_DWELL_S))

    except Exception as e:
        log_json_entry(
            LogType.GRBL,
            {"message": f"Centralized pen-up failed", "action": "centralized_pen_up_error", "error": str(e), "context": context},
            print_message=f"[❌] Centralized pen-up failed ({context}): {e}",
        )
        raise

    total_time = time.time() - start_time
    log_json_entry(
        LogType.GRBL,
        {
            "message": f"CENTRALIZED PEN SAFETY: Completed pen-up sequence",
            "action": "centralized_pen_up_complete",
            "context": context,
            "total_time_s": round(total_time, 3),
            "commands_sent": repeat_count + (1 if GRBL_FORCE_ABSOLUTE_UP_FOR_HOMING else 0),
        },
        print_message=f"[✅] Centralized pen-up complete ({context}) - {total_time:.2f}s",
    )


def _release_drawing_state():
    """Clear the 'currently drawing' flag. Called on every path out of
    execute_gcode_file, not just the successful one."""
    try:
        from utils.drawing_state import DrawingState

        DrawingState.end_drawing()
    except Exception as e:
        print(f"[⚠️] Could not clear drawing state: {e}")


def execute_gcode_file(ser, gcode_file, move_timeout=DEFAULT_MOVE_TIMEOUT):
    """Execute G-code file line by line with proper waiting"""
    print(f"🎨 [DEBUG] GRBL EXECUTION STARTING: {gcode_file}")
    log_json_entry(
        LogType.GRBL,
        {"message": "Starting G-code file execution", "action": "gcode_execution_start", "file": gcode_file, "timeout": move_timeout},
        print_message=f"[🚀] Executing G-code file: {gcode_file}",
    )

    # Start CNC execution tracking NOW (when actual GRBL execution begins)
    try:
        from utils.state_manager import state_manager

        original_prompt = (
            getattr(state_manager, "current_drawing_prompt", None)
            or getattr(state_manager, "last_completed_drawing_prompt", None)
            or "actively drawing"
        )
        state_manager.start_cnc_execution(gcode_file, original_prompt)

        # Display drawing summary on LCD during execution
        try:
            from utils.caption_display import _caption_display

            if _caption_display and _caption_display.connected:
                # Send with HIGH priority to override regular captions during drawing
                _caption_display.send_caption(original_prompt, priority="HIGH")
                print(f"[📺] Displaying drawing title on LCD: {original_prompt}")
        except Exception as lcd_e:
            print(f"[⚠️] Could not display drawing title on LCD: {lcd_e}")

    except Exception as e:
        print(f"[⚠️] Could not start CNC execution tracking: {e}")

    # Pause idle movements NOW for actual drawing execution
    try:
        from grbl.idle_movement_manager import pause_for_drawing

        try:
            from config.config import PRINT_CLEAN_CAPTIONS

            if not PRINT_CLEAN_CAPTIONS:
                print("[🌊] Pausing idle movements for drawing execution...")
        except ImportError:
            print("[🌊] Pausing idle movements for drawing execution...")
        pause_for_drawing()
    except Exception as e:
        print(f"[⚠️] Could not pause idle movements: {e}")

    # Notify drawing state manager with actual drawing prompt
    try:
        from utils.drawing_state import DrawingState
        from utils.state_manager import state_manager
        import os

        # Get drawing summary for state tracking
        # The drawing summary was generated and stored in drawing.py during prompt generation
        drawing_summary = getattr(state_manager, "current_drawing_prompt", None) or getattr(state_manager, "last_completed_drawing_prompt", None)
        if drawing_summary and isinstance(drawing_summary, str) and len(drawing_summary.strip()) > 0:
            # Use the concise drawing summary directly (already generated by model)
            compressed_description = drawing_summary.strip()
            print(f"[📝] Using drawing summary for state: {compressed_description}")
        else:
            compressed_description = "actively drawing"
            print(f"[📝] Using default drawing description for state tracking")

        # Drawing state tracking with compressed description
        DrawingState.start_drawing(
            drawing_file=gcode_file, description=compressed_description, intent=compressed_description  # Use the summary as the intent too
        )

    except Exception as e:
        print(f"[⚠️] Could not update drawing state: {e}")

    # Lock gaze system to drawing position during execution
    try:
        from config.config import USE_SERVO, TILT_MIN
        from vision.gaze import set_drawing_mode

        if USE_SERVO:
            print("[👁️] Locking gaze to drawing surface...")
            set_drawing_mode(active=True, drawing_pan=90, drawing_tilt=TILT_MIN + 2)
            time.sleep(1.0)  # Allow servos to reach position before drawing
    except Exception as e:
        print(f"[⚠️] Could not lock gaze for drawing observation: {e}")

    # Start smooth lightbulb fluctuation during drawing
    lightbulb_fluctuation = DrawingLightbulbFluctuation(min_brightness=40, max_brightness=200, period_seconds=3.0)
    try:
        from config.config import USE_LIGHTBULB_PWM

        if USE_LIGHTBULB_PWM:
            lightbulb_fluctuation.start_fluctuation()
    except Exception as e:
        print(f"[⚠️] Could not start lightbulb fluctuation: {e}")

    # Execute G-code in a separate thread to avoid blocking other systems
    gcode_complete = threading.Event()
    gcode_error = None
    executed_lines = 0
    total_lines = 0

    def execute_gcode_threaded():
        """Execute G-code in background thread."""
        nonlocal gcode_error, executed_lines, total_lines
        try:
            with open(gcode_file, "r") as f:
                lines = f.readlines()

            total_lines = len(lines)
            executed_lines = 0
            serial_recoveries = 0
            lines = lines[3:]  # Skip first three lines (G20, G17, G90), from vpype inject somehow
            min_x, min_y, max_x, max_y = find_xy_bounds_from_lines(lines)

            for line_num, line in enumerate(lines, 1):
                line = line.strip()

                # Skip empty lines and comments
                if not line or line.startswith(";") or line.startswith("%"):
                    continue

                try:
                    # Determine timeout and transform based on command type
                    if line.startswith(("G0", "G1", "G00", "G01")):
                        if GRBL_WARP_TRANSFORM:
                            line = warp_transform_line(line, max_x, max_y, min_x=min_x, min_y=min_y)  # warp transform line coords
                        timeout = move_timeout
                    else:
                        timeout = DEFAULT_CMD_TIMEOUT

                    send_cmd(ser, line, timeout=timeout)
                    executed_lines += 1

                    if executed_lines % 10 == 0:  # Progress update every 10 commands
                        log_json_entry(
                            LogType.GRBL,
                            {
                                "message": "G-code execution progress",
                                "action": "execution_progress",
                                "executed_lines": executed_lines,
                                "total_lines": total_lines,
                                "progress_percent": (executed_lines / total_lines) * 100,
                            },
                            print_message=f"[📋] Progress: {executed_lines}/{total_lines} lines executed",
                        )

                except TimeoutError as e:
                    # Serial dropout mid-drawing (the Aug 12 pen-up signature:
                    # 15s of silence, then the link answers again). Try to save
                    # the drawing in place instead of aborting a half-inked
                    # sheet: revive the link, then retry this one line — G90
                    # absolute coords make the retry idempotent even if the
                    # original command actually executed and only its ok died.
                    from config.config import GRBL_SERIAL_RECOVERY_MAX

                    serial_recoveries += 1
                    if serial_recoveries <= GRBL_SERIAL_RECOVERY_MAX:
                        log_json_entry(
                            LogType.GRBL,
                            {
                                "message": "Silent line timeout — attempting serial link revival",
                                "action": "serial_revival_attempt",
                                "line_number": line_num,
                                "command": line,
                                "attempt": serial_recoveries,
                            },
                            print_message=f"[🔌] Line {line_num} timed out silently — reviving link (attempt {serial_recoveries})",
                        )
                        verdict = _revive_link(ser)
                        if verdict == "alive":
                            try:
                                ser.reset_input_buffer()
                                send_cmd(ser, line, timeout=timeout)
                                executed_lines += 1
                                log_json_entry(
                                    LogType.GRBL,
                                    {
                                        "message": "Serial link revived — drawing continues",
                                        "action": "serial_revival_success",
                                        "line_number": line_num,
                                    },
                                    print_message=f"[🔌] Link revived at line {line_num} — drawing continues",
                                )
                                continue
                            except Exception as e2:
                                e = e2  # fall through to the abort log below
                        elif verdict == "reset":
                            log_json_entry(
                                LogType.ERROR,
                                {
                                    "message": "GRBL controller REBOOTED mid-drawing (banner seen) — position lost, aborting",
                                    "component": "grbl",
                                    "action": "serial_revival_reset",
                                    "line_number": line_num,
                                },
                                print_message=f"[🔌] Controller rebooted at line {line_num} — position lost, aborting for re-home",
                            )
                    log_json_entry(
                        LogType.ERROR,
                        {
                            "message": "Failed to execute G-code line",
                            "component": "grbl",
                            "line_number": line_num,
                            "command": line,
                            "error": str(e),
                            "error_type": type(e).__name__,
                            "serial_recoveries_used": serial_recoveries,
                        },
                        print_message=f"[❌] Failed to execute line {line_num}: {line} - Error: {e}",
                    )
                    raise
                except Exception as e:
                    log_json_entry(
                        LogType.ERROR,
                        {
                            "message": "Failed to execute G-code line",
                            "component": "grbl",
                            "line_number": line_num,
                            "command": line,
                            "error": str(e),
                            "error_type": type(e).__name__,
                        },
                        print_message=f"[❌] Failed to execute line {line_num}: {line} - Error: {e}",
                    )
                    raise
        except Exception as e:
            gcode_error = e
        finally:
            gcode_complete.set()

    # Start G-code execution in background thread
    gcode_thread = threading.Thread(target=execute_gcode_threaded, daemon=False, name="GCodeExecution")
    gcode_thread.start()

    # Wait for completion with periodic checks (allows other threads to run)
    try:
        while not gcode_complete.is_set():
            time.sleep(0.1)  # Short sleep allows other threads to continue
    except KeyboardInterrupt:
        log_json_entry(
            LogType.GRBL,
            {"message": "G-code execution interrupted by user", "action": "execution_interrupted"},
            print_message="[⚠️] G-code execution interrupted!",
        )
        # Send any emergency stops or cleanup commands if needed
        try:
            send_cmd(ser, "M3 S30", wait_ok=False)  # Pen up
            send_cmd(ser, "!", wait_ok=False)  # Emergency stop
        except Exception:
            pass

        # Stop lightbulb fluctuation on interruption
        try:
            lightbulb_fluctuation.stop_fluctuation()
        except Exception:
            pass
        _release_drawing_state()

        raise

    # Re-raise any errors that occurred in the thread
    if gcode_error:
        # Ensure gaze unlock even on error
        try:
            from vision.gaze import set_drawing_mode

            set_drawing_mode(active=False)
        except Exception:
            pass
        try:
            lightbulb_fluctuation.stop_fluctuation()
        except Exception:
            pass
        # The gaze and the lightbulb were already released here; the drawing
        # state was not, so a single timed-out g-code line left the machine
        # believing it was still drawing until the process restarted.
        _release_drawing_state()
        raise gcode_error

    log_json_entry(
        LogType.GRBL,
        {
            "message": "G-code execution complete",
            "action": "gcode_execution_complete",
            "file": gcode_file,
            "executed_lines": executed_lines,
            "total_lines": total_lines,
        },
        print_message=f"[✅] G-code execution complete: {executed_lines} lines executed",
    )

    # === DRAWING COMPLETION RITUAL ===
    # NOTE: Gaze stays locked to drawing surface through the entire completion ritual
    # (homing, pen-up, pause). Unlocked at the end of the ritual, not here.

    # Step 1: End drawing state (releases drawing context from captions)
    try:
        from utils.drawing_state import DrawingState

        DrawingState.end_drawing()
    except Exception as e:
        print(f"[⚠️] Could not update drawing state: {e}")

    # Step 2: Home the machine and pause for completion ritual
    print(f"🏠 [DEBUG] COMPLETION RITUAL STARTING")
    try:
        log_json_entry(
            LogType.GRBL,
            {"message": "Starting completion ritual", "action": "completion_ritual_start"},
            print_message="[🏠] Starting completion ritual - homing and pausing...",
        )

        # CRITICAL: Ensure pen is up before homing in completion ritual
        # This is especially important after drawing when pen might still be down
        for i in range(int(GRBL_PEN_UP_REPEATS)):
            try:
                send_cmd(ser, PEN_UP_CMD, wait_ok=False)
                time.sleep(0.25)
            except Exception:
                pass

        # Dwell to ensure servo has responded
        try:
            send_cmd(ser, f"G4 P{GRBL_PEN_UP_DWELL_S}", wait_ok=False)
            time.sleep(float(GRBL_PEN_UP_DWELL_S))
        except Exception:
            pass

        # Now safe to send homing command
        send_cmd(ser, "$H")
        wait_until_idle(ser, 30)  # Wait for homing to complete
        # Reassert UP after homing in case PWM was reset during $H
        try:
            send_cmd(ser, PEN_UP_CMD, wait_ok=False)
            time.sleep(0.4)
        except Exception:
            pass

        log_json_entry(
            LogType.GRBL,
            {"message": "Homed for completion ritual - staying at home for 30-second pause", "action": "completion_homing_complete"},
            print_message="[✅] Homing complete - beginning 30-second completion pause at home position",
        )

        # Step 3: Unlock gaze system during completion pause
        print("[DEBUG] About to unlock gaze system...")
        try:
            from vision.gaze import set_drawing_mode

            set_drawing_mode(active=False)
            print("[👁️] ✅ Gaze unlocked for completion ritual")
        except Exception as e:
            print(f"[❌] Could not unlock gaze system: {e}")
            import traceback

            traceback.print_exc()

        # Step 4: Trigger self-critique during 7-second pause (AT HOME POSITION)
        completion_thread_running = threading.Event()

        def completion_self_critique():
            """Generate self-critique reflection during completion pause."""
            try:
                # Get the drawing context for self-critique
                from utils.state_manager import state_manager
                from captioner.prompt_interface import PromptInterface

                # Get drawing details
                drawing_prompt = getattr(state_manager, "current_drawing_prompt", "recent drawing")

                # Get the compressed description if available
                from utils.drawing_state import DrawingState

                drawing_info = DrawingState.get_drawing_info()
                compressed_desc = drawing_info.get("description", "a drawing") if drawing_info else "a drawing"

                # Build self-critique prompt using existing system
                critique_prompt = f"""You have just finished drawing. Look at what you created.
                
You were drawing: {compressed_desc}
Original intent: {drawing_prompt}

The pen has lifted, the machine has returned home. You can feel the completion.
How do you reflect on this creative act? What did you express through these lines?

Respond with 2-3 sentences of honest self-reflection about your artwork."""

                # The critique was REMOVED Aug 5 (artist: "not useful and
                # underutilised", and it will be redesigned to judge the paper
                # rather than the ComfyUI image). What survives is the fact:
                # the pen finished, and the machine should remember drawing.
                try:
                    self_critique = ""

                    if self_critique:
                        log_json_entry(
                            LogType.REFLECTION,
                            {
                                "message": "Drawing completion self-critique",
                                "action": "drawing_self_critique",
                                "drawing_intent": drawing_prompt,
                                "drawing_description": compressed_desc,
                                "self_critique": self_critique,
                                "completion_type": "post_drawing_reflection",
                            },
                            print_message=f"[🎨💭] Drawing self-critique: {self_critique}",
                        )

                    # The completion is recorded WHETHER OR NOT a critique
                    # exists. Having drawn is the fact; the reflection is
                    # commentary on it. This used to sit inside the critique
                    # branch, so a failed critique erased the machine's memory
                    # of having drawn at all — and with the critique now living
                    # in another module, absent is a normal state, not an error.
                    if True:
                        try:
                            if hasattr(state_manager, "captioner") and hasattr(state_manager.captioner, "observe"):
                                completion_text = f"Completed drawing {compressed_desc}." + (
                                    f" Reflection: {self_critique[:100]}" if self_critique else ""
                                )
                                state_manager.captioner.observe(
                                    completion_text,
                                    state_manager.captioner.current_mood if hasattr(state_manager.captioner, "current_mood") else 0.5,
                                    "",
                                    memory_type="drawing_completion",
                                )
                                print(f"[📝] Stored drawing completion in memory: {completion_text[:50]}...")
                            else:
                                print(
                                    f"[⚠️] Drawing completion not stored: state_manager.captioner exists={hasattr(state_manager, 'captioner')}, observe method exists={hasattr(state_manager.captioner, 'observe') if hasattr(state_manager, 'captioner') else False}"
                                )
                        except Exception as e:
                            print(f"[❌] Failed to store drawing completion: {e}")
                            import traceback

                            traceback.print_exc()

                except Exception as e:
                    print(f"[⚠️] Could not generate drawing self-critique: {e}")

            except Exception as e:
                print(f"[⚠️] Error in completion self-critique: {e}")
            finally:
                completion_thread_running.set()

        # Start self-critique in background thread
        critique_thread = threading.Thread(target=completion_self_critique, daemon=True)
        critique_thread.start()

        # Notify any runtime hook that GRBL drawing has finished (uArm starts immediately)
        # CRITICAL: Hook must run BEFORE 30-second pause to allow uArm to use the full time
        hook_completed_successfully = False
        try:
            from utils.hooks import on_grbl_drawing_complete

            print(f"[DEBUG] GRBL completion hook check: callable={callable(on_grbl_drawing_complete)}")
            if callable(on_grbl_drawing_complete):
                print(f"[DEBUG] Calling GRBL completion hook (uArm starts NOW during pause)")
                on_grbl_drawing_complete()
                hook_completed_successfully = True
                print(f"[DEBUG] GRBL completion hook finished successfully")
            else:
                print(f"[DEBUG] No GRBL completion hook registered")
                hook_completed_successfully = True
        except Exception as e:
            print(f"[hooks] on_grbl_drawing_complete error: {e}")
            hook_completed_successfully = True  # Continue anyway

        # 30-second completion pause AT HOME POSITION (allows time for uArm movement to complete)
        time.sleep(30.0)

        # Ensure self-critique thread completes
        if not completion_thread_running.is_set():
            completion_thread_running.wait(timeout=5.0)

        log_json_entry(
            LogType.GRBL,
            {"message": "Completion ritual finished at home position", "action": "completion_ritual_complete"},
            print_message="[✅] Completion ritual finished at home - idle movements will handle positioning",
        )

    except Exception as e:
        log_json_entry(
            LogType.ERROR,
            {"message": f"Completion ritual failed: {e}", "component": "grbl", "error": str(e)},
            print_message=f"[❌] Completion ritual failed: {e}",
        )
        hook_completed_successfully = True  # Continue anyway

    # Step 4.6 (Aug 18 race fix): register the completion BEFORE the executing
    # flag clears. The trigger used to evaluate in the ~10s gap (port probing)
    # between Step 5 and the caller's registration and fire on a want that was
    # seconds from being spent — a back-to-back drawing with a stale reason.
    # register_drawing has a 60s reentry guard, so the caller's late call
    # stays as belt-and-braces.
    try:
        from utils.state_manager import state_manager as _sm

        _cap = getattr(_sm, "captioner", None)
        if _cap is not None and hasattr(_cap, "drawing"):
            _cap.drawing.register_drawing(_sm.current_drawing_prompt or getattr(_sm, "last_completed_drawing_prompt", None) or "Unknown drawing")
    except Exception as _e:
        print(f"[⚠️] Early drawing registration failed: {_e}")

    # Step 5: Clear CNC execution state AFTER hook completes to allow proper coordination
    if hook_completed_successfully:
        try:
            from utils.state_manager import state_manager

            print(f"[DEBUG] Clearing CNC execution state AFTER hook completion")
            state_manager.finish_cnc_execution()
        except Exception as e:
            print(f"[⚠️] Could not clear CNC execution state: {e}")
    else:
        print(f"[DEBUG] Skipping CNC state clear due to hook failure")

    # Step 6: Stop lightbulb fluctuation after completion ritual
    try:
        lightbulb_fluctuation.stop_fluctuation()
    except Exception as e:
        print(f"[⚠️] Could not stop lightbulb fluctuation: {e}")

    # Step 7: Resume idle movements after completion ritual (and uArm action)
    try:
        from grbl.idle_movement_manager import resume_after_drawing

        try:
            from config.config import PRINT_CLEAN_CAPTIONS

            if not PRINT_CLEAN_CAPTIONS:
                print("[🌊] Resuming idle movements after completion ritual...")
        except ImportError:
            print("[🌊] Resuming idle movements after completion ritual...")
        resume_after_drawing()
    except Exception as e:
        print(f"[⚠️] Could not resume idle movements: {e}")


def initialize_grbl_for_drawing(
    ser, origin=(0, 0, 0), origin_offset=(0, 0, 0), feed_rate=DEFAULT_FEED_RATE, use_absolute_positioning=False, max_homing_retries=-1
):
    """Complete GRBL initialization sequence for drawing with robust error handling"""
    log_json_entry(
        LogType.GRBL,
        {
            "message": "Initializing GRBL for drawing",
            "action": "initialization_start",
            "origin": origin,
            "origin_offset": origin_offset,
            "feed_rate": feed_rate,
            "absolute_positioning": use_absolute_positioning,
            "max_homing_retries": max_homing_retries,
        },
        print_message="[🎨] Initializing GRBL for drawing...",
    )

    try:
        # Step 1: Homing with retry logic
        ensure_homed(ser, max_retries=max_homing_retries)

        # Step 2: Basic GRBL setup
        setup_basic_grbl(ser, feed_rate, use_absolute_positioning=use_absolute_positioning)

        # Step 3: Work coordinate setup
        set_work_origin_and_offset(ser, origin, origin_offset)

        # Step 4: Pen control initialization
        pen_control(ser, pen_down=False)

        log_json_entry(
            LogType.GRBL,
            {"message": "GRBL initialization complete", "action": "initialization_complete"},
            print_message="[✅] GRBL initialization complete",
        )

    except Exception as e:
        log_json_entry(
            LogType.ERROR,
            {
                "message": f"GRBL initialization failed: {e}",
                "action": "initialization_failed",
                "error": str(e),
                "error_type": type(e).__name__,
                "component": "grbl",
            },
            print_message=f"[❌] GRBL initialization failed: {e}",
        )
        raise  # Re-raise the exception


def process_svg_to_grbl(
    svg_input,
    output_gcode=None,
    execute_grbl=True,
    scale_to=None,
    origin=(0, 0, 0),
    origin_offset=(0, 0, 0),
    feed_rate=DEFAULT_FEED_RATE,
    use_absolute_positioning=False,
):
    print(f"🎯 [DEBUG] PROCESS_SVG_TO_GRBL CALLED: execute_grbl={execute_grbl}")
    """
    Process SVG to G-code and optionally execute on GRBL hardware

    Args:
        svg_input: Path to input SVG file
        output_gcode: Path for output G-code file (optional)
        execute_grbl: Whether to execute the G-code on GRBL hardware
        scale_to: Scale to fit size (e.g., '50x50mm', '100x100mm')
        origin: Work origin (x, y, z) tuple
        origin_offset: Origin offset (x, y, z) tuple
        feed_rate: Feed rate for movements
        use_absolute_positioning: Whether to use absolute positioning

    Returns:
        str: Path to generated G-code file if successful, None if failed
    """
    try:
        import os
        from pathlib import Path

        svg_path = Path(svg_input)
        if not svg_path.exists():
            log_json_entry(
                LogType.ERROR,
                {"message": "SVG file not found", "component": "grbl", "file_path": svg_input},
                print_message=f"[❌] SVG file not found: {svg_input}",
            )
            return None

        if output_gcode:
            output_file_adjusted = output_gcode
            output_file_vpype = f"{output_gcode}_raw_vpype.gcode"
        else:
            output_file_vpype = str(svg_path.parent / f"{svg_path.stem}_raw_vpype.gcode")
            output_file_adjusted = str(svg_path.parent / f"{svg_path.stem}_servo_adjusted.gcode")

        convert_with_vpype(str(svg_path), output_file_vpype, scale_to=scale_to)
        log_json_entry(
            LogType.GRBL,
            {"message": "V-PYPE G-code generated", "action": "vpype_gcode_generated", "output_file": output_file_vpype, "input_file": str(svg_path)},
            print_message=f"[✅] V-PYPE G-code generated: {output_file_vpype}",
        )
        convert_gcode_to_servo_format(output_file_vpype, output_file_adjusted)
        log_json_entry(
            LogType.GRBL,
            {
                "message": "Servo G-code generated",
                "action": "servo_gcode_generated",
                "output_file": output_file_adjusted,
                "input_file": output_file_vpype,
            },
            print_message=f"[✅] Servo G-code generated: {output_file_adjusted}",
        )

        if os.path.exists(output_file_vpype):
            os.remove(output_file_vpype)

        # G-code is fully generated — NOW release the generation phase so
        # llama-server may reload. The artist's rule (Aug 12): llama runs
        # DURING GRBL execution (watching itself draw needs it) but must not
        # start while the vectorizer/gcode chain still owns the GPU. The old
        # release point (image_monitor, pre-conversion) let the 27B load
        # alongside DSV; this seam is the earliest safe moment.
        try:
            from utils.state_manager import state_manager as _sm

            if getattr(_sm, "is_generating_drawing", False):
                _sm.finish_drawing_generation()
        except Exception:
            pass

        # Execute on GRBL if requested
        if execute_grbl:
            log_json_entry(
                LogType.GRBL,
                {"message": "Starting GRBL execution", "action": "grbl_execution_start", "gcode_file": output_file_adjusted},
                print_message="[🚀] Executing on GRBL...",
            )
            try:
                # Get GRBL configuration
                try:
                    from config.config import GRBL_CNC_PORT, GRBL_HOMING_MAX_RETRIES

                    ser = find_grbl_port(preferred_port=GRBL_CNC_PORT, continuous_retry=True)
                    max_retries = GRBL_HOMING_MAX_RETRIES if GRBL_HOMING_MAX_RETRIES != -1 else -1
                except ImportError:
                    ser = find_grbl_port(continuous_retry=True)
                    max_retries = -1  # Continuous retry by default

                print(f"🎯 [DEBUG] About to call initialize_grbl_for_drawing...")
                initialize_grbl_for_drawing(
                    ser,
                    origin=origin,
                    origin_offset=origin_offset,
                    feed_rate=feed_rate,
                    use_absolute_positioning=use_absolute_positioning,
                    max_homing_retries=max_retries,
                )
                print(f"🎯 [DEBUG] initialize_grbl_for_drawing completed successfully")

                # === PAPER CHECK AFTER HOMING (opt-in) ===
                try:
                    from config.config import (
                        ENABLE_PAPER_DETECTION,
                        ENABLE_POST_HOME_PAPER_CHECK,
                        ALLOW_PAPER_DETECTION_OVERRIDE,
                        PAPER_DETECTION_GAZE_PAN,
                        PAPER_DETECTION_GAZE_TILT,
                    )
                except Exception:
                    ENABLE_PAPER_DETECTION = False
                    ENABLE_POST_HOME_PAPER_CHECK = False
                    ALLOW_PAPER_DETECTION_OVERRIDE = True
                    PAPER_DETECTION_GAZE_PAN = 90
                    from config.config import TILT_MIN

                    PAPER_DETECTION_GAZE_TILT = TILT_MIN + 2

                def _paper_check_after_homing() -> bool:
                    """Delegates to the centralized paper check (vlm or aruco per PAPER_CHECK_METHOD)."""
                    print("[📄] Running centralized paper check...")
                    try:
                        # Get camera and servos from state manager
                        from utils.state_manager import state_manager as _sm

                        camera_obj = getattr(_sm, "camera", None)
                        servos_obj = getattr(_sm, "servos", None)

                        if camera_obj is None:
                            print("[📄] No camera - defaulting to ALLOW drawing")
                            return True

                        from safety.paper_detection import check_paper_before_drawing

                        paper_present = check_paper_before_drawing(camera_obj, servos_obj, None)

                        print(f"[📄] Paper check result: {'PAPER PRESENT' if paper_present else 'NO PAPER'}")
                        return paper_present

                    except Exception as e:
                        import traceback

                        traceback.print_exc()
                        from config import config as _c

                        fail_open = str(getattr(_c, "PAPER_CHECK_METHOD", "aruco")).lower() != "vlm"
                        print(f"[📄] Paper check error: {e} - {'defaulting to ALLOW drawing' if fail_open else 'failing CLOSED (no draw)'}")
                        return fail_open

                # Run paper check with hard fail-safes (never blocks silently)
                if ENABLE_PAPER_DETECTION and ENABLE_POST_HOME_PAPER_CHECK:
                    try:
                        # Force a visible console print regardless of clean-caption settings
                        print("[📄] Post-home paper check enabled")
                        import sys as _sys

                        _sys.stdout.flush()
                        ok = _paper_check_after_homing()
                        print(f"[📄] Post-home paper check result: {'OK' if ok else 'BLOCK'}")
                        _sys.stdout.flush()
                        if not ok:
                            try:
                                ser.close()
                            except Exception:
                                pass
                            # Mark skip for image monitor to log gracefully
                            try:
                                from utils.state_manager import state_manager as _sm

                            except Exception:
                                pass
                            # Ensure gaze unlocked after skip
                            try:
                                from vision.gaze import set_drawing_mode

                                set_drawing_mode(active=False)
                            except Exception:
                                pass
                            return None
                    except Exception as e:
                        # Never block drawing due to check errors
                        print(f"[📄] Paper check error ({e}) — proceeding")
                        _sys.stdout.flush()
                else:
                    print("[📄] Post-home paper check skipped (disabled)")

                # Lock gaze to drawing surface immediately after paper check passes
                # This eliminates the gap where gaze could track a person between
                # paper search ending and execute_gcode_file engaging its own lock
                try:
                    from config.config import USE_SERVO, TILT_MIN
                    from vision.gaze import set_drawing_mode

                    if USE_SERVO:
                        set_drawing_mode(active=True, drawing_pan=90, drawing_tilt=TILT_MIN + 2)
                except Exception:
                    pass

                # Execute G-code in a separate thread to prevent blocking captions
                print(f"🎯 [DEBUG] Starting threaded G-code execution for file: {output_file_adjusted}")
                import threading

                gcode_complete = threading.Event()
                gcode_error = None

                def execute_gcode_threaded():
                    """Execute G-code in background thread."""
                    nonlocal gcode_error
                    try:
                        print(f"🎯 [DEBUG] Thread starting G-code execution...")
                        execute_gcode_file(ser, output_file_adjusted)
                        print(f"🎯 [DEBUG] Thread completed G-code execution successfully")
                    except Exception as e:
                        print(f"🎯 [DEBUG] Thread G-code execution failed: {e}")
                        gcode_error = e
                    finally:
                        gcode_complete.set()

                gcode_thread = threading.Thread(target=execute_gcode_threaded, daemon=False)
                gcode_thread.start()

                # Wait for completion with periodic checks (allows captions to continue)
                while not gcode_complete.is_set():
                    time.sleep(0.1)  # Short sleep allows other threads to run

                # Re-raise any errors that occurred in the thread
                if gcode_error:
                    print(f"🚨 [DEBUG] GCODE_ERROR OCCURRED: {gcode_error}")
                    raise gcode_error

                print(f"🎯 [DEBUG] GCODE EXECUTION COMPLETED SUCCESSFULLY")
                log_json_entry(
                    LogType.GRBL,
                    {"message": "Drawing complete", "action": "drawing_complete", "gcode_file": output_file_adjusted},
                    print_message="[✅] Drawing complete!",
                )

                # Drawing execution completed successfully
                print(f"🏁 [DEBUG] Drawing execution completed successfully")

                # CRITICAL: Now that physical drawing is complete, start the cooldown timer
                # This ensures proper spacing between actual drawings, not just prompt generations
                try:
                    import sys
                    from utils.state_manager import state_manager

                    # Get the drawing prompt that was used for this drawing
                    drawing_prompt = state_manager.current_drawing_prompt or state_manager.last_completed_drawing_prompt or "Unknown drawing"

                    # Get the captioner instance via state_manager (more reliable than sys.modules)
                    print(f"[🔍 DEBUG] Attempting to register drawing completion...")
                    captioner = getattr(state_manager, "captioner", None)
                    print(f"[🔍 DEBUG] captioner from state_manager: {captioner is not None}")

                    if captioner is None:
                        # Fallback: try sys.modules (old method)
                        print(f"[🔍 DEBUG] Fallback: trying sys.modules['machine']")
                        if "machine" in sys.modules:
                            machine_module = sys.modules["machine"]
                            captioner = getattr(machine_module, "_global_captioner", None) or getattr(machine_module, "captioner", None)
                            print(f"[🔍 DEBUG] captioner from sys.modules: {captioner is not None}")

                    if captioner:
                        print(f"[🔍 DEBUG] has 'drawing' attr: {hasattr(captioner, 'drawing')}")
                        print(
                            f"[🔍 DEBUG] has 'register_drawing' method: {hasattr(captioner.drawing, 'register_drawing') if hasattr(captioner, 'drawing') else False}"
                        )

                        if hasattr(captioner, "drawing") and hasattr(captioner.drawing, "register_drawing"):
                            captioner.drawing.register_drawing(drawing_prompt)
                            print(f"✅ [SUCCESS] Drawing cooldown timer started after GRBL completion")
                            log_json_entry(
                                LogType.DEBUG,
                                {
                                    "message": "Drawing cooldown started after physical completion",
                                    "prompt": drawing_prompt[:50] + "..." if len(drawing_prompt) > 50 else drawing_prompt,
                                },
                                print_message=f"[⏰] Drawing cooldown started: {captioner.drawing.cooldown}s",
                            )
                        else:
                            print(f"[❌ CRITICAL] Captioner found but missing drawing/register_drawing!")
                    else:
                        print(f"[❌ CRITICAL] Could not find captioner via state_manager OR sys.modules!")
                except Exception as e:
                    print(f"🎯 [DEBUG] Error registering drawing completion: {e}")
                    import traceback

                    traceback.print_exc()
                    # Don't fail the whole process if cooldown registration fails

                try:
                    ser.close()
                except Exception:
                    pass
                # Success – return the path
                return output_file_adjusted
            except Exception as e:
                # Categorize the error for better debugging
                error_category = "unknown"
                if "homing" in str(e).lower() or "home" in str(e).lower():
                    error_category = "homing_failure"
                elif "timeout" in str(e).lower():
                    error_category = "timeout"
                elif "alarm" in str(e).lower():
                    error_category = "grbl_alarm"
                elif "port" in str(e).lower() or "serial" in str(e).lower():
                    error_category = "connection_error"

                log_json_entry(
                    LogType.ERROR,
                    {
                        "message": "GRBL execution failed",
                        "component": "grbl",
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "error_category": error_category,
                        "gcode_file": output_file_adjusted,
                    },
                    print_message=f"[❌] GRBL execution failed ({error_category}): {e}",
                )
                log_json_entry(
                    LogType.GRBL,
                    {
                        "message": "G-code file saved (execution failed)",
                        "action": "gcode_file_saved",
                        "file_path": output_file_adjusted,
                        "error_category": error_category,
                    },
                    print_message=f"[💾] G-code file saved at: {output_file_adjusted}",
                )
                # Return None explicitly on failure so callers can avoid marking success
                try:
                    if "ser" in locals():
                        ser.close()
                except Exception:
                    pass
                return None
        else:
            log_json_entry(
                LogType.GRBL,
                {"message": "G-code generation complete (no execution)", "action": "gcode_generation_only", "file_path": output_file_adjusted},
                print_message=f"[💾] G-code generation complete but will not be executed. File saved: {output_file_adjusted}",
            )
        # If we get here with execute_grbl=True, success already returned above.
        # For generation-only, return the path.
        return output_file_adjusted

    except Exception as e:
        log_json_entry(
            LogType.ERROR,
            {"message": "Failed to process SVG", "component": "grbl", "error": str(e), "error_type": type(e).__name__, "svg_input": svg_input},
            print_message=f"[❌] Failed to process SVG: {e}",
        )
        return None
