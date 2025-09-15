"""
GRBL Utility Functions
Shared functions for GRBL communication and control
"""

import subprocess
import threading
import time

import serial
from serial.tools import list_ports

from event_logging.event_logger import log_json_entry
from event_logging.log_type import LogType
from .warp_transform import warp_transform_line

# Import pen servo configuration
try:
    from config.config import GRBL_PEN_DOWN_S, GRBL_PEN_UP_S, GRBL_SPINDLE_MAX_S, GRBL_SPINDLE_MIN_S, GRBL_WARP_TRANSFORM
except Exception:
    GRBL_PEN_UP_S, GRBL_PEN_DOWN_S, GRBL_SPINDLE_MAX_S, GRBL_SPINDLE_MIN_S = 30, 50, 255, 0
    GRBL_WARP_TRANSFORM = True

# Default configuration
DEFAULT_BAUD = 115200
DEFAULT_STATUS_POLL = 0.1
DEFAULT_HOME_TIMEOUT = 120  # seconds
DEFAULT_MOVE_TIMEOUT = 15  # seconds
DEFAULT_CMD_TIMEOUT = 5.0  # seconds
DEFAULT_FEED_RATE = 12000  # Balanced speed for clean drawing

PEN_DOWN_CMD = f"M3 S{GRBL_PEN_DOWN_S} ; PEN DOWN"  # Command to lower pen
PEN_UP_CMD = f"M3 S{GRBL_PEN_UP_S} ; PEN UP"  # Command to raise pen


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
            ser = serial.Serial(preferred_port, baud, timeout=timeout)
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
            "port_count": len(filtered_ports)
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
            ser = serial.Serial(p.device, baud, timeout=timeout)
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
    
    # CRITICAL SAFETY: Multiple pen up commands with delays to ensure servo responds
    # This is essential when switching between idle and drawing states
    try:
        log_json_entry(
            LogType.GRBL,
            {"message": "SAFETY: Ensuring pen is raised before homing", "action": "pen_safety_sequence", "repeats": GRBL_PEN_UP_REPEATS, "dwell_s": GRBL_PEN_UP_DWELL_S},
            print_message="[⚠️ SAFETY] Ensuring pen is raised before homing sequence...",
        )
        
        # Send multiple pen up commands with delays to ensure servo catches the signal
        for i in range(int(GRBL_PEN_UP_REPEATS)):
            try:
                send_cmd(ser, PEN_UP_CMD, wait_ok=False)
                time.sleep(0.25)  # Give servo time to respond
            except Exception:
                pass
        
        # Additional dwell to ensure servo has fully moved
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

            # Clear any existing alarm state
            status = get_status(ser)
            if parse_state(status) == "Alarm":
                log_json_entry(
                    LogType.GRBL,
                    {"message": "Clearing alarm state", "action": "clear_alarm", "command": "$X", "status": status, "attempt": attempt + 1},
                    print_message="[⚠️] Clearing alarm state with $X",
                )
                send_cmd(ser, "$X", wait_ok=True)
                time.sleep(0.5)  # Give more time after clearing alarm
                # Raise pen again after clearing alarm, in case prior command was ignored
                try:
                    send_cmd(ser, PEN_UP_CMD, wait_ok=False)
                    time.sleep(0.2)
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

        # Add scaling if specified
        if scale_to:
            cmd.extend(["layout", "--fit-to-margins", "1cm", scale_to])

        cmd.extend(["linemerge", "--tolerance", "0.1mm", "linesort", "gwrite", "--profile", "gcodemm", output_file])

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
    """Convert vpype-generated G-code to servo format"""
    try:
        with open(input_gcode, "r") as f:
            lines = f.readlines()

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


def execute_gcode_file(ser, gcode_file, move_timeout=DEFAULT_MOVE_TIMEOUT):
    """Execute G-code file line by line with proper waiting"""
    log_json_entry(
        LogType.GRBL,
        {"message": "Starting G-code file execution", "action": "gcode_execution_start", "file": gcode_file, "timeout": move_timeout},
        print_message=f"[🚀] Executing G-code file: {gcode_file}",
    )
    
    
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
        
        # Simple drawing state tracking
        DrawingState.start_drawing(
            drawing_file=gcode_file,
            description="actively drawing",
            intent="drawing based on observations"
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

    try:
        with open(gcode_file, "r") as f:
            lines = f.readlines()
    except FileNotFoundError:
        raise FileNotFoundError(f"G-code file not found: {gcode_file}")

    total_lines = len(lines)
    executed_lines = 0

    lines = lines[3:]  # Skip first three lines (G20, G17, G90), from vpype inject somehow

    try:
        for line_num, line in enumerate(lines, 1):
            line = line.strip()

            # Skip empty lines and comments
            if not line or line.startswith(";") or line.startswith("%"):
                continue

            try:
                # Determine timeout and transform based on command type
                if line.startswith(("G0", "G1", "G00", "G01")):
                    if GRBL_WARP_TRANSFORM:
                        line = warp_transform_line(line)  # warp transform line coords
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

    except KeyboardInterrupt:
        log_json_entry(
            LogType.GRBL,
            {
                "message": "G-code execution interrupted by user",
                "action": "execution_interrupted",
                "executed_lines": executed_lines,
                "total_lines": total_lines,
                "progress_percent": (executed_lines / total_lines) * 100 if total_lines > 0 else 0,
            },
            print_message=f"[⚠️] G-code execution interrupted! Executed {executed_lines}/{total_lines} lines",
        )
        # Send any emergency stops or cleanup commands if needed
        try:
            send_cmd(ser, "M3 S30", wait_ok=False)  # Pen up
            send_cmd(ser, "!", wait_ok=False)  # Emergency stop
        except Exception:
            pass
        
        
        raise

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
    
    # Step 1: End drawing state (releases drawing context from captions)
    try:
        from utils.drawing_state import DrawingState
        DrawingState.end_drawing()
    except Exception as e:
        print(f"[⚠️] Could not update drawing state: {e}")
    
    # Step 2: Home the machine and pause for completion ritual
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
            {"message": "Homed for completion ritual - staying at home for 7-second pause", "action": "completion_homing_complete"},
            print_message="[✅] Homing complete - beginning 7-second completion pause at home position",
        )
        
        # Step 3: Unlock gaze system during completion pause
        try:
            from vision.gaze import set_drawing_mode
            set_drawing_mode(active=False)
            print("[👁️] Gaze unlocked for completion ritual")
        except Exception as e:
            print(f"[⚠️] Could not unlock gaze system: {e}")
        
        # Step 4: Trigger self-critique during 7-second pause (AT HOME POSITION)
        completion_thread_running = threading.Event()
        
        def completion_self_critique():
            """Generate self-critique reflection during completion pause."""
            try:
                # Get the drawing context for self-critique
                from utils.state_manager import state_manager
                from captioner.prompt_interface import PromptInterface
                
                # Get drawing details
                drawing_prompt = getattr(state_manager, 'current_drawing_prompt', 'recent drawing')
                
                # Get the compressed description if available
                from utils.drawing_state import DrawingState
                drawing_info = DrawingState.get_drawing_info()
                compressed_desc = drawing_info.get('description', 'a drawing') if drawing_info else 'a drawing'
                
                # Build self-critique prompt using existing system
                critique_prompt = f"""You have just finished drawing. Look at what you created.
                
You were drawing: {compressed_desc}
Original intent: {drawing_prompt}

The pen has lifted, the machine has returned home. You can feel the completion.
How do you reflect on this creative act? What did you express through these lines?

Respond with 2-3 sentences of honest self-reflection about your artwork."""
                
                # Use the captioner system to generate reflection
                try:
                    from captioner.captioner import Captioner
                    from utils.ollama import query_ollama
                    from config import config
                    
                    # Create critique using reflection system
                    model_options = {"temperature": 0.8, "top_p": 0.9, "num_predict": 100}
                    
                    # Import consolidated system prompt
                    from captioner.prompts import SELF_CRITIQUE_SYSTEM_PROMPT
                    
                    self_critique = query_ollama(
                        critique_prompt,
                        model=config.OLLAMA_MODEL,
                        system_prompt=SELF_CRITIQUE_SYSTEM_PROMPT,
                        options=model_options,
                        timeout=30
                    )
                    
                    if self_critique and self_critique.strip():
                        log_json_entry(
                            LogType.REFLECTION,
                            {
                                "message": "Drawing completion self-critique",
                                "action": "drawing_self_critique", 
                                "drawing_intent": drawing_prompt,
                                "drawing_description": compressed_desc,
                                "self_critique": self_critique.strip(),
                                "completion_type": "post_drawing_reflection"
                            },
                            print_message=f"[🎨💭] Drawing self-critique: {self_critique.strip()}",
                        )
                        
                        # Store the completion memory for future reference
                        try:
                            if hasattr(state_manager, 'captioner') and hasattr(state_manager.captioner, 'observe'):
                                state_manager.captioner.observe(
                                    f"Completed drawing {compressed_desc}. Reflection: {self_critique.strip()[:100]}",
                                    state_manager.captioner.current_mood if hasattr(state_manager.captioner, 'current_mood') else 0.5,
                                    "",
                                    memory_type="drawing_completion"
                                )
                                print(f"[📝] Stored drawing completion in memory")
                        except Exception as e:
                            print(f"[⚠️] Could not store completion memory: {e}")
                    
                except Exception as e:
                    print(f"[⚠️] Could not generate drawing self-critique: {e}")
                    
            except Exception as e:
                print(f"[⚠️] Error in completion self-critique: {e}")
            finally:
                completion_thread_running.set()
        
        # Start self-critique in background thread
        critique_thread = threading.Thread(target=completion_self_critique, daemon=True)
        critique_thread.start()
        
        # 7-second completion pause AT HOME POSITION (allows time for self-critique)
        time.sleep(7.0)
        
        # Ensure self-critique thread completes
        if not completion_thread_running.is_set():
            completion_thread_running.wait(timeout=5.0)
        
        log_json_entry(
            LogType.GRBL,
            {"message": "Completion ritual finished at home position", "action": "completion_ritual_complete"},
            print_message="[✅] Completion ritual finished at home - idle movements will handle positioning",
        )
        # Notify any runtime hook that GRBL drawing has finished
        try:
            from utils.hooks import on_grbl_drawing_complete
            if callable(on_grbl_drawing_complete):
                on_grbl_drawing_complete()
        except Exception as e:
            print(f"[hooks] on_grbl_drawing_complete error: {e}")
        
    except Exception as e:
        log_json_entry(
            LogType.ERROR,
            {"message": f"Completion ritual failed: {e}", "component": "grbl", "error": str(e)},
            print_message=f"[❌] Completion ritual failed: {e}",
        )
    
    # Step 5: Clear CNC execution state to allow idle movements to resume
    try:
        from utils.state_manager import state_manager
        state_manager.finish_cnc_execution()
    except Exception as e:
        print(f"[⚠️] Could not clear CNC execution state: {e}")
    
    # Step 6: Resume idle movements after completion ritual (and uArm action)
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

                initialize_grbl_for_drawing(
                    ser,
                    origin=origin,
                    origin_offset=origin_offset,
                    feed_rate=feed_rate,
                    use_absolute_positioning=use_absolute_positioning,
                    max_homing_retries=max_retries,
                )
                
                # Execute G-code in a separate thread to prevent blocking captions
                import threading
                import time
                
                gcode_complete = threading.Event()
                gcode_error = None
                
                def execute_gcode_threaded():
                    """Execute G-code in background thread."""
                    nonlocal gcode_error
                    try:
                        execute_gcode_file(ser, output_file_adjusted)
                    except Exception as e:
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
                    raise gcode_error
                log_json_entry(
                    LogType.GRBL,
                    {"message": "Drawing complete", "action": "drawing_complete", "gcode_file": output_file_adjusted},
                    print_message="[✅] Drawing complete!",
                )
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
