"""
GRBL Utility Functions
Shared functions for GRBL communication and control
"""

import time
import serial
from serial.tools import list_ports
import subprocess
from event_logging.event_logger import log_json_entry
from event_logging.log_type import LogType

# Default configuration
DEFAULT_BAUD = 115200
DEFAULT_STATUS_POLL = 0.1
DEFAULT_HOME_TIMEOUT = 120  # seconds
DEFAULT_MOVE_TIMEOUT = 15  # seconds
DEFAULT_CMD_TIMEOUT = 5.0  # seconds
DEFAULT_FEED_RATE = 5000

PEN_DOWN_CMD = "M3 S50 ; PEN DOWN"  # Command to lower pen
PEN_UP_CMD = "M3 S30 ; PEN UP"  # Command to raise pen


def find_grbl_port(baud=DEFAULT_BAUD, timeout=0.5):
    """Find and connect to GRBL controller port"""
    ports = list(list_ports.comports())
    if not ports:
        raise RuntimeError("No serial ports found")

    log_json_entry(
        LogType.GRBL,
        {
            "message": "Available serial ports",
            "action": "port_discovery",
            "available_ports": [p.device for p in ports],
            "port_count": len(ports)
        },
        print_message=f"[🔌] Available ports: {', '.join(p.device for p in ports)}"
    )

    for p in ports:
        try:
            log_json_entry(
                LogType.GRBL,
                {
                    "message": "Testing port for GRBL",
                    "action": "port_test",
                    "port": p.device,
                    "baud": baud
                },
                print_message=f"[🔌] Testing {p.device}..."
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
                    {
                        "message": "GRBL port found",
                        "action": "port_found",
                        "port": p.device,
                        "response": line,
                        "baud": baud
                    },
                    print_message=f"[✅] {p.device} responds as GRBL: {line}"
                )
                return ser
            ser.close()
        except Exception as e:
            log_json_entry(
                LogType.GRBL,
                {
                    "message": "Port test failed",
                    "action": "port_test_failed",
                    "port": p.device,
                    "error": str(e),
                    "error_type": type(e).__name__
                },
                print_message=f"[❌] {p.device} failed ({e})"
            )

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
    print(f"[📤] {cmd}")
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


def ensure_homed(ser, home_timeout=DEFAULT_HOME_TIMEOUT):
    """Ensure GRBL is homed and setup coordinate system"""
    status = get_status(ser)
    if parse_state(status) == "Alarm":
        log_json_entry(
            LogType.GRBL,
            {
                "message": "Clearing alarm state",
                "action": "clear_alarm",
                "command": "$X",
                "status": status
            },
            print_message="[⚠️] Clearing alarm state with $X"
        )
        send_cmd(ser, "$X", wait_ok=True)
        time.sleep(0.2)

    log_json_entry(
        LogType.GRBL,
        {
            "message": "Running homing cycle",
            "action": "homing_start",
            "command": "$H",
            "timeout": home_timeout
        },
        print_message="[🏠] Running homing cycle ($H)..."
    )
    send_cmd(ser, "$H", wait_ok=False)

    start = time.time()
    while time.time() - start < home_timeout:
        status = get_status(ser)
        state = parse_state(status)
        if state == "Idle":
            log_json_entry(
                LogType.GRBL,
                {
                    "message": "Homing complete",
                    "action": "homing_complete",
                    "final_status": status,
                    "duration": time.time() - start
                },
                print_message="[✅] Homing complete"
            )
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
                    "origin": "0,0,0"
                },
                print_message="[📍] Work coordinate system G54 set to 0,0,0 at home position"
            )
            return
        if state == "Alarm":
            raise RuntimeError(f"Homing failed: {status}")
        time.sleep(DEFAULT_STATUS_POLL)

    raise TimeoutError("Homing took too long")


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
                "command": ' '.join(cmd),
                "input_file": svg_file,
                "output_file": output_file,
                "scale": scale_to
            },
            print_message=f"[🔧] Running vpype with gcode plugin: {' '.join(cmd)}"
        )
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        log_json_entry(
            LogType.GRBL,
            {
                "message": "vpype gcode conversion successful",
                "action": "vpype_success",
                "input_file": svg_file,
                "output_file": output_file,
                "result": str(result)
            },
            print_message="[✅] vpype gcode conversion successful"
        )
        return True

    except subprocess.CalledProcessError as e:
        log_json_entry(
            LogType.ERROR,
            {
                "message": "vpype gcode conversion failed",
                "component": "grbl",
                "error": e.stderr,
                "input_file": svg_file,
                "output_file": output_file
            },
            print_message=f"[❌] vpype gcode conversion failed: {e.stderr}"
        )
    except FileNotFoundError:
        log_json_entry(
            LogType.ERROR,
            {
                "message": "vpype or vpype-gcode not installed",
                "component": "grbl",
                "input_file": svg_file,
                "output_file": output_file
            },
            print_message="[❌] vpype or vpype-gcode not installed"
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
                "output_file": output_gcode
            },
            print_message=f"[✅] G-code converted to servo format: {output_gcode}"
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
                "output_file": output_gcode
            },
            print_message=f"[❌] Servo formatting failed: {e}"
        )
        return False


def setup_basic_grbl(ser, feed_rate=DEFAULT_FEED_RATE, use_absolute_positioning=False):
    """Setup basic GRBL configuration"""
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
            {
                "message": "Moving to work origin",
                "action": "move_to_origin",
                "origin_x": origin[0],
                "origin_y": origin[1]
            },
            print_message=f"[📍] Moving to work origin: X{origin[0]} Y{origin[1]}"
        )
        send_cmd(ser, f"G0 X{origin[0]} Y{origin[1]}", timeout=move_timeout)
        wait_until_idle(ser, move_timeout)
        send_cmd(ser, "G55")
        wait_until_idle(ser, DEFAULT_CMD_TIMEOUT)
        send_cmd(ser, f"G10 L20 P2 X0 Y0 Z0")
        wait_until_idle(ser, DEFAULT_CMD_TIMEOUT)
        log_json_entry(
            LogType.GRBL,
            {
                "message": "Work origin set",
                "action": "origin_set",
                "coordinate_system": "G55",
                "origin": origin
            },
            print_message="[📍] Work origin set in G55 coordinate system"
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
        {
            "message": "Starting G-code file execution",
            "action": "gcode_execution_start",
            "file": gcode_file,
            "timeout": move_timeout
        },
        print_message=f"[🚀] Executing G-code file: {gcode_file}"
    )

    try:
        with open(gcode_file, "r") as f:
            lines = f.readlines()
    except FileNotFoundError:
        raise FileNotFoundError(f"G-code file not found: {gcode_file}")

    total_lines = len(lines)
    executed_lines = 0

    lines = lines[3:]  # Skip first three lines (G20, G17, G90), from vpype inject somehow
    for line_num, line in enumerate(lines, 1):
        line = line.strip()

        # Skip empty lines and comments
        if not line or line.startswith(";") or line.startswith("%"):
            continue

        try:
            # Determine timeout based on command type
            if line.startswith(("G0", "G1", "G00", "G01")):
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
                        "progress_percent": (executed_lines / total_lines) * 100
                    },
                    print_message=f"[📋] Progress: {executed_lines}/{total_lines} lines executed"
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
                    "error_type": type(e).__name__
                },
                print_message=f"[❌] Failed to execute line {line_num}: {line} - Error: {e}"
            )
            raise

    log_json_entry(
        LogType.GRBL,
        {
            "message": "G-code execution complete",
            "action": "gcode_execution_complete",
            "file": gcode_file,
            "executed_lines": executed_lines,
            "total_lines": total_lines
        },
        print_message=f"[✅] G-code execution complete: {executed_lines} lines executed"
    )


def initialize_grbl_for_drawing(ser, origin=(0, 0, 0), origin_offset=(0, 0, 0), feed_rate=DEFAULT_FEED_RATE, use_absolute_positioning=False):
    """Complete GRBL initialization sequence for drawing"""
    log_json_entry(
        LogType.GRBL,
        {
            "message": "Initializing GRBL for drawing",
            "action": "initialization_start",
            "origin": origin,
            "origin_offset": origin_offset,
            "feed_rate": feed_rate,
            "absolute_positioning": use_absolute_positioning
        },
        print_message="[🎨] Initializing GRBL for drawing..."
    )

    ensure_homed(ser)
    setup_basic_grbl(ser, feed_rate, use_absolute_positioning=use_absolute_positioning)
    set_work_origin_and_offset(ser, origin, origin_offset)
    pen_control(ser, pen_down=False)

    log_json_entry(
        LogType.GRBL,
        {
            "message": "GRBL initialization complete",
            "action": "initialization_complete"
        },
        print_message="[✅] GRBL initialization complete"
    )


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
        from pathlib import Path
        import os

        svg_path = Path(svg_input)
        if not svg_path.exists():
            log_json_entry(
                LogType.ERROR,
                {
                    "message": "SVG file not found",
                    "component": "grbl",
                    "file_path": svg_input
                },
                print_message=f"[❌] SVG file not found: {svg_input}"
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
            {
                "message": "V-PYPE G-code generated",
                "action": "vpype_gcode_generated",
                "output_file": output_file_vpype,
                "input_file": str(svg_path)
            },
            print_message=f"[✅] V-PYPE G-code generated: {output_file_vpype}"
        )
        convert_gcode_to_servo_format(output_file_vpype, output_file_adjusted)
        log_json_entry(
            LogType.GRBL,
            {
                "message": "Servo G-code generated",
                "action": "servo_gcode_generated",
                "output_file": output_file_adjusted,
                "input_file": output_file_vpype
            },
            print_message=f"[✅] Servo G-code generated: {output_file_adjusted}"
        )

        if os.path.exists(output_file_vpype):
            os.remove(output_file_vpype)

        # Execute on GRBL if requested
        if execute_grbl:
            log_json_entry(
                LogType.GRBL,
                {
                    "message": "Starting GRBL execution",
                    "action": "grbl_execution_start",
                    "gcode_file": output_file_adjusted
                },
                print_message="[🚀] Executing on GRBL..."
            )
            try:
                ser = find_grbl_port()
                initialize_grbl_for_drawing(
                    ser, origin=origin, origin_offset=origin_offset, feed_rate=feed_rate, use_absolute_positioning=use_absolute_positioning
                )
                execute_gcode_file(ser, output_file_adjusted)
                log_json_entry(
                    LogType.GRBL,
                    {
                        "message": "Drawing complete",
                        "action": "drawing_complete",
                        "gcode_file": output_file_adjusted
                    },
                    print_message="[✅] Drawing complete!"
                )
                ser.close()
            except Exception as e:
                log_json_entry(
                    LogType.ERROR,
                    {
                        "message": "GRBL execution failed",
                        "component": "grbl",
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "gcode_file": output_file_adjusted
                    },
                    print_message=f"[❌] GRBL execution failed: {e}"
                )
                log_json_entry(
                    LogType.GRBL,
                    {
                        "message": "G-code file saved (execution failed)",
                        "action": "gcode_file_saved",
                        "file_path": output_file_adjusted
                    },
                    print_message=f"[💾] G-code file saved at: {output_file_adjusted}"
                )
        else:
            log_json_entry(
                LogType.GRBL,
                {
                    "message": "G-code generation complete (no execution)",
                    "action": "gcode_generation_only",
                    "file_path": output_file_adjusted
                },
                print_message=f"[💾] G-code generation complete but will not be executed. File saved: {output_file_adjusted}"
            )

        return output_file_adjusted

    except Exception as e:
        log_json_entry(
            LogType.ERROR,
            {
                "message": "Failed to process SVG",
                "component": "grbl",
                "error": str(e),
                "error_type": type(e).__name__,
                "svg_input": svg_input
            },
            print_message=f"[❌] Failed to process SVG: {e}"
        )
        return None
