"""
GRBL Utility Functions
Shared functions for GRBL communication and control
"""

import time
import serial
from serial.tools import list_ports
import subprocess

# import shutil
# import os

# Default configuration
DEFAULT_BAUD = 115200
DEFAULT_STATUS_POLL = 0.2
DEFAULT_HOME_TIMEOUT = 300
DEFAULT_MOVE_TIMEOUT = 120
DEFAULT_CMD_TIMEOUT = 5.0
DEFAULT_FEED_RATE = 3000

PEN_DOWN_CMD = "M3 S50 ; PEN DOWN"  # Command to lower pen
PEN_UP_CMD = "M3 S30 ; PEN UP"  # Command to raise pen


def find_grbl_port(baud=DEFAULT_BAUD, timeout=0.5):
    """Find and connect to GRBL controller port"""
    ports = list(list_ports.comports())
    if not ports:
        raise RuntimeError("No serial ports found")

    print(f"[INFO] Available ports: {', '.join(p.device for p in ports)}")

    for p in ports:
        try:
            print(f"[INFO] Testing {p.device}...")
            ser = serial.Serial(p.device, baud, timeout=timeout)
            time.sleep(2.0)
            ser.reset_input_buffer()
            ser.write(b"?")
            ser.flush()
            line = ser.readline().decode(errors="ignore").strip()
            if line.startswith("<") or "Grbl" in line:
                print(f"[INFO] {p.device} responds as GRBL: {line}")
                return ser
            ser.close()
        except Exception as e:
            print(f"[WARNING] {p.device} failed ({e})")

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
    print(f"[SEND] {cmd}")
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
        print("[INFO] Clearing alarm state with $X")
        send_cmd(ser, "$X", wait_ok=True)
        time.sleep(0.2)

    print("[INFO] Running homing cycle ($H)...")
    send_cmd(ser, "$H", wait_ok=False)

    start = time.time()
    while time.time() - start < home_timeout:
        status = get_status(ser)
        state = parse_state(status)
        if state == "Idle":
            print("[INFO] Homing complete")
            send_cmd(ser, "G54")
            wait_until_idle(ser, DEFAULT_CMD_TIMEOUT)
            send_cmd(ser, "G10 L20 P1 X0 Y0 Z0")
            wait_until_idle(ser, DEFAULT_CMD_TIMEOUT)
            print("[INFO] Work coordinate system G54 set to 0,0,0 at home position")
            return
        if state == "Alarm":
            raise RuntimeError(f"Homing failed: {status}")
        time.sleep(DEFAULT_STATUS_POLL)

    raise TimeoutError("Homing took too long")


def convert_with_vpype(svg_file, output_file):
    """Convert SVG to G-code using vpype with vpype-gcode plugin"""
    try:
        # Use one of the built-in profiles, then post-process for servo commands
        # temp_gcode = output_file + ".temp"

        cmd = ["vpype", "read", svg_file, "linemerge", "--tolerance", "0.1mm", "linesort", "gwrite", "--profile", "gcode", output_file]

        print(f"[INFO] Kör vpype med gcode plugin: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"[INFO] vpype gcode konvertering lyckades", result)
        return True

        # # Post-process the G-code to add servo commands and origin
        # if convert_gcode_to_servo_format(temp_gcode, output_file, origin):
        #     # Clean up temp file
        #     try:
        #         os.remove(temp_gcode)
        #     except ValueError:
        #         pass
        #     return True
        # else:
        #     return False

    except subprocess.CalledProcessError as e:
        print(f"[FEL] vpype gcode misslyckades: {e.stderr}")
        # Fallback to our custom converter
        # print("[INFO] Faller tillbaka till anpassad konverterare...")
        # return convert_with_vpype_fallback(svg_file, output_file, origin)
    except FileNotFoundError:
        print("[FEL] vpype eller vpype-gcode inte installerat")
        return False


# REFERENCE
# def convert_z_to_servo(input_file, output_file):
#     """Convert Z commands to servo commands for pen up/down control"""
#     print("[INFO] Konverterar Z-kommandon till servo...")
#     try:
#         current_pen_state = None
#         with open(input_file, "r") as infile, open(output_file, "w") as outfile:
#             for line in infile:
#                 clean = line.strip()
#                 if clean.startswith("G0"):
#                     if current_pen_state != "up":
#                         outfile.write(f"{PEN_UP_CMD}\n")
#                         current_pen_state = "up"
#                     outfile.write(line)
#                 elif clean.startswith("G1"):
#                     if current_pen_state != "down":
#                         outfile.write(f"{PEN_DOWN_CMD}\n")
#                         current_pen_state = "down"
#                     outfile.write(line)
#                 elif "Z" in clean:
#                     for part in clean.split():
#                         if part.startswith("Z"):
#                             try:
#                                 z = float(part[1:])
#                                 if z > 0 and current_pen_state != "up":
#                                     outfile.write(f"{PEN_UP_CMD}\n")
#                                     current_pen_state = "up"
#                                 elif z <= 0 and current_pen_state != "down":
#                                     outfile.write(f"{PEN_DOWN_CMD}\n")
#                                     current_pen_state = "down"
#                             except ValueError:
#                                 print(f"[FEL] Kunde inte konvertera Z-värde: {part}")
#                                 pass
#                     outfile.write(line)
#                 else:
#                     outfile.write(line)
#         print(f"[INFO] Optimerad G-kod sparad: {output_file}")
#         return True
#     except Exception as e:
#         print(f"[FEL] Z-to-servo konvertering misslyckades: {e}")
#         return False


def convert_gcode_to_servo_format(input_gcode, output_gcode, origin):
    """Convert vpype-generated G-code to servo format"""
    try:
        with open(input_gcode, "r") as f:
            lines = f.readlines()

        with open(output_gcode, "w") as f:
            # Write header
            # f.write("; G-code generated with vpype-gcode, optimized for servo control\n")
            # f.write("G21 ; Set units to millimeters\n")
            # f.write("G90 ; Absolute positioning\n")
            # f.write("G28 ; Home all axes\n")
            # f.write(f"G92 X{origin[0]} Y{origin[1]} Z{origin[2]} ; Set origin offset\n")
            # f.write("M3 S30 ; PEN UP (initial state)\n")
            # f.write("\n")

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

            # Write footer
            # f.write("\n")
            # f.write(f"{PEN_UP_CMD}\n")
            # f.write("G28 ; Return home\n")
            # f.write("M30 ; Program end\n")

        print(f"[INFO] G-code konverterad till servo-format: {output_gcode}")
        return True

    except Exception as e:
        print(f"[FEL] Servo-formatering misslyckades: {e}")
        return False


def setup_basic_grbl(ser, feed_rate=DEFAULT_FEED_RATE):
    """Setup basic GRBL configuration"""
    send_cmd(ser, "G21")  # mm units
    wait_until_idle(ser, DEFAULT_CMD_TIMEOUT)

    # send_cmd(ser, "G90")  # absolute positioning
    # wait_until_idle(ser, DEFAULT_CMD_TIMEOUT)

    send_cmd(ser, "G17")  # XY-plane
    wait_until_idle(ser, DEFAULT_CMD_TIMEOUT)
    send_cmd(ser, "G54")  # coordinate system
    wait_until_idle(ser, DEFAULT_CMD_TIMEOUT)
    send_cmd(ser, f"F{feed_rate}")  # set feed rate
    wait_until_idle(ser, DEFAULT_CMD_TIMEOUT)


def set_work_origin(ser, origin_x, origin_y, origin_z=0.0, move_timeout=DEFAULT_MOVE_TIMEOUT):
    """Move to and set work origin offset"""
    if origin_x != 0 or origin_y != 0:
        print(f"[INFO] Moving to work origin: X{origin_x} Y{origin_y}")
        send_cmd(ser, f"G0 X{origin_x} Y{origin_y}", timeout=move_timeout)
        wait_until_idle(ser, move_timeout)
        send_cmd(ser, "G55")
        wait_until_idle(ser, DEFAULT_CMD_TIMEOUT)
        send_cmd(ser, f"G10 L20 P2 X0 Y0 Z{origin_z}")
        wait_until_idle(ser, DEFAULT_CMD_TIMEOUT)
        print("[INFO] Work origin set in G55 coordinate system")
        return True
    return False


def pen_control(ser, pen_down=True, pen_down_cmd=PEN_DOWN_CMD, pen_up_cmd=PEN_UP_CMD):
    """Control pen up/down using servo commands"""
    if pen_down:
        send_cmd(ser, pen_down_cmd)
    else:
        send_cmd(ser, pen_up_cmd)
    wait_until_idle(ser, DEFAULT_CMD_TIMEOUT)


def execute_gcode_file(ser, gcode_file, move_timeout=DEFAULT_MOVE_TIMEOUT):
    """Execute G-code file line by line with proper waiting"""
    print(f"[INFO] Executing G-code file: {gcode_file}")

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
            wait_until_idle(ser, timeout)
            executed_lines += 1

            if executed_lines % 10 == 0:  # Progress update every 10 commands
                print(f"[INFO] Progress: {executed_lines}/{total_lines} lines executed")

        except Exception as e:
            print(f"[ERROR] Failed to execute line {line_num}: {line}")
            print(f"[ERROR] Error: {e}")
            raise

    print(f"[INFO] G-code execution complete: {executed_lines} lines executed")


def initialize_grbl_for_drawing(ser, origin_x=0, origin_y=0, origin_z=0, feed_rate=DEFAULT_FEED_RATE):
    """Complete GRBL initialization sequence for drawing"""
    print("[INFO] Initializing GRBL for drawing...")

    # Home and setup coordinate system
    ensure_homed(ser)

    # Basic GRBL setup with feed rate
    setup_basic_grbl(ser, feed_rate)

    # Set work origin if specified
    if origin_x != 0 or origin_y != 0:
        set_work_origin(ser, origin_x, origin_y, origin_z)

    # Initial pen up
    pen_control(ser, pen_down=False)

    print("[INFO] GRBL initialization complete")
