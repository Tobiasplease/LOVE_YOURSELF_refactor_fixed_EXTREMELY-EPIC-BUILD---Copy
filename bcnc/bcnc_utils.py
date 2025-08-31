"""
bCNC Utility Functions
Shared functions for G-code processing and bCNC CLI operations
"""

import shutil
import subprocess


def convert_z_to_servo(input_file, output_file):
    """Convert Z commands to servo commands for pen up/down control"""
    print("[INFO] Konverterar Z-kommandon till servo...")
    try:
        current_pen_state = None
        with open(input_file, "r") as infile, open(output_file, "w") as outfile:
            for line in infile:
                clean = line.strip()
                if clean.startswith("G0"):
                    if current_pen_state != "up":
                        outfile.write("M3 S40 ; PEN UP\n")
                        current_pen_state = "up"
                    outfile.write(line)
                elif clean.startswith("G1"):
                    if current_pen_state != "down":
                        outfile.write("M3 S50 ; PEN DOWN\n")
                        current_pen_state = "down"
                    outfile.write(line)
                elif "Z" in clean:
                    for part in clean.split():
                        if part.startswith("Z"):
                            try:
                                z = float(part[1:])
                                if z > 0 and current_pen_state != "up":
                                    outfile.write("M3 S30 ; PEN UP\n")
                                    current_pen_state = "up"
                                elif z <= 0 and current_pen_state != "down":
                                    outfile.write("M3 S90 ; PEN DOWN\n")
                                    current_pen_state = "down"
                            except ValueError:
                                print(f"[FEL] Kunde inte konvertera Z-värde: {part}")
                                pass
                    outfile.write(line)
                else:
                    outfile.write(line)
        print(f"[INFO] Optimerad G-kod sparad: {output_file}")
        return True
    except Exception as e:
        print(f"[FEL] Z-to-servo konvertering misslyckades: {e}")
        return False


def try_bcnc_cli_run(gcode_file):
    """Try to run G-code file using bCNC CLI with filename argument"""
    try:
        # Try different methods to start bCNC with the file
        cli_commands = [
            # Method 1: Load file and run immediately
            ["bcnc", "--run", gcode_file],
            ["bCNC", "--run", gcode_file],
            # Method 2: Just load the file (user can run manually)
            ["bcnc", gcode_file],
            ["bCNC", gcode_file],
        ]

        for cmd in cli_commands:
            try:
                print(f"[INFO] Försöker: {' '.join(cmd)}")
                # Start bCNC in background
                process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                print(f"[INFO] bCNC startad med PID {process.pid}")

                if "--run" in cmd:
                    print("[INFO] Filen kommer köras automatiskt")
                else:
                    print("[INFO] Filen laddad - tryck RUN i bCNC för att starta")
                return True
            except FileNotFoundError:
                continue

    except Exception as e:
        print(f"[DEBUG] bCNC CLI inte tillgängligt: {e}")

    return False


def check_bcnc_available():
    """Check if bCNC is available in the system"""
    bcnc_variants = ["bcnc", "bCNC"]

    for variant in bcnc_variants:
        if shutil.which(variant):
            print(f"[INFO] Hittade {variant}: {shutil.which(variant)}")
            return variant

    print("[FEL] bCNC inte hittad i PATH")
    return None


def get_servo_gcode_header():
    """Get standard G-code header with servo setup"""
    return [
        "; G-code generated with servo control",
        "G21 ; Set units to millimeters",
        "G90 ; Absolute positioning",
        "G28 ; Home all axes",
        "M3 S30 ; PEN UP (initial state)",
        "",
    ]


def get_servo_gcode_footer():
    """Get standard G-code footer with servo cleanup"""
    return ["", "M3 S30 ; PEN UP", "G28 ; Return home", "M30 ; Program end"]


def parse_svg_path_simple(d):
    """Simple SVG path parser for G-code generation"""
    gcode = []
    if not d:
        return gcode

    import re

    commands = re.findall(r"[MLZ][^MLZ]*", d)
    pen_down = False

    for cmd in commands:
        cmd = cmd.strip()
        if cmd.startswith("M"):
            coords = re.findall(r"-?\d+\.?\d*", cmd[1:])
            if len(coords) >= 2:
                x, y = float(coords[0]), float(coords[1])
                if pen_down:
                    gcode.append("M3 S30 ; PEN UP")
                    pen_down = False
                gcode.append(f"G0 X{x} Y{y} ; Move to")
        elif cmd.startswith("L"):
            coords = re.findall(r"-?\d+\.?\d*", cmd[1:])
            if len(coords) >= 2:
                x, y = float(coords[0]), float(coords[1])
                if not pen_down:
                    gcode.append("M3 S50 ; PEN DOWN")
                    pen_down = True
                gcode.append(f"G1 X{x} Y{y} ; Draw to")
        elif cmd.startswith("Z"):
            if pen_down:
                gcode.append("M3 S30 ; PEN UP")
                pen_down = False

    return gcode


def parse_svg_points_simple(points_str):
    """Simple SVG points parser for G-code generation"""
    gcode = []
    import re

    points = re.findall(r"-?\d+\.?\d*,-?\d+\.?\d*", points_str)

    if points:
        first_point = points[0].split(",")
        x, y = float(first_point[0]), float(first_point[1])
        gcode.append(f"G0 X{x} Y{y} ; Move to start")
        gcode.append("M3 S50 ; PEN DOWN")

        for point in points[1:]:
            coords = point.split(",")
            x, y = float(coords[0]), float(coords[1])
            gcode.append(f"G1 X{x} Y{y} ; Draw to")

        gcode.append("M3 S30 ; PEN UP")

    return gcode


def convert_gcode_to_servo_format(input_gcode, output_gcode, origin):
    """Convert standard G-code to servo format with proper pen control"""
    try:
        with open(input_gcode, "r") as f:
            lines = f.readlines()

        with open(output_gcode, "w") as f:
            # Write header
            f.write("; G-code optimized for servo control\n")
            f.write("G21 ; Set units to millimeters\n")
            f.write("G90 ; Absolute positioning\n")
            f.write("G28 ; Home all axes\n")
            f.write(f"G92 X{origin[0]} Y{origin[1]} Z{origin[2]} ; Set origin offset\n")
            f.write("M3 S30 ; PEN UP (initial state)\n")
            f.write("\n")

            pen_down = False
            for line in lines:
                line = line.strip()

                # Skip headers and comments
                if line.startswith(";") or line.startswith("%") or not line:
                    continue

                # Handle movement commands
                if line.startswith("G0") or line.startswith("G00"):
                    # Rapid move - pen should be up
                    if pen_down:
                        f.write("M3 S30 ; PEN UP\n")
                        pen_down = False
                    f.write(f"{line}\n")
                elif line.startswith("G1") or line.startswith("G01"):
                    # Linear move - pen should be down
                    if not pen_down:
                        f.write("M3 S50 ; PEN DOWN\n")
                        pen_down = True
                    f.write(f"{line}\n")
                else:
                    # Pass through other commands
                    f.write(f"{line}\n")

            # Write footer
            f.write("\n")
            f.write("M3 S30 ; PEN UP\n")
            f.write("G28 ; Return home\n")
            f.write("M30 ; Program end\n")

        print(f"[INFO] G-code konverterad till servo-format: {output_gcode}")
        return True

    except Exception as e:
        print(f"[FEL] Servo-formatering misslyckades: {e}")
        return False
