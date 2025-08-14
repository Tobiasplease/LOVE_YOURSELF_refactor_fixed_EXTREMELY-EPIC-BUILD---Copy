"""
bCNC Utility Functions
Shared functions for G-code processing and bCNC CLI operations
"""

import subprocess
import shutil


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
