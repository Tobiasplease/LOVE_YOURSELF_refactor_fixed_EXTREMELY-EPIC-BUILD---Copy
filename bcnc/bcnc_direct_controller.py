# import os
import subprocess
import xml.etree.ElementTree as ET
import re

# from pathlib import Path

# === FILVÄGAR ===
base_path = "/home/jbe/Dropbox/_outputs"
svg_input = f"{base_path}/impostor-20250725_185854_00001_.png.svg"
output_gcode = f"{base_path}/drawing.ngc"
origin_offset = (-40, -40, 0)


def try_bcnc_cli(svg_file, output_file, origin=(0, 0, 0)):
    """Try using bCNC command line interface if available"""
    try:
        # Try different possible bCNC CLI commands
        cli_commands = [
            ["bcnc", "--load", svg_file, "--save", output_file],
            ["bCNC", "--load", svg_file, "--save", output_file],
            ["python", "-m", "bCNC", "--load", svg_file, "--save", output_file],
        ]

        for cmd in cli_commands:
            try:
                print(f"[INFO] Försöker: {' '.join(cmd)}")
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                if result.returncode == 0:
                    print(f"[INFO] bCNC CLI lyckades: {output_file}")
                    return True
                else:
                    print(f"[DEBUG] Kommando misslyckades: {result.stderr}")
            except (subprocess.TimeoutExpired, FileNotFoundError):
                continue

    except Exception as e:
        print(f"[DEBUG] bCNC CLI inte tillgängligt: {e}")

    return False


def svg_to_gcode_simple(svg_file, output_file, origin=(0, 0, 0)):
    """Simple SVG to G-code converter"""
    print("[INFO] Konverterar SVG till G-code direkt...")

    try:
        tree = ET.parse(svg_file)
        root = tree.getroot()

        gcode_lines = [
            "; G-code generated from SVG",
            "G21 ; Set units to millimeters",
            "G90 ; Absolute positioning",
            "G28 ; Home all axes",
            f"G92 X{origin[0]} Y{origin[1]} Z{origin[2]} ; Set origin",
            "M3 S30 ; PEN UP",
            "",
        ]

        # Find SVG namespace
        ns = {"svg": "http://www.w3.org/2000/svg"}
        if root.tag.startswith("{"):
            ns_match = re.match(r"\{([^}]+)\}", root.tag)
            if ns_match:
                ns["svg"] = ns_match.group(1)

        # Process paths and polylines
        for elem in root.iter():
            if elem.tag.endswith("path"):
                d = elem.get("d", "")
                gcode_lines.extend(parse_svg_path(d))
            elif elem.tag.endswith("polyline") or elem.tag.endswith("polygon"):
                points = elem.get("points", "")
                gcode_lines.extend(parse_svg_points(points))
            elif elem.tag.endswith("line"):
                x1, y1 = float(elem.get("x1", 0)), float(elem.get("y1", 0))
                x2, y2 = float(elem.get("x2", 0)), float(elem.get("y2", 0))
                gcode_lines.extend([f"G0 X{x1} Y{y1} ; Move to start", "M3 S50 ; PEN DOWN", f"G1 X{x2} Y{y2} ; Draw line", "M3 S30 ; PEN UP"])

        gcode_lines.extend(["", "M3 S30 ; PEN UP", "G28 ; Return home", "M30 ; Program end"])

        with open(output_file, "w") as f:
            f.write("\n".join(gcode_lines))

        print(f"[INFO] G-code sparad: {output_file}")
        return True

    except Exception as e:
        print(f"[FEL] SVG konvertering misslyckades: {e}")
        return False


def parse_svg_path(d):
    """Parse SVG path data to G-code"""
    gcode = []
    if not d:
        return gcode

    # Simple path parser - handles M (move) and L (line) commands
    commands = re.findall(r"[MLZ][^MLZ]*", d)
    pen_down = False

    for cmd in commands:
        cmd = cmd.strip()
        if cmd.startswith("M"):
            # Move command
            coords = re.findall(r"-?\d+\.?\d*", cmd[1:])
            if len(coords) >= 2:
                x, y = float(coords[0]), float(coords[1])
                if pen_down:
                    gcode.append("M3 S30 ; PEN UP")
                    pen_down = False
                gcode.append(f"G0 X{x} Y{y} ; Move to")
        elif cmd.startswith("L"):
            # Line command
            coords = re.findall(r"-?\d+\.?\d*", cmd[1:])
            if len(coords) >= 2:
                x, y = float(coords[0]), float(coords[1])
                if not pen_down:
                    gcode.append("M3 S50 ; PEN DOWN")
                    pen_down = True
                gcode.append(f"G1 X{x} Y{y} ; Draw to")
        elif cmd.startswith("Z"):
            # Close path
            if pen_down:
                gcode.append("M3 S30 ; PEN UP")
                pen_down = False

    return gcode


def parse_svg_points(points_str):
    """Parse SVG points to G-code"""
    gcode = []
    points = re.findall(r"-?\d+\.?\d*,-?\d+\.?\d*", points_str)

    if points:
        # Move to first point
        first_point = points[0].split(",")
        x, y = float(first_point[0]), float(first_point[1])
        gcode.append(f"G0 X{x} Y{y} ; Move to start")
        gcode.append("M3 S50 ; PEN DOWN")

        # Draw to remaining points
        for point in points[1:]:
            coords = point.split(",")
            x, y = float(coords[0]), float(coords[1])
            gcode.append(f"G1 X{x} Y{y} ; Draw to")

        gcode.append("M3 S30 ; PEN UP")

    return gcode


def run_gcode_via_bcnc_cli(gcode_file):
    """Try to run G-code via bCNC CLI"""
    try:
        cli_commands = [["bcnc", "--run", gcode_file], ["bCNC", "--run", gcode_file]]

        for cmd in cli_commands:
            try:
                print(f"[INFO] Kör G-code: {' '.join(cmd)}")
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                if result.returncode == 0:
                    print("[INFO] G-code körning startad")
                    return True
                else:
                    print(f"[DEBUG] Körning misslyckades: {result.stderr}")
            except (subprocess.TimeoutExpired, FileNotFoundError):
                continue

    except Exception as e:
        print(f"[FEL] Kunde inte köra G-code: {e}")

    print("[INFO] Använd GUI för att köra G-code manuellt")
    return False


def main():
    """Main function - try CLI first, fallback to direct conversion"""
    print("[INFO] Försöker CLI-metod först...")

    # Try bCNC CLI first
    if try_bcnc_cli(svg_input, output_gcode, origin_offset):
        run_gcode_via_bcnc_cli(output_gcode)
        return

    print("[INFO] CLI inte tillgängligt, konverterar direkt...")

    # Fallback to direct conversion
    if svg_to_gcode_simple(svg_input, output_gcode, origin_offset):
        if not run_gcode_via_bcnc_cli(output_gcode):
            print(f"[INFO] G-code klar: {output_gcode}")
            print("[INFO] Ladda filen manuellt i bCNC för körning")
    else:
        print("[FEL] Konvertering misslyckades")


if __name__ == "__main__":
    main()
