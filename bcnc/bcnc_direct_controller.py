# import os
import subprocess
import xml.etree.ElementTree as ET
import re
from bcnc_utils import convert_z_to_servo, try_bcnc_cli_run, get_servo_gcode_header, get_servo_gcode_footer

# from pathlib import Path

# === FILVÄGAR ===
base_path = "/home/jbe/Dropbox/_outputs"

# Check if file exists and use fallback if needed
import os

svg_candidates = [f"{base_path}/impostor-20250725_185854_00001_.png.svg", f"{base_path}/test.svg", f"{base_path}/drawing.svg"]

svg_input = None
for candidate in svg_candidates:
    if os.path.exists(candidate):
        svg_input = candidate
        print(f"[INFO] Använder SVG-fil: {svg_input}")
        break

if not svg_input:
    print(f"[FEL] Ingen SVG-fil hittades. Sökta platser:")
    for candidate in svg_candidates:
        print(f"  - {candidate}")
    print("Skapa en test-SVG eller uppdatera sökvägen.")
    svg_input = svg_candidates[0]  # Use first as fallback

output_gcode = f"{base_path}/drawing.ngc"
origin_offset = (-40, -40, 0)




def svg_to_gcode_simple(svg_file, output_file, origin=(0, 0, 0)):
    """Simple SVG to G-code converter"""
    print("[INFO] Konverterar SVG till G-code direkt...")

    try:
        tree = ET.parse(svg_file)
        root = tree.getroot()

        gcode_lines = get_servo_gcode_header()
        gcode_lines.append(f"G92 X{origin[0]} Y{origin[1]} Z{origin[2]} ; Set origin")

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

        gcode_lines.extend(get_servo_gcode_footer())

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


def main():
    """Main function - direct SVG to G-code conversion with CLI execution"""
    print("[INFO] Konverterar SVG till G-code...")

    # Direct conversion (no bCNC CLI for conversion, only execution)
    if svg_to_gcode_simple(svg_input, output_gcode, origin_offset):
        print(f"[INFO] G-code genererad: {output_gcode}")

        # Try to run with bCNC CLI --run option
        if try_bcnc_cli_run(output_gcode):
            print("[INFO] bCNC startad för att köra G-code")
        else:
            print("[INFO] Kunde inte starta bCNC automatiskt")
            print("[INFO] Starta bCNC manuellt och ladda filen:")
            print(f"       {output_gcode}")
    else:
        print("[FEL] SVG konvertering misslyckades")


if __name__ == "__main__":
    main()
