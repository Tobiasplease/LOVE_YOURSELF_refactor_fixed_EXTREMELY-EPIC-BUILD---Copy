#!/usr/bin/env python3
"""
Standalone SVG to G-code converter using external tools
Bypasses bCNC GUI entirely while still providing servo conversion
"""

import os

# import re
import subprocess
import shutil

# from cycler import V
from bcnc_utils import convert_z_to_servo, try_bcnc_cli_run, check_bcnc_available

# from pathlib import Path

# === Configuration ===
base_path = "/home/jbe/Dropbox/_outputs"
# base_path = "/Users/jbe/Dropbox/_outputs/"
svg_input = f"{base_path}/impostor-20250725_185854_00001_.png.svg"
# output_gcode = f"{base_path}/drawing.ngc"
output_gcode = f"{svg_input}.ngc"
origin_offset = (-40, -40, 0)


def check_external_tools():
    """Check which SVG to G-code tools are available"""
    tools = {
        "vpype": shutil.which("vpype"),
        "svg2gcode": shutil.which("svg2gcode"),
        "inkscape": shutil.which("inkscape"),
        "f-engrave": shutil.which("f-engrave.py"),
    }

    available = {name: path for name, path in tools.items() if path}
    print(f"[INFO] Tillgängliga verktyg: {list(available.keys())}")
    return available


def convert_with_vpype(svg_file, output_file, origin=(0, 0, 0)):
    """Convert SVG to G-code using vpype with vpype-gcode plugin"""
    try:
        # Use one of the built-in profiles, then post-process for servo commands
        temp_gcode = output_file + ".temp"

        cmd = ["vpype", "read", svg_file, "linemerge", "--tolerance", "0.1mm", "linesort", "gwrite", "--profile", "gcode", temp_gcode]

        print(f"[INFO] Kör vpype med gcode plugin: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"[INFO] vpype gcode konvertering lyckades", result)

        # Post-process the G-code to add servo commands and origin
        if convert_gcode_to_servo_format(temp_gcode, output_file, origin):
            # Clean up temp file
            try:
                os.remove(temp_gcode)
            except ValueError:
                pass
            return True
        else:
            return False

    except subprocess.CalledProcessError as e:
        print(f"[FEL] vpype gcode misslyckades: {e.stderr}")
        # Fallback to our custom converter
        print("[INFO] Faller tillbaka till anpassad konverterare...")
        return convert_with_vpype_fallback(svg_file, output_file, origin)
    except FileNotFoundError:
        print("[FEL] vpype eller vpype-gcode inte installerat")
        return False


def convert_gcode_to_servo_format(input_gcode, output_gcode, origin):
    """Convert vpype-generated G-code to servo format"""
    try:
        with open(input_gcode, "r") as f:
            lines = f.readlines()

        with open(output_gcode, "w") as f:
            # Write header
            f.write("; G-code generated with vpype-gcode, optimized for servo control\n")
            f.write("G21 ; Set units to millimeters\n")
            f.write("G90 ; Absolute positioning\n")
            f.write("G28 ; Home all axes\n")
            f.write(f"G92 X{origin[0]} Y{origin[1]} Z{origin[2]} ; Set origin offset\n")
            f.write("M3 S30 ; PEN UP (initial state)\n")
            f.write("\n")

            pen_down = False
            for line in lines:
                line = line.strip()

                # Skip vpype headers and comments
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


def create_vpype_gcode_config(origin):
    """Create a temporary vpype config file with servo settings"""
    import tempfile

    config_content = f"""
[gcode_writer]

[gcode_writer.servo]
document_start = '''
; G-code generated with vpype-gcode for servo control
G21 ; Set units to millimeters
G90 ; Absolute positioning
G28 ; Home all axes
G92 X{origin[0]} Y{origin[1]} Z{origin[2]} ; Set origin offset
M3 S30 ; PEN UP (initial state)
'''

segment_first = '''
G0 X%f Y%f ; Move to start
M3 S50 ; PEN DOWN
'''

segment_line = '''
G1 X%f Y%f ; Draw line
'''

segment_end = '''
M3 S30 ; PEN UP
'''

document_end = '''
M3 S30 ; PEN UP
G28 ; Return home
M30 ; Program end
'''

feed_rate = 1000
"""

    # Create temporary config file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
        f.write(config_content)
        return f.name


def convert_with_vpype_fallback(svg_file, output_file, origin=(0, 0, 0)):
    """Fallback: Use built-in direct converter if vpype-gcode fails"""
    print("[INFO] vpype-gcode inte tillgängligt, använder direkt konverterare...")

    # Import the direct converter from our other module
    try:
        import xml.etree.ElementTree as ET

        # import re
        from bcnc_utils import get_servo_gcode_header, get_servo_gcode_footer

        tree = ET.parse(svg_file)
        root = tree.getroot()

        gcode_lines = get_servo_gcode_header()
        gcode_lines.append(f"G92 X{origin[0]} Y{origin[1]} Z{origin[2]} ; Set origin offset")

        # Simple SVG processing (no vpype optimization)
        for elem in root.iter():
            if elem.tag.endswith("path"):
                d = elem.get("d", "")
                gcode_lines.extend(parse_svg_path_simple(d))
            elif elem.tag.endswith("polyline"):
                points = elem.get("points", "")
                gcode_lines.extend(parse_svg_points_simple(points))

        gcode_lines.extend(get_servo_gcode_footer())

        with open(output_file, "w") as f:
            f.write("\n".join(gcode_lines))

        print(f"[INFO] G-code genererad med direkt konverterare: {output_file}")
        return True

    except Exception as e:
        print(f"[FEL] Direkt konvertering misslyckades: {e}")
        return False


def parse_svg_path_simple(d):
    """Simple SVG path parser"""
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
    """Simple SVG points parser"""
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


def apply_origin_offset_to_gcode(gcode_file, origin):
    """Apply origin offset to existing G-code file"""
    try:
        with open(gcode_file, "r") as f:
            lines = f.readlines()

        with open(gcode_file, "w") as f:
            # Add origin command at the beginning after any header comments
            header_written = False
            for line in lines:
                if not header_written and (line.startswith("G") or line.startswith("M")):
                    # Insert origin command before first G/M command
                    #   1. vpype generates clean G-code without trying to misuse the translate command
                    #   2. Origin offset is applied properly by adding a G92 command to the generated G-code,
                    #   which is the correct G-code way to set coordinate system offsets
                    #   3. Matches bCNC behavior - G92 sets the current position as the specified coordinates,
                    #   effectively creating an origin offset

                    #   This is equivalent to what bCNC's origin command does - it tells the machine "consider
                    #   your current position to be X-40 Y-40 Z0" rather than trying to move the geometry around.
                    f.write(f"G92 X{origin[0]} Y{origin[1]} Z{origin[2]} ; Set origin offset\n")
                    header_written = True
                f.write(line)

        print(f"[INFO] Origin offset tillagt: X{origin[0]} Y{origin[1]} Z{origin[2]}")

    except Exception as e:
        print(f"[FEL] Kunde inte lägga till origin offset: {e}")


def convert_with_svg2gcode(svg_file, output_file, origin=(0, 0, 0)):
    """Convert SVG to G-code using svg2gcode tool"""
    try:
        cmd = [
            "svg2gcode",
            "--input",
            svg_file,
            "--output",
            output_file,
            "--origin",
            f"{origin[0]},{origin[1]}",
            "--pen-up",
            "M3 S30",
            "--pen-down",
            "M3 S50",
        ]

        print(f"[INFO] Kör svg2gcode: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"[INFO] svg2gcode lyckades: {output_file}")
        print(result)
        return True

    except subprocess.CalledProcessError as e:
        print(f"[FEL] svg2gcode misslyckades: {e.stderr}")
        return False
    except FileNotFoundError:
        print("[FEL] svg2gcode inte installerat")
        return False


def convert_with_inkscape(svg_file, output_file, origin=(0, 0, 0)):
    """Convert SVG to G-code using Inkscape with gcodetools extension"""
    try:
        # First, try to convert to intermediate format
        temp_file = output_file + ".temp.svg"

        # Inkscape can export paths, but we need gcodetools extension for G-code
        cmd = ["inkscape", "--without-gui", "--file", svg_file, "--export-plain-svg", temp_file]

        print(f"[INFO] Preprocessing med Inkscape...")
        subprocess.run(cmd, check=True)

        # Note: This requires gcodetools extension to be installed
        print("[INFO] Inkscape preprocessing klar, men G-code konvertering kräver gcodetools")
        return False

    except subprocess.CalledProcessError as e:
        print(f"[FEL] Inkscape misslyckades: {e}")
        return False
    except FileNotFoundError:
        print("[FEL] Inkscape inte installerat")
        return False


def install_recommendations():
    """Print installation recommendations for SVG to G-code tools"""
    print("\n[INFO] Rekommenderade verktyg för SVG till G-code:")
    print("\n1. vpype med vpype-gcode (Bästa alternativet):")
    print("   pip install vpype[all] vpype-gcode")
    print("   Eller: sudo apt install python3-vpype && pip install vpype-gcode")

    print("\n2. svg2gcode:")
    print("   npm install -g svg2gcode")

    print("\n3. Inkscape med gcodetools:")
    print("   sudo apt install inkscape")
    print("   Installera gcodetools extension manuellt")

    print("\n4. F-Engrave:")
    print("   Ladda ner från: https://www.scorchworks.com/Fengrave/fengrave.html")


def main():
    """Main conversion function"""
    print("[INFO] Standalone SVG till G-code konverterare")

    # Check what tools are available
    available_tools = check_external_tools()

    if not available_tools:
        print("[FEL] Inga SVG till G-code verktyg hittades!")
        install_recommendations()
        return False

    # Try conversion with available tools in order of preference
    converters = [("vpype", convert_with_vpype), ("svg2gcode", convert_with_svg2gcode), ("inkscape", convert_with_inkscape)]

    for tool_name, converter_func in converters:
        if tool_name in available_tools:
            print(f"\n[INFO] Försöker konvertering med {tool_name}...")
            if converter_func(svg_input, output_gcode, origin_offset):
                print(f"[INFO] Konvertering lyckades med {tool_name}!")

                # Apply servo conversion to handle any Z-axis commands
                temp_file = output_gcode + ".temp"
                if convert_z_to_servo(output_gcode, temp_file):
                    import os

                    os.replace(temp_file, output_gcode)
                    print("[INFO] Servo-konvertering klar!")

                print(f"[INFO] G-code sparad: {output_gcode}")
                return True

    print("[FEL] Alla konverteringsmetoder misslyckades")
    install_recommendations()
    return False


if __name__ == "__main__":
    success = main()
    if success:
        print("\n[INFO] Konvertering klar!")

        # Check if bCNC is available and try to run
        bcnc_cmd = check_bcnc_available()
        if bcnc_cmd:
            print(f"[INFO] Försöker köra med bCNC...")
            if not try_bcnc_cli_run(output_gcode):
                print(f"[INFO] Automatisk körning misslyckades. Kör manuellt:")
                print(f"       {bcnc_cmd} --run {output_gcode}")
        else:
            print(f"[INFO] Installera bCNC för att köra G-code automatiskt")
    else:
        print("\n[FEL] Konvertering misslyckades")
