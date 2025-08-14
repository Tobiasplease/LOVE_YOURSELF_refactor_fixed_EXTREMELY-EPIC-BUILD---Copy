#!/usr/bin/env python3
"""
Standalone SVG to G-code converter using external tools
Bypasses bCNC GUI entirely while still providing servo conversion
"""

# import os
# import re
import subprocess
import shutil
from bcnc_utils import convert_z_to_servo, try_bcnc_cli_run, check_bcnc_available

# from pathlib import Path

# === Configuration ===
base_path = "/home/jbe/Dropbox/_outputs"
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
    """Convert SVG to G-code using vpype (vector graphics processor)"""
    try:
        # vpype generates G-code directly, so we'll add origin offset in post-processing
        cmd = [
            "vpype",
            "read",
            svg_file,
            "linemerge",
            "--tolerance",
            "0.1mm",
            "linesort",
            "write",
            "--format",
            "gcode",
            "--device",
            "custom",
            "--pen-up",
            "M3 S30",
            "--pen-down",
            "M3 S50",
            "--feed-rate",
            "1000",
            output_file,
        ]

        print(f"[INFO] Kör vpype: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"[INFO] vpype lyckades: {output_file}")
        print(result)

        # Apply origin offset by modifying the generated G-code
        if origin[0] != 0 or origin[1] != 0:
            print(f"[INFO] Lägger till origin offset: {origin[:2]}")
            apply_origin_offset_to_gcode(output_file, origin)

        return True

    except subprocess.CalledProcessError as e:
        print(f"[FEL] vpype misslyckades: {e.stderr}")
        return False
    except FileNotFoundError:
        print("[FEL] vpype inte installerat")
        return False


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
    print("\n1. vpype (Bästa alternativet):")
    print("   pip install vpype[all]")
    print("   Eller: sudo apt install python3-vpype")

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
                print("[INFO] Konverterar Z-kommandon till servo...")
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
