#!/usr/bin/env python3
"""
SVG to G-code converter with modular converter backends
Main entry point for converting SVG files to G-code for servo-controlled plotters
"""

# import os
# import shutil
from pathlib import Path
from bcnc_utils import try_bcnc_cli_run, check_bcnc_available

# Import converter modules
from converters.vpype_converter import VpypeConverter
from converters.svg2gcode_converter import Svg2GcodeConverter
from converters.inkscape_converter import InkscapeConverter
from converters.fengrave_converter import FEngraveConverter


def check_available_converters():
    """Check which converter backends are available"""
    converters = {
        "vpype": VpypeConverter.is_available(),
        "svg2gcode": Svg2GcodeConverter.is_available(),
        "inkscape": InkscapeConverter.is_available(),
        "f-engrave": FEngraveConverter.is_available(),
    }

    available = {name: status for name, status in converters.items() if status}
    print(f"[INFO] Tillgängliga konverterare: {list(available.keys())}")
    return available


def convert_svg_to_gcode(svg_input, output_gcode=None, origin_offset=(-40, -40, 0), preferred_converter=None):
    """
    Convert SVG to G-code using available converters

    Args:
        svg_input: Path to input SVG file
        output_gcode: Path for output G-code file (defaults to svg_input + .ngc)
        origin_offset: Tuple of (x, y, z) origin offset
        preferred_converter: Preferred converter name ('vpype', 'svg2gcode', etc.)

    Returns:
        str: Path to generated G-code file if successful, None if failed
    """
    svg_path = Path(svg_input)
    if not svg_path.exists():
        print(f"[FEL] SVG-fil hittades inte: {svg_input}")
        return None

    # Default output path
    if output_gcode is None:
        output_gcode = str(svg_path) + ".ngc"

    print(f"[INFO] Konverterar {svg_input} till G-code...")

    # Check available converters
    available = check_available_converters()
    if not available:
        print("[FEL] Inga konverterare tillgängliga!")
        print_installation_recommendations()
        return None

    # Define converter priority (or use preferred)
    if preferred_converter and preferred_converter in available:
        converter_order = [preferred_converter]
    else:
        converter_order = ["vpype", "svg2gcode", "inkscape", "f-engrave"]

    # Try converters in order
    for converter_name in converter_order:
        if converter_name not in available:
            continue

        print(f"\n[INFO] Försöker med {converter_name}...")

        try:
            if converter_name == "vpype":
                converter = VpypeConverter()
            elif converter_name == "svg2gcode":
                converter = Svg2GcodeConverter()
            elif converter_name == "inkscape":
                converter = InkscapeConverter()
            elif converter_name == "f-engrave":
                converter = FEngraveConverter()
            else:
                continue

            if converter.convert(svg_input, output_gcode, origin_offset):
                print(f"[INFO] Konvertering lyckades med {converter_name}!")
                print(f"[INFO] G-code sparad: {output_gcode}")
                return output_gcode

        except Exception as e:
            print(f"[FEL] {converter_name} misslyckades: {e}")
            continue

    print("[FEL] Alla konverterare misslyckades")
    return None


def print_installation_recommendations():
    """Print installation recommendations for converters"""
    print("\n[INFO] Rekommenderade verktyg för SVG till G-code:")
    print("\n1. vpype med vpype-gcode (Bästa alternativet):")
    print("   pip install vpype[all] vpype-gcode")

    print("\n2. svg2gcode:")
    print("   npm install -g svg2gcode")

    print("\n3. Inkscape med gcodetools:")
    print("   sudo apt install inkscape")

    print("\n4. F-Engrave:")
    print("   Ladda ner från: https://www.scorchworks.com/Fengrave/fengrave.html")


def main(svg_input=None, output_gcode=None, origin_offset=(-40, -40, 0), auto_run=True):
    """
    Main function for SVG to G-code conversion

    Args:
        svg_input: Path to SVG file to convert
        output_gcode: Output path for G-code
        origin_offset: Origin offset tuple
        auto_run: Whether to automatically run in bCNC

    Returns:
        str: Path to generated G-code file if successful, None if failed
    """
    if svg_input is None:
        # Default for standalone usage
        base_path = "/home/jbe/Dropbox/_outputs"
        svg_input = f"{base_path}/impostor-20250725_185854_00001_.png.svg"

    # Convert SVG to G-code
    gcode_path = convert_svg_to_gcode(svg_input, output_gcode, origin_offset)

    if gcode_path and auto_run:
        # Try to run automatically in bCNC
        bcnc_cmd = check_bcnc_available()
        if bcnc_cmd:
            print(f"[INFO] Försöker köra med bCNC...")
            if not try_bcnc_cli_run(gcode_path):
                print(f"[INFO] Automatisk körning misslyckades. Kör manuellt:")
                print(f"       {bcnc_cmd} --run {gcode_path}")
        else:
            print(f"[INFO] Installera bCNC för att köra G-code automatiskt")

    return gcode_path


if __name__ == "__main__":
    result = main()
    if result:
        print(f"\n[INFO] Konvertering klar: {result}")
    else:
        print("\n[FEL] Konvertering misslyckades")
