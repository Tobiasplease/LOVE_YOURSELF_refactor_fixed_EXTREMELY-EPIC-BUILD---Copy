"""
Inkscape converter - Uses Inkscape with gcodetools extension
Requires manual setup of gcodetools extension
"""

import os
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from bcnc_utils import get_servo_gcode_footer, get_servo_gcode_header, parse_svg_path_simple, parse_svg_points_simple


class InkscapeConverter:
    """Converter using Inkscape preprocessing with custom G-code generation"""

    @staticmethod
    def is_available():
        """Check if Inkscape is available"""
        return shutil.which("inkscape") is not None

    def convert(self, svg_file, output_file, origin=(0, 0, 0)):
        """Convert SVG to G-code using Inkscape preprocessing"""
        try:
            # Use Inkscape to preprocess the SVG (simplify paths, etc.)
            temp_svg = output_file + ".temp.svg"

            cmd = ["inkscape", "--without-gui", "--file", svg_file, "--export-plain-svg", temp_svg]

            print(f"[INFO] Förbehandlar med Inkscape...")
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"[INFO] Inkscape förbehandling lyckades")

            # Convert the preprocessed SVG to G-code
            if self._svg_to_gcode(temp_svg, output_file, origin):
                # Clean up temp file
                import os

                try:
                    os.remove(temp_svg)
                except:
                    pass
                return True
            else:
                return False

        except subprocess.CalledProcessError as e:
            print(f"[FEL] Inkscape misslyckades: {e.stderr}")
            return False
        except Exception as e:
            print(f"[FEL] Inkscape konvertering misslyckades: {e}")
            return False

    def _svg_to_gcode(self, svg_file, output_file, origin):
        """Convert preprocessed SVG to G-code"""
        try:
            tree = ET.parse(svg_file)
            root = tree.getroot()

            gcode_lines = get_servo_gcode_header()
            gcode_lines.append(f"G92 X{origin[0]} Y{origin[1]} Z{origin[2]} ; Set origin offset")

            # Process SVG elements
            for elem in root.iter():
                if elem.tag.endswith("path"):
                    d = elem.get("d", "")
                    gcode_lines.extend(parse_svg_path_simple(d))
                elif elem.tag.endswith("polyline"):
                    points = elem.get("points", "")
                    gcode_lines.extend(parse_svg_points_simple(points))
                elif elem.tag.endswith("line"):
                    x1, y1 = float(elem.get("x1", 0)), float(elem.get("y1", 0))
                    x2, y2 = float(elem.get("x2", 0)), float(elem.get("y2", 0))
                    gcode_lines.extend([f"G0 X{x1} Y{y1} ; Move to start", "M3 S50 ; PEN DOWN", f"G1 X{x2} Y{y2} ; Draw line", "M3 S30 ; PEN UP"])

            gcode_lines.extend(get_servo_gcode_footer())

            with open(output_file, "w") as f:
                f.write("\n".join(gcode_lines))

            print(f"[INFO] Inkscape G-code genererad: {output_file}")
            return True

        except Exception as e:
            print(f"[FEL] Inkscape SVG parsing misslyckades: {e}")
            return False
