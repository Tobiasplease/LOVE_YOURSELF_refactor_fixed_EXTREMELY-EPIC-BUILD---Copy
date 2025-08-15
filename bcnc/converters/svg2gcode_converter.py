"""
SVG2GCode converter - Node.js based SVG to G-code conversion
Uses svg2gcode npm package for conversion
"""

import subprocess
import shutil
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from bcnc_utils import convert_z_to_servo


class Svg2GcodeConverter:
    """Converter using svg2gcode npm package"""
    
    @staticmethod
    def is_available():
        """Check if svg2gcode is available"""
        return shutil.which("svg2gcode") is not None
    
    def convert(self, svg_file, output_file, origin=(0, 0, 0)):
        """Convert SVG to G-code using svg2gcode"""
        try:
            cmd = [
                "svg2gcode",
                "--input", svg_file,
                "--output", output_file,
                "--origin", f"{origin[0]},{origin[1]}",
                "--pen-up", "M3 S30",
                "--pen-down", "M3 S50"
            ]
            
            print(f"[INFO] Kör svg2gcode: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"[INFO] svg2gcode konvertering lyckades")
            
            # Apply Z-to-servo conversion to handle any Z-axis commands
            temp_file = output_file + ".temp"
            if convert_z_to_servo(output_file, temp_file):
                import os
                os.replace(temp_file, output_file)
                print("[INFO] Z-to-servo konvertering klar")
            
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"[FEL] svg2gcode misslyckades: {e.stderr}")
            return False
        except Exception as e:
            print(f"[FEL] svg2gcode konvertering misslyckades: {e}")
            return False