"""
VPype converter - Professional vector graphics processing with G-code export
Uses vpype for optimization and vpype-gcode plugin for G-code generation
"""

import os
import shutil
import subprocess

# import tempfile
import sys

# import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from bcnc_utils import convert_gcode_to_servo_format


class VpypeConverter:
    """Converter using vpype with vpype-gcode plugin"""

    @staticmethod
    def is_available():
        """Check if vpype and vpype-gcode are available"""
        if not shutil.which("vpype"):
            return False

        # Test if vpype-gcode plugin is installed
        try:
            result = subprocess.run(["vpype", "--help"], capture_output=True, text=True, check=True)
            return "gwrite" in result.stdout
        except subprocess.CalledProcessError:
            return False

    def convert(self, svg_file, output_file, origin=(0, 0, 0)):
        """Convert SVG to G-code using vpype with optimization"""
        try:
            # Import experimental settings
            try:
                from config.config import GRBL_EXPERIMENTAL_SIMPLIFICATION, GRBL_MERGE_TOLERANCE, GRBL_SIMPLIFICATION_TOLERANCE
            except ImportError:
                GRBL_EXPERIMENTAL_SIMPLIFICATION = False
                GRBL_SIMPLIFICATION_TOLERANCE = 0.05
                GRBL_MERGE_TOLERANCE = 0.1

            # Use built-in gcode profile, then post-process for servo commands
            temp_gcode = output_file + ".temp"

            # Build command based on experimental settings
            cmd = ["vpype", "read", svg_file]

            if GRBL_EXPERIMENTAL_SIMPLIFICATION:
                print(f"[EXPERIMENTAL] Using path simplification (tolerance: {GRBL_SIMPLIFICATION_TOLERANCE}mm)")
                cmd.extend(
                    [
                        "linemerge",
                        "--tolerance",
                        f"{GRBL_MERGE_TOLERANCE}mm",
                        "linesimplify",
                        "--tolerance",
                        f"{GRBL_SIMPLIFICATION_TOLERANCE}mm",
                        "linesort",
                    ]
                )
            else:
                print("[STANDARD] Using high-quality mode (no simplification)")
                cmd.extend(["linemerge", "--tolerance", "0.1mm", "linesort"])

            cmd.extend(["gwrite", "--profile", "gcode", temp_gcode])

            print(f"[INFO] Kör vpype optimering: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"[INFO] vpype G-code generering lyckades")
            print(result)

            # Convert to servo format with origin offset
            if convert_gcode_to_servo_format(temp_gcode, output_file, origin):
                # Clean up temp file
                try:
                    os.remove(temp_gcode)
                except Exception:
                    pass
                return True
            else:
                return False

        except subprocess.CalledProcessError as e:
            print(f"[FEL] vpype misslyckades: {e.stderr}")
            return False
        except Exception as e:
            print(f"[FEL] Vpype konvertering misslyckades: {e}")
            return False
