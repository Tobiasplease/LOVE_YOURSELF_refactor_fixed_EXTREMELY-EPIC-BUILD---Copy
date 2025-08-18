#!/usr/bin/env python3
"""
SVG to GRBL Script
Converts SVG files to G-code, applies servo control, and executes on GRBL
"""

import os
import sys
import argparse
import tempfile
from pathlib import Path

# Add paths for imports
bcnc_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "bcnc")
grbl_path = os.path.dirname(__file__)
sys.path.insert(0, bcnc_path)
sys.path.insert(0, grbl_path)

from grbl_utils import find_grbl_port, initialize_grbl_for_drawing, execute_gcode_file
import bcnc_utils
import svg_to_gcode


def main():
    parser = argparse.ArgumentParser(description="Convert SVG to G-code and execute on GRBL")
    parser.add_argument("svg_file", help="Input SVG file")
    parser.add_argument("-o", "--output", help="Output G-code file (optional)")
    parser.add_argument("-x", "--offset-x", type=float, default=0.0, help="X offset (default: 0.0)")
    parser.add_argument("-y", "--offset-y", type=float, default=0.0, help="Y offset (default: 0.0)")
    parser.add_argument("--origin-x", type=float, default=66.0, help="Work origin X (default: 66.0)")
    parser.add_argument("--origin-y", type=float, default=-2.0, help="Work origin Y (default: -2.0)")
    parser.add_argument("--feed-rate", type=int, default=3000, help="Feed rate (default: 3000)")
    parser.add_argument("--no-execute", action="store_true", help="Generate G-code only, don't execute")
    parser.add_argument("--temp-dir", help="Directory for temporary files (default: system temp)")

    args = parser.parse_args()

    # Validate input file
    if not os.path.exists(args.svg_file):
        print(f"[ERROR] SVG file not found: {args.svg_file}")
        sys.exit(1)

    svg_path = Path(args.svg_file)

    # Determine output file
    if args.output:
        output_file = args.output
    else:
        temp_dir = args.temp_dir or tempfile.gettempdir()
        output_file = os.path.join(temp_dir, f"{svg_path.stem}_servo.gcode")

    try:
        # Step 1: Convert SVG to G-code using bcnc converter
        print("[STEP 1] Converting SVG to G-code...")
        temp_gcode = output_file.replace(".gcode", "_temp.gcode")

        # Use bcnc's convert_svg_to_gcode with origin offset
        origin_offset = (args.offset_x, args.offset_y, 0)
        converted_gcode = svg_to_gcode.convert_svg_to_gcode(args.svg_file, output_gcode=temp_gcode, origin_offset=origin_offset)

        if not converted_gcode or not os.path.exists(temp_gcode):
            print("[ERROR] SVG to G-code conversion failed")
            sys.exit(1)

        # Step 3: Apply servo conversion
        print("[STEP 2] Applying servo control conversion...")
        if not bcnc_utils.convert_z_to_servo(temp_gcode, output_file):
            print("[ERROR] Servo conversion failed")
            sys.exit(1)

        # Clean up temp file
        if os.path.exists(temp_gcode):
            os.remove(temp_gcode)

        print(f"[SUCCESS] G-code generated: {output_file}")

        # Step 4: Execute on GRBL (if requested)
        if not args.no_execute:
            print("[STEP 3] Executing on GRBL...")

            try:
                # Connect to GRBL
                ser = find_grbl_port()

                # Initialize GRBL
                initialize_grbl_for_drawing(
                    ser, 
                    origin_x=args.origin_x, 
                    origin_y=args.origin_y, 
                    feed_rate=args.feed_rate
                )

                # Execute G-code
                execute_gcode_file(ser, output_file)

                print("[SUCCESS] Drawing complete!")

            except Exception as e:
                print(f"[ERROR] GRBL execution failed: {e}")
                print(f"[INFO] G-code file saved at: {output_file}")
                print("[INFO] You can manually load and run this file in bCNC")
                sys.exit(1)

            finally:
                if "ser" in locals():
                    ser.close()
        else:
            print(f"[INFO] G-code generation complete. File saved: {output_file}")
            print("[INFO] Use --no-execute flag was used. To execute, run:")
            print(f"[INFO]   python svg_to_grbl.py {args.svg_file} --output {output_file}")

    except Exception as e:
        print(f"[ERROR] Failed to process SVG: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
