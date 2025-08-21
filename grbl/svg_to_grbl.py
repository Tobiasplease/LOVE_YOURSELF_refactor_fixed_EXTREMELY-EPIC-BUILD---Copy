#!/usr/bin/env python3
"""
SVG to GRBL Script
Converts SVG files to G-code, applies servo control, and executes on GRBL
"""

import os
import sys
import argparse
from pathlib import Path

# Add paths for imports
# bcnc_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "bcnc")
grbl_path = os.path.dirname(__file__)
# sys.path.insert(0, bcnc_path)
sys.path.insert(0, grbl_path)

from grbl_utils import find_grbl_port, initialize_grbl_for_drawing, execute_gcode_file, convert_with_vpype, convert_gcode_to_servo_format


def main():
    parser = argparse.ArgumentParser(description="Convert SVG to G-code and execute on GRBL")
    parser.add_argument("svg_file", help="Input SVG file")
    parser.add_argument("-o", "--output", help="Output G-code file (optional)")
    parser.add_argument("-x", "--offset-x", type=float, default=0.0, help="X offset (default: 0.0)")
    parser.add_argument("-y", "--offset-y", type=float, default=0.0, help="Y offset (default: 0.0)")
    parser.add_argument("--origin-x", type=float, default=0, help="Work origin X (default: 0)")
    parser.add_argument("--origin-y", type=float, default=0, help="Work origin Y (default: 0)")
    parser.add_argument("--feed-rate", type=int, default=5000, help="Feed rate (default: 5000)")
    parser.add_argument("--scale-to", help="Scale to fit size (e.g., '50x50mm', '100x100mm')")
    parser.add_argument("--no-execute", action="store_true", help="Generate G-code only, don't execute")
    parser.add_argument("--use-absolute", action="store_true", help="Use absolute positioning")

    args = parser.parse_args()

    # Validate input file
    if not os.path.exists(args.svg_file):
        print(f"[ERROR] SVG file not found: {args.svg_file}")
        sys.exit(1)

    svg_path = Path(args.svg_file)

    if args.output:
        output_file_vpype = args.output
        output_file_adjusted = f"{output_file_vpype}_servo_adjusted.gcode"
    else:
        output_file_vpype = str(svg_path.parent / f"{svg_path.stem}_raw_vpype.gcode")
        output_file_adjusted = str(svg_path.parent / f"{svg_path.stem}_servo_adjusted.gcode")

    try:
        convert_with_vpype(args.svg_file, output_file_vpype, scale_to=args.scale_to)
        print(f"[SUCCESS] V-PYPE G-code generated: {output_file_vpype}")
        convert_gcode_to_servo_format(output_file_vpype, output_file_adjusted)
        print(f"[SUCCESS] Servo G-code generated: {output_file_adjusted}")
        os.remove(output_file_vpype)  # Clean up raw G-code file

        # Execute on GRBL (if requested)
        if not args.no_execute:
            print("[INFO] Executing on GRBL...")
            origin = (args.origin_x, args.origin_y, 0)
            origin_offset = (args.offset_x, args.offset_y, 0)
            try:
                ser = find_grbl_port()
                initialize_grbl_for_drawing(
                    ser, origin=origin, origin_offset=origin_offset, feed_rate=args.feed_rate, use_absolute_positioning=args.use_absolute
                )

                # Execute G-code, skips first three lines!
                execute_gcode_file(ser, output_file_adjusted)

                print("[SUCCESS] Drawing complete!")

            except Exception as e:
                print(f"[ERROR] GRBL execution failed: {e}")
                print(f"[INFO] G-code file saved at: {output_file_adjusted}")
                print("[INFO] You can manually load and run this file in bCNC")
                sys.exit(1)

            finally:
                if "ser" in locals():
                    ser.close()
        else:
            print(f"[INFO] G-code generation complete. File saved: {output_file_adjusted}")
            print("[INFO] Use --no-execute flag was used.")

    except Exception as e:
        print(f"[ERROR] Failed to process SVG: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
