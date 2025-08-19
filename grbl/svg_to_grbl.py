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
bcnc_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "bcnc")
grbl_path = os.path.dirname(__file__)
sys.path.insert(0, bcnc_path)
sys.path.insert(0, grbl_path)

from grbl_utils import find_grbl_port, initialize_grbl_for_drawing, execute_gcode_file, convert_with_vpype, convert_gcode_to_servo_format

#   G20 is inches
#   G21 (millimeters) is standard for most plotters since:
#   - SVG coordinates are typically in mm or pixels
#   - Easier to work with small precise movements


#   G17 (XY plane) is essential because:
#   - We're doing 2D plotting on the XY plane
#   - Required for proper arc/curve interpretation

#   G90 (absolute positioning) is preferred because:
#   - SVG coordinates are absolute positions
#   - Easier to debug and predict movements
#   - Less accumulation of positioning errors


#   G00 X5 Y5        ; Move rapidly to start position (pen up)
#   M3 S50           ; Lower pen
#   G01 X15 Y5       ; Draw line to X=15 (pen down, controlled speed)
#   G01 X15 Y15      ; Draw line to Y=15
#   G01 X5 Y15       ; Draw line back to X=5
#   G01 X5 Y5        ; Draw line back to start (complete rectangle)
#   M3 S30           ; Raise pen
#   G00 X0 Y0        ; Move rapidly back to origin
#   M2               ; End program

#   Use M2 when:
#   - Single drawing job
#   - Want machine to stay in place after drawing
#   - Manual control after completion

#   Use M30 when:
#   - Production/batch jobs (run same drawing multiple times)
#   - Want machine to return to safe home position
#   - Automated workflows


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
    parser.add_argument("--temp-dir", help="Directory for temporary files (default: system temp)")

    args = parser.parse_args()

    # Validate input file
    if not os.path.exists(args.svg_file):
        print(f"[ERROR] SVG file not found: {args.svg_file}")
        sys.exit(1)

    svg_path = Path(args.svg_file)

    # Determine output file
    # if args.output:
    #     output_file = args.output
    # else:
    #     if args.temp_dir:
    #         # Use temp dir if explicitly specified
    #         output_file = os.path.join(args.temp_dir, f"{svg_path.stem}_servo.gcode")
    #     else:
    #         # Default: write beside input file
    output_file_vpype = str(svg_path.parent / f"{svg_path.stem}_raw_vpype.gcode")
    output_file_adjusted = str(svg_path.parent / f"{svg_path.stem}_servo_adjusted.gcode")

    try:
        origin_offset = (args.offset_x, args.offset_y, 0)

        convert_with_vpype(args.svg_file, output_file_vpype, scale_to=args.scale_to)
        print(f"[SUCCESS] V-PYPE G-code generated: {output_file_vpype}")
        convert_gcode_to_servo_format(output_file_vpype, output_file_adjusted)
        print(f"[SUCCESS] Servo G-code generated: {output_file_adjusted}")

        # Step 4: Execute on GRBL (if requested)
        if not args.no_execute:
            print("[INFO] Executing on GRBL...")

            try:
                # Connect to GRBL
                ser = find_grbl_port()

                # Initialize GRBL
                # todo ORIGIN OFFSET?!
                initialize_grbl_for_drawing(
                    ser, origin_x=args.origin_x, origin_y=args.origin_y, feed_rate=args.feed_rate, use_absolute_positioning=args.use_absolute
                )

                # Execute G-code
                # skips first three lines!
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
            # print(f"[INFO]   python svg_to_grbl.py {args.svg_file} --output {output_file}")

    except Exception as e:
        print(f"[ERROR] Failed to process SVG: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
