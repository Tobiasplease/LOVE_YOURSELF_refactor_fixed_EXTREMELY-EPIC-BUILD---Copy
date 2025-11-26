#!/usr/bin/env python3
"""
SVG to GRBL Script
Converts SVG files to G-code, applies servo control, and executes on GRBL
"""

import argparse
import os
import sys
from pathlib import Path

# Add paths for imports
# bcnc_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "bcnc")
grbl_path = os.path.dirname(__file__)
# sys.path.insert(0, bcnc_path)
sys.path.insert(0, grbl_path)

from grbl_utils import process_svg_to_grbl


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

    if not os.path.exists(args.svg_file):
        print(f"[ERROR] SVG file not found: {args.svg_file}")
        sys.exit(1)

    svg_path = Path(args.svg_file)

    if args.output:
        output_file_adjusted = args.output
    else:
        output_file_adjusted = str(svg_path.parent / f"{svg_path.stem}_servo_adjusted.gcode")

    try:
        result = process_svg_to_grbl(
            svg_input=args.svg_file,
            output_gcode=output_file_adjusted,
            execute_grbl=not args.no_execute,
            scale_to=args.scale_to,
            origin=(args.origin_x, args.origin_y, 0),
            origin_offset=(args.offset_x, args.offset_y, 0),
            feed_rate=args.feed_rate,
            use_absolute_positioning=args.use_absolute,
        )

        if result:
            if args.no_execute:
                print(f"[INFO] G-code generation complete. File saved: {result}")
                print("[INFO] Use --no-execute flag was used.")
            else:
                print(f"[SUCCESS] Processing complete. File saved: {result}")
        else:
            print("[ERROR] Failed to process SVG")
            sys.exit(1)

    except Exception as e:
        print(f"[ERROR] Failed to process SVG: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
