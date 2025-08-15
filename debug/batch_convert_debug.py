#!/usr/bin/env python3
"""
Debug script for batch processing folders of PNGs and SVGs
- Converts PNGs to centerline SVGs
- Converts all SVGs (existing + new centerlines) to G-code
- Does NOT auto-run G-code (for debugging)
"""

import os
import sys
import glob
import argparse
from pathlib import Path

# Add project root to path
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_root)

from bcnc import raster_to_centerline_svg, svg_to_gcode


def process_folder(input_folder, output_folder=None, auto_run=False):
    """Process a folder of PNGs and SVGs"""
    
    if not os.path.exists(input_folder):
        print(f"❌ Input folder not found: {input_folder}")
        return
    
    # Use input folder as output folder if not specified
    if output_folder is None:
        output_folder = input_folder
    else:
        os.makedirs(output_folder, exist_ok=True)
    
    print(f"📁 Processing folder: {input_folder}")
    print(f"📁 Output folder: {output_folder}")
    print(f"🔧 Auto-run G-code: {auto_run}")
    print("-" * 50)
    
    # Find all PNGs and SVGs
    png_files = glob.glob(os.path.join(input_folder, "*.png"))
    png_files.extend(glob.glob(os.path.join(input_folder, "*.PNG")))
    svg_files = glob.glob(os.path.join(input_folder, "*.svg"))
    svg_files.extend(glob.glob(os.path.join(input_folder, "*.SVG")))
    
    print(f"Found {len(png_files)} PNG files and {len(svg_files)} SVG files")
    
    # Step 1: Convert PNGs to centerline SVGs
    centerline_svgs = []
    for png_file in png_files:
        print(f"\n🖼️ Processing PNG: {os.path.basename(png_file)}")
        
        base_name = os.path.splitext(os.path.basename(png_file))[0]
        centerline_svg = os.path.join(output_folder, f"{base_name}_centerlined.svg")
        
        try:
            raster_to_centerline_svg(
                input_path=png_file,
                output_path=centerline_svg,
                threshold_value=180,
                blur_kernel=(1, 1),
                do_dilate=True,
                dilation_iterations=1,
                scale=1.0,
            )
            centerline_svgs.append(centerline_svg)
            print(f"✅ Centerline SVG created: {os.path.basename(centerline_svg)}")
            
        except Exception as e:
            print(f"❌ Failed to convert PNG {os.path.basename(png_file)}: {e}")
    
    # Step 2: Convert all SVGs (existing + new centerlines) to G-code
    all_svgs = svg_files + centerline_svgs
    
    print(f"\n🔄 Converting {len(all_svgs)} SVG files to G-code...")
    
    successful_conversions = 0
    for svg_file in all_svgs:
        print(f"\n📐 Processing SVG: {os.path.basename(svg_file)}")
        
        base_name = os.path.splitext(os.path.basename(svg_file))[0]
        gcode_file = os.path.join(output_folder, f"{base_name}.gcode")
        
        try:
            result = svg_to_gcode(
                svg_input=svg_file,
                output_gcode=gcode_file,
                auto_run=auto_run
            )
            
            if result:
                successful_conversions += 1
                print(f"✅ G-code created: {os.path.basename(gcode_file)}")
                if auto_run:
                    print("🚀 G-code sent to bCNC")
            else:
                print(f"❌ Failed to convert SVG: {os.path.basename(svg_file)}")
                
        except Exception as e:
            print(f"❌ Failed to convert SVG {os.path.basename(svg_file)}: {e}")
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 PROCESSING SUMMARY")
    print("=" * 50)
    print(f"PNGs processed: {len(png_files)}")
    print(f"Centerline SVGs created: {len(centerline_svgs)}")
    print(f"Total SVGs to convert: {len(all_svgs)}")
    print(f"Successful G-code conversions: {successful_conversions}")
    print(f"Output folder: {output_folder}")


def main():
    parser = argparse.ArgumentParser(description="Batch convert PNGs and SVGs to G-code")
    parser.add_argument("input_folder", help="Folder containing PNG and SVG files")
    parser.add_argument("-o", "--output", help="Output folder (default: same as input)")
    parser.add_argument("--auto-run", action="store_true", help="Auto-run G-code in bCNC")
    parser.add_argument("--no-centerline", action="store_true", help="Skip PNG to centerline conversion")
    
    args = parser.parse_args()
    
    if args.no_centerline:
        print("⚠️ Skipping PNG to centerline conversion")
        # Just convert existing SVGs
        svg_files = glob.glob(os.path.join(args.input_folder, "*.svg"))
        svg_files.extend(glob.glob(os.path.join(args.input_folder, "*.SVG")))
        
        output_folder = args.output or args.input_folder
        os.makedirs(output_folder, exist_ok=True)
        
        print(f"Converting {len(svg_files)} SVG files to G-code...")
        for svg_file in svg_files:
            base_name = os.path.splitext(os.path.basename(svg_file))[0]
            gcode_file = os.path.join(output_folder, f"{base_name}.gcode")
            
            try:
                result = svg_to_gcode(
                    svg_input=svg_file,
                    output_gcode=gcode_file,
                    auto_run=args.auto_run
                )
                print(f"✅ {os.path.basename(svg_file)} → {os.path.basename(gcode_file)}")
            except Exception as e:
                print(f"❌ Failed: {os.path.basename(svg_file)} - {e}")
    else:
        process_folder(args.input_folder, args.output, args.auto_run)


if __name__ == "__main__":
    main()