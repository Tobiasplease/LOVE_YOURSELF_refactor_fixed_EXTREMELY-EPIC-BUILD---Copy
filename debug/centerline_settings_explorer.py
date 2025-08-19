#!/usr/bin/env python3
"""
Debug script for exploring centerline conversion settings
Processes a folder of PNGs with different parameter combinations and
clearly labels output files with the settings used.

Parameters tested:
- blur_kernel: (1,1), (2,2), (3,3)
- threshold_value: 120, 140, 160, 180, 200
- do_dilate: True, False

Output format: {basename}_blur{N}_thresh{N}_dilate{T/F}.svg
"""

import os
import sys
import glob
import argparse

# from pathlib import Path
from itertools import product

# Add project root to path
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_root)

from bcnc import raster_to_centerline_svg


def explore_settings(input_folder, output_folder=None):
    """Explore different centerline settings on all PNGs in folder"""

    if not os.path.exists(input_folder):
        print(f"❌ Input folder not found: {input_folder}")
        return

    # Use input folder + '_centerline_exploration' as output if not specified
    if output_folder is None:
        output_folder = input_folder + "_centerline_exploration"

    os.makedirs(output_folder, exist_ok=True)

    print(f"📁 Input folder: {input_folder}")
    print(f"📁 Output folder: {output_folder}")
    print("-" * 60)

    # Find all PNGs
    png_files = glob.glob(os.path.join(input_folder, "*.png"))
    png_files.extend(glob.glob(os.path.join(input_folder, "*.PNG")))

    if not png_files:
        print("❌ No PNG files found in input folder")
        return

    print(f"Found {len(png_files)} PNG files")

    # Define parameter combinations to test
    blur_kernels = [(1, 1), (2, 2), (3, 3)]
    thresholds = [120, 140, 160, 180, 200]
    dilate_options = [True, False]

    total_combinations = len(blur_kernels) * len(thresholds) * len(dilate_options)
    print(f"Testing {total_combinations} parameter combinations per PNG")
    print(f"Total conversions: {len(png_files) * total_combinations}")
    print("-" * 60)

    successful_conversions = 0
    failed_conversions = 0

    # Process each PNG with all parameter combinations
    for png_idx, png_file in enumerate(png_files, 1):
        base_name = os.path.splitext(os.path.basename(png_file))[0]
        print(f"\n🖼️ [{png_idx}/{len(png_files)}] Processing: {base_name}")

        combination_count = 0

        # Test all parameter combinations
        for blur, threshold, dilate in product(blur_kernels, thresholds, dilate_options):
            combination_count += 1

            # Create descriptive filename with settings
            blur_str = f"blur{blur[0]}"
            thresh_str = f"thresh{threshold}"
            dilate_str = f"dilate{'T' if dilate else 'F'}"

            output_filename = f"{base_name}_{blur_str}_{thresh_str}_{dilate_str}.svg"
            output_path = os.path.join(output_folder, output_filename)

            print(f"  [{combination_count:2d}/{total_combinations}] {blur_str}_{thresh_str}_{dilate_str}... ", end="")

            try:
                raster_to_centerline_svg(
                    input_path=png_file,
                    output_path=output_path,
                    threshold_value=threshold,
                    blur_kernel=blur,
                    do_dilate=dilate,
                    dilation_iterations=1,  # Keep this constant
                    scale=1.0,  # Keep this constant
                )
                print("✅")
                successful_conversions += 1

            except Exception as e:
                print(f"❌ Error: {str(e)[:50]}...")
                failed_conversions += 1

    # Create summary report
    summary_file = os.path.join(output_folder, "SETTINGS_SUMMARY.txt")
    with open(summary_file, "w") as f:
        f.write("CENTERLINE SETTINGS EXPLORATION SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Input folder: {input_folder}\n")
        f.write(f"Output folder: {output_folder}\n")
        f.write(f"PNG files processed: {len(png_files)}\n")
        f.write(f"Parameter combinations tested: {total_combinations}\n")
        f.write(f"Successful conversions: {successful_conversions}\n")
        f.write(f"Failed conversions: {failed_conversions}\n\n")

        f.write("PARAMETERS TESTED:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Blur kernels: {blur_kernels}\n")
        f.write(f"Thresholds: {thresholds}\n")
        f.write(f"Dilate options: {dilate_options}\n\n")

        f.write("FILENAME FORMAT:\n")
        f.write("-" * 20 + "\n")
        f.write("{basename}_blur{N}_thresh{N}_dilate{T/F}.svg\n\n")

        f.write("EXAMPLES:\n")
        f.write("-" * 20 + "\n")
        f.write("drawing_blur1_thresh160_dilateT.svg\n")
        f.write("drawing_blur3_thresh200_dilateF.svg\n\n")

        f.write("FIXED PARAMETERS:\n")
        f.write("-" * 20 + "\n")
        f.write("dilation_iterations: 1\n")
        f.write("scale: 1.0\n")

    # Print final summary
    print("\n" + "=" * 60)
    print("📊 EXPLORATION COMPLETE")
    print("=" * 60)
    print(f"PNGs processed: {len(png_files)}")
    print(f"Total combinations tested: {total_combinations} per PNG")
    print(f"Successful conversions: {successful_conversions}")
    print(f"Failed conversions: {failed_conversions}")
    print(f"Output folder: {output_folder}")
    print(f"Summary saved: {summary_file}")
    print("\n💡 TIP: Compare SVG outputs to find optimal settings for your images!")


def create_comparison_grid(output_folder, png_name=None):
    """Create a quick visual comparison guide"""
    if png_name is None:
        print("Specify a PNG basename with --compare to create comparison grid")
        return

    print(f"\n🎨 Creating comparison guide for: {png_name}")

    # Find all variants for this PNG
    pattern = f"{png_name}_blur*_thresh*_dilate*.svg"
    svg_files = glob.glob(os.path.join(output_folder, pattern))

    if not svg_files:
        print(f"❌ No SVG variants found for {png_name}")
        return

    print(f"Found {len(svg_files)} variants")

    # Group by settings for easy comparison
    comparison_file = os.path.join(output_folder, f"COMPARISON_{png_name}.txt")
    with open(comparison_file, "w") as f:
        f.write(f"COMPARISON GUIDE FOR: {png_name}\n")
        f.write("=" * 50 + "\n\n")

        for svg_file in sorted(svg_files):
            basename = os.path.basename(svg_file)
            # Extract settings from filename
            parts = basename.replace(f"{png_name}_", "").replace(".svg", "").split("_")
            f.write(f"📄 {basename}\n")
            f.write(f"   Settings: {' | '.join(parts)}\n\n")

    print(f"✅ Comparison guide saved: {comparison_file}")


def main():
    parser = argparse.ArgumentParser(description="Explore centerline conversion settings")
    parser.add_argument("input_folder", help="Folder containing PNG files")
    parser.add_argument("-o", "--output", help="Output folder (default: input_folder + '_centerline_exploration')")
    parser.add_argument("--compare", help="Create comparison guide for specific PNG basename")

    args = parser.parse_args()

    if args.compare:
        output_folder = args.output or (args.input_folder + "_centerline_exploration")
        create_comparison_grid(output_folder, args.compare)
    else:
        explore_settings(args.input_folder, args.output)


if __name__ == "__main__":
    main()
