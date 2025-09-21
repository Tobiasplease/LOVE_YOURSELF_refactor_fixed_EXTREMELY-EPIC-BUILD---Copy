#!/usr/bin/env python3
"""
Test SVG parsing for the transform UI
"""
import sys
import os
from pathlib import Path

# Add the project root to the path
sys.path.append('/home/impostor/LOVE_YOURSELF_refactor_fixed_EXTREMELY-EPIC-BUILD---Copy')

from tools.svg_transform_ui import SVGTransformUI
import tkinter as tk

def test_svg_parsing():
    """Test if SVG parsing is working correctly"""

    # Create a minimal tkinter root (required for the UI class)
    root = tk.Tk()
    root.withdraw()  # Hide the window

    # Create the UI object (just for parsing)
    ui = SVGTransformUI(root)

    # Test with a known centerlined SVG file
    svg_path = Path("/home/impostor/ComfyUI/output/impostor-20250920_005332_00001__center_lined.svg")

    if not svg_path.exists():
        print(f"❌ SVG file not found: {svg_path}")
        return False

    print(f"🧪 Testing SVG parsing...")
    print(f"📁 SVG file: {svg_path}")

    # Set the input SVG and parse
    ui.input_svg = svg_path
    ui.parse_svg()

    print(f"✅ Parsing complete!")
    print(f"📊 Total paths parsed: {len(ui.svg_paths)}")

    if ui.svg_paths:
        # Show first few paths for debugging
        for i, path in enumerate(ui.svg_paths[:3]):
            print(f"   Path {i+1}: {len(path)} points")
            if path:
                print(f"      First point: {path[0]}")
                print(f"      Last point: {path[-1]}")

    # Test transform
    if ui.svg_paths:
        print(f"🔄 Testing transformation...")
        ui.scale_x_var.set(1.5)
        ui.translate_x_var.set(5)
        ui.rotation_var.set(3)

        test_path = ui.svg_paths[0]
        transformed = ui.apply_transform(test_path)

        print(f"   Original first point: {test_path[0]}")
        print(f"   Transformed first point: {transformed[0]}")

        return True
    else:
        print("❌ No paths were parsed!")
        return False

if __name__ == "__main__":
    test_svg_parsing()