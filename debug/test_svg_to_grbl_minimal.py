#!/usr/bin/env python3
"""
Quick test of SVG to GRBL execution
"""
import sys
import os

# Add the project root to the path
sys.path.append('/home/impostor/LOVE_YOURSELF_refactor_fixed_EXTREMELY-EPIC-BUILD---Copy')

from grbl import svg_to_grbl

def test_svg_to_grbl():
    """Test if SVG to GRBL execution is working."""

    # Use the latest generated SVG
    svg_path = "/home/impostor/ComfyUI/output/impostor-20250920_033503_00001__center_lined.svg"

    if not os.path.exists(svg_path):
        print(f"❌ SVG file not found: {svg_path}")
        return False

    print(f"🧪 Testing SVG to GRBL execution...")
    print(f"📁 SVG file: {svg_path}")

    try:
        result = svg_to_grbl(
            svg_input=svg_path,
            execute_grbl=True,
            scale_to="50x50mm"
        )

        if result:
            print(f"✅ SVG to GRBL execution successful: {result}")
            return True
        else:
            print(f"❌ SVG to GRBL execution failed")
            return False

    except Exception as e:
        print(f"❌ SVG to GRBL execution error: {e}")
        return False

if __name__ == "__main__":
    test_svg_to_grbl()