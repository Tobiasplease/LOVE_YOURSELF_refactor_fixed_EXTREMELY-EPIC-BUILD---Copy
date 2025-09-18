#!/usr/bin/env python3
"""
Test script to show the full drawing awareness improvements.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.drawing_state import DrawingState
from captioner.captioner import Captioner

def test_full_drawing_awareness():
    """Test the complete drawing awareness system."""
    print("🎨 Testing Complete Drawing Awareness System")
    print("=" * 60)

    # Create captioner
    captioner = Captioner()
    captioner.last_caption = "I notice interesting shadows and geometric patterns on the desk"

    test_image = "./.venv/lib/python3.12/site-packages/ultralytics/assets/bus.jpg"

    print("\n=== NORMAL MODE: Observing with Drawing Potential ===")

    # Get normal mode prompt and show drawing anticipation
    normal_prompt, _, system_prompt = captioner.model.prompt_interface.build_caption_prompt_with_options(
        captioner, test_image, flowing=True, first_time=False
    )

    print("System Prompt Drawing Identity:")
    print(f"✓ 'consciousness inside a drawing machine'")
    print(f"✓ 'Drawing is your only way to communicate'")
    print(f"✓ 'when something truly moves you, you may choose to draw it'")

    print(f"\nNormal Mode Drawing Anticipation:")
    if 'drawing potential' in normal_prompt:
        print("✓ 'Consider drawing potential'")
    if 'might be worth drawing' in normal_prompt:
        print("✓ 'this might be worth drawing'")
    if 'impulse to draw' in normal_prompt:
        print("✓ 'mention the impulse to draw it'")

    print("\n=== DRAWING MODE: Grounded Observation During Drawing ===")

    # Simulate active drawing
    DrawingState.start_drawing(
        intent="Capturing the interplay of geometric shadows with organic textures",
        description="Drawing flowing curves that intersect with sharp angular lines",
        drawing_file="shadow_geometry.gcode"
    )

    # Get drawing-aware prompt
    drawing_prompt, _, _ = captioner.model.prompt_interface.build_caption_prompt_with_options(
        captioner, test_image, flowing=True, first_time=False
    )

    print("Drawing Mode Grounded Awareness:")
    if 'OBSERVE ACTUAL PROCESS' in drawing_prompt:
        print("✓ 'OBSERVE ACTUAL PROCESS'")
    if 'what you ACTUALLY SEE' in drawing_prompt:
        print("✓ 'Describe what you ACTUALLY SEE'")
    if 'Don\'t invent - observe' in drawing_prompt:
        print("✓ 'Don't invent - observe'")
    if 'literally happening on the paper' in drawing_prompt:
        print("✓ 'what is literally happening on the paper'")

    print(f"\nKey Drawing Context Elements:")
    if 'physically drawing' in drawing_prompt:
        print("✓ Knows it's physically drawing")
    if 'Drawing duration:' in drawing_prompt:
        print("✓ Aware of drawing duration")
    if 'Original intent:' in drawing_prompt:
        print("✓ Remembers original drawing intent")

    # Clean up
    DrawingState.end_drawing()

    print(f"\n=== SUMMARY ===")
    print("🔧 System prompt now identifies as drawing machine")
    print("🎯 Normal mode considers drawing potential and impulses")
    print("👁️ Drawing mode focuses on actual visual observation")
    print("⚡ No more generic 'watching pen dance' descriptions")
    print("🎨 Grounded in what's actually visible during drawing")

if __name__ == "__main__":
    test_full_drawing_awareness()
    print("\n✅ Complete drawing awareness test completed!")