#!/usr/bin/env python3
"""
Test script to verify drawing state integration in prompts.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.drawing_state import DrawingState
from captioner.captioner import Captioner

def test_drawing_state_integration():
    """Test drawing state awareness in prompts."""
    print("🎨 Testing Drawing State Integration in Prompts")
    print("=" * 50)

    # Create captioner
    captioner = Captioner()
    captioner.last_caption = "I see workspace with papers and monitor"

    print("\n=== Testing Normal Mode (Not Drawing) ===")

    # Get normal prompt (use existing test image)
    test_image = "./.venv/lib/python3.12/site-packages/ultralytics/assets/bus.jpg"
    normal_prompt, _, _ = captioner.model.prompt_interface.build_caption_prompt_with_options(
        captioner, test_image, flowing=True, first_time=False
    )

    if normal_prompt:
        print("✓ Normal prompt generated")
        print(f"Contains 'Looking through your camera': {'Looking through your camera' in normal_prompt}")
        print(f"Contains 'MACHINE AWARENESS': {'MACHINE AWARENESS' in normal_prompt}")
    else:
        print("✗ Failed to generate normal prompt")

    print("\n=== Testing Drawing Mode (Active Drawing) ===")

    # Simulate active drawing
    DrawingState.start_drawing(
        intent="Testing line quality and composition",
        description="Creating precise geometric forms",
        drawing_file="test_drawing.gcode"
    )

    # Get drawing-aware prompt
    drawing_prompt, _, _ = captioner.model.prompt_interface.build_caption_prompt_with_options(
        captioner, test_image, flowing=True, first_time=False
    )

    if drawing_prompt:
        print("✓ Drawing-aware prompt generated")
        print(f"Contains 'physically drawing': {'physically drawing' in drawing_prompt}")
        print(f"Contains 'MACHINE AWARENESS': {'MACHINE AWARENESS' in drawing_prompt}")
        print(f"Contains 'pen servo': {'pen servo' in drawing_prompt}")
        print(f"Contains drawing duration: {'Drawing duration:' in drawing_prompt}")
    else:
        print("✗ Failed to generate drawing-aware prompt")

    # Clean up
    DrawingState.end_drawing()

if __name__ == "__main__":
    test_drawing_state_integration()
    print("\n✅ Drawing state prompt test completed!")