#!/usr/bin/env python3
"""
Test script to verify the new visual-grounded drawing prompt system works.
"""

import sys
import os
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from captioner.captioner import Captioner
from captioner.prompts import build_drawing_prompt

def test_visual_grounded_drawing():
    """Test drawing prompt generation with actual image."""
    print("🎨 Testing Visual-Grounded Drawing Prompts")
    print("=" * 50)

    # Create captioner with some state
    captioner = Captioner()
    captioner.last_caption = "I see a workspace with a computer, papers scattered, and warm lighting from a desk lamp"
    captioner.current_mood = 0.7

    # Add some fake drawing history
    captioner.observe("Drawing intent: Captured the interplay of shadows and light across the desk surface", 0.6, "", memory_type="drawing_intent")
    captioner.observe("Drawing intent: Focused on the organic curves of scattered papers against geometric monitor edges", 0.8, "", memory_type="drawing_intent")

    # Test with image
    test_image = "./.venv/lib/python3.12/site-packages/ultralytics/assets/bus.jpg"

    print(f"\n=== Testing with image: {test_image} ===")
    if os.path.exists(test_image):
        print("✓ Test image found")

        # Test the build_drawing_prompt function directly with image
        start_time = time.time()
        prompt = build_drawing_prompt(captioner, extra="Current lighting seems particularly dramatic", image_path=test_image)
        elapsed = time.time() - start_time

        print(f"✓ Visual prompt generated in {elapsed:.2f}s")
        print(f"Prompt length: {len(prompt)} characters")
        print(f"\n--- VISUAL-GROUNDED PROMPT ---")
        print(prompt)
        print("--- END PROMPT ---\n")

        # Test the full model integration with image
        print("=== Testing full model integration with image ===")
        start_time = time.time()
        model_prompt = captioner.model.generate_drawing_prompt(extra="Current scene has interesting contrasts", image_path=test_image)
        elapsed = time.time() - start_time

        print(f"✓ Full model response in {elapsed:.2f}s")
        print(f"Model response length: {len(model_prompt)} characters")
        print(f"\n--- MODEL RESPONSE ---")
        print(model_prompt)
        print("--- END RESPONSE ---")
    else:
        print("✗ Test image not found, using text-only mode")

        # Test without image (fallback mode)
        start_time = time.time()
        prompt = build_drawing_prompt(captioner, extra="Testing fallback mode")
        elapsed = time.time() - start_time

        print(f"✓ Text-only prompt generated in {elapsed:.2f}s")
        print(f"Prompt length: {len(prompt)} characters")
        print(f"\n--- TEXT-ONLY PROMPT ---")
        print(prompt)
        print("--- END PROMPT ---")

if __name__ == "__main__":
    test_visual_grounded_drawing()
    print("\n✅ Visual drawing prompt test completed!")