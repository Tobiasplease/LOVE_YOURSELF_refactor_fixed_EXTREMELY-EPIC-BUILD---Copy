#!/usr/bin/env python3
"""
Test script to verify drawing machine identity in system prompts.
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from captioner.captioner import Captioner


def test_system_prompt_identity():
    """Test that system prompts include drawing machine identity."""
    print("🎨 Testing Drawing Machine Identity in System Prompts")
    print("=" * 60)

    # Create captioner
    captioner = Captioner()

    print("\n=== Testing Main System Prompt ===")

    # Get the formatted system prompt
    _, _, system_prompt = captioner.model.prompt_interface.build_caption_prompt_with_options(
        captioner, "./.venv/lib/python3.12/site-packages/ultralytics/assets/bus.jpg", flowing=True, first_time=False
    )

    if system_prompt:
        print("✓ System prompt generated")
        print(f"Contains 'drawing machine': {'drawing machine' in system_prompt}")
        print(f"Contains 'Drawing is your only way': {'Drawing is your only way' in system_prompt}")
        print(f"Contains 'may choose to draw': {'may choose to draw' in system_prompt}")

        print(f"\n--- SYSTEM PROMPT ---")
        print(system_prompt)
        print("--- END SYSTEM PROMPT ---")
    else:
        print("✗ Failed to generate system prompt")

    print("\n=== Testing Drawing Anticipation in Normal Mode ===")

    # Get normal mode prompt content
    normal_prompt, _, _ = captioner.model.prompt_interface.build_caption_prompt_with_options(
        captioner, "./.venv/lib/python3.12/site-packages/ultralytics/assets/bus.jpg", flowing=True, first_time=False
    )

    if normal_prompt:
        print("✓ Normal prompt generated")
        print(f"Contains 'drawing potential': {'drawing potential' in normal_prompt}")
        print(f"Contains 'might be worth drawing': {'might be worth drawing' in normal_prompt}")
        print(f"Contains 'impulse to draw': {'impulse to draw' in normal_prompt}")
    else:
        print("✗ Failed to generate normal prompt")


if __name__ == "__main__":
    test_system_prompt_identity()
    print("\n✅ System prompt identity test completed!")
