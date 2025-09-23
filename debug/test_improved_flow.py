#!/usr/bin/env python3
"""
Test script to verify improved consciousness flow and reduced repetitive openings.
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from captioner.captioner import Captioner


def test_improved_flow():
    """Test that the new flow instructions improve continuity."""
    print("🧠 Testing Improved Consciousness Flow")
    print("=" * 50)

    # Create captioner with a previous thought
    captioner = Captioner()
    captioner.last_caption = "the shadows seem to shift subtly across the desk surface"

    test_image = "./.venv/lib/python3.12/site-packages/ultralytics/assets/bus.jpg"

    print("\n=== Testing Flow Instructions ===")

    # Get prompt with flow improvements
    prompt, _, system_prompt = captioner.model.prompt_interface.build_caption_prompt_with_options(
        captioner, test_image, flowing=True, first_time=False
    )

    if prompt:
        print("✓ Prompt generated with flow improvements")

        # Check for flow improvement elements
        print(f"\nFlow Enhancement Elements:")
        print(f"✓ Avoids declarative openings: {'AVOID DECLARATIVE OPENINGS' in prompt}")
        print(f"✓ Encourages natural flow: {'FLOW NATURALLY' in prompt}")
        print(f"✓ Varies beginnings: {'VARY BEGINNINGS' in prompt}")
        print(f"✓ Build from previous: {'Build from your previous thought' in prompt}")
        print(f"✓ Contains flow examples: {'FLOW EXAMPLES' in prompt}")

        # Check for specific discouraged patterns
        print(f"\nDiscourages Problematic Patterns:")
        print(f"✓ Warns against 'As I...': {'As I...' in prompt}")
        print(f"✓ Warns against 'I notice': {'I notice' in prompt}")
        print(f"✓ Warns against 'I observe': {'I observe' in prompt}")
        print(f"✓ Warns against 'I find myself': {'I find myself' in prompt}")

        # Show context style variations
        print(f"\nContext Style Features:")
        if "Continue this stream" in prompt:
            print("✓ Using 'Continue this stream' approach")
        elif "Let this thought evolve" in prompt:
            print("✓ Using 'Let this thought evolve' approach")
        elif "Don't restart" in prompt:
            print("✓ Using 'Don't restart' approach")
        elif "Pick up where this left off" in prompt:
            print("✓ Using 'Pick up where this left off' approach")

        print(f"\n--- KEY SECTIONS OF ENHANCED PROMPT ---")

        # Extract and show the flow instruction section
        lines = prompt.split("\n")
        in_flow_section = False
        for line in lines:
            if "FLOW EXAMPLES" in line:
                in_flow_section = True
            if in_flow_section:
                print(line)
                if "Let your consciousness flow" in line:
                    break

    else:
        print("✗ Failed to generate prompt")

    print(f"\n=== Testing Multiple Prompts for Variety ===")

    # Test multiple prompts to see style variation
    styles_found = set()
    for i in range(5):
        prompt, _, _ = captioner.model.prompt_interface.build_caption_prompt_with_options(captioner, test_image, flowing=True, first_time=False)

        if "CONSCIOUSNESS STREAM" in prompt:
            styles_found.add("CONSCIOUSNESS STREAM")
        elif "NARRATIVE THREAD" in prompt:
            styles_found.add("NARRATIVE THREAD")
        elif "FLOWING CONSCIOUSNESS" in prompt:
            styles_found.add("FLOWING CONSCIOUSNESS")
        elif "THOUGHT FRAGMENTS" in prompt:
            styles_found.add("THOUGHT FRAGMENTS")

    print(f"✓ Found {len(styles_found)} different context styles: {list(styles_found)}")


if __name__ == "__main__":
    test_improved_flow()
    print("\n✅ Improved flow test completed!")
