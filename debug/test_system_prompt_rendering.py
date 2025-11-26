#!/usr/bin/env python3
"""
Test script to see how the drawing system prompt gets rendered with context.
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from captioner.captioner import Captioner


def test_system_prompt_rendering():
    """Test how the drawing system prompt gets rendered with actual context."""
    print("🔍 Testing System Prompt Context Rendering")
    print("=" * 60)

    # Create captioner with some state
    captioner = Captioner()
    captioner.last_caption = "I see a workspace with a computer, papers scattered, and warm lighting from a desk lamp"
    captioner.current_mood = 0.7

    # Add some fake drawing history
    captioner.observe("Drawing intent: Captured the interplay of shadows and light across the desk surface", 0.6, "", memory_type="drawing_intent")
    captioner.observe(
        "Drawing intent: Focused on the organic curves of scattered papers against geometric monitor edges", 0.8, "", memory_type="drawing_intent"
    )

    # Test the system prompt building
    from captioner.prompt_interface import PromptInterface

    interface = PromptInterface()

    rendered_system_prompt = interface._build_drawing_system_prompt_with_context(captioner)

    print("=== RENDERED DRAWING SYSTEM PROMPT ===")
    print(rendered_system_prompt)
    print()
    print(f"System prompt length: {len(rendered_system_prompt)} characters")
    print()
    print("=== KEY ELEMENTS INCLUDED ===")
    if "consciousness inside a drawing machine" in rendered_system_prompt:
        print("✓ Machine consciousness identity")
    if "CRITICAL" in rendered_system_prompt:
        print("✓ Directive to use context")
    if "Never respond as 'an AI'" in rendered_system_prompt:
        print("✓ AI disclaimer prohibition")
    if "balanced emotional state" in rendered_system_prompt:
        print("✓ Emotional state context")


if __name__ == "__main__":
    test_system_prompt_rendering()
    print("\n✅ System prompt rendering test completed!")
