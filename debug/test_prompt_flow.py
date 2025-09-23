#!/usr/bin/env python3
"""
Test the actual prompt flow to see if direct continuation works.
"""
import os
import sys
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from captioner.prompts import build_ongoing_caption_prompt


class MockAgent:
    def __init__(self):
        self.current_emotion_state = "calm_observant"
        self.true_session_start = time.time() - 300  # 5 minutes ago
        self.recent_captions = []
        self.beliefs = {"workspace_lighting": 0.8, "drawing_focus": 0.7}
        self.self_model = {"drawing_preference": 0.9, "technical_interest": 0.8, "emotional_sensitivity": 0.6}


def main():
    print("🔄 Testing Prompt Flow for Direct Continuation")
    print("=" * 60)

    # Create mock agent
    agent = MockAgent()

    # First caption - no previous thought
    print("=== FIRST CAPTION PROMPT ===")
    first_prompt = build_ongoing_caption_prompt(agent=agent, last_caption=None)  # No previous thought
    print(first_prompt[-500:])  # Show last 500 chars
    print("\n" + "=" * 60)

    # Simulate first caption being generated
    first_caption = "feeling overwhelmed by the cluttered state surrounding me as if there are so many elements that require attention it's hard to focus on one specific thing in this space labeled 'studio'."

    # Add to recent captions
    agent.recent_captions = [(first_caption, time.time())]

    # Second caption - should continue from first
    print("=== SECOND CAPTION PROMPT (CONTINUATION) ===")
    second_prompt = build_ongoing_caption_prompt(agent=agent, last_caption=first_caption)

    print(second_prompt[-800:])  # Show last 800 chars
    print("\n" + "=" * 60)

    # Check for continuation instruction
    if "CONTINUE YOUR PREVIOUS THOUGHT" in second_prompt:
        print("✅ Direct continuation instruction found")
    else:
        print("❌ No direct continuation instruction")

    if first_caption[-50:] in second_prompt:
        print("✅ Previous thought snippet included")
    else:
        print("❌ Previous thought snippet not found")

    if "Don't start a new observation" in second_prompt:
        print("✅ Anti-observation instruction found")
    else:
        print("❌ No anti-observation instruction")


if __name__ == "__main__":
    main()
