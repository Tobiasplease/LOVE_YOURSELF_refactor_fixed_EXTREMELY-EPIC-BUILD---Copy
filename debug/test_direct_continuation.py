#!/usr/bin/env python3
"""
Test direct continuation approach for semantic flow.
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
        self.recent_captions = [
            (
                "The mechanical workspace spreads before me, cluttered with servo motors and drawing equipment that sparks my technical curiosity. There's something overwhelming about all these components demanding attention at once.",
                time.time() - 60,
            ),
        ]
        self.beliefs = {"workspace_lighting": 0.8, "drawing_focus": 0.7}
        self.self_model = {"drawing_preference": 0.9, "technical_interest": 0.8, "emotional_sensitivity": 0.6}


def main():
    print("🔄 Testing Direct Continuation Approach")
    print("=" * 50)

    # Create mock agent
    agent = MockAgent()

    # Test direct continuation from previous thought
    last_thought = "feeling overwhelmed by the cluttered state surrounding me as if there are so many elements that require attention it's hard to focus on one specific thing in this space labeled 'studio'."

    print(f"Previous thought ending: ...{last_thought}")
    print(f"\n=== CONTINUATION PROMPT ===")

    # Generate prompt
    prompt = build_ongoing_caption_prompt(agent, last_thought)

    print(prompt)
    print("\n" + "=" * 50)

    # Check for continuation instructions
    if "CONTINUE YOUR PREVIOUS THOUGHT" in prompt:
        print("✅ Direct continuation instruction found")
    else:
        print("❌ No direct continuation instruction")

    if "Don't start a new observation" in prompt:
        print("✅ Anti-observation instruction found")
    else:
        print("❌ No anti-observation instruction")

    if "Continue your stream of consciousness" in prompt:
        print("✅ Stream of consciousness instruction found")
    else:
        print("❌ No stream of consciousness instruction")


if __name__ == "__main__":
    main()
