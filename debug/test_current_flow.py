#!/usr/bin/env python3
"""
Test the current prompt flow to diagnose why 'As I...' patterns persist.
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

        # Create scenarios with "As I..." patterns
        self.recent_captions = [
            ("As I look around this workspace, everything feels familiar yet somehow different.", time.time() - 120),
            ("As I observe the mechanical components before me, I notice subtle changes in lighting.", time.time() - 60),
        ]

        self.beliefs = {"workspace_lighting": 0.8, "drawing_focus": 0.7}
        self.self_model = {"drawing_preference": 0.9, "technical_interest": 0.8, "emotional_sensitivity": 0.6}

        # Mock memory ref with motif counter
        self.memory_ref = MockMemoryRef()


class MockMemoryRef:
    def __init__(self):
        from collections import Counter

        self.motif_counter = Counter({"workspace": 15, "lighting": 12, "mechanical": 8})

    def get_top_motifs(self, n):
        return ["workspace focus", "lighting patterns", "mechanical precision"]


def main():
    print("🔍 Testing Current Flow with 'As I...' Pattern Detection")
    print("=" * 60)

    agent = MockAgent()

    # Test with "As I..." pattern already present
    last_caption = "As I continue observing this creative space, the familiar arrangement of tools draws my attention."

    print("=== TESTING WITH 'AS I...' PATTERN HISTORY ===")
    print(f"Recent captions contain 'As I...' patterns: {len([c for c in agent.recent_captions if 'As I' in c[0]])}")
    print(f"Last caption: '{last_caption}'")

    # Generate prompt
    prompt = build_ongoing_caption_prompt(agent, last_caption)

    print("\n=== GENERATED PROMPT ===")
    print(prompt[-1000:])  # Show last 1000 chars

    # Check if anti-pattern mechanisms triggered
    if "BREAK PATTERN" in prompt:
        print("\n✅ Anti-'As I...' pattern detection triggered")
        print("✅ Alternative openings provided")
    else:
        print("\n❌ Anti-'As I...' pattern detection NOT triggered")

    if "CONTINUE YOUR PREVIOUS THOUGHT" in prompt:
        print("✅ Direct continuation instruction found")
    else:
        print("❌ Direct continuation instruction missing")

    if last_caption[-50:] in prompt:
        print("✅ Last thought snippet included")
    else:
        print("❌ Last thought snippet not found")


if __name__ == "__main__":
    main()
