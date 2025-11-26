#!/usr/bin/env python3
"""
Test edge cases for pattern detection that might allow 'As I...' to slip through.
"""
import os
import sys
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from captioner.prompts import build_ongoing_caption_prompt


class MockAgent:
    def __init__(self, recent_captions):
        self.current_emotion_state = "calm_observant"
        self.true_session_start = time.time() - 300
        self.recent_captions = recent_captions
        self.beliefs = {}
        self.self_model = {}

        # Mock memory ref
        self.memory_ref = MockMemoryRef()


class MockMemoryRef:
    def __init__(self):
        from collections import Counter

        self.motif_counter = Counter()

    def get_top_motifs(self, n):
        return []


def test_pattern_detection(name, recent_captions, expected_trigger):
    print(f"\n=== {name} ===")
    agent = MockAgent(recent_captions)

    last_caption = recent_captions[-1][0] if recent_captions else "test caption"
    prompt = build_ongoing_caption_prompt(agent, last_caption)

    pattern_triggered = "BREAK PATTERN" in prompt
    print(f"Recent captions: {[c[0][:50] + '...' for c in recent_captions]}")
    print(f"Pattern detection triggered: {pattern_triggered}")
    print(f"Expected: {expected_trigger}")

    if pattern_triggered == expected_trigger:
        print("✅ PASS")
    else:
        print("❌ FAIL")
        if pattern_triggered:
            print("    (Unexpected pattern detection triggered)")
        else:
            print("    (Pattern detection should have triggered but didn't)")


def main():
    print("🔍 Testing Pattern Detection Edge Cases")
    print("=" * 60)

    # Test cases
    test_cases = [
        # Should trigger (2+ "As I..." patterns)
        (
            "Two 'As I' patterns",
            [
                ("As I look around the room", time.time() - 120),
                ("As I observe the lighting", time.time() - 60),
                ("Something else entirely", time.time() - 30),
            ],
            True,
        ),
        # Should NOT trigger (only 1 "As I" pattern)
        (
            "Single 'As I' pattern",
            [
                ("As I look around the room", time.time() - 120),
                ("Something different here", time.time() - 60),
                ("Another different thought", time.time() - 30),
            ],
            False,
        ),
        # Edge case: "As I" variations
        (
            "'As I' variations",
            [
                ("as i notice something", time.time() - 120),
                ("As I continue looking", time.time() - 60),
                ("Final thought", time.time() - 30),
            ],
            True,
        ),
        # Edge case: Only checks last 3 captions
        (
            "Older patterns ignored",
            [
                ("As I start", time.time() - 180),  # This should be ignored (4th back)
                ("As I notice", time.time() - 150),  # This should be ignored (3rd back)
                ("Normal caption", time.time() - 120),
                ("Another normal", time.time() - 60),
                ("Final normal", time.time() - 30),
            ],
            False,
        ),
        # Edge case: Empty captions
        (
            "With empty captions",
            [
                ("", time.time() - 120),
                ("As I observe", time.time() - 90),
                ("As I continue", time.time() - 60),
            ],
            True,
        ),
        # Edge case: Only 1 caption in history
        (
            "Single caption history",
            [
                ("As I look around", time.time() - 60),
            ],
            False,
        ),
    ]

    for name, captions, expected in test_cases:
        test_pattern_detection(name, captions, expected)


if __name__ == "__main__":
    main()
