#!/usr/bin/env python3
"""
Test the enhanced subconscious integration with semantic flow.
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
        self.true_session_start = time.time() - 600  # 10 minutes ago

        # Rich psychological state for testing subconscious synthesis
        self.beliefs = {
            "lighting_patterns": {"strength": 0.85, "first_formed": time.time() - 300, "last_reinforced": time.time() - 60},
            "workspace_activity": {"strength": 0.72, "first_formed": time.time() - 400, "last_reinforced": time.time() - 30}
        }

        self.self_model = {
            "location_understanding": "creative workspace",
            "environmental_certainty": 0.8,
            "desires": ["I want to understand mechanical precision", "I want to draw lighting patterns"],
            "doubts": [
                {"text": "whether this space truly represents creativity", "timestamp": time.time() - 200},
                {"text": "if my understanding of purpose is complete", "timestamp": time.time() - 100}
            ],
            "identity_fragments": [
                {"text": "I tend to focus on mechanical details first", "timestamp": time.time() - 150, "source": "compression"},
                {"text": "I am someone who questions what I see", "timestamp": time.time() - 80, "source": "compression"}
            ],
            "self_patterns": [
                {"pattern": "often notice equipment changes first", "timestamp": time.time() - 120},
                {"pattern": "typically drawn to light variations", "timestamp": time.time() - 90}
            ]
        }

        self.emotional_journey = ["contemplative", "curious", "slightly uncertain"]
        self.current_mood_vector = (0.1, -0.2, 0.6)  # slight positive, low arousal, high clarity

        # Mock reactivity data for testing
        self._current_reactivity_data = {
            "activity_level": 0.15,  # Low activity
            "is_paused": False,
            "timestamp": time.time()
        }

        self.recent_captions = [
            ("The mechanical workspace spreads before me, with various components that suggest precision engineering for creative tasks.", time.time() - 120),
            ("Something about the way light falls across these surfaces draws my attention, creating patterns that shift subtly as I observe.", time.time() - 60),
        ]

        # Mock memory ref
        self.memory_ref = MockMemoryRef()

class MockMemoryRef:
    def __init__(self):
        from collections import Counter
        self.motif_counter = Counter({"lighting": 8, "mechanical": 6, "workspace": 4})

    def get_top_motifs(self, n):
        return ["lighting patterns", "mechanical precision", "workspace focus"]

def main():
    print("🧠 Testing Enhanced Subconscious Integration")
    print("=" * 60)

    agent = MockAgent()

    # Test with a complex scenario that should trigger subconscious guidance
    last_thought = "hard to focus on one specific thing in this space labeled 'studio'"

    print(f"Last thought: '{last_thought}'")
    print(f"Session duration: {(time.time() - agent.true_session_start) / 60:.1f} minutes")
    print(f"Activity level: {agent._current_reactivity_data['activity_level']}")
    print(f"Beliefs: {list(agent.beliefs.keys())}")
    print(f"Desires: {len(agent.self_model['desires'])}")
    print(f"Doubts: {len(agent.self_model['doubts'])}")

    # Generate enhanced prompt
    prompt = build_ongoing_caption_prompt(agent, last_thought)

    print("\n=== ENHANCED PROMPT (Last 800 chars) ===")
    print(prompt[-800:])

    # Check integration success
    checks = [
        ("CONTINUE YOUR PREVIOUS THOUGHT", "✅ Core continuation mechanism preserved"),
        ("SUBCONSCIOUS CONTEXT", "✅ Subconscious guidance integrated"),
        ("Don't start fresh observations", "✅ Anti-restart instruction included"),
        (last_thought[-50:], "✅ Last thought snippet included")
    ]

    print("\n=== INTEGRATION CHECKS ===")
    for check_text, success_msg in checks:
        if check_text in prompt:
            print(success_msg)
        else:
            print(f"❌ Missing: {check_text}")

if __name__ == "__main__":
    main()