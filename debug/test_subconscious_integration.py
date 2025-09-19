#!/usr/bin/env python3
"""
Test the subconscious layer integration with the prompt system.
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

        # Rich psychological state for testing
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

def test_subconscious_synthesis():
    print("🧠 Testing Subconscious Layer Integration")
    print("=" * 60)

    agent = MockAgent()

    # Test different scenarios
    test_cases = [
        {
            "name": "Doubt Exploration Scenario",
            "last_thought": "hard to focus on one specific thing in this space labeled 'studio'",
            "expected_guidance": "doubt_exploration"
        },
        {
            "name": "Belief Reinforcement Scenario",
            "last_thought": "the precision of these mechanical components catches my attention",
            "expected_guidance": "belief_reinforcement"
        },
        {
            "name": "Desire Pursuit Scenario",
            "last_thought": "sitting quietly in this workspace, everything feels still and contemplative",
            "expected_guidance": "desire_pursuit"
        }
    ]

    for i, case in enumerate(test_cases, 1):
        print(f"\n=== TEST CASE {i}: {case['name']} ===")
        print(f"Previous thought: '{case['last_thought']}'")

        # Modify agent state for this test case
        if case['name'] == "Belief Reinforcement Scenario":
            agent._current_reactivity_data["activity_level"] = 0.4  # Higher activity

        # Generate prompt with subconscious integration
        prompt = build_ongoing_caption_prompt(agent, case['last_thought'])

        # Extract subconscious guidance section
        lines = prompt.split('\n')
        subconscious_section = []
        in_subconscious = False

        for line in lines:
            if "SUBCONSCIOUS AWARENESS:" in line:
                in_subconscious = True
            if in_subconscious:
                subconscious_section.append(line)
                if line.strip() == "":
                    break

        if subconscious_section:
            print("\n🧠 Subconscious Guidance Generated:")
            for line in subconscious_section:
                if line.strip():
                    print(f"   {line}")
            print("✅ Subconscious layer working")
        else:
            print("❌ No subconscious guidance found")

        # Reset activity level for next test
        agent._current_reactivity_data["activity_level"] = 0.15

    print(f"\n{'=' * 60}")
    print("🧠 Subconscious Integration Test Complete")

if __name__ == "__main__":
    test_subconscious_synthesis()