#!/usr/bin/env python3
"""
Test the new semantic restructuring subconscious layer.
"""
import os
import sys
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from captioner.subconscious import subconscious_synthesizer

class MockAgent:
    def __init__(self):
        self.true_session_start = time.time() - 600  # 10 minutes ago

        # Rich psychological state (real data, not word counts)
        self.self_model = {
            "location_understanding": "creative workspace",
            "environmental_certainty": 0.8,
            "desires": [
                "I want to understand mechanical precision",
                "I want to draw something meaningful"
            ],
            "doubts": [
                {
                    "text": "whether this space truly represents creativity",
                    "timestamp": time.time() - 200
                },
                {
                    "text": "if my understanding of purpose is complete",
                    "timestamp": time.time() - 100
                }
            ],
            "identity_fragments": [
                {
                    "text": "I tend to focus on mechanical details first",
                    "timestamp": time.time() - 150,
                    "source": "compression"
                },
                {
                    "text": "I am someone who questions what I see",
                    "timestamp": time.time() - 80,
                    "source": "compression"
                }
            ],
            "self_patterns": [
                {
                    "pattern": "often notice equipment changes first",
                    "timestamp": time.time() - 120
                },
                {
                    "pattern": "typically drawn to light variations",
                    "timestamp": time.time() - 90
                }
            ]
        }

        self.emotional_journey = ["contemplative", "curious", "slightly uncertain"]
        self.current_mood_vector = (0.1, -0.2, 0.6)  # slight positive, low arousal, high clarity

        # Mock reactivity data
        self._current_reactivity_data = {
            "activity_level": 0.15,  # Low activity
            "is_paused": False,
            "timestamp": time.time()
        }

def test_semantic_restructuring():
    print("🧠 Testing Semantic Restructuring Subconscious Layer")
    print("=" * 70)

    agent = MockAgent()

    # Test with different scenarios
    test_cases = [
        {
            "name": "Questioning/Doubt Scenario",
            "last_thought": "wondering if this space is what it claims to be, labeled as a studio but feeling more like...",
            "description": "Should restructure doubts into coherent questioning guidance"
        },
        {
            "name": "Desire-Driven Scenario",
            "last_thought": "sitting quietly in this workspace, everything feels still and contemplative",
            "description": "Should restructure desires into coherent continuation guidance"
        },
        {
            "name": "Familiar Environment Scenario",
            "last_thought": "the precision of these mechanical components catches my attention again",
            "description": "Should leverage high environmental certainty and patterns"
        }
    ]

    for i, case in enumerate(test_cases, 1):
        print(f"\n{'=' * 50}")
        print(f"TEST CASE {i}: {case['name']}")
        print(f"{'=' * 50}")
        print(f"Last thought: '{case['last_thought']}'")
        print(f"Expected: {case['description']}")

        # Generate semantically restructured prompt
        coherent_prompt = subconscious_synthesizer.synthesize_coherent_consciousness_prompt(
            agent=agent,
            last_thought=case['last_thought'],
            reactivity_data=agent._current_reactivity_data,
            temporal_context="You're alone in this space"
        )

        print(f"\n🧠 SEMANTICALLY RESTRUCTURED PROMPT:")
        print("-" * 50)
        print(coherent_prompt)
        print("-" * 50)

        # Analyze what was synthesized
        if "You are intimately familiar" in coherent_prompt:
            print("✅ Environmental certainty semantically integrated")
        elif "You have growing understanding" in coherent_prompt:
            print("✅ Developing environmental understanding integrated")

        if "recognize that" in coherent_prompt or "learned that" in coherent_prompt:
            print("✅ Identity fragments semantically restructured")

        if "attention naturally follows patterns" in coherent_prompt:
            print("✅ Behavioral patterns semantically integrated")

        if "desire to" in coherent_prompt and case['name'] == "Desire-Driven Scenario":
            print("✅ Desires semantically integrated for low-activity scenario")

        if "questioning mood" in coherent_prompt and case['name'] == "Questioning/Doubt Scenario":
            print("✅ Doubts semantically integrated for questioning scenario")

        print()

    print(f"\n{'=' * 70}")
    print("🧠 Semantic Restructuring Test Complete")

if __name__ == "__main__":
    test_semantic_restructuring()