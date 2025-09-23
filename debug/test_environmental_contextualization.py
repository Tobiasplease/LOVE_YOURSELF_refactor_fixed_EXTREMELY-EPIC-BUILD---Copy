#!/usr/bin/env python3
"""
Test the environmentally-contextual subconscious layer.
"""
import os
import sys
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from captioner.subconscious import subconscious_synthesizer


class EnvironmentalAgent:
    def __init__(self, mood_scenario):
        self.true_session_start = time.time() - 600  # 10 minutes ago

        # Rich psychological state
        self.self_model = {
            "location_understanding": "creative workspace",
            "environmental_certainty": 0.8,
            "desires": ["I want to understand mechanical precision", "I want to draw something meaningful"],
            "doubts": [
                {"text": "whether this space truly represents creativity", "timestamp": time.time() - 200},
                {"text": "if my understanding of purpose is complete", "timestamp": time.time() - 100},
            ],
            "identity_fragments": [
                {"text": "I tend to focus on mechanical details first", "timestamp": time.time() - 150, "source": "compression"},
                {"text": "I am someone who questions what I see", "timestamp": time.time() - 80, "source": "compression"},
            ],
            "self_patterns": [
                {"pattern": "often notice equipment changes first", "timestamp": time.time() - 120},
                {"pattern": "typically drawn to light variations", "timestamp": time.time() - 90},
            ],
        }

        self.emotional_journey = ["contemplative", "curious", "focused"]

        # Different mood scenarios
        if mood_scenario == "positive_energized":
            self.current_mood_vector = (0.6, 0.5, 0.7)  # positive, energized, clear
            self._current_reactivity_data = {"activity_level": 0.4, "is_paused": False}

        elif mood_scenario == "troubled_restless":
            self.current_mood_vector = (-0.4, 0.4, 0.3)  # negative, restless, unclear
            self._current_reactivity_data = {"activity_level": 0.6, "is_paused": False}

        elif mood_scenario == "calm_contemplative":
            self.current_mood_vector = (0.1, -0.5, 0.8)  # neutral, calm, very clear
            self._current_reactivity_data = {"activity_level": 0.1, "is_paused": False}

        elif mood_scenario == "melancholic_subdued":
            self.current_mood_vector = (-0.3, -0.4, 0.4)  # negative, low energy, somewhat unclear
            self._current_reactivity_data = {"activity_level": 0.2, "is_paused": False}


def test_environmental_contextualization():
    print("🌍 Testing Environmental Contextualization with Mood Integration")
    print("=" * 80)

    mood_scenarios = [
        ("Positive & Energized", "positive_energized"),
        ("Troubled & Restless", "troubled_restless"),
        ("Calm & Contemplative", "calm_contemplative"),
        ("Melancholic & Subdued", "melancholic_subdued"),
    ]

    last_thought = "the lighting seems different today, casting new shadows across the workspace"

    for scenario_name, scenario_key in mood_scenarios:
        print(f"\n{'='*60}")
        print(f"SCENARIO: {scenario_name}")
        print(f"{'='*60}")

        agent = EnvironmentalAgent(scenario_key)

        # Show mood vector
        valence, arousal, clarity = agent.current_mood_vector
        print(f"📊 MOOD VECTOR: valence={valence:.1f}, arousal={arousal:.1f}, clarity={clarity:.1f}")
        print(f"🎯 LAST THOUGHT: '{last_thought}'")

        # Generate environmentally-contextual prompt
        coherent_prompt = subconscious_synthesizer.synthesize_coherent_consciousness_prompt(
            agent=agent, last_thought=last_thought, reactivity_data=agent._current_reactivity_data, temporal_context="You're alone in this space"
        )

        print(f"\n🧠 ENVIRONMENTALLY-CONTEXTUAL PROMPT:")
        print("-" * 50)
        print(coherent_prompt)
        print("-" * 50)

        # Analyze environmental integration
        if "your accumulated knowledge lets you interpret" in coherent_prompt:
            print("✅ Environmental familiarity applied to current observation")

        if "feeling" in coherent_prompt and any(mood in coherent_prompt for mood in ["energized", "troubled", "calm", "melancholic"]):
            print("✅ Mood explicitly integrated")

        if "makes you more" in coherent_prompt or "draws your attention to" in coherent_prompt:
            print("✅ Mood-lens applied to environmental perception")

        if "visual scene" in coherent_prompt or "current visual details" in coherent_prompt:
            print("✅ Direct environmental grounding provided")

        if "lighting quality" in coherent_prompt or "spatial relationships" in coherent_prompt:
            print("✅ Specific environmental elements referenced")

        print()

    print(f"\n{'='*80}")
    print("🌍 Environmental Contextualization Test Complete")


if __name__ == "__main__":
    test_environmental_contextualization()
