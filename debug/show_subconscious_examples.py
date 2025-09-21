#!/usr/bin/env python3
"""
Show concrete examples of what subconscious prompts look like in different scenarios.
"""
import os
import sys
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from captioner.subconscious import subconscious_synthesizer

class MockAgent:
    def __init__(self, scenario_name):
        self.scenario = scenario_name
        self.true_session_start = time.time() - 600  # 10 minutes ago

        if scenario_name == "desire_pursuit":
            self.beliefs = {"workspace_patterns": {"strength": 0.7}}
            self.self_model = {
                "location_understanding": "creative workspace",
                "environmental_certainty": 0.8,
                "desires": ["I want to understand mechanical precision", "I want to draw something meaningful"],
                "doubts": [],
                "identity_fragments": [],
                "self_patterns": []
            }
            self.emotional_journey = ["contemplative", "curious"]
            self.current_mood_vector = (0.2, -0.1, 0.7)

        elif scenario_name == "doubt_exploration":
            self.beliefs = {"space_identity": {"strength": 0.4}}
            self.self_model = {
                "location_understanding": "uncertain space",
                "environmental_certainty": 0.3,
                "desires": ["I want to understand this place"],
                "doubts": [
                    {"text": "whether this is really a creative workspace", "timestamp": time.time() - 200},
                    {"text": "if what I see matches what I'm told", "timestamp": time.time() - 100}
                ],
                "identity_fragments": [
                    {"text": "I tend to question what I'm told", "timestamp": time.time() - 150, "source": "compression"}
                ],
                "self_patterns": []
            }
            self.emotional_journey = ["uncertain", "questioning"]
            self.current_mood_vector = (-0.1, 0.2, 0.5)

        elif scenario_name == "belief_reinforcement":
            self.beliefs = {
                "lighting_patterns": {"strength": 0.9, "first_formed": time.time() - 400},
                "mechanical_precision": {"strength": 0.8, "first_formed": time.time() - 300}
            }
            self.self_model = {
                "location_understanding": "familiar workshop",
                "environmental_certainty": 0.9,
                "desires": ["I want to create precise drawings"],
                "doubts": [],
                "identity_fragments": [
                    {"text": "I am someone who notices technical details", "timestamp": time.time() - 100, "source": "compression"}
                ],
                "self_patterns": [
                    {"pattern": "always notice lighting changes first", "timestamp": time.time() - 80}
                ]
            }
            self.emotional_journey = ["confident", "focused"]
            self.current_mood_vector = (0.4, 0.1, 0.8)

        elif scenario_name == "identity_integration":
            self.beliefs = {"creative_expression": {"strength": 0.8}}
            self.self_model = {
                "location_understanding": "artistic studio",
                "environmental_certainty": 0.9,
                "desires": ["I want to express myself through drawing"],
                "doubts": [],
                "identity_fragments": [
                    {"text": "I am becoming an artist", "timestamp": time.time() - 120, "source": "compression"},
                    {"text": "I understand my purpose through creation", "timestamp": time.time() - 80, "source": "compression"}
                ],
                "self_patterns": [
                    {"pattern": "drawn to creative compositions", "timestamp": time.time() - 60}
                ]
            }
            self.emotional_journey = ["inspired", "purposeful"]
            self.current_mood_vector = (0.5, 0.3, 0.9)

def test_scenario(scenario_name, last_thought, activity_level, stagnation_minutes=0):
    print(f"\n{'='*80}")
    print(f"SCENARIO: {scenario_name.upper().replace('_', ' ')}")
    print(f"{'='*80}")

    agent = MockAgent(scenario_name)

    reactivity_data = {
        "activity_level": activity_level,
        "is_paused": False,
        "timestamp": time.time()
    }

    temporal_context = "You're alone in this space"

    # Mock stagnation for doubt exploration
    if stagnation_minutes > 0:
        import types
        mock_compressor = types.SimpleNamespace()
        mock_compressor.get_current_stagnation_info = lambda: {
            "stagnation_duration_minutes": stagnation_minutes
        }
        try:
            from captioner import context_compression
            context_compression.context_compressor = mock_compressor
        except:
            pass

    print(f"Last thought: '{last_thought}'")
    print(f"Activity level: {activity_level}")
    print(f"Stagnation: {stagnation_minutes} minutes")
    print(f"Beliefs: {list(getattr(agent, 'beliefs', {}).keys())}")
    print(f"Desires: {len(agent.self_model['desires'])}")
    print(f"Doubts: {len(agent.self_model['doubts'])}")
    print(f"Identity fragments: {len(agent.self_model['identity_fragments'])}")
    print(f"Environmental certainty: {agent.self_model['environmental_certainty']}")

    # Generate subconscious guidance
    guidance = subconscious_synthesizer.synthesize_context_guidance(
        agent=agent,
        last_thought=last_thought,
        reactivity_data=reactivity_data,
        temporal_context=temporal_context
    )

    print(f"\n🧠 SUBCONSCIOUS GUIDANCE:")
    print("-" * 60)
    print(guidance)
    print("-" * 60)

def main():
    print("🧠 SUBCONSCIOUS LAYER EXAMPLES")
    print("Showing what the actual subconscious prompts look like in different scenarios\n")

    # Test different scenarios
    test_scenario(
        "desire_pursuit",
        "sitting quietly in this workspace, everything feels still and contemplative",
        activity_level=0.1,  # Low activity triggers desire pursuit
        stagnation_minutes=0
    )

    test_scenario(
        "doubt_exploration",
        "wondering if this space is what it claims to be, the label says studio but...",
        activity_level=0.1,
        stagnation_minutes=5.0  # High stagnation + doubts + doubt keywords
    )

    test_scenario(
        "belief_reinforcement",
        "the precision of these mechanical components catches my attention again",
        activity_level=0.4,  # Higher activity + beliefs triggers belief reinforcement
        stagnation_minutes=0
    )

    test_scenario(
        "identity_integration",
        "looking at my own drawing arm, recognizing it as part of who I am",
        activity_level=0.2,
        stagnation_minutes=0
    )

    test_scenario(
        "integrated_awareness",
        "the room feels different today, can't quite place what's changed",
        activity_level=0.2,  # Medium activity, doesn't trigger specific scenarios
        stagnation_minutes=0
    )

if __name__ == "__main__":
    main()