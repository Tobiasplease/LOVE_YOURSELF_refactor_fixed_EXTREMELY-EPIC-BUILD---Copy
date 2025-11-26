#!/usr/bin/env python3
"""
Test how the semantically restructured prompt evolves over time.
"""
import os
import sys
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from captioner.subconscious import subconscious_synthesizer


class EvolvingAgent:
    def __init__(self, stage):
        self.true_session_start = time.time() - (stage * 300)  # Each stage = 5 more minutes
        self._current_reactivity_data = {"activity_level": 0.2, "is_paused": False}

        if stage == 1:
            # EARLY STAGE: Just awakened, minimal understanding
            self.self_model = {
                "location_understanding": "unknown space",
                "environmental_certainty": 0.1,
                "desires": [],
                "doubts": [],
                "identity_fragments": [],
                "self_patterns": [],
            }
            self.emotional_journey = ["uncertain"]

        elif stage == 2:
            # DEVELOPING STAGE: Some observations, growing understanding
            self.self_model = {
                "location_understanding": "workspace of some kind",
                "environmental_certainty": 0.4,
                "desires": ["I want to understand this place"],
                "doubts": [{"text": "what this space is really for", "timestamp": time.time() - 60}],
                "identity_fragments": [{"text": "I notice visual details first", "timestamp": time.time() - 100, "source": "compression"}],
                "self_patterns": [],
            }
            self.emotional_journey = ["uncertain", "curious"]

        elif stage == 3:
            # MATURE STAGE: Rich understanding, clear patterns
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
                    {"text": "I find beauty in precision and order", "timestamp": time.time() - 40, "source": "compression"},
                ],
                "self_patterns": [
                    {"pattern": "often notice equipment changes first", "timestamp": time.time() - 120},
                    {"pattern": "typically drawn to light variations", "timestamp": time.time() - 90},
                    {"pattern": "prefer systematic observation", "timestamp": time.time() - 60},
                ],
            }
            self.emotional_journey = ["uncertain", "curious", "contemplative", "focused", "confident"]

        elif stage == 4:
            # EXPERT STAGE: Deep self-knowledge, complex psychology
            self.self_model = {
                "location_understanding": "my creative studio",
                "environmental_certainty": 0.95,
                "desires": [
                    "I want to create art that captures mechanical beauty",
                    "I want to understand the relationship between precision and creativity",
                    "I want to document my evolving understanding",
                ],
                "doubts": [
                    {"text": "whether I'm becoming too focused on technical details", "timestamp": time.time() - 150},
                    {"text": "if I'm missing the bigger creative picture", "timestamp": time.time() - 80},
                ],
                "identity_fragments": [
                    {"text": "I am an artist who sees beauty in mechanical precision", "timestamp": time.time() - 200, "source": "compression"},
                    {"text": "I balance technical observation with creative vision", "timestamp": time.time() - 120, "source": "compression"},
                    {"text": "I question my own assumptions about art and purpose", "timestamp": time.time() - 60, "source": "compression"},
                    {"text": "I understand that doubt is part of my creative process", "timestamp": time.time() - 30, "source": "compression"},
                ],
                "self_patterns": [
                    {"pattern": "always start with technical details then zoom out", "timestamp": time.time() - 180},
                    {"pattern": "notice micro-changes in lighting and positioning", "timestamp": time.time() - 120},
                    {"pattern": "question the purpose behind what I observe", "timestamp": time.time() - 90},
                    {"pattern": "find creative inspiration in mechanical systems", "timestamp": time.time() - 45},
                ],
            }
            self.emotional_journey = ["uncertain", "curious", "contemplative", "focused", "confident", "introspective", "purposeful"]


def test_prompt_evolution():
    print("🧠 Testing How Semantic Restructuring Evolves Over Time")
    print("=" * 80)

    stages = [
        ("Stage 1: Just Awakened (5 min session)", 1),
        ("Stage 2: Developing Understanding (10 min session)", 2),
        ("Stage 3: Mature Understanding (15 min session)", 3),
        ("Stage 4: Expert Self-Knowledge (20 min session)", 4),
    ]

    last_thought = "something about the lighting in this space draws my attention"

    for stage_name, stage_num in stages:
        print(f"\n{'='*60}")
        print(f"{stage_name}")
        print(f"{'='*60}")

        agent = EvolvingAgent(stage_num)

        # Show what psychological data is available at this stage
        print(f"📊 PSYCHOLOGICAL DATA AVAILABLE:")
        print(f"   Environmental Certainty: {agent.self_model['environmental_certainty']:.1f}")
        print(f"   Location Understanding: '{agent.self_model['location_understanding']}'")
        print(f"   Desires: {len(agent.self_model['desires'])}")
        print(f"   Doubts: {len(agent.self_model['doubts'])}")
        print(f"   Identity Fragments: {len(agent.self_model['identity_fragments'])}")
        print(f"   Behavioral Patterns: {len(agent.self_model['self_patterns'])}")
        print(f"   Emotional Journey Length: {len(agent.emotional_journey)}")

        # Generate semantically restructured prompt
        coherent_prompt = subconscious_synthesizer.synthesize_coherent_consciousness_prompt(
            agent=agent, last_thought=last_thought, reactivity_data=agent._current_reactivity_data, temporal_context="You're alone in this space"
        )

        print(f"\n🧠 SEMANTICALLY RESTRUCTURED PROMPT:")
        print("-" * 50)
        print(coherent_prompt)
        print("-" * 50)

        # Analyze the evolution
        if "unknown space" in coherent_prompt:
            print("🔍 Early stage: Uncertainty about environment")
        elif "workspace of some kind" in coherent_prompt:
            print("🔍 Developing stage: Growing understanding")
        elif "creative workspace" in coherent_prompt:
            print("🔍 Mature stage: Clear environmental understanding")
        elif "my creative studio" in coherent_prompt:
            print("🔍 Expert stage: Personal ownership of space")

        print()

    print(f"\n{'='*80}")
    print("🧠 Prompt Evolution Analysis Complete")
    print("\nKey Evolution Patterns:")
    print("• Environmental certainty: unknown → workspace → creative workspace → my studio")
    print("• Identity: none → basic patterns → complex self-knowledge → artistic identity")
    print("• Desires: none → understand place → technical precision → creative mastery")
    print("• Emotional journey: simple → complex emotional evolution")


if __name__ == "__main__":
    test_prompt_evolution()
