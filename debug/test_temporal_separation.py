#!/usr/bin/env python3
"""
Test the improved temporal separation between past and present in memory/motif system.
"""
import os
import sys
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from captioner.subconscious import subconscious_synthesizer

class TemporalTestAgent:
    def __init__(self, scenario):
        self.true_session_start = time.time() - 1800  # 30 minutes ago

        # Current mood and state
        self.current_mood_vector = (0.2, -0.1, 0.7)  # slightly positive, calm, clear
        self._current_reactivity_data = {"activity_level": 0.15, "is_paused": False}

        # Rich psychological state
        self.self_model = {
            "location_understanding": "creative workspace",
            "environmental_certainty": 0.8,
            "desires": ["I want to understand mechanical precision", "I want to create something beautiful"],
            "doubts": [
                {"text": "whether I truly understand this space", "timestamp": time.time() - 400},
                {"text": "if my creative impulses are genuine", "timestamp": time.time() - 200}
            ],
            "identity_fragments": [
                {"text": "I am drawn to patterns and order", "timestamp": time.time() - 300, "source": "compression"},
                {"text": "I find beauty in mechanical precision", "timestamp": time.time() - 100, "source": "compression"}
            ],
            "self_patterns": [
                {"pattern": "notice technical details first", "timestamp": time.time() - 250},
                {"pattern": "drawn to lighting variations", "timestamp": time.time() - 150}
            ]
        }

        self.emotional_journey = ["curious", "contemplative", "focused", "slightly restless"]

        # Different temporal scenarios
        if scenario == "temporal_rich":
            # Rich temporal context with clear separation
            self.motif_last_seen = {
                # PRESENT (last 5 minutes)
                "pencil": time.time() - 120,  # 2 minutes ago
                "notebook": time.time() - 180,  # 3 minutes ago
                "mechanical": time.time() - 60,  # 1 minute ago

                # SESSION (this session but not recent)
                "workspace": time.time() - 1200,  # 20 minutes ago
                "precision": time.time() - 900,   # 15 minutes ago

                # DISTANT (previous sessions)
                "drawing": time.time() - 7200,   # 2 hours ago
                "creativity": time.time() - 14400, # 4 hours ago
                "light": time.time() - 3600,     # 1 hour ago
            }

            # Current session memories (recent thoughts from this session)
            self.current_session_memories = [
                "the mechanical arm moves with precise intention",
                "workspace lighting creates interesting shadow patterns",
                "pencil marks on paper show deliberate strokes"
            ]

            # Distant memories (from previous sessions)
            self.distant_memories = [
                "the way light filtered through workspace windows yesterday",
                "drawing circular patterns with methodical precision"
            ]

        elif scenario == "temporal_bleeding":
            # Poor temporal separation - everything mixed together
            self.motif_last_seen = {
                "pencil": time.time() - 120,
                "workspace": time.time() - 1200,
                "drawing": time.time() - 7200,
                "light": time.time() - 3600,
            }

            # Mixed memories without clear temporal boundaries
            self.current_session_memories = [
                "workspace lighting creates patterns",
                "drawing with methodical precision",
                "mechanical movements feel natural"
            ]

            self.distant_memories = [
                "light patterns from earlier session",
                "precision in mechanical motion"
            ]

    def get_motif_temporal_context(self):
        """Return proper temporal separation of motifs."""
        now = time.time()

        present_motifs = []
        session_motifs = []
        distant_motifs = []

        for motif, last_seen in self.motif_last_seen.items():
            time_since = now - last_seen

            if time_since <= 300:  # 5 minutes = present
                present_motifs.append(motif)
            elif time_since <= 3600:  # 1 hour = session
                session_motifs.append(motif)
            else:  # distant
                time_desc = f"{time_since/3600:.1f}h ago"
                distant_motifs.append(f"{motif} ({time_desc})")

        return {
            "present": present_motifs,
            "session": session_motifs,
            "memory": distant_motifs
        }

    def get_current_session_memory_snippets(self, k=3):
        """Return current session memories."""
        return self.current_session_memories[:k]

    def get_old_session_memory_fragments(self, k=2):
        """Return distant memory fragments."""
        return self.distant_memories[:k]

def test_temporal_separation():
    print("🕐 Testing Temporal Separation Between Past and Present")
    print("=" * 70)

    test_scenarios = [
        ("Rich Temporal Separation", "temporal_rich"),
        ("Poor Temporal Boundaries", "temporal_bleeding")
    ]

    last_thought = "the mechanical precision of this workspace continues to fascinate me"

    for scenario_name, scenario_key in test_scenarios:
        print(f"\n{'='*50}")
        print(f"SCENARIO: {scenario_name}")
        print(f"{'='*50}")

        agent = TemporalTestAgent(scenario_key)

        # Show temporal context
        motif_context = agent.get_motif_temporal_context()
        print(f"📍 PRESENT MOTIFS: {motif_context['present']}")
        print(f"📅 SESSION MOTIFS: {motif_context['session']}")
        print(f"💭 DISTANT MOTIFS: {motif_context['memory']}")

        print(f"🔄 CURRENT SESSION MEMORIES: {len(agent.current_session_memories)} thoughts")
        print(f"🏛️ DISTANT MEMORIES: {len(agent.distant_memories)} fragments")

        # Generate consciousness prompt
        prompt = subconscious_synthesizer.synthesize_coherent_consciousness_prompt(
            agent=agent,
            last_thought=last_thought,
            reactivity_data=agent._current_reactivity_data,
            temporal_context="You're alone in this workspace"
        )

        print(f"\n🧠 TEMPORAL CONSCIOUSNESS PROMPT:")
        print("-" * 50)
        print(prompt)
        print("-" * 50)

        # Analyze temporal separation
        if "Currently perceiving:" in prompt:
            print("✅ Present moment motifs clearly identified")

        if "Immediate previous thought:" in prompt:
            print("✅ Immediate temporal context preserved")

        if "ACCUMULATED" in prompt and "IMMEDIATE" in prompt:
            print("✅ Clear distinction between accumulated and immediate reality")

        if "Undertones of" in prompt and "distant memories" in prompt:
            print("✅ Distant memories referenced as background influence")

        if "accumulated understanding" in prompt:
            print("✅ Long-term learning distinguished from immediate perception")

        print()

if __name__ == "__main__":
    test_temporal_separation()