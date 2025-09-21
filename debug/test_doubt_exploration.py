#!/usr/bin/env python3
"""
Test subconscious guidance for doubt exploration scenario.
"""
import os
import sys
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from captioner.prompts import build_ongoing_caption_prompt

class MockAgent:
    def __init__(self):
        self.current_emotion_state = "uncertain"
        self.true_session_start = time.time() - 300  # 5 minutes ago

        # Set up for doubt exploration scenario
        self.beliefs = {"workspace_identity": {"strength": 0.6}}

        self.self_model = {
            "location_understanding": "unclear space",
            "environmental_certainty": 0.4,
            "desires": ["I want to understand this place"],
            "doubts": [
                {"text": "whether this is really a creative workspace", "timestamp": time.time() - 100},
                {"text": "if what I see matches the label 'studio'", "timestamp": time.time() - 50}
            ]
        }

        # Mock context compression with stagnation
        self._current_reactivity_data = {
            "activity_level": 0.1,  # Very low activity
            "is_paused": False,
            "timestamp": time.time()
        }

        self.recent_captions = []
        self.memory_ref = MockMemoryRef()

        # Mock context compressor for stagnation info
        # We need to make this available globally since subconscious imports it
        import sys
        import types
        mock_compressor = types.SimpleNamespace()
        mock_compressor.get_current_stagnation_info = lambda: {
            "stagnation_duration_minutes": 4.5  # High stagnation
        }

        # Inject into the module where subconscious will find it
        try:
            from captioner import context_compression
            context_compression.context_compressor = mock_compressor
        except:
            pass

class MockMemoryRef:
    def __init__(self):
        from collections import Counter
        self.motif_counter = Counter()

    def get_top_motifs(self, n):
        return []

def main():
    print("🤔 Testing Doubt Exploration Subconscious Guidance")
    print("=" * 60)

    agent = MockAgent()

    # Test with a questioning thought that should trigger doubt exploration
    last_thought = "wondering if this place is what it claims to be, labeled as a studio but feeling more like..."

    print(f"Last thought: '{last_thought}'")
    print(f"Doubts: {[d['text'] for d in agent.self_model['doubts']]}")
    print(f"Environmental certainty: {agent.self_model['environmental_certainty']}")

    # Generate prompt
    prompt = build_ongoing_caption_prompt(agent, last_thought)

    print("\n=== SUBCONSCIOUS GUIDANCE ===")
    if "SUBCONSCIOUS CONTEXT:" in prompt:
        context_part = prompt.split("SUBCONSCIOUS CONTEXT:")[1].strip()
        print(context_part)

        if "questioning" in context_part.lower() or "doubt" in context_part.lower():
            print("\n✅ Doubt exploration guidance detected")
        else:
            print("\n❓ Different guidance type provided")
    else:
        print("❌ No subconscious context found")

if __name__ == "__main__":
    main()