#!/usr/bin/env python3
"""
Simple test of the enhanced subconscious integration.
"""
import os
import sys
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from captioner.prompts import _build_semantic_bridge

class MockAgent:
    def __init__(self):
        self.true_session_start = time.time() - 600  # 10 minutes ago
        self.beliefs = {"lighting_patterns": {"strength": 0.85}}
        self.self_model = {
            "desires": ["I want to understand light"],
            "doubts": [{"text": "whether this is real", "timestamp": time.time()}]
        }
        self._current_reactivity_data = {"activity_level": 0.15}

def main():
    print("✨ Testing Simple Enhanced Integration")
    print("=" * 50)

    agent = MockAgent()
    last_thought = "this lighting seems different today, wondering what changed"
    social_context = "You're alone in the workspace"

    # Test the enhanced semantic bridge directly
    result = _build_semantic_bridge(last_thought, social_context, agent)

    print("=== ENHANCED SEMANTIC BRIDGE ===")
    print(result)

    # Check both components are present
    if "CONTINUE YOUR PREVIOUS THOUGHT:" in result:
        print("\n✅ Core continuation mechanism preserved")
    else:
        print("\n❌ Core mechanism missing")

    if "SUBCONSCIOUS CONTEXT:" in result:
        print("✅ Subconscious guidance added")
    else:
        print("❌ Subconscious guidance missing")

if __name__ == "__main__":
    main()