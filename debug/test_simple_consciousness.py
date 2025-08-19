#!/usr/bin/env python3
"""
Test the new simplified consciousness system (MVC approach).
"""

import time
from captioner.prompts import build_simple_caption_prompt


class MockMemory:
    """Mock memory for testing."""
    def get_top_motifs(self, k):
        return ["ceiling_damage", "light_fixtures", "desk_activity"]


class MockAgent:
    """Mock agent for testing."""
    
    def __init__(self):
        self.true_session_start = time.time() - 1800  # 30 minutes ago
        self.emotional_journey = ["calm", "alert", "engaged", "curious"]
        self.memory_ref = MockMemory()


def test_simple_consciousness():
    """Test the new simplified consciousness prompt."""
    
    print("=== Testing Simplified Consciousness System ===\n")
    
    # Create mock agent
    agent = MockAgent()
    
    # Test different mood states
    test_cases = [
        {
            "name": "Energized and Engaged",
            "mood_vector": (0.8, 0.7, 0.9),
            "last_caption": "The person at the desk is typing rapidly, creating a rhythmic sound."
        },
        {
            "name": "Withdrawn and Distant", 
            "mood_vector": (-0.5, 0.2, 0.3),
            "last_caption": "The room feels quiet and still."
        },
        {
            "name": "Curious and Alert",
            "mood_vector": (0.3, 0.6, 0.8),
            "last_caption": "I notice the way light reflects off the ceiling tiles."
        }
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"--- Test Case {i}: {test['name']} ---")
        
        prompt = build_simple_caption_prompt(
            agent, 
            test['mood_vector'], 
            test['last_caption']
        )
        
        print("Generated Prompt:")
        print(prompt)
        print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    test_simple_consciousness()
