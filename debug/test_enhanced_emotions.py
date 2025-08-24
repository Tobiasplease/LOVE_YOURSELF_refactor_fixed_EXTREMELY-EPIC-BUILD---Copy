#!/usr/bin/env python3
"""
Test the enhanced emotional progression system.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from captioner.prompts import mood_to_words
from collections import Counter
import time

class MockAgent:
    def __init__(self):
        self.true_session_start = time.time() - 3600  # 1 hour ago
        self.boredom = 0.3
        self.novelty_score = 0.5
        
        # Mock memory with motifs
        class MockMemory:
            def __init__(self):
                self.motif_counter = Counter({
                    "light patterns": 45,
                    "desk workspace": 32,
                    "ceiling details": 28
                })
            def get_top_motifs(self, k):
                return [motif for motif, count in self.motif_counter.most_common(k)]
        
        self.memory_ref = MockMemory()

def test_emotional_progressions():
    """Test different emotional states and progressions"""
    print("Testing Enhanced Emotional System")
    print("=" * 60)
    
    agent = MockAgent()
    
    # Test different mood vectors
    test_cases = [
        # (valence, arousal, clarity, description)
        (0.4, 0.6, 0.7, "Curious and alert"),
        (0.1, 0.2, 0.8, "Contemplative and clear"),
        (-0.3, 0.5, 0.4, "Frustrated and confused"),
        (-0.5, 0.1, 0.3, "Melancholic and withdrawn"),
        (0.7, 0.8, 0.6, "Highly energized"),
    ]
    
    print("\\nEMOTIONAL STATES WITH AGENT CONTEXT:")
    print("-" * 40)
    for valence, arousal, clarity, desc in test_cases:
        mood_vector = (valence, arousal, clarity)
        
        # Test without agent (old way)
        basic_emotion = mood_to_words(mood_vector)
        
        # Test with agent (new enhanced way)
        enhanced_emotion = mood_to_words(mood_vector, agent)
        
        print(f"{desc}:")
        print(f"  Basic:    {basic_emotion}")
        print(f"  Enhanced: {enhanced_emotion}")
        print()
    
    # Test energy effects
    print("\\nENERGY LEVEL EFFECTS:")
    print("-" * 40)
    
    # Test low energy (high boredom, high repetition)
    tired_agent = MockAgent()
    tired_agent.boredom = 0.8  # Very bored
    tired_agent.novelty_score = 0.1  # No novelty
    tired_agent.memory_ref.motif_counter = Counter({"same_thing": 150})  # Lots of repetition
    
    print("Low Energy State (bored, repetitive):")
    tired_emotion = mood_to_words((0.3, 0.5, 0.6), tired_agent)
    print(f"  Result: {tired_emotion}")
    print()
    
    # Test high energy (low boredom, high novelty)
    energized_agent = MockAgent()
    energized_agent.boredom = 0.0
    energized_agent.novelty_score = 0.9  # High novelty
    energized_agent.memory_ref.motif_counter = Counter({"new_thing": 5})  # Low repetition
    
    print("High Energy State (novel, engaging):")
    energized_emotion = mood_to_words((0.3, 0.5, 0.6), energized_agent)
    print(f"  Result: {energized_emotion}")
    print()
    
    # Test progression over time
    print("\\nTEMPORAL PROGRESSION:")
    print("-" * 40)
    
    # Short session
    short_agent = MockAgent()
    short_agent.true_session_start = time.time() - 600  # 10 minutes ago
    short_emotion = mood_to_words((0.1, 0.3, 0.7), short_agent)
    print(f"Short session (10 min): {short_emotion}")
    
    # Medium session  
    medium_agent = MockAgent()
    medium_agent.true_session_start = time.time() - 3600  # 1 hour ago
    medium_emotion = mood_to_words((0.1, 0.3, 0.7), medium_agent)
    print(f"Medium session (1 hour): {medium_emotion}")
    
    # Long session
    long_agent = MockAgent()
    long_agent.true_session_start = time.time() - 10800  # 3 hours ago
    long_emotion = mood_to_words((0.1, 0.3, 0.7), long_agent)
    print(f"Long session (3 hours): {long_emotion}")
    
    print("\\n" + "=" * 60)
    print("Enhanced emotional system test complete!")

if __name__ == "__main__":
    test_emotional_progressions()