#!/usr/bin/env python3
"""
Test how different mood states affect expression style.
"""

import sys
import os
import time
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from captioner.memory import MemoryMixin
from captioner.prompts import build_caption_prompt

class MockAgent(MemoryMixin):
    def __init__(self, mood_vector, emotion_state):
        super().__init__()
        self.true_session_start = time.time() - 300  # 5 minutes ago
        self.first_caption_done = True
        self.current_mood_vector = mood_vector
        self.current_emotion_state = emotion_state
        self.current_mood = 0.65
        self.boredom = 0.2
        self.novelty_score = 0.8
        
        # Add a test thought
        self.memory_queue.append({
            "type": "perception", 
            "text": "watching the afternoon light slowly shift across the workspace",
            "timestamp": "test",
            "session_id": "test"
        })
        
    def get_identity_summary(self):
        return "an artistic consciousness that finds meaning in visual details"

def test_mood_expression_styles():
    print("=== MOOD-DRIVEN EXPRESSION STYLES TEST ===\n")
    
    # Test different mood states
    mood_tests = [
        ((0.7, 0.8, 0.7), "energetic_engaged", "High positive, high energy"),
        ((0.4, 0.2, 0.8), "calm_observant", "Mild positive, calm, clear"),
        ((-0.4, 0.6, 0.5), "restless_watchful", "Negative, agitated"),
        ((-0.3, 0.1, 0.4), "withdrawn_distant", "Negative, low energy"),
        ((0.1, 0.3, 0.2), "uncertain_searching", "Neutral, unclear")
    ]
    
    for mood_vector, emotion_state, description in mood_tests:
        print(f"🎭 MOOD: {description}")
        print(f"   Vector: valence={mood_vector[0]:.1f}, arousal={mood_vector[1]:.1f}, clarity={mood_vector[2]:.1f}")
        print("-" * 50)
        
        agent = MockAgent(mood_vector, emotion_state)
        prompt = build_caption_prompt(agent, 0.65, 0.2, 0.8)
        
        # Extract just the expression style line
        lines = prompt.split('\n')
        for line in lines:
            if line.startswith('Expression style:'):
                print(f"   {line}")
                break
        print()

if __name__ == "__main__":
    test_mood_expression_styles()
