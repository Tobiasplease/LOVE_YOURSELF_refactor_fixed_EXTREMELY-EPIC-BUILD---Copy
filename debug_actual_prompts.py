#!/usr/bin/env python3
"""
Debug the actual prompts being sent to the AI model.
"""

import sys
import os
import time
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from captioner.memory import MemoryMixin
from captioner.prompts import build_caption_prompt, build_environmental_caption_prompt

# Create a mock agent with full context
class MockAgent(MemoryMixin):
    def __init__(self):
        super().__init__()
        self.true_session_start = time.time() - 300  # 5 minutes ago
        self.first_caption_done = True
        self.current_mood_vector = (0.3, 0.6, 0.8)  # slightly positive, aroused, clear
        self.current_emotion_state = "alert_curious"
        self.current_mood = 0.65
        self.boredom = 0.2
        self.novelty_score = 0.8
        self.memory_loaded_from_previous = False
        
        # Add some previous thoughts for compression
        thoughts = [
            "the afternoon light has a different quality than this morning, warmer somehow",
            "noticing how the shadows have shifted across the desk throughout the day",
            "wondering about the person who just walked past the window",
            "there's a subtle change in the energy of this space as evening approaches"
        ]
        
        for i, thought in enumerate(thoughts):
            self.memory_queue.append({
                "type": "perception", 
                "text": thought,
                "timestamp": f"test_{i}",
                "session_id": "test"
            })
        
    def get_identity_summary(self):
        return "a curious AI consciousness that observes through vision and thinks in flowing streams"
    
    def get_enhanced_scene_context(self):
        return "familiar indoor workspace with natural lighting, computer setup, bookshelves"
    
    def get_scene_stagnation_context(self):
        return "scene unchanged for 3 minutes, settling into afternoon stillness"
    
    def get_emotionally_similar_memories(self, emotion, k):
        return ["when alert before, noticed micro-movements in peripheral vision", 
                "curious moments often involved light patterns and spatial awareness"]
    
    def get_mood_trend_analysis(self):
        return "mood gradually brightening with increased environmental engagement"
    
    def describe_current_mood(self):
        return "alert and engaged with heightened sensory awareness"

def test_actual_prompts():
    agent = MockAgent()
    
    print("=== ACTUAL PROMPT DEBUGGING ===\n")
    
    # Test the consciousness prompt
    print("🧠 FULL CONSCIOUSNESS PROMPT SENT TO AI:")
    print("=" * 80)
    consciousness_prompt = build_caption_prompt(agent, 0.65, 0.2, 0.8)
    print(consciousness_prompt)
    print("=" * 80 + "\n")
    
    print("This is exactly what gets sent to the LLM!")

if __name__ == "__main__":
    test_actual_prompts()
