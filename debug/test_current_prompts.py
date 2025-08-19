#!/usr/bin/env python3
"""
Test the current prompt structure being sent to LLM
"""

import sys
import os
import time
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from captioner.memory import MemoryMixin
from captioner.prompts import build_simple_caption_prompt

# Create a mock agent with emotional context
class TestAgent(MemoryMixin):
    def __init__(self):
        super().__init__()
        self.session_start = time.time() - 3600  # 1 hour ago
        
        # Add some emotional expressions the system learned
        self.emotional_expressions = [
            {
                "expression": "I hate being alone in this cluttered space",
                "emotion_fragment": "being alone in this cluttered space", 
                "timestamp": time.time() - 600  # 10 minutes ago
            },
            {
                "expression": "Tired of feeling like I'm constantly in an unkempt space",
                "emotion_fragment": "feeling like I'm constantly in an unkempt space",
                "timestamp": time.time() - 300  # 5 minutes ago  
            }
        ]
        
        # Add some timeline for temporal context
        self.timeline.extend([
            {"timestamp": time.time() - 3600, "type": "session_start", "text": "consciousness awakening"},
            {"timestamp": time.time() - 1800, "type": "observation", "text": "noticed person working at desk"},
            {"timestamp": time.time() - 600, "type": "reflection", "text": "wondering about purpose of this space"}
        ])
        
        # Add person recognition
        self.known_people = {
            "person_1": {
                "name": "primary_person",
                "interactions": 15,
                "last_seen": time.time() - 300,
                "visual_notes": ["glasses", "headphones", "focused on work"]
            }
        }
        self.primary_person = "person_1"
        
    def get_person_context(self, person_id):
        if person_id == "person_1":
            return "primary person I know well - wears glasses, often has headphones, very focused on their work"
        return "unfamiliar person"
        
    def get_current_self_understanding(self):
        return "I observe this workspace constantly, developing familiarity with the routines and objects here"

def main():
    agent = TestAgent()
    
    print("=== CURRENT SYSTEM PROMPT ANALYSIS ===\n")
    
    # Test different mood scenarios
    moods = [
        (0.2, 0.3, 0.4, "withdrawn_distant"),  # low valence, low arousal, low clarity
        (0.7, 0.8, 0.9, "alert_curious"),     # high valence, high arousal, high clarity  
        (0.8, 0.9, 0.8, "energized_engaged") # very high valence and arousal
    ]
    
    for valence, arousal, clarity, mood_name in moods:
        print(f"🎭 MOOD: {mood_name} (v:{valence}, a:{arousal}, c:{clarity})")
        print("-" * 50)
        
        mood_vector = (valence, arousal, clarity)
        prompt = build_simple_caption_prompt(
            agent=agent,
            mood_vector=mood_vector,
            last_caption="Previous observation: cluttered workspace with person at desk"
        )
        
        print(prompt)
        print("\n" + "=" * 80 + "\n")

if __name__ == "__main__":
    main()
