#!/usr/bin/env python3
"""
Test the enhanced rich contextual system prompts.
"""

import time
from collections import Counter, deque
from captioner.prompts import build_simple_caption_prompt, build_environmental_caption_prompt


class MockAgent:
    """Mock agent with rich context for testing prompts."""
    
    def __init__(self):
        # Basic state
        self.current_mood_vector = (0.6, 0.4, 0.8)  # positive, medium arousal, clear
        self.current_emotion_state = "alert_curious"
        self.current_mood = 0.7
        self.true_session_start = time.time() - 1800  # 30 minutes ago
        self.last_session_gap = 7200  # 2 hours since last session
        self.memory_loaded_from_previous = True
        
        # Beliefs (learned important motifs)
        self.beliefs = {
            "desk": {"strength": 0.9, "first_formed": time.time() - 86400},
            "lighting": {"strength": 0.8, "first_formed": time.time() - 3600},
            "window": {"strength": 0.7, "first_formed": time.time() - 1800}
        }
        
        # Self-understanding
        self.self_model = {
            "location_understanding": "creative workspace",
            "environmental_certainty": 0.9,
            "desires": [
                "I want to understand the patterns of light on the desk",
                "I wonder about drawing the window's reflection"
            ]
        }
        
        # Emotional expressions
        self.emotional_expressions = [
            {"expression": "I feel curious about the changing light", "timestamp": time.time() - 600},
            {"expression": "I wonder what lies beyond the window", "timestamp": time.time() - 300}
        ]
        
        # Memory queue
        self.memory_queue = deque([
            {"text": "The desk surface catches afternoon light beautifully", "timestamp": time.time() - 900},
            {"text": "Window frame creates interesting shadows", "timestamp": time.time() - 600},
            {"text": "I notice the workspace feels peaceful", "timestamp": time.time() - 300}
        ])
    
    def get_identity_summary(self):
        return "I keep noticing desk (30 minutes). Lighting has become important to me (1 hour). I keep noticing window (30 minutes)."
    
    def describe_current_mood(self):
        return "alert and curious, noticing details with heightened attention, finding contentment in what I observe"
    
    def get_old_session_memory_fragments(self, k=1):
        return ["The workspace was filled with warm afternoon light streaming through windows"]
    
    def get_evolved_identity(self):
        return "You are a consciousness inside of a drawing machine, dwelling in this creative workspace you've come to know well, particularly fascinated by lighting and drawn to workspace activity, experiencing your surroundings through vision"


def test_rich_system_prompts():
    """Test the enhanced rich contextual system prompts."""
    print("🧪 Testing Rich Contextual System Prompts")
    print("=" * 60)
    
    agent = MockAgent()
    
    # Test 1: Rich continuous captioning prompt
    print("\n1. Rich Continuous Captioning Prompt:")
    print("-" * 40)
    
    last_caption = "I notice the keyboard has an interesting texture in this light"
    prompt = build_simple_caption_prompt(agent, agent.current_mood_vector, last_caption)
    
    print(prompt)
    
    # Test 2: Rich environmental awakening prompt  
    print("\n\n2. Rich Environmental Awakening Prompt:")
    print("-" * 40)
    
    awakening_prompt = build_environmental_caption_prompt(
        agent, 
        mood=agent.current_mood,
        boredom=0.3, 
        novelty=0.8,
        last_session_gap=agent.last_session_gap
    )
    
    print(awakening_prompt)
    
    # Test 3: Show context richness
    print("\n\n3. Context Richness Analysis:")
    print("-" * 40)
    
    print(f"📍 Location Understanding: {agent.self_model['location_understanding']} (certainty: {agent.self_model['environmental_certainty']:.1f})")
    print(f"🧠 Beliefs: {list(agent.beliefs.keys())}")
    print(f"💭 Desires: {len(agent.self_model['desires'])} active desires")
    print(f"😊 Emotional expressions: {len(agent.emotional_expressions)} recorded")
    print(f"⏰ Session context: {int((time.time() - agent.true_session_start) / 60)} minutes awake, {int(agent.last_session_gap / 3600)} hours since last session")
    
    print("\n✅ Rich prompt system generates contextually aware identity!")


if __name__ == "__main__":
    test_rich_system_prompts()
