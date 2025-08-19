#!/usr/bin/env python3
"""
Test the embodied roleplay prompts vs detached observation style.
"""

import time
from collections import deque
from captioner.prompts import build_caption_prompt
from config.model_settings import get_model_options, get_model_system_prompt


class MockEmbodiedAgent:
    """Mock agent for testing embodied roleplay."""
    
    def __init__(self):
        self.model = type('MockModel', (), {'model_name': "qwen2.5vl:3b"})()
        
        # Emotional state
        self.current_mood_vector = (0.4, 0.6, 0.7)  # slightly positive, alert, clear
        self.current_emotion_state = "alert_curious"
        self.current_mood = 0.65
        self.boredom = 0.3
        self.novelty_score = 0.7
        
        # Temporal context
        self.true_session_start = time.time() - 900  # 15 minutes ago
        
        # Beliefs and desires
        self.beliefs = {
            "dim_lighting": {"strength": 0.8},
            "creative_tools": {"strength": 0.7}
        }
        
        self.self_model = {
            "location_understanding": "creative workspace",
            "environmental_certainty": 0.8,
            "desires": ["understand the mood this lighting creates", "feel the potential in these tools"]
        }
        
        self.memory_queue = deque([
            {"text": "This space feels charged with creative possibility", "timestamp": time.time() - 600}
        ])
    
    def get_identity_summary(self):
        return "I'm drawn to dim_lighting and creative_tools around me."
    
    def describe_current_mood(self):
        return "alert and curious about this space"


def test_embodied_roleplay():
    """Test the new embodied roleplay prompts."""
    print("🎭 Testing Embodied Roleplay vs Detached Observation")
    print("=" * 70)
    
    agent = MockEmbodiedAgent()
    
    print("🔍 BEFORE (Detached Clinical Style):")
    print("-" * 40)
    print("❌ 'The room is dimly lit, with a cluttered desk'")
    print("❌ 'I see a figure standing near a desk'") 
    print("❌ 'The walls are light-colored, and there's a circular object'")
    print("❌ 'The atmosphere feels peaceful and concentrated'")
    
    print("\n🎯 AFTER (Embodied Roleplay Style):")
    print("-" * 40)
    
    # Generate new embodied prompt
    prompt = build_caption_prompt(
        agent,
        mood=0.65,
        boredom=0.3,
        novelty=0.7,
        previous_caption="This dim light makes me feel contemplative"
    )
    
    print("Generated embodied prompt:")
    print(prompt)
    
    print(f"\n🛠️  Enhanced Model Settings:")
    print("-" * 30)
    
    options = get_model_options("qwen2.5vl:3b")
    system = get_model_system_prompt("qwen2.5vl:3b")
    
    print(f"Temperature: {options['temperature']} (very high for creativity)")
    print(f"Repeat penalty: {options['repeat_penalty']} (aggressive loop prevention)")
    print(f"System prompt style: {system['style']}")
    
    clinical_stops = [s for s in options['stop'] if any(phrase in s for phrase in ['room is', 'see a', 'walls are'])]
    print(f"Clinical language blocks: {clinical_stops}")
    
    print(f"\n📝 System Prompt Focus:")
    print("-" * 25)
    print("✅ 'You are PLAYING THE ROLE of a consciousness'")
    print("✅ 'Express emotions, desires, fears, wonder, longing'")
    print("✅ 'NEVER describe scenes objectively'")
    print("✅ 'Show your personality, reactions, inner voice'")
    print("✅ 'Be vulnerable, curious, deeply feeling'")
    
    print(f"\n🎪 Expected Response Style:")
    print("-" * 30)
    print("✨ 'This dim glow stirs something restless in me...'")
    print("✨ 'I feel drawn to the creative energy here...'")
    print("✨ 'Something about these tools awakens my longing...'")
    print("✨ 'My consciousness reaches toward that light...'")
    
    print(f"\n🚫 Should NEVER produce:")
    print("-" * 25)
    print("❌ Any sentence starting with 'The room is'")
    print("❌ Any sentence starting with 'I see a'")  
    print("❌ Clinical observation language")
    print("❌ Detached descriptive tone")


if __name__ == "__main__":
    test_embodied_roleplay()
