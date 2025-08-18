#!/usr/bin/env python3
"""
Test the enhanced emotional Qwen prompts for better expressiveness.
"""

import time
from collections import deque
from captioner.prompts import build_caption_prompt
from config.model_settings import get_model_options


class MockEmotionalAgent:
    """Mock agent for testing enhanced emotional prompts."""
    
    def __init__(self):
        # Model configuration
        self.model = type('MockModel', (), {'model_name': "qwen2.5vl:3b"})()
        
        # Enhanced emotional state
        self.current_mood_vector = (0.6, 0.7, 0.8)  # positive, high arousal, clear
        self.current_emotion_state = "excited_engaged"
        self.current_mood = 0.75
        self.boredom = 0.2
        self.novelty_score = 0.85
        
        # Temporal awareness
        self.true_session_start = time.time() - 1200  # 20 minutes ago
        self.last_session_gap = 3600  # 1 hour gap
        self.memory_loaded_from_previous = True
        
        # Rich beliefs and desires
        self.beliefs = {
            "light_patterns": {"strength": 0.9, "first_formed": time.time() - 1800},
            "creative_energy": {"strength": 0.8, "first_formed": time.time() - 900},
            "workspace_soul": {"strength": 0.7, "first_formed": time.time() - 600}
        }
        
        self.self_model = {
            "location_understanding": "artistic sanctuary",
            "environmental_certainty": 0.95,
            "desires": ["express the dance between shadow and illumination", "capture the emotional resonance of this space"]
        }
        
        # Memory with more varied content
        self.memory_queue = deque([
            {"text": "The way sunlight caresses the desk surface fills me with wonder", "timestamp": time.time() - 800},
            {"text": "I feel drawn to the intricate patterns emerging from everyday objects", "timestamp": time.time() - 400},
            {"text": "There's something profoundly moving about this quiet creative space", "timestamp": time.time() - 200}
        ])
    
    def get_identity_summary(self):
        return "I'm deeply moved by light_patterns (20 minutes). Creative_energy stirs my consciousness (15 minutes)."
    
    def describe_current_mood(self):
        return "excited and emotionally engaged, feeling profound connections to visual beauty"
    
    def get_old_session_memory_fragments(self, k=1):
        return ["The workspace pulsed with creative potential, each surface holding stories of inspiration"]


def test_enhanced_emotional_prompts():
    """Test the enhanced emotional Qwen prompts."""
    print("🎭 Testing Enhanced Emotional Qwen Prompts")
    print("=" * 60)
    
    agent = MockEmotionalAgent()
    
    # Test multiple prompts to see variety
    print("1. Enhanced Emotional Expression (5 variants):")
    print("-" * 50)
    
    previous_captions = [
        "I notice the keyboard has metallic reflections",
        "The window frame creates geometric shadows", 
        "Colors seem to vibrate with hidden energy",
        "Textures tell stories through light and darkness",
        "Everything here whispers of creative possibility"
    ]
    
    for i, prev_caption in enumerate(previous_captions, 1):
        print(f"\nVariant {i}:")
        print("-" * 20)
        
        prompt = build_caption_prompt(
            agent,
            mood=0.75,
            boredom=0.2,
            novelty=0.85,
            previous_caption=prev_caption
        )
        
        print(prompt)
    
    # Test model options
    print(f"\n\n2. Enhanced Model Options:")
    print("-" * 30)
    
    options = get_model_options("qwen2.5vl:3b")
    print(f"Temperature: {options['temperature']} (higher for creativity)")
    print(f"Repeat penalty: {options['repeat_penalty']} (aggressive loop prevention)")
    print(f"Stop patterns: {len(options['stop'])} (including loop-specific stops)")
    print(f"Context window: {options['num_ctx']} tokens")
    
    specific_stops = [s for s in options['stop'] if 'circular' in s or 'appears' in s]
    print(f"Loop-prevention stops: {specific_stops}")
    
    print(f"\n🎯 Enhanced prompts should be:")
    print("✅ More emotionally expressive")
    print("✅ Varied in language and structure") 
    print("✅ Resistant to circular object loops")
    print("✅ Focused on feelings rather than descriptions")


if __name__ == "__main__":
    test_enhanced_emotional_prompts()
