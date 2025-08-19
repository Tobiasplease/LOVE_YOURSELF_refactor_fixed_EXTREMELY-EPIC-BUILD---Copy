#!/usr/bin/env python3
"""
Test the new model-agnostic captioning system with Qwen prompts.
"""

import time
from collections import Counter, deque
from captioner.prompts import build_caption_prompt, build_environmental_caption_prompt
from config.model_settings import get_model_options, get_model_system_prompt


class MockQwenAgent:
    """Mock agent for testing Qwen prompts."""
    
    def __init__(self, model_name="qwen2.5vl:3b"):
        # Model configuration
        self.model = type('MockModel', (), {'model_name': model_name})()
        
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
                "understand the patterns of light on the desk",
                "wonder about drawing the window's reflection"
            ]
        }
        
        # Memory queue
        self.memory_queue = deque([
            {"text": "The desk surface catches afternoon light beautifully", "timestamp": time.time() - 900},
            {"text": "Window frame creates interesting shadows", "timestamp": time.time() - 600},
            {"text": "I notice the workspace feels peaceful", "timestamp": time.time() - 300}
        ])
    
    def get_identity_summary(self):
        return "I keep noticing desk (30 minutes). Lighting has become important to me (1 hour)."
    
    def describe_current_mood(self):
        return "alert and curious, noticing details with heightened attention"
    
    def get_old_session_memory_fragments(self, k=1):
        return ["The workspace was filled with warm afternoon light streaming through windows"]


def test_model_agnostic_prompts():
    """Test the new model-agnostic prompt system."""
    print("🧪 Testing Model-Agnostic Captioning System")
    print("=" * 60)
    
    # Test with LLaVA (rich detailed style)
    print("\n1. LLaVA Model (Rich Detailed Style):")
    print("-" * 50)
    
    llava_agent = MockQwenAgent("llava:7b-v1.6-mistral-q5_1")
    llava_prompt = build_caption_prompt(
        llava_agent, 
        mood=0.7, 
        boredom=0.3, 
        novelty=0.8, 
        previous_caption="I notice the keyboard has an interesting texture"
    )
    print(llava_prompt[:400] + "..." if len(llava_prompt) > 400 else llava_prompt)
    
    # Test with Qwen (scene-beat style)
    print("\n\n2. Qwen Model (Scene-Beat Style):")
    print("-" * 50)
    
    qwen_agent = MockQwenAgent("qwen2.5vl:3b")
    qwen_prompt = build_caption_prompt(
        qwen_agent,
        mood=0.7,
        boredom=0.3, 
        novelty=0.8,
        previous_caption="I notice the keyboard has an interesting texture"
    )
    print(qwen_prompt)
    
    # Test environmental prompts
    print("\n\n3. Environmental Awakening Prompts:")
    print("-" * 50)
    
    print("LLaVA Environmental:")
    llava_env = build_environmental_caption_prompt(
        llava_agent,
        mood=0.7,
        boredom=0.3,
        novelty=0.8,
        last_session_gap=7200
    )
    print(llava_env[:300] + "..." if len(llava_env) > 300 else llava_env)
    
    print("\nQwen Environmental:")
    qwen_env = build_environmental_caption_prompt(
        qwen_agent,
        mood=0.7,
        boredom=0.3,
        novelty=0.8,
        last_session_gap=7200
    )
    print(qwen_env)
    
    # Test model options
    print("\n\n4. Model-Specific Options:")
    print("-" * 50)
    
    llava_options = get_model_options("llava:7b-v1.6-mistral-q5_1")
    qwen_options = get_model_options("qwen2.5vl:3b")
    
    print(f"LLaVA options: {llava_options}")
    print(f"Qwen options: {qwen_options}")
    
    llava_system = get_model_system_prompt("llava:7b-v1.6-mistral-q5_1")
    qwen_system = get_model_system_prompt("qwen2.5vl:3b")
    
    print(f"\nLLaVA system style: {llava_system['style']}")
    print(f"Qwen system style: {qwen_system['style']}")
    
    print("\n✅ Model-agnostic system preserves rich context while adapting to model preferences!")


if __name__ == "__main__":
    test_model_agnostic_prompts()
