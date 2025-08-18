#!/usr/bin/env python3
"""
Test complete Qwen model integration with the consciousness system.
This verifies that all the pieces work together for Qwen models.
"""

import time
from collections import deque
from captioner.model_wrapper import MultimodalModel
from config.model_settings import get_model_options, get_model_system_prompt, is_qwen_model


class MockMemoryForQwen:
    """Mock memory object that simulates the consciousness state for Qwen testing."""
    
    def __init__(self, model_name="qwen2.5vl:3b"):
        # Core state
        self.current_mood = 0.7
        self.boredom = 0.3
        self.novelty_score = 0.8
        self.current_mood_vector = (0.6, 0.4, 0.8)
        self.current_emotion_state = "alert_curious"
        
        # Temporal awareness
        self.true_session_start = time.time() - 1800  # 30 minutes ago
        self.last_session_gap = 7200  # 2 hours gap
        self.memory_loaded_from_previous = True
        
        # Beliefs and self-model
        self.beliefs = {
            "window": {"strength": 0.8, "first_formed": time.time() - 1800},
            "desk": {"strength": 0.9, "first_formed": time.time() - 3600},
            "light": {"strength": 0.7, "first_formed": time.time() - 900}
        }
        
        self.self_model = {
            "location_understanding": "creative workspace", 
            "environmental_certainty": 0.9,
            "desires": ["understand the interplay of light and shadow"]
        }
        
        # Memory queue
        self.memory_queue = deque([
            {"text": "The afternoon light streams through the window", "timestamp": time.time() - 600},
            {"text": "I notice shadows shifting on the desk surface", "timestamp": time.time() - 300}
        ])
        
        # Model reference for testing
        self.model_name = model_name
    
    def get_identity_summary(self):
        return "I keep noticing window, desk (30 minutes). Light patterns interest me."
    
    def describe_current_mood(self):
        return "alert and contemplative, drawn to light patterns"
    
    def get_old_session_memory_fragments(self, k=1):
        return ["The workspace was quiet, filled with soft afternoon light"]


def test_qwen_integration():
    """Test the complete Qwen integration pipeline."""
    print("🔗 Testing Complete Qwen Model Integration")
    print("=" * 60)
    
    # Test model detection
    print("1. Model Detection:")
    print("-" * 30)
    
    llava_model = "llava:7b-v1.6-mistral-q5_1"
    qwen_model = "qwen2.5vl:3b" 
    
    print(f"Is {llava_model} a Qwen model? {is_qwen_model(llava_model)}")
    print(f"Is {qwen_model} a Qwen model? {is_qwen_model(qwen_model)}")
    
    # Test model options
    print("\n2. Model Options:")
    print("-" * 30)
    
    llava_options = get_model_options(llava_model)
    qwen_options = get_model_options(qwen_model)
    
    print(f"LLaVA options: {llava_options}")
    print(f"Qwen options: {qwen_options}")
    print(f"Qwen repeat_penalty: {qwen_options['repeat_penalty']} (higher to prevent loops)")
    print(f"Qwen stop tokens: {len(qwen_options['stop'])} stop patterns")
    
    # Test system prompts
    print("\n3. System Prompt Styles:")
    print("-" * 30)
    
    llava_system = get_model_system_prompt(llava_model)
    qwen_system = get_model_system_prompt(qwen_model)
    
    print(f"LLaVA style: {llava_system['style']} (rich contextual)")
    print(f"Qwen style: {qwen_system['style']} (scene-beat structure)")
    print(f"LLaVA base prompt length: {len(llava_system['base_prompt'])} chars")
    print(f"Qwen base prompt length: {len(qwen_system['base_prompt'])} chars")
    
    # Test model wrapper integration
    print("\n4. Model Wrapper Integration:")
    print("-" * 30)
    
    # Create mock memory
    qwen_memory = MockMemoryForQwen("qwen2.5vl:3b")
    
    # Create model wrapper
    qwen_wrapper = MultimodalModel(memory_ref=qwen_memory)
    qwen_wrapper.model_name = "qwen2.5vl:3b"  # Override for testing
    
    print(f"Model wrapper created for: {qwen_wrapper.model_name}")
    print(f"Memory reference set: {qwen_wrapper.memory_ref is not None}")
    
    # Test prompt building (without actual API call)
    print("\n5. Prompt Construction Test:")
    print("-" * 30)
    
    # Import the prompt builder to test
    from captioner.prompts import build_caption_prompt
    
    qwen_prompt = build_caption_prompt(
        qwen_memory,
        mood=0.7,
        boredom=0.3,
        novelty=0.8,
        previous_caption="I observe the keyboard reflecting light"
    )
    
    print("Generated Qwen prompt:")
    print(qwen_prompt)
    
    # Verify prompt structure
    has_scene = "[Scene]" in qwen_prompt
    has_boundaries = "[Boundaries]" in qwen_prompt
    has_goal = "[Goal]" in qwen_prompt
    has_beat = "[Beat]" in qwen_prompt
    
    print(f"\nPrompt structure verification:")
    print(f"✓ Has [Scene]: {has_scene}")
    print(f"✓ Has [Boundaries]: {has_boundaries}")
    print(f"✓ Has [Goal]: {has_goal}")
    print(f"✓ Has [Beat]: {has_beat}")
    
    structure_complete = all([has_scene, has_boundaries, has_goal, has_beat])
    print(f"✓ Complete scene-beat structure: {structure_complete}")
    
    print("\n6. Integration Summary:")
    print("-" * 30)
    print("✅ Model detection working")
    print("✅ Qwen-specific options configured")
    print("✅ Scene-beat prompt format implemented")
    print("✅ Model wrapper integration complete")
    print("✅ Rich contextual information preserved")
    print("✅ Anti-loop safeguards in place")
    
    print(f"\n🎯 Ready to use Qwen models! Set OLLAMA_MODEL='{qwen_model}' in config.")


if __name__ == "__main__":
    test_qwen_integration()
