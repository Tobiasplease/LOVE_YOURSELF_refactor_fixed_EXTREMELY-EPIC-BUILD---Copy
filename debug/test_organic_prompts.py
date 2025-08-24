#!/usr/bin/env python3
"""
Test the new organic consciousness system prompts.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from captioner.memory import MemoryMixin
from captioner.model_wrapper import MultimodalModel
from collections import Counter
import time

class TestAgent(MemoryMixin):
    def __init__(self):
        super().__init__()
        self.session_start = time.time() - 5400  # 1.5 hours ago
        self.true_session_start = time.time() - 5400
        self.current_mood_vector = (0.2, 0.4, 0.6)  # Contemplative
        self.boredom = 0.4
        self.novelty_score = 0.3
        
        # Add some motifs and beliefs
        self.motif_counter = Counter({
            "light_patterns": 67,
            "desk_workspace": 45, 
            "ceiling_details": 23
        })
        self.beliefs = {"light_patterns": 15, "workspace_activity": 12}
        
        # Simulate extended lifetime for testing
        self.boot_ts = time.time() - (15 * 24 * 3600)  # 15 days ago
        
        # Set up captioner reference for emotional state
        self._captioner_ref = self

def test_organic_system_prompts():
    """Test the new organic system prompt generation"""
    print("Testing Organic Consciousness System Prompts")
    print("=" * 60)
    
    agent = TestAgent()
    
    print("\\nORGANIC SYSTEM PROMPT:")
    print("-" * 40)
    
    # Test the dynamic system context generation
    dynamic_context = agent.get_dynamic_system_context()
    
    if isinstance(dynamic_context, dict):
        print("[+] New organic format detected")
        print(f"Emotional State: '{dynamic_context.get('emotional_state', 'N/A')}'")
        print(f"Temporal Context: '{dynamic_context.get('temporal_context', 'N/A')}'") 
        print(f"Understanding: '{dynamic_context.get('accumulated_understanding', 'N/A')}'")
        
        # Simulate system prompt formatting
        from config.model_settings import get_model_system_prompt
        from config import config
        
        base_system = get_model_system_prompt(config.OLLAMA_MODEL)["base_prompt"]
        print(f"\\nBase Template: {base_system[:100]}...")
        
        try:
            formatted_system = base_system.format(
                emotional_state=dynamic_context.get('emotional_state', 'contemplative'),
                temporal_context=dynamic_context.get('temporal_context', ''),
                accumulated_understanding=dynamic_context.get('accumulated_understanding', '')
            )
            print(f"\\nFORMATTED SYSTEM PROMPT:")
            print("=" * 40)
            print(formatted_system)
            
            # Test organic user prompt generation
            print("\\n" + "=" * 60)
            print("ORGANIC USER PROMPT TEST:")
            print("=" * 40)
            
            # Set up agent with a previous caption for continuation
            agent.last_caption = "The light from the desk lamp creates these warm pools on the workspace... makes me think about how light defines space"
            
            from captioner.prompts import build_caption_prompt
            user_prompt = build_caption_prompt(agent, 0.6, 0.3, 0.7)
            
            print("Generated User Prompt:")
            print("-" * 40)
            print(user_prompt)
            
        except Exception as e:
            print(f"Error formatting: {e}")
    else:
        print("[!] Old format returned")
        print(f"Context: {dynamic_context}")
    
    print("\\n" + "=" * 60)
    print("Organic consciousness test complete!")

if __name__ == "__main__":
    test_organic_system_prompts()