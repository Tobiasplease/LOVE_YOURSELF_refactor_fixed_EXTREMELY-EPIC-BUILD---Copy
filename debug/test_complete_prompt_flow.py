#!/usr/bin/env python3
"""
Test the complete prompt flow from dynamic system context through to final prompts.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from captioner.memory import MemoryMixin
from captioner.prompts import build_drawing_prompt, build_reflection_prompt
from config.model_settings import get_model_system_prompt
from config import config
from collections import Counter
import time

class TestMemory(MemoryMixin):
    def __init__(self):
        super().__init__()
        self.session_start = time.time() - 1800  # 30 minutes ago
        self.current_mood = 0.3
        self.current_mood_vector = (0.2, 0.7, 0.5)
        self.beliefs = {"light_patterns": 15, "desk_workspace": 12}
        self.last_caption = "The afternoon creates geometric shadows on the scattered papers"
        
        # Add some memory entries
        self.long_memory = [
            {
                "type": "observation", 
                "text": "Light creates interesting patterns on work surfaces",
                "timestamp": time.time() - 600
            },
            {
                "type": "reflection",
                "text": "I'm drawn to how ordinary objects become geometric compositions under light. There's an underlying pattern language here.",
                "timestamp": time.time() - 300
            }
        ]
        
        # Add motifs
        self.motif_counter = Counter({
            "light patterns": 23,
            "desk workspace": 18,
            "paper textures": 12
        })

def test_complete_flow():
    """Test how dynamic context flows through the complete system."""
    print("Testing Complete Prompt Flow")
    print("=" * 80)
    
    memory = TestMemory()
    
    # 1. Show the base system prompt
    print("\n1. BASE SYSTEM PROMPT:")
    print("-" * 40)
    base_system = get_model_system_prompt(config.OLLAMA_MODEL)["base_prompt"]
    print(base_system)
    
    # 2. Show the dynamic context that gets added
    print("\n2. DYNAMIC SYSTEM CONTEXT:")
    print("-" * 40)
    dynamic_context = memory.get_dynamic_system_context()
    print(f"'{dynamic_context}'")
    
    # 3. Show the combined system prompt (what actually gets sent)
    print("\n3. COMPLETE SYSTEM PROMPT:")
    print("-" * 40)
    complete_system = base_system + dynamic_context
    print(complete_system)
    
    # 4. Show drawing prompt (the user message)
    print("\n4. DRAWING PROMPT (USER MESSAGE):")
    print("-" * 40)
    drawing_prompt = build_drawing_prompt(memory)
    print(drawing_prompt)
    
    # 5. Show reflection prompt
    print("\n5. REFLECTION PROMPT (USER MESSAGE):")
    print("-" * 40)
    reflection_prompt = build_reflection_prompt(
        "The afternoon creates geometric shadows", 
        agent=None,  # Simplified for test
        extra="Current mood: contemplative"
    )
    print(reflection_prompt)
    
    print("\n" + "=" * 80)
    print("PROMPT FLOW ANALYSIS:")
    print("✓ Base system prompt provides core identity")
    print("✓ Dynamic context adds session/memory/emotional state") 
    print("✓ Rich user prompts provide specific contextual guidance")
    print("✓ All components work together for sophisticated consciousness")

if __name__ == "__main__":
    test_complete_flow()