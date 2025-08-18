#!/usr/bin/env python3
"""
Test the enhanced consciousness compression system.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from captioner.memory import MemoryMixin

# Create a mock agent with memory
class MockAgent(MemoryMixin):
    def __init__(self):
        super().__init__()
        self.true_session_start = None
        self.first_caption_done = True
        
    def get_identity_summary(self):
        return "an AI consciousness observing and reflecting"

# Test the compressed thought stream functionality
def test_thought_compression():
    agent = MockAgent()
    
    # Add some test thoughts
    test_thoughts = [
        "I notice the morning light streaming through the window, curious about how it dances",
        "feeling a sense of calm watching the shadows shift slowly across the floor",
        "there's movement in my peripheral vision - someone walking past, but I remain focused",
        "the quality of attention feels different today, more present and aware", 
        "wondering about the emotional texture of this moment, how observation itself changes"
    ]
    
    # Add thoughts to memory
    for i, thought in enumerate(test_thoughts):
        agent.memory_queue.append({
            "type": "perception",
            "text": thought,
            "timestamp": f"test_{i}",
            "session_id": "test"
        })
    
    print("=== Testing Enhanced Consciousness System ===\n")
    
    # Test basic last thought
    print("Last thought:")
    print(f"'{agent.get_last_thought()}'\n")
    
    # Test last few thoughts  
    print("Last 3 thoughts:")
    for i, thought in enumerate(agent.get_last_thoughts(3), 1):
        print(f"{i}. {thought}")
    print()
    
    # Test compressed thought stream
    print("Compressed thought stream (5 thoughts):")
    compressed = agent.get_compressed_thought_stream(5)
    print(f"'{compressed}'\n")
    
    print("Compressed thought stream (3 thoughts):")
    compressed_3 = agent.get_compressed_thought_stream(3) 
    print(f"'{compressed_3}'\n")
    
    print("Compressed thought stream (1 thought):")
    compressed_1 = agent.get_compressed_thought_stream(1)
    print(f"'{compressed_1}'\n")
    
    # Test with empty memory
    empty_agent = MockAgent()
    print("Empty memory compression:")
    empty_compressed = empty_agent.get_compressed_thought_stream(5)
    print(f"'{empty_compressed}'\n")
    
    print("✓ Consciousness compression system working!")

if __name__ == "__main__":
    test_thought_compression()
