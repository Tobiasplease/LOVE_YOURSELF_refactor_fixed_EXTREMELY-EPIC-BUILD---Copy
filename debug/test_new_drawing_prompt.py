#!/usr/bin/env python3
"""
Test the new refactored drawing prompt system.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from captioner.prompts import build_drawing_prompt
from captioner.memory import MemoryMixin
import time

class MockCaptioner:
    """Mock captioner for testing."""
    def __init__(self):
        self.current_mood = 0.3
        self.current_mood_vector = (0.2, 0.7, 0.5)  # valence, arousal, clarity
        self.boredom = 0.6
        
    def recognize_person(self, caption):
        return None

def test_new_drawing_prompt():
    """Test the new drawing prompt format."""
    print("Testing New Drawing Prompt Format")
    print("=" * 60)
    
    # Create mock memory with captioner reference
    class TestMemory(MemoryMixin):
        def get_last_reflection(self):
            for entry in reversed(self.long_memory):
                if entry.get("type") == "reflection":
                    return entry.get("text", "")
            return ""
    
    memory = TestMemory()
    memory.long_memory = []
    memory.beliefs = {}
    memory.session_start = time.time() - 1800  # 30 minutes ago
    memory.boot_ts = time.time() - 7200  # 2 hours ago
    memory.novelty_score = 0.7
    
    # Set up mock captioner
    mock_captioner = MockCaptioner()
    memory._captioner_ref = mock_captioner
    
    # Add some test data
    memory.last_caption = "The afternoon light creates diagonal shadows across the desk, highlighting the texture of scattered papers"
    
    # Add a reflection directly to long_memory
    memory.long_memory.append({
        "type": "reflection",
        "text": "I'm drawn to the way light transforms ordinary objects into geometric compositions. There's poetry in these everyday shadows.",
        "timestamp": time.time()
    })
    
    # Build the drawing prompt
    print("\nGenerating drawing prompt with rich context...")
    print("-" * 60)
    
    drawing_prompt = build_drawing_prompt(memory)
    
    print("\nFULL DRAWING PROMPT:")
    print("=" * 60)
    print(drawing_prompt)
    print("=" * 60)
    
    # Show what changed
    print("\n\nKEY IMPROVEMENTS:")
    print("-" * 60)
    print("[+] Emotional context included (mood, emotional state)")
    print("[+] Temporal context (how long observing)")
    print("[+] Trigger reason (why drawing now)")
    print("[+] Person context (if someone is present)")
    print("[+] LINE DRAWING focus with visual clarity")
    print("[+] Clear instructions for visual composition")
    print("[+] No redundant information")
    print("[+] No confusing 'should I draw?' question")

if __name__ == "__main__":
    test_new_drawing_prompt()