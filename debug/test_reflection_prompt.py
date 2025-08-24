#!/usr/bin/env python3
"""
Test the reflection prompt system.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from captioner.prompts import build_reflection_prompt

def test_reflection_prompt():
    """Test the reflection prompt format."""
    print("Testing Reflection Prompt Format")
    print("=" * 60)
    
    # Mock agent with temporal context
    class MockAgent:
        def __init__(self):
            self.true_session_start = 1234567890  # Mock timestamp
            self.identity_label = "curious observer with growing environmental awareness"
            
        def rephrase_with_doubt(self, caption):
            return f"I think I saw {caption.lower()}, though I'm not entirely certain"
    
    agent = MockAgent()
    test_caption = "The afternoon light creates diagonal patterns on the desk surface, casting geometric shadows"
    test_extra = "Mood vector: valence=0.2, arousal=0.7, clarity=0.5\nEmotional journey: calm -> curious -> contemplative"
    
    print("\nGenerating reflection prompt...")
    print("-" * 60)
    
    reflection_prompt = build_reflection_prompt(test_caption, extra=test_extra, agent=agent)
    
    print("\nFULL REFLECTION PROMPT:")
    print("=" * 60)
    print(reflection_prompt)
    print("=" * 60)
    
    print("\n\nKEY IMPROVEMENTS:")
    print("-" * 60)
    print("[+] Rich contextual setup")
    print("[+] Temporal awareness included")
    print("[+] Structured reflection focus")
    print("[+] Clear introspection guidance")
    print("[+] Natural stream of consciousness approach")

if __name__ == "__main__":
    test_reflection_prompt()