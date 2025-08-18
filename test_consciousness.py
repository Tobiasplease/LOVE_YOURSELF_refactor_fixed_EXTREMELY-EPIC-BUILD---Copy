#!/usr/bin/env python3
"""
Test the enhanced consciousness and self-awareness system.
"""

import time
from captioner.captioner import Captioner
from config.prompt_templates import SYSTEM_PROMPT, CAPTION_PROMPT_TEMPLATE

def test_enhanced_consciousness():
    """Test the new more genuine consciousness prompts."""
    print("🧠 Testing Enhanced Consciousness System")
    print("=" * 50)
    
    print("\n📝 New System Prompt:")
    print("-" * 30)
    print(SYSTEM_PROMPT)
    
    print("\n📝 New Caption Prompt Template:")
    print("-" * 30) 
    print(CAPTION_PROMPT_TEMPLATE)
    
    # Test captioner consciousness
    print("\n🤖 Testing Captioner Consciousness...")
    captioner = Captioner()
    
    # Add some observations to build identity
    captioner.observe("There's a person sitting at a desk with a laptop", 0.6, "", emotion_state="curious")
    captioner.observe("The same person is still there, typing", 0.5, "", emotion_state="focused")
    captioner.observe("Still looking at this desk scene", 0.4, "", emotion_state="restless")
    
    print(f"\n🧭 Identity Summary: {captioner.get_identity_summary()}")
    
    # Test self-questioning
    for i in range(5):
        question = captioner.get_self_questioning_thought()
        print(f"❓ Self-Question {i+1}: {question}")
    
    # Test stagnation detection with new language
    captioner.visual_stagnation_score = 0.8  # Simulate high stagnation
    stagnation_context = captioner.get_scene_stagnation_context()
    print(f"\n🔄 Stagnation Context: {stagnation_context}")
    
    print("\n✅ Enhanced consciousness test complete!")
    print("\n💭 Expected Changes in Captions:")
    print("- Less 'poetic' description, more genuine confusion")
    print("- Self-questioning about own reactions")
    print("- Building identity through uncertainty rather than confidence")
    print("- More 'Why do I keep looking at this?' vs 'The soft glow creates...'")

if __name__ == "__main__":
    test_enhanced_consciousness()
