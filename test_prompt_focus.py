#!/usr/bin/env python3
"""
Test the new real-time caption prompt structure
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from captioner.captioner import Captioner
    from captioner.prompts import build_caption_prompt
    import time
    print("✅ Imports successful")
except Exception as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

def test_prompt_focus():
    """Test how the new prompt structure prioritizes current vs memory"""
    try:
        print("=== TESTING REAL-TIME PROMPT FOCUS ===\n")
        
        # Initialize captioner with some fake history
        print("1. Initializing captioner...")
        captioner = Captioner()
        
        # Give it a moment for threads to start
        time.sleep(1)
        
        print("2. Adding fake historical context...")
        # Simulate some old observations that might be causing the delay issue
        captioner.observe("User was drawing on unmade bed", 0.6, "", memory_type="perception")
        captioner.observe("Dog was lying on bed looking at user", 0.7, "", memory_type="perception")
        
        print("3. Building prompt...")
        # Build the new prompt
        prompt = build_caption_prompt(
            agent=captioner,
            mood=0.6,
            boredom=0.2, 
            novelty=0.8,
            previous_caption="The room appears quiet and peaceful."
        )
        
        print(f"\n🎯 NEW PROMPT STRUCTURE:")
        print("=" * 60)
        print(prompt)
        print("=" * 60)
        
        print(f"\n✅ Test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_prompt_focus()
