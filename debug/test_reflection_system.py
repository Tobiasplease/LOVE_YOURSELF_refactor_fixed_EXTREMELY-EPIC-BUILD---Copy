#!/usr/bin/env python3
"""
Test the reflection system to see if it's working properly.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from captioner.model_wrapper import MultimodalModel
from captioner.prompts import build_reflection_prompt
import time

def test_reflection():
    """Test if the reflection system works."""
    print("Testing reflection system...")
    
    # Create a model wrapper
    model = MultimodalModel()
    
    # Test basic reflection
    test_caption = "I notice the familiar arrangement of objects on this desk..."
    test_context = """Mood: 0.65
Boredom: 0.2
Novelty: 0.4
Identity: An observing consciousness 
Recent memory: 
- Earlier observations of workspace
- Patterns of light and shadow
Top motifs: desk, keyboard, screen, workspace
Emotional evolution: calm_observant → contemplative → focused"""

    print(f"Test caption: {test_caption}")
    print(f"Test context: {test_context}")
    
    try:
        # Build the reflection prompt
        prompt = build_reflection_prompt(test_caption, extra=test_context)
        print(f"\nGenerated prompt length: {len(prompt)} chars")
        print("Prompt preview:", prompt[:200] + "..." if len(prompt) > 200 else prompt)
        
        # Test the reflection call with shorter timeout
        print("\nCalling reason_about_caption...")
        start_time = time.time()
        
        reflection = model.reason_about_caption(
            test_caption, 
            mood_text="I feel contemplative and focused",
            extra=test_context
        )
        
        elapsed = time.time() - start_time
        print(f"\nReflection completed in {elapsed:.1f}s")
        print(f"Reflection length: {len(reflection)} chars")
        print(f"Reflection: {reflection}")
        
        return True
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_reflection()
    print(f"\nTest {'PASSED' if success else 'FAILED'}")