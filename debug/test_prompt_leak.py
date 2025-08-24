#!/usr/bin/env python3
"""
Test to reproduce prompt leakage issue.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from captioner.model_wrapper import MultimodalModel
from captioner.captioner import Captioner
import time

def test_prompt_leak():
    """Simulate conditions that cause prompt leakage."""
    print("Testing conditions that might cause prompt leakage...")
    
    # Create a mock captioner
    try:
        captioner = Captioner()
        model = MultimodalModel(memory_ref=captioner)
        
        # Create a fake image path (this won't exist, which might cause errors)
        fake_image_path = "nonexistent_image.jpg"
        
        print("Testing caption_image with non-existent image...")
        result = model.caption_image(fake_image_path, flowing=True, first_time=False)
        
        print(f"Result length: {len(result)}")
        print(f"Result preview: {result[:200]}...")
        
        # Check if result contains prompt template elements
        if "FLOW GUIDANCE" in result or "PERSON:" in result or "SELF:" in result:
            print("❌ PROMPT LEAKAGE DETECTED!")
            print("The result contains internal prompt template elements:")
            print(result)
        else:
            print("✅ No prompt leakage detected")
            
    except Exception as e:
        print(f"Test error: {e}")

if __name__ == "__main__":
    test_prompt_leak()