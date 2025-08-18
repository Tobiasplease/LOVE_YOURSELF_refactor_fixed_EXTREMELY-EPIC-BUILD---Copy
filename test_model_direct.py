#!/usr/bin/env python3
"""
Debug captioning with detailed logging
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from captioner.model_wrapper import MultimodalModel
from captioner.memory import MemoryMixin
import cv2
import numpy as np

def test_model_directly():
    print("=== TESTING MODEL WRAPPER DIRECTLY ===\n")
    
    # Create test image
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    test_image[100:380, 100:540] = [50, 100, 150]  # Add a rectangle
    
    # Save test image
    test_path = "test_image.jpg"
    cv2.imwrite(test_path, test_image)
    print(f"Created test image: {test_path}")
    
    # Create memory mixin
    memory = MemoryMixin()
    print("Created memory mixin")
    
    # Create model
    model = MultimodalModel(memory_ref=memory)
    print(f"Created model: {model.model_name}")
    
    try:
        print("Calling caption_image...")
        result = model.caption_image(test_path, flowing=True, first_time=True)
        print(f"Result: {result}")
    except Exception as e:
        print(f"Error during captioning: {e}")
        import traceback
        traceback.print_exc()
    
    # Cleanup
    if os.path.exists(test_path):
        os.remove(test_path)

if __name__ == "__main__":
    test_model_directly()
