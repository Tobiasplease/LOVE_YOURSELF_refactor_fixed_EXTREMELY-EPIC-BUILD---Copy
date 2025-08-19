#!/usr/bin/env python3
"""
Test captioning system to see what's broken
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from captioner.captioner import Captioner
import numpy as np
import cv2

def test_captioning():
    print("=== TESTING CAPTIONING SYSTEM ===\n")
    
    # Create a test image
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    test_image[100:380, 100:540] = [50, 100, 150]  # Add a rectangle
    
    print("Creating captioner...")
    captioner = Captioner()
    
    print("Updating with test frame...")
    captioner.update(test_image, person_present=True, mood=0.5)
    
    print("Waiting for caption processing...")
    import time
    time.sleep(12)  # Wait longer than CAPTION_INTERVAL
    
    print(f"Last caption: {captioner.last_caption}")
    print(f"Memory queue length: {len(captioner.memory_queue)}")
    print(f"Is processing: {captioner.is_processing}")

if __name__ == "__main__":
    test_captioning()
