#!/usr/bin/env python3
"""
Debug caption worker with detailed error catching
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from captioner.captioner import Captioner
import numpy as np
import cv2
import time

# Monkey patch the _process_frame method to add debugging
original_process_frame = Captioner._process_frame

def debug_process_frame(self, frame, reactivity_data=None):
    print("DEBUG: _process_frame called")
    try:
        now = time.time()
        print(f"DEBUG: now={now}, last_caption_time={self.last_caption_time}")
        print(f"DEBUG: interval check: {now - self.last_caption_time} < 10 = {now - self.last_caption_time < 10}")
        
        result = original_process_frame(self, frame, reactivity_data)
        print(f"DEBUG: _process_frame completed, last_caption='{self.last_caption}'")
        return result
    except Exception as e:
        print(f"DEBUG: Exception in _process_frame: {e}")
        import traceback
        traceback.print_exc()
        raise

Captioner._process_frame = debug_process_frame

def test_captioning_debug():
    print("=== DEBUGGING CAPTIONING SYSTEM ===\n")
    
    # Create a test image
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    test_image[100:380, 100:540] = [50, 100, 150]  # Add a rectangle
    
    print("Creating captioner...")
    captioner = Captioner()
    
    print("Setting initial caption time to 0 to force immediate processing...")
    captioner.last_caption_time = 0
    
    print("Updating with test frame...")
    captioner.update(test_image, person_present=True, mood=0.5)
    
    print("Waiting for processing...")
    time.sleep(15)
    
    print(f"Last caption: '{captioner.last_caption}'")
    print(f"Memory queue length: {len(captioner.memory_queue)}")
    print(f"Is processing: {captioner.is_processing}")

if __name__ == "__main__":
    test_captioning_debug()
