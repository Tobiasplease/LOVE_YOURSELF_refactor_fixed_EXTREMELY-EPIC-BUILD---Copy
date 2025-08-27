#!/usr/bin/env python3
"""
Test script to verify the caption repetition fix without camera dependency.
This simulates the caption printing mechanism to ensure no repetitive output.
"""
import os
import sys
import threading
import time
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from captioner.captioner import Captioner
from config import config

def create_dummy_frame():
    """Create a dummy frame for testing."""
    return np.zeros((480, 640, 3), dtype=np.uint8)

def test_caption_repetition_fix():
    """Test that the caption repetition fix works."""
    print("Testing caption repetition fix...")
    print("=" * 50)
    
    # Create captioner instance
    captioner = Captioner()
    
    # Create dummy frame
    dummy_frame = create_dummy_frame()
    
    print("Simulating 5 rapid caption updates to test for repetition...")
    print("Each caption should appear only ONCE:")
    print("-" * 30)
    
    # Simulate rapid updates that previously caused repetition
    for i in range(5):
        print(f"\nUpdate {i+1}:")
        
        # Update captioner with dummy data
        captioner.update(
            frame=dummy_frame,
            person_present=False,
            mood=0.5,
            mood_vector=(0.5, 0.0, 0.0),
            emotion_state="neutral"
        )
        
        # Wait briefly to let processing complete
        time.sleep(2)
        
        # Show what the last caption was
        if hasattr(captioner, 'last_caption') and captioner.last_caption:
            print(f"Last caption stored: {captioner.last_caption[:50]}...")
        else:
            print("No caption generated yet")
    
    print("\n" + "=" * 50)
    print("Test complete. Check output above for repetitive printing.")
    print("SUCCESS: If each caption appears only once, the fix works!")
    print("FAILED: If captions are repeated multiple times, fix needs adjustment.")

if __name__ == "__main__":
    test_caption_repetition_fix()