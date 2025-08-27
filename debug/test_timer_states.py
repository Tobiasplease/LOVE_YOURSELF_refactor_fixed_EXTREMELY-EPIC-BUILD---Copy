#!/usr/bin/env python3
"""
Debug script to check timer states in the captioner system
"""

import sys
import os
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from captioner.captioner import Captioner
from config.config import REASON_INTERVAL, DRAWING_INTERVAL

def test_timer_states():
    """Test current timer states"""
    print("=== Testing Timer States ===")
    
    captioner = Captioner()
    current_time = time.time()
    
    print(f"Current time: {current_time:.0f}")
    print(f"Last reason time: {captioner.last_reason_time:.0f}")
    print(f"Last drawing time: {captioner.last_drawing_time:.0f}")
    
    reason_diff = current_time - captioner.last_reason_time
    drawing_diff = current_time - captioner.last_drawing_time
    
    print(f"\nTime since last reason: {reason_diff:.0f}s (threshold: {REASON_INTERVAL}s)")
    print(f"Time since last drawing: {drawing_diff:.0f}s (threshold: {DRAWING_INTERVAL}s)")
    
    print(f"\nReason trigger: {'YES' if reason_diff > REASON_INTERVAL else 'NO'}")
    print(f"Drawing trigger: {'YES' if drawing_diff > DRAWING_INTERVAL else 'NO'}")
    
    # Test what happens after a few seconds
    print(f"\nWaiting 3 seconds...")
    time.sleep(3)
    
    current_time = time.time()
    reason_diff = current_time - captioner.last_reason_time
    drawing_diff = current_time - captioner.last_drawing_time
    
    print(f"After 3s - Reason diff: {reason_diff:.0f}s, Drawing diff: {drawing_diff:.0f}s")

if __name__ == "__main__":
    test_timer_states()