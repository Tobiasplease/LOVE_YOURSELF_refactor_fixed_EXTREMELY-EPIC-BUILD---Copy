#!/usr/bin/env python3
"""
Quick test of dynamic gaze behavior
"""
import os
import sys
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from vision.gaze import update_gaze

def test_quick_gaze():
    """Quick test of gaze movement"""
    print("=== QUICK GAZE TEST ===")
    
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    last_position = (90, 90)
    
    print("Testing 10 iterations of idle behavior...")
    print("Iteration | Pan | Tilt | Movement")
    print("----------|-----|------|----------")
    
    for i in range(10):
        person_present, pan, tilt = update_gaze(frame, None, current_mood=0.0)
        
        if (pan, tilt) != last_position:
            distance = ((pan - last_position[0])**2 + (tilt - last_position[1])**2)**0.5
            print(f"{i:9d} | {pan:3d} | {tilt:4d} | {distance:6.1f}°")
        else:
            print(f"{i:9d} | {pan:3d} | {tilt:4d} | No change")
            
        last_position = (pan, tilt)
        time.sleep(0.2)
    
    print("\n✅ Quick test complete")

if __name__ == "__main__":
    test_quick_gaze()