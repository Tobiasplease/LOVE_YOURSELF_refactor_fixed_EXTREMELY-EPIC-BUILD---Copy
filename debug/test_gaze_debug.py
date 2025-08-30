#!/usr/bin/env python3
"""
Debug the gaze decision making process
"""
import os
import sys
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

def debug_gaze_decisions():
    """Debug what decisions the gaze system is making"""
    print("=== GAZE DECISION DEBUG ===")
    
    # Import and get access to internal state
    from vision import gaze
    
    # Reset state to force new decisions
    gaze.state = "idle"
    gaze.idle_next_move_time = 0
    gaze.servo_x = 90
    gaze.servo_y = 90
    gaze.target_x = 90
    gaze.target_y = 90
    
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    print("Monitoring decision making process...\n")
    
    for i in range(8):
        print(f"--- Iteration {i+1} ---")
        print(f"Before: servo=({gaze.servo_x:.1f}, {gaze.servo_y:.1f}), target=({gaze.target_x:.1f}, {gaze.target_y:.1f})")
        print(f"State: {gaze.state}, next_move_time: {gaze.idle_next_move_time - time.time():.1f}s")
        
        # Call the update function
        person_present, pan, tilt = gaze.update_gaze(frame, None, current_mood=0.0)
        
        print(f"After:  servo=({gaze.servo_x:.1f}, {gaze.servo_y:.1f}), target=({gaze.target_x:.1f}, {gaze.target_y:.1f})")
        print(f"Output: pan={pan}, tilt={tilt}")
        
        # Check if target changed (indicates new decision)
        distance_to_target = ((gaze.target_x - gaze.servo_x)**2 + (gaze.target_y - gaze.servo_y)**2)**0.5
        if distance_to_target > 10:
            print(f"🎯 NEW TARGET SET! Distance to target: {distance_to_target:.1f}°")
        else:
            print(f"Moving toward existing target (distance: {distance_to_target:.1f}°)")
        
        print()
        time.sleep(0.5)

if __name__ == "__main__":
    debug_gaze_decisions()