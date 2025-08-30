#!/usr/bin/env python3
"""
Test the new dynamic gaze behavior
"""
import os
import sys
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from vision.gaze import update_gaze

def test_idle_gaze_behavior():
    """Test the idle gaze movement to see if it's more dynamic"""
    print("=== DYNAMIC GAZE BEHAVIOR TEST ===")
    print("Testing idle movement (no face detection)...")
    print("This will simulate 30 seconds of idle behavior\n")
    
    # Create a dummy frame
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    start_time = time.time()
    last_position = (90, 90)
    movement_count = 0
    big_movements = 0
    
    print("Time  | Pan | Tilt | Movement | Type")
    print("------|-----|------|----------|----------")
    
    for i in range(60):  # 60 iterations over 30 seconds
        # No face detected (idle behavior)
        person_present, pan, tilt = update_gaze(frame, None, current_mood=0.0)
        
        # Check for movement
        if (pan, tilt) != last_position:
            movement_count += 1
            
            # Calculate movement distance
            distance = ((pan - last_position[0])**2 + (tilt - last_position[1])**2)**0.5
            
            if distance > 20:
                movement_type = "BIG SWEEP"
                big_movements += 1
            elif distance > 10:
                movement_type = "Medium"
            else:
                movement_type = "Small"
                
            print(f"{i*0.5:5.1f}s | {pan:3d} | {tilt:4d} | {distance:6.1f}° | {movement_type}")
            
        last_position = (pan, tilt)
        
        # Sleep to simulate real-time
        time.sleep(0.5)
    
    print("\n=== ANALYSIS ===")
    print(f"Total movements: {movement_count}")
    print(f"Big sweeping movements: {big_movements}")
    print(f"Movement frequency: {movement_count/30:.1f} per second")
    print(f"Sweep frequency: {big_movements/30:.2f} per second")
    
    if big_movements > 0:
        print("✅ Dynamic sweeping movements detected!")
    else:
        print("⚠️ No big sweeping movements - may need tuning")
        
    if movement_count > 10:
        print("✅ Good movement activity")
    else:
        print("⚠️ Low movement activity - may be too static")

def test_face_tracking():
    """Test face tracking responsiveness"""
    print("\n=== FACE TRACKING TEST ===")
    print("Testing face tracking responsiveness...")
    
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Simulate face moving across screen
    positions = [
        (100, 100, 200, 200),  # Left side
        (200, 150, 300, 250),  # Moving right
        (400, 200, 500, 300),  # Right side
        (300, 100, 400, 200),  # Moving back
        (200, 200, 300, 300),  # Center
    ]
    
    print("Face Position    | Pan | Tilt | Response")
    print("-----------------|-----|------|----------")
    
    for i, face_box in enumerate(positions):
        person_present, pan, tilt = update_gaze(frame, face_box, current_mood=0.0)
        
        face_center_x = (face_box[0] + face_box[2]) // 2
        face_center_y = (face_box[1] + face_box[3]) // 2
        
        print(f"({face_center_x:3d}, {face_center_y:3d})      | {pan:3d} | {tilt:4d} | {'✅ Tracked' if person_present else '❌ Lost'}")
        
        time.sleep(0.1)  # Brief pause
    
    print("✅ Face tracking test complete")

if __name__ == "__main__":
    test_idle_gaze_behavior()
    test_face_tracking()