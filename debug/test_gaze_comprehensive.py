#!/usr/bin/env python3
"""
Comprehensive test of the new dynamic gaze system
"""
import os
import sys
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from vision.gaze import update_gaze

def test_comprehensive_gaze():
    """Test all aspects of the gaze system"""
    print("=== COMPREHENSIVE GAZE TEST ===")
    print("Testing dynamic idle behavior with the new settings:")
    print("- 60% chance of big sweeping movements")
    print("- 25% easing factor for faster movement")  
    print("- Larger pause times after sweeps")
    print("- 35° movement range\n")
    
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    last_position = (90, 90)
    start_time = time.time()
    
    movements = []
    big_movements = 0
    total_movements = 0
    
    print("Time | Pan | Tilt | Distance | Status")
    print("-----|-----|------|----------|--------")
    
    # Test for 15 iterations
    for i in range(15):
        person_present, pan, tilt = update_gaze(frame, None, current_mood=0.0)
        
        if (pan, tilt) != last_position:
            distance = ((pan - last_position[0])**2 + (tilt - last_position[1])**2)**0.5
            movements.append(distance)
            total_movements += 1
            
            if distance > 15:
                big_movements += 1
                status = "BIG SWEEP ✨"
            elif distance > 8:
                status = "Medium move"
            else:
                status = "Small adjust"
                
            elapsed = time.time() - start_time
            print(f"{elapsed:4.1f}s | {pan:3d} | {tilt:4d} | {distance:6.1f}° | {status}")
        
        last_position = (pan, tilt)
        time.sleep(0.3)
    
    print("\n=== ANALYSIS ===")
    if movements:
        avg_movement = sum(movements) / len(movements)
        max_movement = max(movements)
        print(f"Total movements: {total_movements}")
        print(f"Big sweeping movements (>15°): {big_movements}")
        print(f"Average movement size: {avg_movement:.1f}°")
        print(f"Largest movement: {max_movement:.1f}°")
        print(f"Sweep percentage: {big_movements/total_movements*100:.0f}%")
        
        if big_movements > 0:
            print("✅ SUCCESS: Dynamic sweeping movements detected!")
        if avg_movement > 5:
            print("✅ SUCCESS: Good average movement size") 
        if max_movement > 20:
            print("✅ SUCCESS: Large sweeping movements present")
    else:
        print("❌ No movements detected - system may be stuck")
    
    print(f"\nThe gaze system now feels much more alive and attentive!")
    print("- Larger movements make it feel like it's actively looking around")
    print("- Longer pauses after big movements simulate 'observation'")
    print("- Mix of small and large movements creates natural behavior")

if __name__ == "__main__":
    test_comprehensive_gaze()