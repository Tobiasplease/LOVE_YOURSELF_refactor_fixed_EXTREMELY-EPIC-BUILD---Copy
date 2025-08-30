#!/usr/bin/env python3
"""
Manual test to trigger big movements
"""
import os
import sys
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

def test_gaze_logic():
    """Test the gaze logic step by step"""
    print("=== MANUAL GAZE LOGIC TEST ===")
    
    # Import after setting path
    from vision import gaze
    import random
    
    # Reset the gaze state
    gaze.state = "idle" 
    gaze.idle_next_move_time = 0
    gaze.servo_x = 90
    gaze.servo_y = 90
    gaze.target_x = 90
    gaze.target_y = 90
    
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    print("Initial state:")
    print(f"  servo_x={gaze.servo_x}, servo_y={gaze.servo_y}")
    print(f"  target_x={gaze.target_x}, target_y={gaze.target_y}")
    print(f"  state={gaze.state}")
    
    # Force a big movement decision
    print("\nForcing big movement logic...")
    
    # Manually trigger the idle movement logic
    current_time = time.time()
    gaze.idle_next_move_time = current_time - 1  # Force it to trigger
    
    # Test the sweep logic directly
    if random.random() < gaze.SWEEP_PROBABILITY:
        print("BIG SWEEP triggered!")
        if random.choice([True, False]):
            # Horizontal sweep
            gaze.target_x = random.choice([gaze.SERVO_MIN + 10, gaze.SERVO_MAX - 10])
            gaze.target_y = gaze.clamp(gaze.IDLE_CENTER_Y + random.randint(-20, 20), gaze.SERVO_MIN, gaze.SERVO_MAX)
            print(f"  Horizontal sweep to: target_x={gaze.target_x}, target_y={gaze.target_y}")
        else:
            # Vertical sweep  
            gaze.target_y = random.choice([gaze.SERVO_MIN + 10, gaze.SERVO_MAX - 10])
            gaze.target_x = gaze.clamp(gaze.IDLE_CENTER_X + random.randint(-20, 20), gaze.SERVO_MIN, gaze.SERVO_MAX)
            print(f"  Vertical sweep to: target_x={gaze.target_x}, target_y={gaze.target_y}")
    else:
        print("Small movement triggered")
        jitter_x = random.randint(-gaze.IDLE_RANGE, gaze.IDLE_RANGE)
        jitter_y = random.randint(-gaze.IDLE_RANGE, gaze.IDLE_RANGE)
        gaze.target_x = gaze.clamp(gaze.IDLE_CENTER_X + jitter_x, gaze.SERVO_MIN, gaze.SERVO_MAX)
        gaze.target_y = gaze.clamp(gaze.IDLE_CENTER_Y + jitter_y, gaze.SERVO_MIN, gaze.SERVO_MAX)
        print(f"  Small movement to: target_x={gaze.target_x}, target_y={gaze.target_y}")
    
    # Now simulate the easing over several steps
    print(f"\nSimulating movement with easing (factor={gaze.IDLE_EASING}):")
    for i in range(10):
        old_x, old_y = gaze.servo_x, gaze.servo_y
        gaze.servo_x = gaze.smooth_step(gaze.servo_x, gaze.target_x, gaze.IDLE_EASING)
        gaze.servo_y = gaze.smooth_step(gaze.servo_y, gaze.target_y, gaze.IDLE_EASING)
        
        distance = ((gaze.servo_x - old_x)**2 + (gaze.servo_y - old_y)**2)**0.5
        print(f"  Step {i+1}: ({gaze.servo_x:.1f}, {gaze.servo_y:.1f}) - moved {distance:.1f}°")
        
        # Stop if we've reached the target
        if abs(gaze.servo_x - gaze.target_x) < 0.5 and abs(gaze.servo_y - gaze.target_y) < 0.5:
            print(f"  Reached target after {i+1} steps")
            break

if __name__ == "__main__":
    test_gaze_logic()