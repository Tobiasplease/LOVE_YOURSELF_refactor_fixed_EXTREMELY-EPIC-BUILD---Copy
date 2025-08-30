#!/usr/bin/env python3
"""
Extreme servo movement test - makes it very obvious if servos are responding.
Tests the full range of both X and Y servos with dramatic movements.
"""
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vision.gaze import update_gaze

def test_extreme_corners():
    """Test all four corners - should produce maximum servo movements"""
    print("=== EXTREME CORNER TEST ===")
    print("Testing maximum servo positions in all corners")
    print("X should range from ~45 to ~135, Y should range from ~45 to ~135")
    print("-" * 60)
    
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Test extreme corners of the frame
    corners = [
        ("TOP-LEFT CORNER", 50, 50),      # Should produce low X, high Y values
        ("TOP-RIGHT CORNER", 590, 50),   # Should produce high X, high Y values  
        ("BOTTOM-LEFT CORNER", 50, 430), # Should produce low X, low Y values
        ("BOTTOM-RIGHT CORNER", 590, 430) # Should produce high X, low Y values
    ]
    
    for label, face_x, face_y in corners:
        face_box = (face_x - 25, face_y - 25, face_x + 25, face_y + 25)
        person_present, servo_x, servo_y = update_gaze(dummy_frame, face_box)
        
        print("{:18} Face({:3d},{:3d}) -> Servos PAN:{:3d}, TILT:{:3d}".format(
            label, face_x, face_y, servo_x, servo_y))
        
        time.sleep(1)  # Pause to let servo reach position

def test_big_movements():
    """Test dramatic servo movements that should be very visible"""
    print("\n=== BIG MOVEMENT TEST ===") 
    print("Servo should make large, obvious movements")
    print("-" * 40)
    
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    movements = [
        ("START CENTER", 320, 240),
        ("HARD LEFT", 50, 240), 
        ("HARD RIGHT", 590, 240),
        ("BACK CENTER", 320, 240),
        ("HARD UP", 320, 50),
        ("HARD DOWN", 320, 430),
        ("BACK CENTER", 320, 240),
    ]
    
    for label, face_x, face_y in movements:
        face_box = (face_x - 25, face_y - 25, face_x + 25, face_y + 25)
        person_present, servo_x, servo_y = update_gaze(dummy_frame, face_box)
        
        print("{:12} -> PAN:{:3d}, TILT:{:3d}  (Face at {}, {})".format(
            label, servo_x, servo_y, face_x, face_y))
        
        time.sleep(2)  # Longer pause for dramatic effect

def test_continuous_sweep():
    """Continuous pan sweep - servo should move smoothly left to right"""
    print("\n=== CONTINUOUS PAN SWEEP ===")
    print("PAN servo should sweep smoothly from left to right")
    print("(TILT should stay roughly the same)")
    print("-" * 50)
    
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    for i in range(11):  # 0 to 10
        progress = i / 10.0  # 0.0 to 1.0
        face_x = int(50 + progress * 540)  # Sweep left to right
        face_y = 240  # Keep Y centered
        
        face_box = (face_x - 25, face_y - 25, face_x + 25, face_y + 25)
        person_present, servo_x, servo_y = update_gaze(dummy_frame, face_box)
        
        print("Step {:2d}: Face X:{:3d} -> PAN:{:3d}, TILT:{:3d}".format(
            i, face_x, servo_x, servo_y))
        
        time.sleep(0.5)

if __name__ == "__main__":
    print("SERVO MOVEMENT TEST - Watch your servos carefully!")
    print("=" * 60)
    
    test_extreme_corners()
    test_big_movements() 
    test_continuous_sweep()
    
    print("\n=== TEST COMPLETE ===")
    print("Did you see both servos move dramatically?")
    print("PAN (X) should have moved left/right")
    print("TILT (Y) should have moved up/down")