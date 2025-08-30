#!/usr/bin/env python3
"""
Test script specifically for Y-axis (tilt) servo movement.
Tests vertical face movement and diagonal movement.
"""
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vision.gaze import update_gaze

def test_vertical_movement(duration=10):
    """Test face moving up and down (Y-axis)"""
    print("=== TESTING: Vertical Face Movement (Y-axis) ===")
    print("Face moving from top to bottom of frame")
    print("-" * 50)
    
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    start_time = time.time()
    while time.time() - start_time < duration:
        elapsed = time.time() - start_time
        
        # Face moving vertically (top to bottom)
        progress = (elapsed / duration) % 1.0  # 0.0 to 1.0
        face_x = 320  # Center X
        face_y = int(50 + progress * 380)  # Move from y=50 to y=430
        
        face_box = (face_x - 50, face_y - 50, face_x + 50, face_y + 50)
        
        person_present, servo_x, servo_y = update_gaze(dummy_frame, face_box)
        
        print("[{:05.1f}s] Face at ({:3d},{:3d}) -> Servos X:{:3d}, Y:{:3d}".format(
            elapsed, face_x, face_y, servo_x, servo_y))
        
        time.sleep(0.5)

def test_diagonal_movement(duration=10):
    """Test face moving diagonally (both X and Y axes)"""
    print("\n=== TESTING: Diagonal Face Movement (X+Y axes) ===")
    print("Face moving diagonally across frame")
    print("-" * 50)
    
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    start_time = time.time()
    while time.time() - start_time < duration:
        elapsed = time.time() - start_time
        
        # Face moving diagonally
        progress = (elapsed / duration) % 1.0  # 0.0 to 1.0
        face_x = int(100 + progress * 440)  # Move from x=100 to x=540
        face_y = int(100 + progress * 280)  # Move from y=100 to y=380
        
        face_box = (face_x - 50, face_y - 50, face_x + 50, face_y + 50)
        
        person_present, servo_x, servo_y = update_gaze(dummy_frame, face_box)
        
        print("[{:05.1f}s] Face at ({:3d},{:3d}) -> Servos X:{:3d}, Y:{:3d}".format(
            elapsed, face_x, face_y, servo_x, servo_y))
        
        time.sleep(0.5)

def test_static_positions():
    """Test specific face positions to verify servo mapping"""
    print("\n=== TESTING: Static Face Positions ===")
    print("Testing corner and center positions")
    print("-" * 50)
    
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    test_positions = [
        ("Top-Left", 160, 120),
        ("Top-Right", 480, 120), 
        ("Center", 320, 240),
        ("Bottom-Left", 160, 360),
        ("Bottom-Right", 480, 360),
    ]
    
    for label, face_x, face_y in test_positions:
        face_box = (face_x - 50, face_y - 50, face_x + 50, face_y + 50)
        person_present, servo_x, servo_y = update_gaze(dummy_frame, face_box)
        
        print("{:12} Face({:3d},{:3d}) -> Servos X:{:3d}, Y:{:3d}".format(
            label, face_x, face_y, servo_x, servo_y))

if __name__ == "__main__":
    test_static_positions()
    test_vertical_movement(10)
    test_diagonal_movement(10)