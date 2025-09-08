#!/usr/bin/env python3
"""
Servo Calibration Tool
Interactive tool to test servo limits and smooth movement
"""

import sys
import time
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from servo_control.servo_control import ServoController


def main():
    print("=== SERVO CALIBRATION TOOL ===")
    print("Testing natural head movement limits:")
    print("- PAN: ±50° range (65-115°)")
    print("- TILT: ±20° range (70-110°)")
    print()
    
    # Connect to servo controller
    servo_port = "/dev/arduino_lunggaze"
    try:
        servos = ServoController(port=servo_port, baudrate=9600)
        if not servos.ser or not servos.ser.is_open:
            print(f"❌ Failed to connect to {servo_port}")
            return
        print(f"✅ Connected to {servo_port}")
    except Exception as e:
        print(f"❌ Connection error: {e}")
        return
    
    # Center position first
    print("\n1. Moving to CENTER position (90, 90)...")
    servos.set_pan(90)
    time.sleep(0.5)  # Small delay between axes
    servos.set_tilt(90)
    time.sleep(2)
    
    # Test PAN range
    print("\n2. Testing PAN range (65-115°)...")
    print("   Moving to LEFT limit (65°)...")
    servos.set_pan(65)
    time.sleep(1.5)
    
    print("   Moving to CENTER (90°)...")
    servos.set_pan(90)
    time.sleep(1)
    
    print("   Moving to RIGHT limit (115°)...")
    servos.set_pan(115)
    time.sleep(1.5)
    
    print("   Returning to CENTER...")
    servos.set_pan(90)
    time.sleep(1)
    
    # Test TILT range
    print("\n3. Testing TILT range (70-110°)...")
    print("   Moving to DOWN limit (70°)...")
    servos.set_tilt(70)
    time.sleep(1.5)
    
    print("   Moving to CENTER (90°)...")
    servos.set_tilt(90)
    time.sleep(1)
    
    print("   Moving to UP limit (110°)...")
    servos.set_tilt(110)
    time.sleep(1.5)
    
    print("   Returning to CENTER...")
    servos.set_tilt(90)
    time.sleep(1)
    
    # Test smooth movement pattern
    print("\n4. Testing smooth natural movement pattern...")
    positions = [
        (85, 85),   # Slight left-down
        (95, 95),   # Slight right-up
        (75, 90),   # Left-center
        (105, 90),  # Right-center
        (90, 80),   # Center-down
        (90, 100),  # Center-up
        (90, 90),   # Return to center
    ]
    
    for i, (pan, tilt) in enumerate(positions):
        print(f"   Position {i+1}: PAN={pan}°, TILT={tilt}°")
        servos.set_pan(pan)
        time.sleep(0.3)  # Small delay between axes for smooth motion
        servos.set_tilt(tilt)
        time.sleep(2.0)  # Longer pause to let servos reach position smoothly
    
    print("\n✅ Calibration complete!")
    print("\nRecommended settings:")
    print("- PAN_MIN = 65°  (natural left limit)")
    print("- PAN_MAX = 115° (natural right limit)")
    print("- TILT_MIN = 70° (natural down limit)")
    print("- TILT_MAX = 110° (natural up limit)")
    print("- CENTER = 90° for both axes")


if __name__ == "__main__":
    main()