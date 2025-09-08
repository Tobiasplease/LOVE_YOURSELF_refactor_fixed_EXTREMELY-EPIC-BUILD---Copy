#!/usr/bin/env python3
"""
Servo Direction Test
Quick test to determine correct FLIP_X and FLIP_Y settings after mechanical changes
"""

import sys
import time
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from servo_control.servo_control import ServoController

def main():
    print("=== SERVO DIRECTION TEST ===")
    print("This will help determine correct FLIP_X and FLIP_Y settings")
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
    
    # Center first
    print("\n📍 Moving to center position (90, 90)...")
    servos.set_pan(90)
    time.sleep(0.1)
    servos.set_tilt(90)
    time.sleep(2)
    
    print("\n🔍 Testing PAN direction:")
    print("The servo should move LEFT when you move LEFT in the camera")
    print("   Moving to 75° (should be LEFT)...")
    servos.set_pan(75)
    time.sleep(2)
    
    print("   Moving back to center...")
    servos.set_pan(90)
    time.sleep(1)
    
    print("   Moving to 105° (should be RIGHT)...")
    servos.set_pan(105)
    time.sleep(2)
    
    print("   Moving back to center...")
    servos.set_pan(90)
    time.sleep(1)
    
    pan_correct = input("Did the servo move in the CORRECT direction? (LEFT then RIGHT) [y/n]: ").strip().lower()
    
    print("\n🔍 Testing TILT direction:")
    print("The servo should move DOWN when you move DOWN in the camera")
    print("   Moving to 80° (should be DOWN)...")
    servos.set_tilt(80)
    time.sleep(2)
    
    print("   Moving back to center...")
    servos.set_tilt(90)
    time.sleep(1)
    
    print("   Moving to 100° (should be UP)...")
    servos.set_tilt(100)
    time.sleep(2)
    
    print("   Moving back to center...")
    servos.set_tilt(90)
    time.sleep(1)
    
    tilt_correct = input("Did the servo move in the CORRECT direction? (DOWN then UP) [y/n]: ").strip().lower()
    
    # Determine settings
    print("\n📋 RECOMMENDED SETTINGS:")
    if pan_correct == 'y':
        print("FLIP_X = False  # PAN direction is correct")
    else:
        print("FLIP_X = True   # PAN direction needs to be flipped")
    
    if tilt_correct == 'y':
        print("FLIP_Y = False  # TILT direction is correct") 
    else:
        print("FLIP_Y = True   # TILT direction needs to be flipped")
    
    print(f"\n💡 Update these settings in config/config.py:")
    print("FLIP_X =", "False" if pan_correct == 'y' else "True")
    print("FLIP_Y =", "False" if tilt_correct == 'y' else "True")

if __name__ == "__main__":
    main()