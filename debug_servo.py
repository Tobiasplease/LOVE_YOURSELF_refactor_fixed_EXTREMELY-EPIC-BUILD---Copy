#!/usr/bin/env python3
"""
Simple servo test script to debug lunggaze communication
"""
import time
from servo_control.servo_control import ServoController

def test_servo():
    print("Testing servo communication...")
    
    # Initialize servo controller
    servo = ServoController("/dev/arduino_lunggaze", 9600)
    
    if not servo.ser or not servo.ser.is_open:
        print("ERROR: Could not open servo port")
        return
        
    print("Servo port opened successfully")
    
    # Give Arduino time to initialize
    time.sleep(2)
    
    # Test sending commands
    print("Sending PAN:90")
    servo.set_pan(90)
    time.sleep(1)
    
    print("Sending TILT:90") 
    servo.set_tilt(90)
    time.sleep(1)
    
    print("Sending LUNG:90")
    servo.set_lung_position(90, force=True)
    time.sleep(1)
    
    # Try some movement
    print("Testing movement...")
    servo.set_pan(120)
    time.sleep(1)
    servo.set_pan(60)
    time.sleep(1)
    servo.set_pan(90)
    
    print("Test complete")

if __name__ == "__main__":
    test_servo()