#!/usr/bin/env python3
"""
Test servo commands as they would be sent in the main loop
"""
import time
from servo_control.servo_control import ServoController

def test_main_loop_servo():
    print("Testing servo commands as in main loop...")
    
    # Initialize servo like machine.py does
    servo = ServoController("/dev/arduino_lunggaze", 9600)
    
    if not servo.ser or not servo.ser.is_open:
        print("ERROR: Could not open servo port")
        return
    
    # Wait like machine.py does
    print("Waiting for Arduino initialization...")
    time.sleep(2.0)
    
    # Initial commands like machine.py
    print("Sending initial servo commands...")
    servo.set_pan(90)
    time.sleep(0.5)
    servo.set_tilt(90)
    time.sleep(0.5)
    servo.set_lung("slow")
    
    # Now simulate main loop commands
    print("Simulating main loop servo commands...")
    for i in range(5):
        # Simulate some pan/tilt values like gaze tracking would generate
        pan = 90 + (i * 10)  # 90, 100, 110, 120, 130
        tilt = 90 + (i * 5)   # 90, 95, 100, 105, 110
        
        print(f"Loop {i}: Sending PAN:{pan}, TILT:{tilt}")
        servo.set_pan(pan)
        servo.set_tilt(tilt)
        time.sleep(1)  # 1 second between updates
    
    print("Test complete - did the servos move during the main loop simulation?")

if __name__ == "__main__":
    test_main_loop_servo()