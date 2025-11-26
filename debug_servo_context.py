#!/usr/bin/env python3
"""
Test servo initialization in machine.py context
"""
import time
import threading
from servo_control.servo_control import ServoController

def simulate_machine_startup():
    print("=== Simulating machine.py startup sequence ===")
    
    # Simulate the initialization order that happens in machine.py
    print("1. Initializing servo controller...")
    servo = ServoController("/dev/arduino_lunggaze", 9600)
    
    if not servo.ser or not servo.ser.is_open:
        print("ERROR: Could not open servo port")
        return None
        
    print("2. Starting background threads (simulated)...")
    # Simulate other startup activity that might interfere
    def dummy_thread():
        for i in range(10):
            time.sleep(0.1)
    
    thread = threading.Thread(target=dummy_thread, daemon=True)
    thread.start()
    
    print("3. Waiting for Arduino initialization...")
    time.sleep(2.0)
    
    print("4. Sending initial servo commands...")
    servo.set_pan(90)
    time.sleep(0.5)
    servo.set_tilt(90)
    time.sleep(0.5) 
    servo.set_lung_position(90, force=True)
    
    print("5. Testing movement...")
    time.sleep(1)
    servo.set_pan(120)
    time.sleep(1)
    servo.set_pan(60)
    time.sleep(1)
    servo.set_pan(90)
    
    return servo

if __name__ == "__main__":
    servo = simulate_machine_startup()
    if servo:
        print("=== Test completed ===")
        print("Did the servos move during this test?")