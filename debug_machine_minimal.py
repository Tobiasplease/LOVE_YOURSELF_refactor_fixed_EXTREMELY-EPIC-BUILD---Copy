#!/usr/bin/env python3
"""
Minimal reproduction of machine.py servo initialization to isolate the issue
"""
import time
import os
from servo_control.servo_control import ServoController
from config.config import BAUD_RATE, USE_SERVO

# Replicate exact machine.py logic
ARDUINO_DEVICES = {
    "LUNGGAZE": "/dev/arduino_lunggaze",
}

print("=== MINIMAL MACHINE.PY SERVO TEST ===")

if USE_SERVO:
    servo_port = ARDUINO_DEVICES["LUNGGAZE"]
    print(f"Checking servo port: {servo_port}")
    print(f"Port exists: {os.path.exists(servo_port)}")
    
    if os.path.exists(servo_port):
        try:
            print("Initializing ServoController...")
            servos = ServoController(port=servo_port, baudrate=BAUD_RATE)
            print(f"Servo controller initialized on {servo_port}")
            
            # Wait a moment
            print("Starting main loop simulation...")
            time.sleep(1)
            
            # Simulate what main loop should do
            for i in range(10):
                pan = 90 + (i * 5)  # 90, 95, 100, 105...
                tilt = 90 + (i * 3)  # 90, 93, 96, 99...
                lung = 90 + (i * 2)  # 90, 92, 94, 96...
                
                print(f"Frame {i}: Sending PAN:{pan}, TILT:{tilt}, LUNG:{lung}")
                
                servos.set_pan(pan)
                servos.set_tilt(tilt) 
                servos.set_lung_position(lung, force=True)
                
                time.sleep(0.2)  # 5 FPS simulation
                
        except Exception as e:
            print(f"ERROR: Servo controller init failed: {e}")
            servos = None
    else:
        print(f"WARNING: Servo controller not found at {servo_port}")
        servos = None
else:
    print("Servo control disabled in config")
    servos = None

print("=== Test complete ===")