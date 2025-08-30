#!/usr/bin/env python3
"""
Minimal test to isolate servo integration issue in machine.py
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import cv2
from config.config import USE_SERVO, SERIAL_PORT, BAUD_RATE, CAMERA_INDEX

print(f"USE_SERVO: {USE_SERVO}")
print(f"SERIAL_PORT: {SERIAL_PORT}")

# Initialize servo controller
if USE_SERVO:
    from servo_control.servo_control import ServoController
    print("Creating ServoController...")
    servos = ServoController(port=SERIAL_PORT, baudrate=BAUD_RATE)
    print("ServoController created successfully")
else:
    servos = None

# Initialize camera (like machine.py does)
print("Opening camera...")
cap = cv2.VideoCapture(CAMERA_INDEX if "CAMERA_INDEX" in globals() else 0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()
print("Camera opened successfully")

# Minimal main loop (like machine.py)
print("Starting main loop...")
frame_count = 0
try:
    while frame_count < 30:  # Run for 30 frames
        ret, frame = cap.read()
        if not ret:
            continue
            
        frame = cv2.resize(frame, (320, 240))
        frame = cv2.flip(frame, 1)
        
        # Test servo commands (like machine.py does)
        if USE_SERVO and servos:
            pan = 90  # Fixed position for test
            tilt = 90  # Fixed position for test
            try:
                servos.set_pan(pan)
                servos.set_tilt(tilt)
                if frame_count % 10 == 0:  # Print every 10 frames
                    print(f"Frame {frame_count}: Sent servo commands pan={pan}, tilt={tilt}")
            except Exception as e:
                print(f"Servo command failed: {e}")
        
        frame_count += 1
        time.sleep(0.1)  # ~10fps
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    print("Test completed successfully!")
    
except KeyboardInterrupt:
    print("Test interrupted by user")
except Exception as e:
    print(f"Test failed with error: {e}")
    import traceback
    traceback.print_exc()
finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Cleanup completed")