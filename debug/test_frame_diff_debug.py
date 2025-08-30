#!/usr/bin/env python3
"""
Test frame difference protocol for lightbulb controller
"""

import serial
import json
import time

port = '/dev/ttyUSB1'
print(f"Testing Frame Diff Lightbulb on {port}")

ser = serial.Serial(port, 9600, timeout=1)
time.sleep(2)

# Read startup message
if ser.in_waiting:
    print("Arduino:", ser.read(ser.in_waiting).decode('utf-8', errors='ignore').strip())

# Test frame differences
for frame_diff in [0.0, 0.3, 0.6, 1.0, 0.0]:
    msg = json.dumps({"frame_diff": frame_diff})
    print(f"Sending: {msg}")
    ser.write((msg + '\n').encode())
    time.sleep(1)
    
    if ser.in_waiting:
        print("Response:", ser.read(ser.in_waiting).decode('utf-8', errors='ignore').strip())

ser.close()
print("Test complete!")
