#!/usr/bin/env python3
"""Test direct serial connection to COM3"""
import serial
import time

try:
    print("Testing direct serial connection to COM3...")
    ser = serial.Serial("COM3", 9600, timeout=1)
    print(f"SUCCESS: Opened COM3 connection")
    
    time.sleep(2)  # Arduino boot time
    
    # Send test command
    test_command = "HAND,90,90,90,90\n"
    ser.write(test_command.encode())
    print(f"SUCCESS: Sent command: {test_command.strip()}")
    
    # Try to read response (if any)
    time.sleep(0.5)
    if ser.in_waiting > 0:
        response = ser.read(ser.in_waiting).decode()
        print(f"Arduino response: {response}")
    else:
        print("No response from Arduino")
    
    ser.close()
    print("SUCCESS: Connection test completed")
    
except Exception as e:
    print(f"ERROR: Direct serial test failed: {e}")
    import traceback
    traceback.print_exc()