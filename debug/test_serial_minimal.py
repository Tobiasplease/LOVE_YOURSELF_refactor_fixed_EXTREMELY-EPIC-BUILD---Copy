#!/usr/bin/env python3
"""Minimal serial test to debug Linux I/O errors."""

import serial
import time
import sys

port = '/dev/ttyUSB1'
baudrate = 9600

print(f"Testing serial port {port} at {baudrate} baud...")
print("-" * 50)

# Test 1: Open without any special flags (like Windows)
print("\nTest 1: Simple connection (Windows-style)")
try:
    ser = serial.Serial()
    ser.port = port
    ser.baudrate = baudrate
    ser.timeout = 1
    ser.setDTR(False)  # Don't reset on connect
    ser.open()
    
    print(f"✓ Port opened successfully")
    print(f"  DTR: {ser.dtr}, RTS: {ser.rts}")
    
    # Wait a bit
    time.sleep(0.5)
    
    # Try to write
    print("  Attempting write...")
    ser.write(b"TEST\n")
    print("  ✓ Write succeeded")
    
    ser.close()
    print("  ✓ Port closed")
    
except Exception as e:
    print(f"  ✗ Failed: {e}")

# Test 2: Try with explicit no flow control
print("\nTest 2: No flow control")
try:
    ser = serial.Serial(
        port=port,
        baudrate=baudrate,
        timeout=1,
        xonxoff=False,
        rtscts=False,
        dsrdtr=False
    )
    
    print(f"✓ Port opened successfully")
    
    # Clear buffers
    ser.reset_input_buffer()
    ser.reset_output_buffer()
    
    # Try to write
    print("  Attempting write...")
    ser.write(b"TEST\n")
    print("  ✓ Write succeeded")
    
    ser.close()
    print("  ✓ Port closed")
    
except Exception as e:
    print(f"  ✗ Failed: {e}")

# Test 3: Try with very short timeout
print("\nTest 3: Short write timeout")
try:
    ser = serial.Serial(
        port=port,
        baudrate=baudrate,
        timeout=0.1,
        write_timeout=0.1,
        dsrdtr=False,
        rtscts=False
    )
    
    print(f"✓ Port opened successfully")
    
    # Try to write
    print("  Attempting write...")
    ser.write(b"TEST\n")
    print("  ✓ Write succeeded")
    
    ser.close()
    print("  ✓ Port closed")
    
except Exception as e:
    print(f"  ✗ Failed: {e}")

print("\n" + "-" * 50)
print("If all tests fail with I/O error, the issue is likely:")
print("1. Arduino sketch not reading serial data (Serial.read())")
print("2. Arduino is stuck/frozen")
print("3. USB cable issue (try a different cable)")
print("4. Linux USB driver issue (try: sudo modprobe -r ch341 && sudo modprobe ch341)")