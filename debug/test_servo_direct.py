#!/usr/bin/env python3
"""
Direct Servo Arduino Test
=========================
Tests direct communication with the gaze/breathing servo Arduino
"""

import sys
import os
import time
import serial

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import SERIAL_PORT

def test_servo_arduino():
    """Test direct servo Arduino communication."""
    print("=" * 50)
    print("SERVO ARDUINO DIRECT TEST")
    print("=" * 50)
    
    print(f"Connecting to servo Arduino at: {SERIAL_PORT}")
    
    try:
        ser = serial.Serial(SERIAL_PORT, 9600, timeout=2)
        time.sleep(2)  # Arduino boot time
        
        print("✅ Connected successfully!")
        
        # Check for startup message
        if ser.in_waiting:
            startup = ser.read(ser.in_waiting).decode('utf-8', errors='ignore').strip()
            print(f"📥 Startup message: '{startup}'")
        else:
            print("📥 No startup message received")
        
        # Test servo commands
        test_commands = [
            "PAN:90",
            "TILT:90", 
            "LUNG:90",
            "PAN:45",
            "TILT:135",
            "LUNG:60",
            "PAN:135",
            "TILT:45", 
            "LUNG:120",
            "PAN:90",
            "TILT:90",
            "LUNG:90"
        ]
        
        print("\n🎯 Testing servo commands...")
        for i, command in enumerate(test_commands):
            print(f"  [{i+1:2d}/12] Sending: {command}")
            
            # Send command
            ser.write((command + "\n").encode())
            time.sleep(0.5)
            
            # Check for response
            if ser.in_waiting:
                response = ser.read(ser.in_waiting).decode('utf-8', errors='ignore').strip()
                print(f"         Response: '{response}'")
            else:
                print("         No response")
                
            time.sleep(1.5)  # Allow time for servo movement
        
        # Test lung breathing modes
        print("\n🫁 Testing lung breathing modes...")
        lung_commands = [
            "LUNG:hold",
            "LUNG:slow" 
        ]
        
        for command in lung_commands:
            print(f"  Sending: {command}")
            ser.write((command + "\n").encode())
            time.sleep(0.5)
            
            if ser.in_waiting:
                response = ser.read(ser.in_waiting).decode('utf-8', errors='ignore').strip()
                print(f"  Response: '{response}'")
            else:
                print("  No response")
                
            time.sleep(2)  # Observe breathing
            
        ser.close()
        print("\n✅ Test complete!")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_servo_arduino()