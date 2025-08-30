#!/usr/bin/env python3
"""
Auto-detect Arduino ports by testing actual functionality
=========================================================
Test each port with the specific commands each device expects.
"""

import serial
import time
from serial.tools import list_ports

def find_servo_port():
    """Find servo controller by testing PAN/TILT commands."""
    print("🔍 Searching for servo controller...")
    
    for port_info in list_ports.comports():
        if 'USB' not in port_info.device:
            continue
            
        port = port_info.device
        try:
            ser = serial.Serial(port, 9600, timeout=1)
            time.sleep(2)  # Arduino boot time
            
            # Test with servo command
            ser.write(b'PAN:90\n')
            time.sleep(0.5)
            
            # Check for servo-style response
            response = ''
            if ser.in_waiting:
                response = ser.read(ser.in_waiting).decode('utf-8', errors='ignore').strip()
            
            if 'Ready for consciousness' in response or 'Hand Controller Ready' in response:
                print(f"✅ Found servo controller on {port}")
                ser.close()
                return port
                
            ser.close()
            
        except Exception as e:
            print(f"❌ {port}: {e}")
            
    return None

def find_lightbulb_port():
    """Find lightbulb controller by testing brightness commands."""
    print("🔍 Searching for lightbulb controller...")
    
    for port_info in list_ports.comports():
        if 'USB' not in port_info.device:
            continue
            
        port = port_info.device
        try:
            ser = serial.Serial(port, 9600, timeout=1)
            time.sleep(2)
            
            # Test with lightbulb command
            ser.write(b'BASE:50\n')
            time.sleep(0.5)
            
            response = ''
            if ser.in_waiting:
                response = ser.read(ser.in_waiting).decode('utf-8', errors='ignore').strip()
            
            # Lightbulb should respond to BASE commands
            if 'brightness' in response.lower() or 'base' in response.lower():
                print(f"✅ Found lightbulb controller on {port}")
                ser.close()
                return port
                
            ser.close()
            
        except Exception as e:
            print(f"❌ {port}: {e}")
            
    return None

def find_hand_controller_port():
    """Find hand controller by testing hand position commands."""
    print("🔍 Searching for hand controller...")
    
    for port_info in list_ports.comports():
        if 'USB' not in port_info.device:
            continue
            
        port = port_info.device
        try:
            ser = serial.Serial(port, 9600, timeout=1)
            time.sleep(2)
            
            # Test with hand command
            ser.write(b'HAND,90,90,90,90\n')
            time.sleep(0.5)
            
            response = ''
            if ser.in_waiting:
                response = ser.read(ser.in_waiting).decode('utf-8', errors='ignore').strip()
            
            # Look for hand-specific responses
            if 'hand' in response.lower() or 'finger' in response.lower():
                print(f"✅ Found hand controller on {port}")
                ser.close()
                return port
                
            ser.close()
            
        except Exception as e:
            print(f"❌ {port}: {e}")
            
    return None

if __name__ == "__main__":
    print("🔍 Auto-detecting Arduino ports...")
    
    servo_port = find_servo_port()
    lightbulb_port = find_lightbulb_port() 
    hand_port = find_hand_controller_port()
    
    print("\n" + "="*50)
    print("DETECTED DEVICES")
    print("="*50)
    print(f"Servo Controller:  {servo_port or 'NOT FOUND'}")
    print(f"Lightbulb:         {lightbulb_port or 'NOT FOUND'}")  
    print(f"Hand Controller:   {hand_port or 'NOT FOUND'}")
    print("="*50)
    
    # Generate config recommendations
    if servo_port or lightbulb_port or hand_port:
        print("\nRecommended config.py updates:")
        if servo_port:
            print(f'SERIAL_PORT = "{servo_port}"')
        if lightbulb_port:
            print(f'LIGHTBULB_SERIAL_PORT = "{lightbulb_port}"')
        if hand_port:
            print(f'HAND_CONTROLLER_PORT = "{hand_port}"')