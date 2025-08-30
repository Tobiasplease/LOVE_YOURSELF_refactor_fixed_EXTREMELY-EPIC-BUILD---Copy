#!/usr/bin/env python3
"""
Direct Arduino communication test
Tests each port with different protocols to identify device types
"""

import serial
import serial.tools.list_ports
import time
import json
import sys

def test_servo_protocol(ser):
    """Test servo/lung Arduino protocol"""
    tests = [
        b"90\n",  # Simple servo position
        b"S90\n",  # Servo command
        b"P90\n",  # Position command
        b"SERVO:90\n",  # Explicit servo command
    ]
    
    for test_cmd in tests:
        ser.write(test_cmd)
        time.sleep(0.1)
        if ser.in_waiting:
            response = ser.read(ser.in_waiting).decode('utf-8', errors='ignore')
            if response.strip():
                return f"Servo response to {test_cmd.strip()}: {response.strip()}"
    return None

def test_lightbulb_protocol(ser):
    """Test lightbulb PWM Arduino protocol"""
    tests = [
        b"BASE:100\n",  # Base brightness command
        b"MOOD:1.0,0.5\n",  # Mood command
        json.dumps({"frame_diff": 0.5}).encode() + b"\n",  # JSON frame diff
        b"B:128\n",  # Direct brightness command
        b"255\n",  # Direct PWM value
    ]
    
    for test_cmd in tests:
        ser.write(test_cmd)
        time.sleep(0.1)
        if ser.in_waiting:
            response = ser.read(ser.in_waiting).decode('utf-8', errors='ignore')
            if response.strip():
                return f"Lightbulb response to {test_cmd.strip()}: {response.strip()}"
    return None

def test_hand_protocol(ser):
    """Test hand controller Arduino protocol"""
    # Hand controller seems to send data continuously
    time.sleep(0.5)
    if ser.in_waiting:
        response = ser.read(ser.in_waiting).decode('utf-8', errors='ignore')
        if "Consciousness" in response or "heart" in response:
            return f"Hand controller detected (sending: {response[:50]}...)"
    return None

def identify_arduino(port):
    """Try to identify what type of Arduino is on this port"""
    print(f"\n{'='*60}")
    print(f"Testing port: {port}")
    print(f"{'='*60}")
    
    try:
        # Try different baud rates
        for baud in [9600, 115200]:
            print(f"Trying {baud} baud...")
            ser = serial.Serial(port, baud, timeout=0.5)
            time.sleep(2)  # Arduino reset time
            
            # Clear buffer
            ser.reset_input_buffer()
            ser.reset_output_buffer()
            
            # Check for automatic data (hand controller)
            result = test_hand_protocol(ser)
            if result:
                print(f"✓ {result}")
                ser.close()
                return "hand_controller", baud
            
            # Test servo protocol
            result = test_servo_protocol(ser)
            if result:
                print(f"✓ {result}")
                ser.close()
                return "servo_lung", baud
            
            # Test lightbulb protocol
            result = test_lightbulb_protocol(ser)
            if result:
                print(f"✓ {result}")
                ser.close()
                return "lightbulb_pwm", baud
            
            ser.close()
        
        print("✗ No recognized protocol found")
        return None, None
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return None, None

def main():
    ports = serial.tools.list_ports.comports()
    usb_ports = [p.device for p in ports if 'USB' in p.device or 'ACM' in p.device]
    
    print("USB Serial Ports Found:", usb_ports)
    
    results = {}
    for port in usb_ports:
        device_type, baud = identify_arduino(port)
        if device_type:
            results[port] = (device_type, baud)
    
    print(f"\n{'='*60}")
    print("IDENTIFICATION RESULTS:")
    print(f"{'='*60}")
    
    if results:
        for port, (device_type, baud) in results.items():
            print(f"{port}: {device_type} @ {baud} baud")
        
        print(f"\n{'='*60}")
        print("SUGGESTED CONFIG.PY SETTINGS:")
        print(f"{'='*60}")
        
        for port, (device_type, baud) in results.items():
            if device_type == "servo_lung":
                print(f'SERIAL_PORT = "{port}"  # Servo/lung system')
            elif device_type == "lightbulb_pwm":
                print(f'LIGHTBULB_SERIAL_PORT = "{port}"  # Lightbulb PWM')
            elif device_type == "hand_controller":
                print(f'HAND_CONTROLLER_PORT = "{port}"  # Hand controller')
    else:
        print("No Arduino devices could be identified")
        print("\nTroubleshooting:")
        print("1. Make sure Arduino sketches are uploaded")
        print("2. Check that no other programs are using the ports")
        print("3. Try unplugging and reconnecting the Arduinos")

if __name__ == "__main__":
    main()