#!/usr/bin/env python3
"""
Precise Arduino Detection - Test each device individually
==========================================================
Test each Arduino with very specific commands to avoid false positives.
Each Arduino should have unique startup messages and command responses.
"""

import serial
import time
from serial.tools import list_ports
from typing import Optional, Dict

def test_lightbulb_arduino(port: str) -> bool:
    """Test if this is the dedicated lightbulb Arduino."""
    try:
        ser = serial.Serial(port, 9600, timeout=2)
        time.sleep(2)  # Arduino boot time
        
        # Read startup message
        startup_msg = ""
        start_time = time.time()
        while time.time() - start_time < 3:
            if ser.in_waiting:
                startup_msg += ser.read(ser.in_waiting).decode('utf-8', errors='ignore')
            time.sleep(0.1)
        
        print(f"  Startup: '{startup_msg.strip()}'")
        
        # Lightbulb-specific startup messages
        lightbulb_indicators = [
            'lightbulb controller ready',
            'frame diff lightbulb',
            'pwm lightbulb',
            'Simple lightbulb controller ready'
        ]
        
        if any(indicator in startup_msg.lower() for indicator in lightbulb_indicators):
            ser.close()
            return True
        
        # Test lightbulb command
        ser.write(b'BASE:100\n')
        time.sleep(0.5)
        
        response = ""
        if ser.in_waiting:
            response = ser.read(ser.in_waiting).decode('utf-8', errors='ignore')
        
        print(f"  BASE:100 response: '{response.strip()}'")
        
        ser.close()
        
        # Should respond with brightness confirmation
        return 'brightness' in response.lower() or 'target' in response.lower()
        
    except Exception as e:
        print(f"  Error: {e}")
        return False

def test_servo_controller_arduino(port: str) -> bool:
    """Test if this is the servo controller (pan/tilt/lung)."""
    try:
        ser = serial.Serial(port, 9600, timeout=2)
        time.sleep(2)
        
        # Read startup message
        startup_msg = ""
        start_time = time.time()
        while time.time() - start_time < 3:
            if ser.in_waiting:
                startup_msg += ser.read(ser.in_waiting).decode('utf-8', errors='ignore')
            time.sleep(0.1)
        
        print(f"  Startup: '{startup_msg.strip()}'")
        
        # Servo controller specific messages (but NOT hand controller)
        servo_indicators = [
            'consciousness mode',
            'ready for consciousness commands',
            '0-180 degree range'
        ]
        
        # Exclude hand controller indicators
        hand_indicators = [
            'hand',
            'finger', 
            'pure consciousness mode'
        ]
        
        has_servo_indicator = any(indicator in startup_msg.lower() for indicator in servo_indicators)
        has_hand_indicator = any(indicator in startup_msg.lower() for indicator in hand_indicators)
        
        if has_servo_indicator and not has_hand_indicator:
            ser.close()
            return True
        
        ser.close()
        return False
        
    except Exception as e:
        print(f"  Error: {e}")
        return False

def test_hand_controller_arduino(port: str) -> bool:
    """Test if this is the hand controller (5 micro servos)."""
    try:
        ser = serial.Serial(port, 9600, timeout=2)
        time.sleep(2)
        
        # Read startup message
        startup_msg = ""
        start_time = time.time()
        while time.time() - start_time < 3:
            if ser.in_waiting:
                startup_msg += ser.read(ser.in_waiting).decode('utf-8', errors='ignore')
            time.sleep(0.1)
        
        print(f"  Startup: '{startup_msg.strip()}'")
        
        # Hand controller specific indicators
        hand_indicators = [
            'hand controller ready',
            'finger',
            'hand',
            'pure consciousness mode'
        ]
        
        if any(indicator in startup_msg.lower() for indicator in hand_indicators):
            ser.close()
            return True
        
        # Test hand-specific command
        ser.write(b'HAND,90,90,90,90,90\n')
        time.sleep(0.5)
        
        response = ""
        if ser.in_waiting:
            response = ser.read(ser.in_waiting).decode('utf-8', errors='ignore')
        
        print(f"  HAND command response: '{response.strip()}'")
        
        ser.close()
        
        # Should acknowledge hand command
        return 'hand' in response.lower() or 'finger' in response.lower() or 'consciousness' in response.lower()
        
    except Exception as e:
        print(f"  Error: {e}")
        return False

def test_grbl_cnc_arduino(port: str) -> bool:
    """Test if this is the GRBL CNC controller."""
    try:
        ser = serial.Serial(port, 115200, timeout=2)  # GRBL uses 115200 baud
        time.sleep(2)
        
        # GRBL sends startup message
        startup_msg = ""
        start_time = time.time()
        while time.time() - start_time < 3:
            if ser.in_waiting:
                startup_msg += ser.read(ser.in_waiting).decode('utf-8', errors='ignore')
            time.sleep(0.1)
        
        print(f"  Startup: '{startup_msg.strip()}'")
        
        if 'grbl' in startup_msg.lower():
            ser.close()
            return True
        
        # Test GRBL status command
        ser.write(b'?\n')
        time.sleep(0.5)
        
        response = ""
        if ser.in_waiting:
            response = ser.read(ser.in_waiting).decode('utf-8', errors='ignore')
        
        print(f"  ? response: '{response.strip()}'")
        
        ser.close()
        
        # GRBL responds with status in angle brackets
        return response.startswith('<') and '>' in response
        
    except Exception as e:
        print(f"  Error: {e}")
        return False

def detect_all_arduinos() -> Dict[str, Optional[str]]:
    """Detect all Arduino types and return port mapping."""
    print("🔍 Precise Arduino Detection")
    print("="*50)
    
    usb_ports = [port.device for port in list_ports.comports() if 'USB' in port.device]
    print(f"Found USB ports: {usb_ports}")
    
    results = {
        'lightbulb': None,
        'servo_controller': None, 
        'hand_controller': None,
        'grbl_cnc': None
    }
    
    for port in usb_ports:
        print(f"\n--- Testing {port} ---")
        
        if test_lightbulb_arduino(port):
            print(f"✅ LIGHTBULB ARDUINO found on {port}")
            results['lightbulb'] = port
        elif test_servo_controller_arduino(port):
            print(f"✅ SERVO CONTROLLER ARDUINO found on {port}")
            results['servo_controller'] = port
        elif test_hand_controller_arduino(port):
            print(f"✅ HAND CONTROLLER ARDUINO found on {port}")
            results['hand_controller'] = port
        elif test_grbl_cnc_arduino(port):
            print(f"✅ GRBL CNC ARDUINO found on {port}")
            results['grbl_cnc'] = port
        else:
            print(f"❌ Unknown/unresponsive device on {port}")
    
    return results

if __name__ == "__main__":
    mapping = detect_all_arduinos()
    
    print("\n" + "="*60)
    print("FINAL ARDUINO MAPPING")
    print("="*60)
    
    for device_type, port in mapping.items():
        status = port if port else "NOT FOUND"
        print(f"{device_type.upper():>20}: {status}")
    
    print("="*60)
    
    # Generate config.py recommendations
    if any(mapping.values()):
        print("\nconfig.py settings:")
        print("-"*30)
        if mapping['servo_controller']:
            print(f'SERIAL_PORT = "{mapping["servo_controller"]}"')
        if mapping['lightbulb']:
            print(f'LIGHTBULB_SERIAL_PORT = "{mapping["lightbulb"]}"')
        if mapping['hand_controller']:
            print(f'HAND_CONTROLLER_PORT = "{mapping["hand_controller"]}"')
        if mapping['grbl_cnc']:
            print(f'GRBL_CNC_PORT = "{mapping["grbl_cnc"]}"')