#!/usr/bin/env python3
"""
Arduino Identification Tool for Linux
Systematically identifies and tests all connected Arduino devices
"""

import serial
import serial.tools.list_ports
import time
import threading
from datetime import datetime

class ArduinoIdentifier:
    def __init__(self):
        self.identified_devices = {}
        
    def list_all_ports(self):
        """List all available serial ports"""
        print("=== Available Serial Ports ===")
        ports = serial.tools.list_ports.comports()
        
        if not ports:
            print("No serial ports found!")
            return []
            
        for port in ports:
            print(f"Port: {port.device}")
            print(f"  Description: {port.description}")
            print(f"  Manufacturer: {port.manufacturer}")
            print(f"  VID:PID: {port.vid:04X}:{port.pid:04X}" if port.vid and port.pid else "  VID:PID: Unknown")
            print(f"  Serial Number: {port.serial_number}")
            print()
        
        return [port.device for port in ports]
    
    def test_arduino_response(self, port, baudrate=9600, timeout=3):
        """Test if a port responds like an Arduino"""
        try:
            print(f"Testing {port}...")
            ser = serial.Serial(port, baudrate, timeout=1)
            time.sleep(2)  # Arduino reset time
            
            # Clear any existing data
            if ser.in_waiting:
                ser.read(ser.in_waiting)
            
            # Test different Arduino types
            responses = {}
            
            # Test 1: Basic Arduino (should stay silent or respond to commands)
            ser.write(b"?\n")
            time.sleep(0.5)
            if ser.in_waiting:
                responses['basic'] = ser.read(ser.in_waiting).decode('utf-8', errors='ignore')
            
            # Test 2: GRBL (responds to $ commands)
            ser.write(b"$\n")
            time.sleep(0.5)
            if ser.in_waiting:
                responses['grbl'] = ser.read(ser.in_waiting).decode('utf-8', errors='ignore')
            
            # Test 3: Custom lightbulb controller
            ser.write(b"BASE:50\n")
            time.sleep(0.5)
            if ser.in_waiting:
                responses['lightbulb'] = ser.read(ser.in_waiting).decode('utf-8', errors='ignore')
            
            # Test 4: Hand controller (might respond to specific commands)
            ser.write(b"STATUS\n")
            time.sleep(0.5)
            if ser.in_waiting:
                responses['hand'] = ser.read(ser.in_waiting).decode('utf-8', errors='ignore')
            
            ser.close()
            return responses
            
        except Exception as e:
            print(f"  Error testing {port}: {e}")
            return None
    
    def identify_device_type(self, port, responses):
        """Identify what type of Arduino this is based on responses"""
        if not responses:
            return "unknown"
        
        device_type = "unknown"
        confidence = "low"
        
        for test_type, response in responses.items():
            response = response.strip().lower()
            
            # GRBL identification
            if 'grbl' in response or 'idle' in response or 'mpos' in response:
                device_type = "grbl_cnc"
                confidence = "high"
                break
            
            # Lightbulb controller identification
            elif 'lightbulb' in response or 'brightness' in response or 'autonomous' in response:
                device_type = "lightbulb_pwm"
                confidence = "high"
                break
            
            # Hand controller identification
            elif 'hand' in response or 'gesture' in response or 'servo' in response:
                device_type = "hand_controller"
                confidence = "medium"
                break
            
            # Basic Arduino (responds but unclear what type)
            elif response and len(response) > 0:
                device_type = "arduino_basic"
                confidence = "low"
        
        return device_type, confidence
    
    def scan_all_arduinos(self):
        """Scan and identify all connected Arduino devices"""
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting Arduino identification scan...\n")
        
        ports = self.list_all_ports()
        
        if not ports:
            return
        
        print("=== Testing Arduino Responses ===")
        
        for port in ports:
            print(f"\n--- Testing {port} ---")
            
            responses = self.test_arduino_response(port)
            if responses:
                device_type, confidence = self.identify_device_type(port, responses)
                
                self.identified_devices[port] = {
                    'type': device_type,
                    'confidence': confidence,
                    'responses': responses
                }
                
                print(f"  Identified as: {device_type} (confidence: {confidence})")
                if responses:
                    for test, response in responses.items():
                        if response.strip():
                            print(f"    {test}: {response.strip()[:100]}")
            else:
                print(f"  No response or connection failed")
    
    def generate_config_recommendations(self):
        """Generate configuration recommendations based on identified devices"""
        print("\n" + "="*60)
        print("CONFIGURATION RECOMMENDATIONS")
        print("="*60)
        
        if not self.identified_devices:
            print("No Arduino devices identified.")
            return
        
        # Create recommendations
        recommendations = {
            'lightbulb_pwm': None,
            'hand_controller': None,
            'servo_lung': None,
            'grbl_cnc': None,
            'uarm_swift': None
        }
        
        for port, info in self.identified_devices.items():
            device_type = info['type']
            
            if device_type == 'lightbulb_pwm':
                recommendations['lightbulb_pwm'] = port
            elif device_type == 'hand_controller':
                recommendations['hand_controller'] = port
            elif device_type == 'grbl_cnc':
                recommendations['grbl_cnc'] = port
            elif device_type == 'arduino_basic':
                # Could be servo lung or uArm Swift
                if recommendations['servo_lung'] is None:
                    recommendations['servo_lung'] = port
                elif recommendations['uarm_swift'] is None:
                    recommendations['uarm_swift'] = port
        
        print("\nRecommended config.py settings:")
        print("-" * 40)
        
        if recommendations['lightbulb_pwm']:
            print(f'LIGHTBULB_SERIAL_PORT = "{recommendations["lightbulb_pwm"]}"')
        else:
            print('LIGHTBULB_SERIAL_PORT = "/dev/ttyUSB0"  # UPDATE MANUALLY')
        
        if recommendations['servo_lung']:
            print(f'SERIAL_PORT = "{recommendations["servo_lung"]}"  # Servo lung')
        else:
            print('SERIAL_PORT = "/dev/ttyUSB1"  # UPDATE MANUALLY - Servo lung')
        
        if recommendations['hand_controller']:
            print(f'HAND_CONTROLLER_PORT = "{recommendations["hand_controller"]}"')
        else:
            print('HAND_CONTROLLER_PORT = "/dev/ttyUSB2"  # UPDATE MANUALLY')
        
        if recommendations['grbl_cnc']:
            print(f'GRBL_CNC_PORT = "{recommendations["grbl_cnc"]}"')
        else:
            print('GRBL_CNC_PORT = "/dev/ttyUSB3"  # UPDATE MANUALLY')
        
        if recommendations['uarm_swift']:
            print(f'UARM_SWIFT_PORT = "{recommendations["uarm_swift"]}"')
        else:
            print('UARM_SWIFT_PORT = "/dev/ttyUSB4"  # UPDATE MANUALLY')
        
        print("\nManual verification recommended for each device!")
        print("Use individual test scripts to confirm each assignment.")

if __name__ == "__main__":
    identifier = ArduinoIdentifier()
    identifier.scan_all_arduinos()
    identifier.generate_config_recommendations()