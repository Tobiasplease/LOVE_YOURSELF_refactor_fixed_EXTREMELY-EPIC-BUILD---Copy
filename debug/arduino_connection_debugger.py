#!/usr/bin/env python3
"""
Arduino Connection Debugger
===========================
Comprehensive tool to identify and test Arduino connections for webcam/lung servos.

Diagnoses:
- Port availability and permissions
- Arduino response and protocol compatibility
- Servo control for gaze/breathing systems
- Connection stability and communication errors

Usage:
    python debug/arduino_connection_debugger.py
    python debug/arduino_connection_debugger.py --test-specific /dev/ttyUSB3
"""

import serial
import time
import sys
import os
import argparse
from serial.tools import list_ports
from typing import Optional, Dict, List, Tuple

# Add parent directory to path for config import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import SERIAL_PORT, HAND_CONTROLLER_PORT, LIGHTBULB_SERIAL_PORT, GRBL_CNC_PORT

class ArduinoConnectionDebugger:
    """Debug Arduino connections with comprehensive testing."""
    
    def __init__(self):
        self.test_results = {}
        self.arduino_ports = {
            "Servo/Lung/Gaze": SERIAL_PORT,
            "Hand Controller": HAND_CONTROLLER_PORT, 
            "Lightbulb PWM": LIGHTBULB_SERIAL_PORT,
            "GRBL CNC": GRBL_CNC_PORT
        }
        
    def scan_available_ports(self) -> List[str]:
        """Scan for available serial ports."""
        print("🔍 Scanning available serial ports...")
        ports = list_ports.comports()
        available = []
        
        for port in ports:
            available.append(port.device)
            print(f"  📍 {port.device}: {port.description}")
            if hasattr(port, 'manufacturer') and port.manufacturer:
                print(f"     Manufacturer: {port.manufacturer}")
                
        if not available:
            print("  ❌ No serial ports found!")
        
        return available
        
    def test_port_permissions(self, port: str) -> bool:
        """Test if we have read/write permissions to the port."""
        try:
            with serial.Serial(port, 9600, timeout=0.5) as ser:
                return True
        except serial.SerialException as e:
            if "Permission denied" in str(e):
                print(f"  ❌ Permission denied - try: sudo chmod 666 {port}")
                return False
            elif "No such file" in str(e):
                print(f"  ❌ Port {port} does not exist")
                return False
            else:
                print(f"  ❌ Serial error: {e}")
                return False
        except Exception as e:
            print(f"  ❌ Unexpected error: {e}")
            return False
            
    def test_arduino_response(self, port: str, timeout: float = 3.0) -> Dict:
        """Test Arduino response and identify protocol."""
        result = {
            "connected": False,
            "responds": False,
            "protocol": "unknown",
            "ready_message": None,
            "test_response": None,
            "error": None
        }
        
        try:
            print(f"  🔗 Connecting to {port}...")
            with serial.Serial(port, 9600, timeout=timeout) as ser:
                time.sleep(2)  # Arduino boot time
                result["connected"] = True
                
                # Check for startup messages
                startup_data = ""
                for _ in range(10):  # Check for 1 second
                    if ser.in_waiting:
                        data = ser.read(ser.in_waiting).decode('utf-8', errors='ignore')
                        startup_data += data
                    time.sleep(0.1)
                    
                if startup_data:
                    result["ready_message"] = startup_data.strip()
                    print(f"     📥 Startup: {startup_data.strip()}")
                    
                # Test servo commands for gaze/lung system
                print("  🎯 Testing servo commands...")
                test_commands = [
                    ("PAN:90", "pan servo"),
                    ("TILT:90", "tilt servo"), 
                    ("LUNG:90", "lung servo"),
                    ("LUNG:hold", "lung hold mode")
                ]
                
                responses = []
                for command, desc in test_commands:
                    print(f"     📤 Sending: {command}")
                    ser.write((command + "\n").encode())
                    time.sleep(0.5)
                    
                    response = ""
                    if ser.in_waiting:
                        response = ser.read(ser.in_waiting).decode('utf-8', errors='ignore').strip()
                        print(f"     📥 Response: {response}")
                        
                    responses.append(f"{command} -> {response}")
                    
                result["test_response"] = "; ".join(responses)
                result["responds"] = True
                
                # Determine protocol based on responses
                if "Received:" in startup_data or any("Received:" in r for r in responses):
                    result["protocol"] = "servo_gaze_lung"
                elif "HAND" in startup_data:
                    result["protocol"] = "hand_controller"
                elif "Grbl" in startup_data:
                    result["protocol"] = "grbl_cnc"
                else:
                    result["protocol"] = "unknown"
                    
        except serial.SerialException as e:
            result["error"] = f"Serial error: {e}"
        except Exception as e:
            result["error"] = f"Unexpected error: {e}"
            
        return result
        
    def test_servo_movement(self, port: str) -> bool:
        """Test actual servo movement for gaze system."""
        try:
            print(f"  🎭 Testing servo movement on {port}...")
            with serial.Serial(port, 9600, timeout=2) as ser:
                time.sleep(2)
                
                # Test sequence: center -> left -> center -> right -> center
                movements = [
                    ("PAN:90", "center"),
                    ("PAN:45", "left"),
                    ("PAN:90", "center"),
                    ("PAN:135", "right"),
                    ("PAN:90", "center"),
                    ("TILT:45", "tilt up"),
                    ("TILT:90", "tilt center"),
                    ("TILT:135", "tilt down"),
                    ("TILT:90", "tilt center")
                ]
                
                for command, desc in movements:
                    print(f"     🎯 {desc}: {command}")
                    ser.write((command + "\n").encode())
                    time.sleep(1.5)  # Allow time for movement
                    
                # Test lung breathing
                print("     🫁 Testing lung movement...")
                ser.write(b"LUNG:60\n")
                time.sleep(1)
                ser.write(b"LUNG:120\n") 
                time.sleep(1)
                ser.write(b"LUNG:90\n")
                time.sleep(1)
                
                return True
                
        except Exception as e:
            print(f"     ❌ Movement test failed: {e}")
            return False
            
    def run_comprehensive_test(self, specific_port: Optional[str] = None):
        """Run comprehensive connection test."""
        print("🔧 Arduino Connection Debugger")
        print("=" * 50)
        
        # Scan available ports
        available_ports = self.scan_available_ports()
        print()
        
        # Test configured ports or specific port
        ports_to_test = {}
        if specific_port:
            ports_to_test["Specific Port"] = specific_port
        else:
            ports_to_test = self.arduino_ports
            
        for name, port in ports_to_test.items():
            print(f"🧪 Testing {name}: {port}")
            print("-" * 40)
            
            # Check if port exists
            if port not in available_ports:
                print(f"  ❌ Port {port} not found in available ports")
                self.test_results[name] = {"error": "Port not found"}
                print()
                continue
                
            # Test permissions
            if not self.test_port_permissions(port):
                self.test_results[name] = {"error": "Permission denied"}
                print()
                continue
                
            # Test Arduino response
            result = self.test_arduino_response(port)
            self.test_results[name] = result
            
            if result["connected"]:
                print(f"  ✅ Connected successfully")
                print(f"  📋 Protocol: {result['protocol']}")
                
                # Test movement if it's a servo controller
                if result["protocol"] == "servo_gaze_lung" and name == "Servo/Lung/Gaze":
                    movement_ok = self.test_servo_movement(port)
                    result["movement_test"] = movement_ok
                    if movement_ok:
                        print("  ✅ Servo movement test passed")
                    else:
                        print("  ⚠️ Servo movement test failed")
            else:
                print(f"  ❌ Connection failed: {result.get('error', 'Unknown error')}")
                
            print()
            
    def print_summary(self):
        """Print test summary and recommendations."""
        print("📊 Test Summary")
        print("=" * 50)
        
        working_ports = []
        problem_ports = []
        
        for name, result in self.test_results.items():
            if result.get("connected") and result.get("responds"):
                working_ports.append(name)
                print(f"✅ {name}: Working")
            else:
                problem_ports.append(name)
                error = result.get("error", "Connection/response failed")
                print(f"❌ {name}: {error}")
                
        print()
        
        if problem_ports:
            print("🔧 Troubleshooting Recommendations:")
            for name in problem_ports:
                port = self.arduino_ports.get(name, "unknown")
                print(f"  📌 {name} ({port}):")
                result = self.test_results.get(name, {})
                
                if "Permission denied" in result.get("error", ""):
                    print(f"     - Run: sudo chmod 666 {port}")
                    print(f"     - Or add user to dialout group: sudo usermod -a -G dialout $USER")
                elif "Port not found" in result.get("error", ""):
                    print(f"     - Check if Arduino is connected")
                    print(f"     - Try different USB port")
                    print(f"     - Update config.py with correct port")
                elif not result.get("responds"):
                    print(f"     - Check Arduino firmware (should be Lint-arduinoserial.ino)")
                    print(f"     - Verify baud rate (9600)")
                    print(f"     - Try Arduino IDE Serial Monitor to test manually")
                    
        print()
        print("📋 Next Steps:")
        if not working_ports:
            print("  1. Fix permission issues with: sudo chmod 666 /dev/ttyUSB*")
            print("  2. Upload correct firmware to Arduinos")
            print("  3. Update config.py port assignments")
        else:
            print("  1. Update config.py to use working ports")
            print("  2. Test with main application: python machine.py")
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Debug Arduino connections")
    parser.add_argument("--test-specific", help="Test specific port only")
    args = parser.parse_args()
    
    debugger = ArduinoConnectionDebugger()
    debugger.run_comprehensive_test(args.test_specific)
    debugger.print_summary()