#!/usr/bin/env python3
"""
Comprehensive Arduino Connection Test
Tests all configured Arduino devices individually
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import (
    SERIAL_PORT, LIGHTBULB_SERIAL_PORT, HAND_CONTROLLER_PORT, 
    GRBL_CNC_PORT, UARM_SWIFT_PORT, USE_LIGHTBULB_PWM, USE_SERVO, USE_HAND_CONTROLLER
)
import serial
import time
from datetime import datetime

class ArduinoTester:
    def __init__(self):
        self.test_results = {}
    
    def test_connection(self, port, device_name, test_commands=None):
        """Test connection to a specific Arduino device"""
        print(f"\n{'='*50}")
        print(f"Testing {device_name}")
        print(f"Port: {port}")
        print(f"{'='*50}")
        
        try:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Opening connection...")
            ser = serial.Serial(port, 9600, timeout=2)
            time.sleep(2)  # Arduino reset time
            
            # Clear any existing data
            if ser.in_waiting:
                initial_data = ser.read(ser.in_waiting).decode('utf-8', errors='ignore')
                print(f"Initial data: {initial_data.strip()}")
            
            success = True
            responses = []
            
            # Send test commands if provided
            if test_commands:
                for cmd, description in test_commands:
                    print(f"\nSending: {cmd} ({description})")
                    ser.write(f"{cmd}\n".encode())
                    time.sleep(0.5)
                    
                    if ser.in_waiting:
                        response = ser.read(ser.in_waiting).decode('utf-8', errors='ignore')
                        print(f"Response: {response.strip()}")
                        responses.append(response.strip())
                    else:
                        print("No response")
                        responses.append("")
            
            ser.close()
            
            self.test_results[device_name] = {
                'port': port,
                'status': 'SUCCESS',
                'responses': responses
            }
            
            print(f"\n✅ {device_name} connection SUCCESS")
            return True
            
        except Exception as e:
            print(f"\n❌ {device_name} connection FAILED: {e}")
            self.test_results[device_name] = {
                'port': port,
                'status': 'FAILED',
                'error': str(e)
            }
            return False
    
    def test_lightbulb_pwm(self):
        """Test lightbulb PWM controller"""
        if not USE_LIGHTBULB_PWM:
            print(f"\n⏭️  Lightbulb PWM disabled in config (USE_LIGHTBULB_PWM = False)")
            return
        
        test_commands = [
            ("BASE:50", "Set base brightness to 50"),
            ("MOOD:0.5:0.2", "Set mood parameters"),
            ("BOOST:500", "Caption boost for 500ms"),
            ("BASE:0", "Turn off")
        ]
        
        return self.test_connection(
            LIGHTBULB_SERIAL_PORT, 
            "Lightbulb PWM Controller", 
            test_commands
        )
    
    def test_servo_lung(self):
        """Test servo lung/gaze system"""
        if not USE_SERVO:
            print(f"\n⏭️  Servo system disabled in config (USE_SERVO = False)")
            return
        
        test_commands = [
            ("90,90,90,90", "Center all servos"),
            ("STATUS", "Request status"),
            ("PING", "Ping test")
        ]
        
        return self.test_connection(
            SERIAL_PORT, 
            "Servo Lung/Gaze System", 
            test_commands
        )
    
    def test_hand_controller(self):
        """Test hand controller"""
        if not USE_HAND_CONTROLLER:
            print(f"\n⏭️  Hand controller disabled in config (USE_HAND_CONTROLLER = False)")
            return
        
        test_commands = [
            ("STATUS", "Request status"),
            ("90,90,90,90", "Center all servos"),
            ("PING", "Ping test")
        ]
        
        return self.test_connection(
            HAND_CONTROLLER_PORT, 
            "Hand Controller", 
            test_commands
        )
    
    def test_grbl_cnc(self):
        """Test GRBL CNC controller"""
        test_commands = [
            ("$", "GRBL help/settings"),
            ("?", "Status query"),
            ("$G", "View G-code parser state")
        ]
        
        return self.test_connection(
            GRBL_CNC_PORT, 
            "GRBL CNC Controller", 
            test_commands
        )
    
    def test_uarm_swift(self):
        """Test uArm Swift Pro controller"""
        test_commands = [
            ("M2400", "Get device info"),
            ("M2401", "Get position"),
            ("P2400", "System status")
        ]
        
        return self.test_connection(
            UARM_SWIFT_PORT, 
            "uArm Swift Pro", 
            test_commands
        )
    
    def run_all_tests(self):
        """Run all Arduino tests"""
        print("ARDUINO CONNECTION TEST SUITE")
        print("="*60)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Test each device
        self.test_lightbulb_pwm()
        self.test_servo_lung()
        self.test_hand_controller()  
        self.test_grbl_cnc()
        self.test_uarm_swift()
        
        # Print summary
        self.print_summary()
    
    def print_summary(self):
        """Print test summary"""
        print(f"\n{'='*60}")
        print("TEST SUMMARY")
        print(f"{'='*60}")
        
        success_count = 0
        total_count = len(self.test_results)
        
        for device_name, result in self.test_results.items():
            status_icon = "✅" if result['status'] == 'SUCCESS' else "❌"
            print(f"{status_icon} {device_name:25} | {result['port']:15} | {result['status']}")
            if result['status'] == 'SUCCESS':
                success_count += 1
        
        print(f"\nResults: {success_count}/{total_count} devices connected successfully")
        
        if success_count == total_count:
            print("🎉 All configured Arduino devices are working!")
        else:
            print(f"\n⚠️  {total_count - success_count} device(s) need attention:")
            print("1. Check physical connections")
            print("2. Verify port assignments in config.py")
            print("3. Check Arduino firmware is uploaded")
            print("4. Run: sudo chmod 666 /dev/ttyUSB*")
            
        print(f"\nTest completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

def main():
    tester = ArduinoTester()
    tester.run_all_tests()

if __name__ == "__main__":
    main()