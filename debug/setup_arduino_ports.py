#!/usr/bin/env python3
"""
Arduino Port Setup Assistant
Helps systematically identify and configure all Arduino devices
"""

import serial.tools.list_ports
import subprocess
import os

def check_permissions():
    """Check if user has permissions to access serial ports"""
    print("🔧 Checking serial port permissions...")
    
    # List USB serial ports
    ports = [port.device for port in serial.tools.list_ports.comports() if 'USB' in port.device]
    
    if not ports:
        print("⚠️  No USB serial ports found. Make sure Arduinos are connected.")
        return
    
    # Check permissions for each port
    need_fix = []
    for port in ports:
        try:
            with open(port, 'r'):
                print(f"✅ {port} - permissions OK")
        except PermissionError:
            print(f"❌ {port} - permission denied")
            need_fix.append(port)
        except Exception as e:
            print(f"⚠️  {port} - {e}")
    
    if need_fix:
        print(f"\n🔧 To fix permissions, run:")
        print(f"sudo chmod 666 {' '.join(need_fix)}")
        print("or")
        print("sudo chmod 666 /dev/ttyUSB*")
        print("sudo chmod 666 /dev/ttyACM*")

def setup_guide():
    """Interactive setup guide"""
    print("ARDUINO SETUP GUIDE")
    print("="*50)
    
    print("\n1. PHYSICAL CONNECTIONS")
    print("   Connect your 5 Arduino devices via USB")
    print("   Recommended order:")
    print("   - Servo lung/gaze system")
    print("   - Lightbulb PWM controller") 
    print("   - Hand controller")
    print("   - GRBL CNC controller")
    print("   - uArm Swift Pro")
    
    print("\n2. CHECK PERMISSIONS")
    check_permissions()
    
    print("\n3. IDENTIFY DEVICES")
    print("   Run: python debug/identify_arduinos.py")
    print("   This will scan all ports and suggest configurations")
    
    print("\n4. TEST CONNECTIONS")
    print("   Run: python debug/test_all_arduinos.py")
    print("   This will test each configured device individually")
    
    print("\n5. UPDATE CONFIG")
    print("   Edit config/config.py with correct port assignments:")
    print("   - SERIAL_PORT (servo lung)")
    print("   - LIGHTBULB_SERIAL_PORT") 
    print("   - HAND_CONTROLLER_PORT")
    print("   - GRBL_CNC_PORT")
    print("   - UARM_SWIFT_PORT")
    
    print("\n6. FINAL VERIFICATION")
    print("   Run: python machine.py --debug")
    print("   Check that all devices initialize properly")

if __name__ == "__main__":
    setup_guide()