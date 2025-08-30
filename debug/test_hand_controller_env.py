#!/usr/bin/env python3
"""
Test hand controller environment variable access
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set the environment variable like machine.py does
import arduino_port_detector
detector = arduino_port_detector.ArduinoPortDetector()
detector.set_environment_variables()

print("=== ENVIRONMENT VARIABLE TEST ===")
print(f"DETECTED_HAND_PORT: {os.environ.get('DETECTED_HAND_PORT', 'NOT SET')}")
print(f"DETECTED_HAND_CONTROLLER_PORT: {os.environ.get('DETECTED_HAND_CONTROLLER_PORT', 'NOT SET')}")

# Test what the hand controller interface sees
try:
    from config.config import HAND_CONTROLLER_PORT as original_port
    print(f"Original config HAND_CONTROLLER_PORT: {original_port}")
    
    # Simulate what hand_control_interface.py does
    HAND_CONTROLLER_PORT = original_port
    if 'DETECTED_HAND_PORT' in os.environ:
        HAND_CONTROLLER_PORT = os.environ['DETECTED_HAND_PORT']
        print(f"[AUTO-DETECT] Using detected hand controller port: {HAND_CONTROLLER_PORT}")
    else:
        print("[NO AUTO-DETECT] Using config file port")
        
    print(f"Final port that GUI should use: {HAND_CONTROLLER_PORT}")
    
except ImportError as e:
    print(f"Error importing config: {e}")

print("\n=== MANUAL CONNECTION TEST ===")
print("Testing if we can connect to the detected port manually...")

try:
    import serial
    import time
    
    detected_port = os.environ.get('DETECTED_HAND_PORT')
    if detected_port:
        print(f"Attempting connection to {detected_port}...")
        ser = serial.Serial(detected_port, 9600, timeout=2)
        time.sleep(1)  # Wait for Arduino
        
        # Send a test command
        test_command = "THUMB:90\n"
        ser.write(test_command.encode())
        time.sleep(0.1)
        
        # Check for response
        if ser.in_waiting > 0:
            response = ser.readline().decode().strip()
            print(f"Arduino responded: {response}")
        else:
            print("No response from Arduino")
            
        ser.close()
        print("✅ Manual connection successful!")
        
    else:
        print("❌ No detected port to test")
        
except Exception as e:
    print(f"❌ Connection failed: {e}")