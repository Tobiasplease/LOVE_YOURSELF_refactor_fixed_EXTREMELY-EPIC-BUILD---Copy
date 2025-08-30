#!/usr/bin/env python3
"""
Direct test of hand Arduino - send commands and monitor response
"""
import os
import sys
import serial
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set environment variables
import arduino_port_detector
detector = arduino_port_detector.ArduinoPortDetector()
detector.set_environment_variables()

detected_port = os.environ.get('DETECTED_HAND_PORT')
print(f"=== DIRECT HAND ARDUINO TEST ===")
print(f"Testing port: {detected_port}")

if not detected_port:
    print("❌ No hand controller detected!")
    exit(1)

try:
    # Connect to Arduino
    ser = serial.Serial(detected_port, 9600, timeout=2)
    print(f"✅ Connected to {detected_port}")
    
    # Wait for Arduino boot
    print("⏳ Waiting for Arduino boot (3 seconds)...")
    time.sleep(3)
    
    # Clear buffers
    ser.reset_input_buffer()
    ser.reset_output_buffer()
    
    print("📡 Monitoring Arduino output and sending test commands...")
    print("Format: [TIME] SENT: command -> RECEIVED: response")
    print("-" * 60)
    
    test_commands = [
        "HAND,90,90,90,90",      # All middle
        "HAND,45,45,45,45",      # All bent
        "HAND,135,135,135,135",  # All straight
        "HAND,90,45,135,90",     # Mixed positions
        "HAND,0,180,0,180",      # Extreme positions
        "HEARTBEAT"              # Test heartbeat
    ]
    
    for i, cmd in enumerate(test_commands):
        # Send command
        full_cmd = cmd + "\n"
        ser.write(full_cmd.encode())
        timestamp = time.time()
        print(f"[{timestamp:.1f}] SENT: {cmd}")
        
        # Wait for response (Arduino should echo)
        time.sleep(0.5)
        
        # Read any responses
        responses = []
        while ser.in_waiting > 0:
            try:
                response = ser.readline().decode().strip()
                if response:
                    responses.append(response)
            except:
                break
        
        if responses:
            for resp in responses:
                print(f"[{time.time():.1f}] RECEIVED: {resp}")
        else:
            print(f"[{time.time():.1f}] RECEIVED: (no response)")
        
        print()
        time.sleep(1)  # Pause between commands
    
    # Monitor for any additional output
    print("🔍 Monitoring for 5 more seconds...")
    end_time = time.time() + 5
    while time.time() < end_time:
        if ser.in_waiting > 0:
            try:
                response = ser.readline().decode().strip()
                if response:
                    print(f"[{time.time():.1f}] SPONTANEOUS: {response}")
            except:
                break
        time.sleep(0.1)
    
    ser.close()
    print("✅ Test completed")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n=== DIAGNOSIS ===")
print("If Arduino is working correctly, you should see:")
print("1. 'Consciousness: 90,90,90,90' type responses")  
print("2. 'Heartbeat acknowledged' for HEARTBEAT command")
print("3. Hand should physically move when commands are sent")
print("\nIf no responses, check:")
print("1. Arduino power LED on")
print("2. USB cable connection") 
print("3. Correct Arduino firmware uploaded")