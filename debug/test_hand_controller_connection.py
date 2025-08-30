#!/usr/bin/env python3
"""
Comprehensive hand controller connection test
"""
import os
import sys
import serial
import time
import subprocess

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("=== COMPREHENSIVE HAND CONTROLLER CONNECTION TEST ===")

# Step 1: Set environment variables
print("1. Setting up environment variables...")
import arduino_port_detector
detector = arduino_port_detector.ArduinoPortDetector()
detector.set_environment_variables()

detected_port = os.environ.get('DETECTED_HAND_PORT')
print(f"   DETECTED_HAND_PORT: {detected_port}")

if not detected_port:
    print("❌ FAILED: No hand controller detected!")
    sys.exit(1)

# Step 2: Check if port is busy
print(f"\n2. Checking if {detected_port} is busy...")
try:
    result = subprocess.run(['lsof', detected_port], capture_output=True, text=True)
    if result.stdout.strip():
        print(f"⚠️  WARNING: Port {detected_port} is in use by:")
        print(result.stdout)
        print("   Try: sudo fuser -k /dev/ttyUSB2")
    else:
        print(f"✅ Port {detected_port} is free")
except FileNotFoundError:
    print("   (lsof not available - can't check port usage)")

# Step 3: Test basic serial connection
print(f"\n3. Testing basic serial connection to {detected_port}...")
try:
    ser = serial.Serial(detected_port, 9600, timeout=2)
    print(f"✅ Serial connection opened successfully")
    
    # Wait for Arduino boot
    print("   Waiting 3 seconds for Arduino boot...")
    time.sleep(3)
    
    # Clear any existing data
    ser.reset_input_buffer()
    ser.reset_output_buffer()
    
    # Send test command
    test_cmd = "HAND:90,90,90,90\n"
    print(f"   Sending test command: {test_cmd.strip()}")
    ser.write(test_cmd.encode())
    time.sleep(0.5)
    
    # Check for response
    if ser.in_waiting > 0:
        response = ser.readline().decode().strip()
        print(f"✅ Arduino responded: '{response}'")
    else:
        print("⚠️  No response from Arduino")
    
    ser.close()
    print("✅ Serial connection test completed")
    
except Exception as e:
    print(f"❌ Serial connection failed: {e}")

# Step 4: Test HandExpressionController
print(f"\n4. Testing HandExpressionController class...")
try:
    from hand_control.hand_expression import HandExpressionController
    
    print("   Creating HandExpressionController...")
    controller = HandExpressionController(clean_output=False)  # Show debug output
    
    if controller.serial_connection:
        print("✅ HandExpressionController connected successfully")
        
        # Test setting positions
        print("   Testing hand position command...")
        controller.set_hand_positions([90, 90, 90, 90])
        time.sleep(1)
        
        # Test emotional expression
        print("   Testing emotional expression...")
        controller.set_expression("neutral")
        time.sleep(1)
        
        print("✅ HandExpressionController test completed")
    else:
        print("❌ HandExpressionController failed to connect")
        
except Exception as e:
    print(f"❌ HandExpressionController test failed: {e}")
    import traceback
    traceback.print_exc()

# Step 5: Final recommendation
print(f"\n=== FINAL DIAGNOSIS ===")
if detected_port:
    print(f"✅ Auto-detection working: Hand controller found on {detected_port}")
    print("✅ Environment variables set correctly")
    print("✅ Hand controller GUI should connect automatically")
    print("\n🔧 NEXT STEPS:")
    print("1. Run: python machine.py")
    print("2. Look for hand controller GUI window")
    print("3. GUI should show 'Connected' status automatically")
    print("4. If still not connecting, check if another program is using the port")
else:
    print("❌ Auto-detection not working - check Arduino connections")