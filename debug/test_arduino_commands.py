#!/usr/bin/env python3
"""Test Arduino finger commands"""
import serial
import time

try:
    print("Connecting to Arduino on COM3...")
    ser = serial.Serial("COM3", 9600, timeout=1)
    print("SUCCESS: Connected to COM3")
    
    time.sleep(2)  # Arduino boot time
    print("Arduino should have booted, testing finger movements...")
    
    # Test different positions
    test_positions = [
        [90, 90, 90, 90],   # Center
        [45, 45, 45, 45],   # Open
        [135, 135, 135, 135], # Close
        [90, 90, 90, 90],   # Center again
    ]
    
    for i, pos in enumerate(test_positions):
        command = f"HAND,{pos[0]},{pos[1]},{pos[2]},{pos[3]}\n"
        print(f"Step {i+1}: Sending {command.strip()}")
        
        ser.write(command.encode())
        time.sleep(2)  # Give time for movement
        
        # Check for any response
        if ser.in_waiting > 0:
            response = ser.read(ser.in_waiting).decode()
            print(f"  Arduino response: {response.strip()}")
        
        # Ask user to confirm movement
        print(f"  Did you see finger movement? (press Enter to continue)")
        input()
    
    # Test continuous commands
    print("\nTesting continuous commands...")
    for i in range(10):
        # Alternate between open and close
        if i % 2 == 0:
            pos = [60, 60, 60, 60]  # More open
        else:
            pos = [120, 120, 120, 120]  # More closed
            
        command = f"HAND,{pos[0]},{pos[1]},{pos[2]},{pos[3]}\n"
        print(f"Continuous {i+1}: {command.strip()}")
        
        ser.write(command.encode())
        time.sleep(0.5)  # Faster movements
    
    ser.close()
    print("SUCCESS: Test completed")
    
except Exception as e:
    print(f"ERROR: Arduino test failed: {e}")
    import traceback
    traceback.print_exc()