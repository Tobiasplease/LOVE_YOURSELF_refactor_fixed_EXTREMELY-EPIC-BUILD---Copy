#!/usr/bin/env python3
"""
Direct servo limit testing tool
Tests tilt servo at various angles to find actual physical limits
"""
import os
import sys
import time

import serial

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("=" * 60)
print("Servo Limit Testing Tool")
print("=" * 60)

# Connect to Arduino
servo_port = "/dev/arduino_lunggaze"
baudrate = 9600

print(f"\nConnecting to {servo_port} at {baudrate} baud...")

try:
    ser = serial.Serial(servo_port, baudrate, timeout=2)
    time.sleep(2)  # Wait for Arduino to initialize

    # Clear any startup messages
    while ser.in_waiting:
        line = ser.readline().decode("utf-8", errors="ignore").strip()
        print(f"Arduino: {line}")

    print("\n✓ Connected to Arduino\n")

    # Test angles from low to high
    test_angles = [40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 130, 135, 140]

    print("Testing tilt angles (watch the servo movement):")
    print("=" * 60)

    for angle in test_angles:
        command = f"TILT:{angle}\n"
        print(f"\nSending: {command.strip()}")
        ser.write(command.encode("utf-8"))
        ser.flush()

        # Read response
        time.sleep(0.1)
        while ser.in_waiting:
            response = ser.readline().decode("utf-8", errors="ignore").strip()
            print(f"  Arduino: {response}")

        # Wait for servo to move
        time.sleep(1.5)

        user_input = input(f"  Did servo move to {angle}°? (y/n/q to quit): ").lower()
        if user_input == "q":
            print("\nTest stopped by user")
            break
        elif user_input == "n":
            print(f"  ⚠️ Servo did NOT reach {angle}° - likely at physical limit")

    print("\n" + "=" * 60)
    print("Test complete")
    print("\nBased on your observations above:")
    print("- What was the LOWEST angle the servo actually reached?")
    print("- What was the HIGHEST angle the servo actually reached?")

    ser.close()

except FileNotFoundError:
    print(f"❌ Error: {servo_port} not found")
    print("Make sure the Arduino is connected and the device path is correct")
except serial.SerialException as e:
    print(f"❌ Serial connection error: {e}")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback

    traceback.print_exc()
