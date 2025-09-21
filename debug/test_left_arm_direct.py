#!/usr/bin/env python3
"""
Direct test of left arm servo control via SERVO commands
"""
import serial
import time

def test_left_arm_servos():
    """Test direct SERVO commands to left arm."""
    try:
        # Connect to Arduino
        ser = serial.Serial('/dev/arduino_lefthand', 9600, timeout=1)
        time.sleep(2)  # Wait for Arduino to initialize

        print("Testing direct SERVO commands...")

        # Test servo positions
        positions = [80, 90, 100, 90]  # Different positions to test

        for i, pos in enumerate(positions):
            print(f"\n--- Test {i+1}: Moving to {pos}° ---")

            # Send commands to both left arm servos
            servo4_cmd = f"SERVO,4,{pos}\n"
            servo5_cmd = f"SERVO,5,{pos}\n"

            print(f"Sending: {servo4_cmd.strip()}")
            ser.write(servo4_cmd.encode())
            ser.flush()

            print(f"Sending: {servo5_cmd.strip()}")
            ser.write(servo5_cmd.encode())
            ser.flush()

            # Read Arduino response
            time.sleep(0.5)
            while ser.in_waiting:
                response = ser.readline().decode().strip()
                if response:
                    print(f"Arduino: {response}")

            time.sleep(2)  # Wait between positions

        print("\n--- Testing HAND command (should work for fingers) ---")
        hand_cmd = "HAND,90,90,90,90\n"
        print(f"Sending: {hand_cmd.strip()}")
        ser.write(hand_cmd.encode())
        ser.flush()

        time.sleep(0.5)
        while ser.in_waiting:
            response = ser.readline().decode().strip()
            if response:
                print(f"Arduino: {response}")

        ser.close()
        print("\nTest complete!")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_left_arm_servos()