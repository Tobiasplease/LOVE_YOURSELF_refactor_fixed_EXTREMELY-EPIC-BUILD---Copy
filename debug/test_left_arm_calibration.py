#!/usr/bin/env python3
"""
Left Arm Calibration Test Script
================================

Slowly moves the left arm servos through their range for position calibration.
Use this to determine the optimal center position and range.
"""

import os
import sys
import time
import serial

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

def find_arduino_port():
    """Find the Arduino port."""
    possible_ports = ["/dev/arduino_lefthand", "/dev/ttyUSB0", "/dev/ttyACM0", "/dev/ttyUSB1", "/dev/ttyACM1"]

    for port in possible_ports:
        if os.path.exists(port):
            try:
                ser = serial.Serial(port, 9600, timeout=1)
                time.sleep(2)  # Allow Arduino to reset
                print(f"✅ Connected to Arduino at {port}")
                return ser
            except Exception as e:
                print(f"❌ Could not connect to {port}: {e}")
                continue

    print("❌ No Arduino found on any port")
    return None

def send_servo_command(ser, pin, angle):
    """Send servo command to Arduino."""
    try:
        command = f"SERVO,{pin},{angle}\n"
        ser.write(command.encode())
        ser.flush()
        print(f"📤 Sent: Pin {pin} -> {angle}°")
        time.sleep(0.1)  # Small delay for serial
        return True
    except Exception as e:
        print(f"❌ Error sending command: {e}")
        return False

def calibration_sweep(ser, pin, start_angle, end_angle, step=2, delay=1.0):
    """Sweep servo from start to end angle."""
    print(f"\n🔄 Sweeping Pin {pin} from {start_angle}° to {end_angle}° (step: {step}°, delay: {delay}s)")

    current = start_angle
    direction = 1 if end_angle > start_angle else -1

    while (direction == 1 and current <= end_angle) or (direction == -1 and current >= end_angle):
        send_servo_command(ser, pin, current)
        time.sleep(delay)
        current += step * direction

def interactive_calibration_bounded(ser, pin):
    """Interactive calibration within safe bounds (80°-100°)."""
    print(f"\n🎛️  Interactive Calibration for Pin {pin} (Range: 80°-100°)")
    print("Commands:")
    print("  +/- : Increase/decrease angle by 1°")
    print("  ++/-- : Increase/decrease angle by 2°")
    print("  [80-100] : Set specific angle within range")
    print("  q : Quit interactive mode")

    current_angle = 90  # Start at center
    send_servo_command(ser, pin, current_angle)

    while True:
        try:
            cmd = input(f"Pin {pin} @ {current_angle}° > ").strip()

            if cmd == 'q':
                break
            elif cmd == '+':
                current_angle = min(100, current_angle + 1)
                send_servo_command(ser, pin, current_angle)
            elif cmd == '-':
                current_angle = max(80, current_angle - 1)
                send_servo_command(ser, pin, current_angle)
            elif cmd == '++':
                current_angle = min(100, current_angle + 2)
                send_servo_command(ser, pin, current_angle)
            elif cmd == '--':
                current_angle = max(80, current_angle - 2)
                send_servo_command(ser, pin, current_angle)
            elif cmd.isdigit():
                angle = int(cmd)
                if 80 <= angle <= 100:
                    current_angle = angle
                    send_servo_command(ser, pin, current_angle)
                else:
                    print("❌ Angle must be between 80° and 100°")
            else:
                print("❌ Unknown command")

        except KeyboardInterrupt:
            break
        except ValueError:
            print("❌ Invalid input")

def main():
    print("🤖 Left Arm Calibration Test")
    print("=" * 40)

    # Connect to Arduino
    ser = find_arduino_port()
    if not ser:
        return

    try:
        # Test both servo pins (4 and 5)
        pins = [4, 5]

        print("\n📋 Calibration Menu (Range: 80°-100°):")
        print("1. Slow range sweep (80°-100°)")
        print("2. Interactive calibration (within range)")
        print("3. Center position test (90°)")
        print("4. Min/Center/Max test")

        choice = input("\nSelect option (1-4): ").strip()

        if choice == '1':
            # Range sweep within bounds
            for pin in pins:
                print(f"\n🔄 Range sweep for Pin {pin} (80°-100°)")
                calibration_sweep(ser, pin, 80, 100, step=1, delay=1.0)
                input("Press Enter to continue to next pin...")

        elif choice == '2':
            # Interactive calibration within bounds
            for pin in pins:
                interactive_calibration_bounded(ser, pin)

        elif choice == '3':
            # Center position test
            print("\n📍 Testing center position (90°)")
            for pin in pins:
                send_servo_command(ser, pin, 90)
                time.sleep(1)
            print("✅ Both servos at 90° center position")

        elif choice == '4':
            # Range test with current settings
            print("\n📏 Testing range positions (80°/90°/100°)")
            for pin in pins:
                print(f"\nPin {pin}:")
                send_servo_command(ser, pin, 80)  # Min
                time.sleep(2)
                send_servo_command(ser, pin, 90)  # Center
                time.sleep(2)
                send_servo_command(ser, pin, 100) # Max
                time.sleep(2)
                send_servo_command(ser, pin, 90)  # Back to center

        print("\n✅ Calibration complete")

    except KeyboardInterrupt:
        print("\n⚠️ Calibration interrupted")
    finally:
        # Return to center position
        print("\n🏠 Returning to center position...")
        for pin in [4, 5]:
            send_servo_command(ser, pin, 90)
        ser.close()
        print("🔌 Serial connection closed")

if __name__ == "__main__":
    main()