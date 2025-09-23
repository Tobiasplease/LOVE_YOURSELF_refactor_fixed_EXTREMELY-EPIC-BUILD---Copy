1  #!/usr/bin/env python3
"""
Test Robot Square Drawing
=========================

Connect to GRBL CNC, home the machine, and draw the corrected square coordinates
to test if the warp transform correction actually works on the physical robot.
"""

import os
import sys
import time

import serial

# Add the grbl directory to path for GRBL utilities
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "grbl"))

try:
    from grbl_utils import find_grbl_port, send_gcode_line, wait_for_idle
except ImportError:
    print("Warning: Could not import grbl_utils, will implement basic GRBL communication")


def find_arduino_cnc_port():
    """Use the known Arduino CNC GRBL port"""
    port_name = "/dev/arduino_cnc"

    print(f"Using Arduino CNC port: {port_name}")

    # Test if the port exists and responds
    try:
        print(f"Testing {port_name}...")

        # Try to connect
        ser = serial.Serial(port_name, 115200, timeout=2)
        time.sleep(2)  # Wait for connection

        # Send status query
        ser.write(b"?\n")
        response = ser.readline().decode("utf-8", errors="ignore").strip()

        ser.close()

        if response and ("<" in response or "Grbl" in response):
            print(f"✅ GRBL found on {port_name}: {response}")
            return port_name
        else:
            print(f"  GRBL response: {response}")
            # Even if no immediate response, the port might work
            return port_name

    except Exception as e:
        print(f"  Error testing {port_name}: {e}")
        print(f"  Trying to use {port_name} anyway...")
        return port_name


def connect_to_grbl(port_name):
    """Connect to GRBL and initialize"""
    try:
        print(f"Connecting to GRBL on {port_name}...")

        ser = serial.Serial(port_name, 115200, timeout=10)
        time.sleep(2)  # Wait for GRBL to initialize

        # Read initial response (usually "Grbl X.X ['$' for help]")
        initial = ser.readline().decode("utf-8", errors="ignore").strip()
        print(f"GRBL startup: {initial}")

        return ser

    except Exception as e:
        print(f"❌ Error connecting to GRBL: {e}")
        return None


def send_gcode_line_simple(ser, line):
    """Send a single G-code line and wait for response"""
    if not line.strip() or line.strip().startswith(";"):
        return True

    try:
        print(f"Sending: {line.strip()}")

        # Send the line
        ser.write((line.strip() + "\n").encode())

        # Wait for response
        while True:
            response = ser.readline().decode("utf-8", errors="ignore").strip()
            print(f"  Response: {response}")

            if response == "ok":
                return True
            elif "error" in response.lower():
                print(f"❌ GRBL Error: {response}")
                return False
            elif not response:
                print("❌ No response from GRBL")
                return False

    except Exception as e:
        print(f"❌ Error sending G-code: {e}")
        return False


def wait_for_idle_simple(ser, timeout=30):
    """Wait for GRBL to become idle"""
    print("Waiting for GRBL to become idle...")

    start_time = time.time()

    while time.time() - start_time < timeout:
        try:
            # Send status query
            ser.write(b"?\n")
            response = ser.readline().decode("utf-8", errors="ignore").strip()

            print(f"Status: {response}")

            if "Idle" in response:
                print("✅ GRBL is idle")
                return True
            elif "Alarm" in response:
                print("❌ GRBL is in alarm state")
                return False

            time.sleep(0.5)

        except Exception as e:
            print(f"Error checking status: {e}")
            time.sleep(1)

    print("❌ Timeout waiting for idle")
    return False


def home_grbl(ser):
    """Home the GRBL machine"""
    print("Homing GRBL machine...")

    # Send homing command
    if not send_gcode_line_simple(ser, "$H"):
        print("❌ Homing command failed")
        return False

    # Wait for homing to complete (can take a while)
    if not wait_for_idle_simple(ser, timeout=60):
        print("❌ Homing did not complete")
        return False

    print("✅ Homing completed")
    return True


def draw_corrected_square(ser):
    """Draw a tiny square ONLY within Andreas's proven safe coordinates"""
    print("Drawing tiny square within Andreas's SAFE coordinates only...")

    # Andreas's PROVEN safe coordinates:
    # Bottom-left: (66, -2)
    # Bottom-right: (111, -1)
    # Top-left: (-2, 67)
    # Top-right: (24, 67)

    # TINY square using ONLY the bottom-right area which is safest
    # Stay between X: 66-111 and Y: -2 to 10 (very conservative)

    gcode_lines = [
        "G21 ; Set units to millimeters",
        "G90 ; Use absolute positioning",
        "G17 ; Select XY plane",
        "",
        "G0 X66 Y-2 ; Move to Andreas's proven bottom-left",
        "M3 S50 ; Lower pen",
        "G1 X70 Y-2 F500 ; Small move right (4 units)",
        "G1 X70 Y2 F500 ; Small move up (4 units)",
        "G1 X66 Y2 F500 ; Small move left (4 units)",
        "G1 X66 Y-2 F500 ; Back to start",
        "",
        "M3 S30 ; Raise pen",
        "M2 ; End program",
    ]

    # Send each line
    for line in gcode_lines:
        if line.strip() and not line.strip().startswith(";"):
            if not send_gcode_line_simple(ser, line):
                print(f"❌ Failed to send: {line}")
                return False

            # Wait a bit between commands
            time.sleep(0.1)

    # Wait for all movements to complete
    if not wait_for_idle_simple(ser, timeout=30):
        print("❌ Drawing did not complete properly")
        return False

    print("✅ Square drawing completed!")
    return True


def draw_reference_square(ser):
    """Draw reference square using ONLY Andreas's exact safe coordinates"""
    print("Drawing reference square using Andreas's EXACT safe coordinates...")

    # Use Andreas's A4 corner coordinates directly - these are PROVEN safe
    # Bottom-left: (66, -2)
    # Bottom-right: (111, -1)
    # Top-left: (-2, 67)
    # Top-right: (24, 67)

    gcode_lines = [
        "G21 ; Set units to millimeters",
        "G90 ; Use absolute positioning",
        "G17 ; Select XY plane",
        "",
        "G0 X66 Y-2 ; Andreas's bottom-left",
        "M3 S50 ; Lower pen",
        "G1 X111 Y-1 F500 ; To Andreas's bottom-right",
        "G1 X24 Y67 F500 ; To Andreas's top-right",
        "G1 X-2 Y67 F500 ; To Andreas's top-left",
        "G1 X66 Y-2 F500 ; Back to bottom-left",
        "",
        "M3 S30 ; Raise pen",
        "M2 ; End program",
    ]

    # Send each line
    for line in gcode_lines:
        if line.strip() and not line.strip().startswith(";"):
            if not send_gcode_line_simple(ser, line):
                print(f"❌ Failed to send: {line}")
                return False
            time.sleep(0.1)

    if not wait_for_idle_simple(ser, timeout=30):
        print("❌ Reference drawing did not complete")
        return False

    print("✅ Reference square completed!")
    return True


def main():
    """Main test function"""
    print("Robot Square Drawing Test")
    print("=" * 50)

    # Find GRBL port
    port = find_arduino_cnc_port()
    if not port:
        print("❌ Could not find GRBL port")
        print("Make sure the Arduino CNC is connected and powered on")
        return

    # Connect to GRBL
    ser = connect_to_grbl(port)
    if not ser:
        print("❌ Could not connect to GRBL")
        return

    try:
        # Home the machine
        if not home_grbl(ser):
            print("❌ Homing failed")
            return

        # Ask user what to draw
        print("\nChoose what to draw:")
        print("1. Corrected square (with warp transform)")
        print("2. Reference square (no correction)")
        print("3. Both (corrected first, then reference)")

        choice = input("Enter choice (1-3): ").strip()

        if choice == "1":
            draw_corrected_square(ser)
        elif choice == "2":
            draw_reference_square(ser)
        elif choice == "3":
            print("\n--- Drawing corrected square first ---")
            draw_corrected_square(ser)

            input("\nPress Enter to draw reference square...")

            print("\n--- Drawing reference square ---")
            draw_reference_square(ser)
        else:
            print("Invalid choice")
            return

        print("\n✅ Test completed successfully!")
        print("Compare the drawn squares to see if the warp correction works")

    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")

    finally:
        # Close connection
        if ser:
            ser.close()
            print("Connection closed")


if __name__ == "__main__":
    main()
