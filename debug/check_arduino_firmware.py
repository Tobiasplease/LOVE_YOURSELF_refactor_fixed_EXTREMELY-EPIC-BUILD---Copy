#!/usr/bin/env python3
"""
Check which Arduino firmware is loaded and test actual servo range.
"""
import serial
import time

print("=" * 60)
print("Arduino Firmware Checker")
print("=" * 60)

port = "/dev/arduino_lunggaze"
baudrate = 9600

try:
    print(f"\nConnecting to {port}...")
    ser = serial.Serial(port, baudrate, timeout=2)

    # Wait for Arduino to boot and send startup messages
    print("Waiting for startup messages...")
    time.sleep(2)

    startup_messages = []
    while ser.in_waiting:
        msg = ser.readline().decode('utf-8', errors='ignore').strip()
        if msg:
            startup_messages.append(msg)
            print(f"  Arduino: {msg}")

    if not startup_messages:
        print("\n⚠️  No startup messages received")
        print("Triggering reset by toggling DTR...")
        ser.setDTR(False)
        time.sleep(0.5)
        ser.setDTR(True)
        time.sleep(2)

        while ser.in_waiting:
            msg = ser.readline().decode('utf-8', errors='ignore').strip()
            if msg:
                startup_messages.append(msg)
                print(f"  Arduino: {msg}")

    print("\n" + "=" * 60)
    print("FIRMWARE TEST RESULTS")
    print("=" * 60)

    # Test constraint behavior
    print("\nTesting TILT constraints:")
    test_angles = [40, 45, 50, 130, 135, 140]

    for angle in test_angles:
        cmd = f"TILT:{angle}\n"
        ser.write(cmd.encode('utf-8'))
        ser.flush()
        time.sleep(0.1)

        # Read response
        responses = []
        while ser.in_waiting:
            resp = ser.readline().decode('utf-8', errors='ignore').strip()
            if resp:
                responses.append(resp)

        print(f"  Sent TILT:{angle} → Arduino: {responses}")

    print("\n" + "=" * 60)
    print("FIRMWARE IDENTIFICATION")
    print("=" * 60)

    print("\nBased on the responses:")
    print("  Lint-arduinoserial.ino       → TILT: 50-130° constraint")
    print("  Lint-arduinoserial-direct.ino → TILT: 45-135° constraint")
    print("\nIf 40° gets constrained to 45° → direct.ino")
    print("If 40° gets constrained to 50° → standard.ino")

    ser.close()

except FileNotFoundError:
    print(f"❌ Error: {port} not found")
except serial.SerialException as e:
    print(f"❌ Serial error: {e}")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
