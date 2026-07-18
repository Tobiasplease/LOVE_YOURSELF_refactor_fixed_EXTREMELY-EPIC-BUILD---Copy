"""Identify which hand-controller firmware is flashed on the lefthand Arduino.

The variants in arduino_src/ behave completely differently for the left arm:

  clean  ("Direct Servo Control")       SERVO,pin,angle works, 5ms/deg slew,
                                        no autonomous movement — what the
                                        motor panel + markov system need
  fixed  ("Improved Left Arm Control")  NO SERVO handler at all; the arm is
                                        driven by an internal random wanderer
                                        (70-110 deg, 0.3-1 deg steps) whenever
                                        LEFT_ARM_ENABLE is active. External
                                        control is impossible.

Resets the board via DTR, reads the boot banner, then sends a SERVO test
command and reports whether the firmware acknowledged it.

Run with machine.py and the motor panel CLOSED (port is exclusive):
    python debug/identify_hand_firmware.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import serial

PORT = os.getenv("HAND_CONTROLLER_PORT", "/dev/arduino_lefthand")


def main():
    print(f"Opening {PORT} (resets the Arduino)…")
    try:
        ser = serial.Serial(PORT, 9600, timeout=1)
    except Exception as e:
        print(f"Cannot open port: {e}")
        print("Is machine.py or the motor panel still running?")
        return

    ser.dtr = False
    time.sleep(0.3)
    ser.dtr = True

    print("Reading boot banner (4s)…\n")
    banner = []
    t0 = time.time()
    while time.time() - t0 < 4:
        line = ser.readline().decode(errors="replace").strip()
        if line:
            banner.append(line)
            print(f"  {line}")

    text = " ".join(banner)
    print()
    if "Improved Left Arm Control" in text:
        print("VERDICT: 'fixed' variant — the arm is firmware-autonomous.")
        print("SERVO commands are ignored; direct control is impossible.")
        print("Flash arduino_src/hand_controller_clean.ino to hand the arm to Python.")
    elif "Direct Servo Control" in text:
        print("VERDICT: 'clean' variant — direct SERVO control available.")
        print("Testing SERVO command echo…")
        ser.reset_input_buffer()
        ser.write(b"SERVO,4,90\n")
        time.sleep(0.5)
        resp = ser.read_all().decode(errors="replace").strip()
        print(f"  response: {resp!r}")
        print("  (an echo like 'Servo 4 (pin 4) -> 90' confirms the handler runs)")
    elif banner:
        print("VERDICT: unrecognized banner — a third variant is flashed.")
        print("Compare the lines above against the files in arduino_src/.")
    else:
        print("No banner received. Wrong port, dead board, or a firmware")
        print("without boot output. Try unplugging/replugging USB and rerunning.")

    ser.close()


if __name__ == "__main__":
    main()
