#!/usr/bin/env python3
"""
Arduino Connection Diagnostic Tool
Checks which Arduino devices are connected and shows their status
"""
import glob
import os
import subprocess
import sys

EXPECTED_DEVICES = {
    "arduino_lightbulb": {"name": "Lightbulb PWM Controller", "port": "USB 0:3.4"},
    "arduino_lunggaze": {"name": "Lung/Gaze Servos", "port": "USB 0:3.3"},
    "arduino_lefthand": {"name": "Left Hand Controller", "port": "USB 0:3.2"},
    "arduino_cnc": {"name": "CNC GRBL Controller", "port": "USB 0:3.1"},
    "arduino_uarm": {"name": "uArm Swift Pro", "port": "USB 0:4"},
    "arduino_lcd": {"name": "LCD Caption Display", "port": "USB 0:3"},
}


def check_symlinks():
    """Check which Arduino symlinks exist"""
    print("=" * 60)
    print("ARDUINO CONNECTION STATUS")
    print("=" * 60)

    connected = []
    missing = []

    for device, info in EXPECTED_DEVICES.items():
        symlink_path = f"/dev/{device}"
        if os.path.exists(symlink_path):
            connected.append((device, info, symlink_path))
            print(f"✓ {info['name']}")
            print(f"  → /dev/{device}")

            # Try to get real device
            try:
                real_device = os.path.realpath(symlink_path)
                print(f"  → {real_device}")
            except Exception:
                pass
            print()
        else:
            missing.append((device, info))

    if missing:
        print("\n" + "=" * 60)
        print("MISSING DEVICES")
        print("=" * 60)
        for device, info in missing:
            print(f"✗ {info['name']}")
            print(f"  Expected: /dev/{device} ({info['port']})")
            print()

    print("=" * 60)
    print(f"Connected: {len(connected)}/{len(EXPECTED_DEVICES)}")
    print("=" * 60)

    return connected, missing


def check_raw_usb():
    """Check raw USB serial devices"""
    print("\nRAW USB SERIAL DEVICES:")
    print("-" * 60)

    usb_devices = glob.glob("/dev/ttyUSB*") + glob.glob("/dev/ttyACM*")

    if not usb_devices:
        print("No USB serial devices found")
        return

    for device in sorted(usb_devices):
        print(f"\n{device}:")

        # Get USB path
        try:
            result = subprocess.run(["udevadm", "info", "-q", "property", "-n", device], capture_output=True, text=True, timeout=2)

            for line in result.stdout.split("\n"):
                if "ID_PATH=" in line or "ID_SERIAL=" in line or "ID_MODEL=" in line:
                    print(f"  {line}")
        except Exception as e:
            print(f"  Error getting info: {e}")


def suggest_fix(missing):
    """Suggest fixes for missing devices"""
    if not missing:
        return

    print("\nTROUBLESHOOTING:")
    print("-" * 60)
    print("1. Check that devices are plugged into the correct USB ports")
    print("2. If devices are plugged in but not showing up:")
    print("   sudo udevadm control --reload-rules")
    print("   sudo udevadm trigger")
    print("3. Check if you're in the dialout group:")
    print("   groups | grep dialout")
    print("4. If not in dialout group:")
    print("   sudo usermod -a -G dialout $USER")
    print("   (then logout/login)")


if __name__ == "__main__":
    print("\nChecking Arduino connections...\n")

    connected, missing = check_symlinks()
    check_raw_usb()
    suggest_fix(missing)

    print("\n" + "=" * 60)
    if len(connected) == len(EXPECTED_DEVICES):
        print("✓ ALL DEVICES CONNECTED - READY FOR FIELD TEST!")
        sys.exit(0)
    else:
        print(f"⚠ {len(missing)} device(s) missing")
        sys.exit(1)
