#!/usr/bin/env python3
"""
Drawing Inspection Calibrator
Interactive tool to find and save optimal angles for looking down at drawing surface
"""

import sys
import time
import os
import json

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from servo_control.servo_control import ServoController

# Configuration file to store calibrated angles
CALIBRATION_FILE = os.path.join(os.path.dirname(__file__), "drawing_inspection_angles.json")

def load_saved_angles():
    """Load previously saved inspection angles"""
    try:
        with open(CALIBRATION_FILE, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def save_angles(angles):
    """Save inspection angles to file"""
    with open(CALIBRATION_FILE, 'w') as f:
        json.dump(angles, f, indent=2)
    print(f"✅ Angles saved to {CALIBRATION_FILE}")

def main():
    print("=== DRAWING INSPECTION CALIBRATOR ===")
    print("Find optimal angles for looking down at drawing surface")
    print()
    
    # Connect to servo controller
    servo_port = "/dev/arduino_lunggaze"
    try:
        servos = ServoController(port=servo_port, baudrate=9600)
        if not servos.ser or not servos.ser.is_open:
            print(f"❌ Failed to connect to {servo_port}")
            return
        print(f"✅ Connected to {servo_port}")
    except Exception as e:
        print(f"❌ Connection error: {e}")
        return
    
    # Load any existing saved angles
    saved_angles = load_saved_angles()
    if saved_angles:
        print("\n📁 Previously saved angles:")
        for name, angles in saved_angles.items():
            print(f"   {name}: PAN={angles['pan']}°, TILT={angles['tilt']}°")
    
    # Start from center position
    current_pan = 90
    current_tilt = 90
    servos.set_pan(current_pan)
    servos.set_tilt(current_tilt)
    time.sleep(1)
    
    print(f"\n🎯 Current position: PAN={current_pan}°, TILT={current_tilt}°")
    print("\nControls:")
    print("  a/d = PAN left/right (±1°)")
    print("  A/D = PAN left/right (±5°)")
    print("  w/s = TILT up/down (±1°)")
    print("  W/S = TILT up/down (±5°)")
    print("  c = center position (90,90)")
    print("  p = print current position")
    print("  save [name] = save current position")
    print("  load [name] = load saved position")
    print("  list = show saved positions")
    print("  test [name] = test saved position")
    print("  q = quit")
    
    print(f"\n💡 Suggested positions to find:")
    print("  - 'drawing_center': Looking at center of drawing area")
    print("  - 'drawing_near': Looking at near edge of drawing")
    print("  - 'drawing_far': Looking at far edge of drawing")
    print("  - 'drawing_left': Looking at left side of drawing")
    print("  - 'drawing_right': Looking at right side of drawing")
    
    while True:
        print(f"\nCurrent: PAN={current_pan}°, TILT={current_tilt}°")
        cmd = input("> ").strip()
        
        if cmd == 'q':
            break
        elif cmd == 'c':
            current_pan = 90
            current_tilt = 90
            servos.set_pan(current_pan)
            time.sleep(0.1)  # Small delay between commands
            servos.set_tilt(current_tilt)
            print("📍 Centered to (90, 90)")
            
        elif cmd == 'p':
            print(f"📍 Current position: PAN={current_pan}°, TILT={current_tilt}°")
            
        elif cmd == 'list':
            saved_angles = load_saved_angles()
            if saved_angles:
                print("📁 Saved positions:")
                for name, angles in saved_angles.items():
                    print(f"   {name}: PAN={angles['pan']}°, TILT={angles['tilt']}°")
            else:
                print("📁 No saved positions")
                
        # Movement commands
        elif cmd == 'a':
            current_pan = max(65, current_pan - 1)
            servos.set_pan(current_pan)
        elif cmd == 'd':
            current_pan = min(115, current_pan + 1)
            servos.set_pan(current_pan)
        elif cmd == 'A':
            current_pan = max(65, current_pan - 5)
            servos.set_pan(current_pan)
        elif cmd == 'D':
            current_pan = min(115, current_pan + 5)
            servos.set_pan(current_pan)
        elif cmd == 'w':
            current_tilt = min(110, current_tilt + 1)
            servos.set_tilt(current_tilt)
        elif cmd == 's':
            current_tilt = max(70, current_tilt - 1)
            servos.set_tilt(current_tilt)
        elif cmd == 'W':
            current_tilt = min(110, current_tilt + 5)
            servos.set_tilt(current_tilt)
        elif cmd == 'S':
            current_tilt = max(70, current_tilt - 5)
            servos.set_tilt(current_tilt)
            
        # Save/load commands
        elif cmd.startswith('save '):
            name = cmd[5:].strip()
            if name:
                saved_angles = load_saved_angles()
                saved_angles[name] = {'pan': current_pan, 'tilt': current_tilt}
                save_angles(saved_angles)
                print(f"💾 Saved '{name}': PAN={current_pan}°, TILT={current_tilt}°")
            else:
                print("❌ Usage: save [name]")
                
        elif cmd.startswith('load '):
            name = cmd[5:].strip()
            saved_angles = load_saved_angles()
            if name in saved_angles:
                current_pan = saved_angles[name]['pan']
                current_tilt = saved_angles[name]['tilt']
                servos.set_pan(current_pan)
                time.sleep(0.1)  # Small delay between commands
                servos.set_tilt(current_tilt)
                print(f"📂 Loaded '{name}': PAN={current_pan}°, TILT={current_tilt}°")
            else:
                print(f"❌ Position '{name}' not found")
                
        elif cmd.startswith('test '):
            name = cmd[5:].strip()
            saved_angles = load_saved_angles()
            if name in saved_angles:
                # Save current position
                orig_pan, orig_tilt = current_pan, current_tilt
                
                # Move to test position
                test_pan = saved_angles[name]['pan']
                test_tilt = saved_angles[name]['tilt']
                servos.set_pan(test_pan)
                time.sleep(0.1)  # Small delay between commands
                servos.set_tilt(test_tilt)
                print(f"🧪 Testing '{name}': PAN={test_pan}°, TILT={test_tilt}°")
                
                input("Press Enter to return to previous position...")
                
                # Return to original position
                current_pan, current_tilt = orig_pan, orig_tilt
                servos.set_pan(current_pan)
                time.sleep(0.1)  # Small delay between commands
                servos.set_tilt(current_tilt)
                print(f"🔙 Returned to: PAN={current_pan}°, TILT={current_tilt}°")
            else:
                print(f"❌ Position '{name}' not found")
        else:
            print("❌ Unknown command")
    
    # Return to center before exiting
    print("\n🏠 Returning to center position...")
    servos.set_pan(90)
    time.sleep(0.1)  # Small delay between commands
    servos.set_tilt(90)
    time.sleep(1)
    
    print("\n✅ Calibration complete!")
    
    # Show final summary
    saved_angles = load_saved_angles()
    if saved_angles:
        print(f"\n📋 Final saved positions:")
        for name, angles in saved_angles.items():
            print(f"   {name}: PAN={angles['pan']}°, TILT={angles['tilt']}°")
            
        print(f"\n💡 Integration example:")
        print("```python")
        print("# Load drawing inspection angles")
        print("import json")
        print(f"with open('{CALIBRATION_FILE}', 'r') as f:")
        print("    inspection_angles = json.load(f)")
        print("")
        print("# Look at drawing center during drawing")
        print("if 'drawing_center' in inspection_angles:")
        print("    pos = inspection_angles['drawing_center']")
        print("    servos.set_pan(pos['pan'])")
        print("    servos.set_tilt(pos['tilt'])")
        print("```")

if __name__ == "__main__":
    main()