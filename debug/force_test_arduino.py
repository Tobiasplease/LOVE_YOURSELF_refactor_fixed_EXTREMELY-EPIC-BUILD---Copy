#!/usr/bin/env python3
"""Force close any Arduino connections and test fresh connection."""

import serial
import serial.tools.list_ports
import time
import psutil

def kill_serial_connections():
    """Kill any processes that might be using serial ports."""
    print("🔍 Checking for processes using serial ports...")
    
    # Check for Python processes
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if proc.info['name'] in ['python.exe', 'pythonw.exe']:
                cmdline = ' '.join(proc.info['cmdline'] or [])
                if 'machine.py' in cmdline or 'hand_control' in cmdline:
                    print(f"🔪 Killing process {proc.info['pid']}: {cmdline}")
                    proc.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

def force_test_arduino():
    """Force test Arduino connection."""
    print("\n🔧 Force testing Arduino connection...")
    
    # Try to open and immediately close connection multiple times
    for i in range(3):
        try:
            print(f"Attempt {i+1}/3...")
            ser = serial.Serial('COM3', 9600, timeout=1)
            print("✅ Opened connection")
            
            # Send test command
            test_cmd = "HAND,45,90,135,180\n"
            print(f"📤 Sending: {test_cmd.strip()}")
            ser.write(test_cmd.encode())
            time.sleep(2)
            
            # Try different positions for visibility
            positions = [
                "HAND,10,10,10,10\n",
                "HAND,170,170,170,170\n",
                "HAND,90,90,90,90\n"
            ]
            
            for pos in positions:
                print(f"📤 Sending: {pos.strip()}")
                ser.write(pos.encode())
                time.sleep(3)  # Longer delay for visible servo movement
            
            ser.close()
            print("✅ Connection test successful!")
            return True
            
        except Exception as e:
            print(f"❌ Attempt {i+1} failed: {e}")
            time.sleep(1)
    
    return False

if __name__ == "__main__":
    kill_serial_connections()
    time.sleep(2)  # Wait for cleanup
    
    if force_test_arduino():
        print("\n🎯 Arduino is responding! The issue was port conflict.")
    else:
        print("\n❌ Arduino still not responding. Check hardware connection.")
