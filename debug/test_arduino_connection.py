#!/usr/bin/env python3
"""Test serial connection to Arduino and list available COM ports."""

import serial
import serial.tools.list_ports
import time

def list_com_ports():
    """List all available COM ports."""
    print("🔍 Available COM ports:")
    ports = serial.tools.list_ports.comports()
    for port in ports:
        print(f"  📍 {port.device}: {port.description}")
        if 'Arduino' in port.description or 'USB' in port.description:
            print(f"    ⭐ Potential Arduino: {port.device}")
    return [port.device for port in ports]

def test_arduino_connection(port, baudrate=9600):
    """Test connection to Arduino on specific port."""
    print(f"\n📡 Testing connection to {port} at {baudrate} baud...")
    try:
        # Open connection
        ser = serial.Serial(port, baudrate, timeout=2)
        time.sleep(3)  # Wait for Arduino to boot
        
        print(f"✅ Connected to {port}")
        
        # Send test command
        test_command = "HAND,90,90,90,90\n"
        print(f"📤 Sending: {test_command.strip()}")
        ser.write(test_command.encode())
        
        # Wait for response or acknowledgment
        time.sleep(1)
        
        # Try a few different positions
        positions = [
            "HAND,30,30,30,30\n",
            "HAND,120,120,120,120\n", 
            "HAND,90,90,90,90\n"
        ]
        
        for pos in positions:
            print(f"📤 Sending: {pos.strip()}")
            ser.write(pos.encode())
            time.sleep(2)  # Wait for servo movement
            
        ser.close()
        print(f"✅ Test completed for {port}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to connect to {port}: {e}")
        return False

if __name__ == "__main__":
    # List all ports
    available_ports = list_com_ports()
    
    if not available_ports:
        print("❌ No COM ports found!")
        exit(1)
    
    # Test each port that might be Arduino
    for port in available_ports:
        if test_arduino_connection(port):
            print(f"\n🎯 Arduino likely connected to: {port}")
            break
    else:
        print("\n❌ No working Arduino connection found")
