#!/usr/bin/env python3
"""
Smart Port Mapper - Identify devices by firmware handshake
============================================================
Instead of relying on port numbers, identify each Arduino by sending
a unique handshake command and checking the response.
"""

import serial
import time
import json
from typing import Dict, Optional, List
from serial.tools import list_ports

class SmartPortMapper:
    """Map Arduino devices by firmware response rather than port number."""
    
    def __init__(self):
        self.device_map = {}
        self.handshakes = {
            'servo': {'command': 'WHO', 'expected': 'Hand Controller Ready'},
            'lightbulb': {'command': 'WHO', 'expected': 'lightbulb'},
            'hand': {'command': 'WHO', 'expected': 'Hand'},
            'grbl': {'command': '$I', 'expected': 'Grbl'}
        }
    
    def scan_and_map(self) -> Dict[str, str]:
        """Scan all USB ports and identify devices by handshake."""
        print("🔍 Scanning for Arduino devices...")
        
        # Get all USB serial ports
        usb_ports = []
        for port in list_ports.comports():
            if 'USB' in port.device:
                usb_ports.append(port.device)
        
        print(f"Found {len(usb_ports)} USB serial ports: {usb_ports}")
        
        device_map = {}
        
        for port in usb_ports:
            print(f"\n--- Testing {port} ---")
            device_type = self.identify_device(port)
            if device_type:
                device_map[device_type] = port
                print(f"✅ {device_type} -> {port}")
            else:
                print(f"❌ Unknown device on {port}")
        
        self.device_map = device_map
        return device_map
    
    def identify_device(self, port: str) -> Optional[str]:
        """Identify device type by sending handshake commands."""
        try:
            ser = serial.Serial(port, 9600, timeout=2)
            time.sleep(1)  # Arduino boot time
            
            # Clear any existing data
            ser.flushInput()
            ser.flushOutput()
            
            # Try each handshake
            for device_type, handshake in self.handshakes.items():
                try:
                    # Send command
                    command = handshake['command'] + '\n'
                    ser.write(command.encode())
                    time.sleep(0.5)
                    
                    # Read response
                    response = ''
                    start_time = time.time()
                    while time.time() - start_time < 1.0:
                        if ser.in_waiting:
                            response += ser.read(ser.in_waiting).decode('utf-8', errors='ignore')
                        time.sleep(0.1)
                    
                    response = response.strip()
                    print(f"  {device_type}: '{command.strip()}' -> '{response}'")
                    
                    if handshake['expected'] in response:
                        ser.close()
                        return device_type
                        
                except Exception as e:
                    print(f"  {device_type}: Error - {e}")
                    continue
            
            ser.close()
            return None
            
        except Exception as e:
            print(f"  Connection failed: {e}")
            return None
    
    def save_mapping(self, filename: str = "device_mapping.json"):
        """Save current device mapping to file."""
        with open(filename, 'w') as f:
            json.dump(self.device_map, f, indent=2)
        print(f"💾 Device mapping saved to {filename}")
    
    def load_mapping(self, filename: str = "device_mapping.json") -> bool:
        """Load device mapping from file."""
        try:
            with open(filename, 'r') as f:
                self.device_map = json.load(f)
            print(f"📁 Device mapping loaded from {filename}")
            return True
        except FileNotFoundError:
            print(f"📁 No saved mapping found at {filename}")
            return False
    
    def get_port(self, device_type: str) -> Optional[str]:
        """Get port for a specific device type."""
        return self.device_map.get(device_type)
    
    def print_mapping(self):
        """Print current device mapping."""
        print("\n" + "="*50)
        print("CURRENT DEVICE MAPPING")
        print("="*50)
        for device_type, port in self.device_map.items():
            print(f"{device_type:>10} -> {port}")
        print("="*50)

if __name__ == "__main__":
    mapper = SmartPortMapper()
    
    # Try to load existing mapping
    if not mapper.load_mapping():
        # No saved mapping, scan for devices
        mapper.scan_and_map()
        mapper.save_mapping()
    
    mapper.print_mapping()
    
    # Verify each device still responds
    print("\n🔍 Verifying saved mapping...")
    for device_type, port in mapper.device_map.items():
        if mapper.identify_device(port) == device_type:
            print(f"✅ {device_type} still at {port}")
        else:
            print(f"❌ {device_type} moved from {port} - rescanning needed")
            mapper.scan_and_map()
            mapper.save_mapping()
            break