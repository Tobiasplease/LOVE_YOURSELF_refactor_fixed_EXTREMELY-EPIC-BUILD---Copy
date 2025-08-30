#!/usr/bin/env python3
"""
Improved Arduino Port Detection System
=====================================

Enhanced Arduino detection system that eliminates DTR reset issues and
provides better integration with the serial port manager.

Key Improvements:
- No unnecessary DTR resets during detection
- Better error handling and recovery
- Integration with serial port manager
- Enhanced stability checking
- Support for future Arduino types
"""

import glob
import time
import logging
from typing import Dict, Optional, List, Tuple, Set
import os
from serial_port_manager import get_serial_manager

logger = logging.getLogger(__name__)

class ImprovedArduinoDetector:
    """Enhanced Arduino detection with no DTR reset issues."""
    
    def __init__(self, debug: bool = False):
        self.debug = debug
        self.serial_manager = get_serial_manager()
        
        # Extended device catalog for future expansion
        self.device_catalog = {
            'SERVO_CONTROLLER': {
                'name': 'Servo Controller (Pan/Tilt/Lung)',
                'type': 'Arduino Nano',
                'firmware': 'Lint-arduinoserial.ino'
            },
            'HAND_CONTROLLER': {
                'name': 'Hand Controller (5 Servos)',
                'type': 'Arduino Nano', 
                'firmware': 'hand_controller.ino'
            },
            'LIGHTBULB_CONTROLLER': {
                'name': 'Lightbulb PWM Controller',
                'type': 'Arduino Nano',
                'firmware': 'lightbulb_simple.ino'
            },
            'GRBL_CNC': {
                'name': 'GRBL CNC Controller',
                'type': 'Arduino Uno',
                'firmware': 'GRBL'
            },
            'UARM_SWIFT': {
                'name': 'uArm Swift Pro Controller', 
                'type': 'Arduino Mega',
                'firmware': 'uArm Swift Pro'
            }
        }
        
        # Cache for detection results
        self._detection_cache = {}
        self._cache_timestamp = 0
        self._cache_timeout = 30  # 30 second cache
        
    def _log(self, message: str, level: str = "INFO"):
        """Enhanced logging with levels."""
        if level == "ERROR":
            logger.error(message)
            if self.debug: print(f"[ERROR] {message}")
        elif level == "WARNING":
            logger.warning(message)
            if self.debug: print(f"[WARN] {message}")
        elif level == "DEBUG":
            if self.debug:
                logger.debug(message)
                print(f"[DEBUG] {message}")
        else:
            logger.info(message)
            if self.debug: print(f"[INFO] {message}")
    
    def _get_available_ports(self) -> List[str]:
        """Get list of available serial ports."""
        ports = []
        
        # Check common Linux serial ports
        patterns = ['/dev/ttyUSB*', '/dev/ttyACM*', '/dev/ttyS*']
        
        for pattern in patterns:
            found_ports = glob.glob(pattern)
            ports.extend(found_ports)
        
        # Sort for consistent ordering
        ports.sort()
        
        if ports:
            self._log(f"Found {len(ports)} potential serial ports: {ports}")
        else:
            self._log("No serial ports found!", "WARNING")
        
        return ports
    
    def _gentle_device_probe(self, port: str) -> Optional[str]:
        """
        Gently probe a device for its ID without causing resets.
        
        This method uses the serial port manager to safely connect and
        read device identification without DTR manipulation.
        """
        self._log(f"Gently probing {port}...", "DEBUG")
        
        # Use serial manager for thread-safe access
        connection = self.serial_manager.acquire_port(
            port, 
            baudrate=9600, 
            timeout=2.0,
            dsrdtr=False,  # Critical: no DTR manipulation
            rtscts=False
        )
        
        if not connection:
            self._log(f"Failed to connect to {port}", "DEBUG")
            return None
        
        try:
            # Wait for any startup messages
            time.sleep(1.0)
            
            # Check for existing data (device may send ID on startup)
            device_id = self._read_device_id(port, timeout=2.0)
            
            if device_id:
                self._log(f"Found device ID from startup: {device_id}", "DEBUG")
                return device_id
            
            # Try gentle probe - just send a newline
            self.serial_manager.send_command(port, "")
            time.sleep(0.5)
            
            # Try to read device ID again
            device_id = self._read_device_id(port, timeout=1.5)
            
            if device_id:
                self._log(f"Found device ID from probe: {device_id}", "DEBUG")
                return device_id
            
            self._log(f"No device ID detected on {port}", "DEBUG")
            return None
            
        except Exception as e:
            self._log(f"Error probing {port}: {e}", "DEBUG")
            return None
        finally:
            # Don't release here - let serial manager handle it
            pass
    
    def _read_device_id(self, port: str, timeout: float = 2.0) -> Optional[str]:
        """Read device ID from a port using the serial manager."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            # Check if we can read from the port
            if self.serial_manager.is_connected(port):
                try:
                    connection = self.serial_manager._connections.get(port)
                    if connection and connection.in_waiting > 0:
                        line = connection.readline().decode('utf-8', errors='ignore').strip()
                        
                        if line.startswith('DEVICE_ID:'):
                            device_type = line.split(':', 1)[1]
                            if device_type in self.device_catalog:
                                return device_type
                            else:
                                self._log(f"Unknown device type: {device_type}", "WARNING")
                        elif line.startswith('Grbl') or 'grbl' in line.lower():
                            return 'GRBL_CNC'
                        elif 'uarm' in line.lower() or 'swift' in line.lower():
                            return 'UARM_SWIFT'
                        
                except Exception as e:
                    self._log(f"Error reading from {port}: {e}", "DEBUG")
            
            time.sleep(0.1)
        
        return None
    
    def _verify_device_stability(self, port: str, device_id: str) -> bool:
        """Verify that device identification is stable."""
        try:
            self._log(f"Verifying stability of {device_id} on {port}", "DEBUG")
            
            # Wait a moment
            time.sleep(0.5)
            
            # Try to communicate again
            if self.serial_manager.health_check(port):
                self._log(f"Device {device_id} on {port} is stable")
                return True
            else:
                self._log(f"Device {device_id} on {port} failed stability check", "WARNING")
                return False
                
        except Exception as e:
            self._log(f"Stability check error for {port}: {e}", "WARNING")
            return False
    
    def detect_arduino_devices(self, force_refresh: bool = False) -> Dict[str, str]:
        """
        Detect Arduino devices with improved methodology.
        
        Returns:
            Dictionary mapping device IDs to port paths
        """
        # Check cache first
        current_time = time.time()
        if not force_refresh and (current_time - self._cache_timestamp) < self._cache_timeout:
            if self._detection_cache:
                self._log("Using cached detection results")
                return self._detection_cache.copy()
        
        self._log("=== Starting Arduino Device Detection ===")
        
        available_ports = self._get_available_ports()
        if not available_ports:
            self._log("No serial ports available for detection!", "ERROR")
            return {}
        
        detected_devices = {}
        
        # Phase 1: Gentle probing (no DTR resets)
        self._log("Phase 1: Gentle device probing")
        
        for port in available_ports:
            device_id = self._gentle_device_probe(port)
            
            if device_id:
                # Verify stability
                if self._verify_device_stability(port, device_id):
                    detected_devices[device_id] = port
                    self._log(f"✓ Detected {self.device_catalog[device_id]['name']} on {port}")
                else:
                    self._log(f"✗ Device {device_id} on {port} failed stability check")
                    # Release unstable connection
                    self.serial_manager.release_port(port)
        
        # Phase 2: Recovery and stabilization
        if detected_devices:
            self._log("Phase 2: Device stabilization")
            time.sleep(2.0)  # Allow all devices to stabilize
            
            # Final verification
            stable_devices = {}
            for device_id, port in detected_devices.items():
                if self.serial_manager.health_check(port):
                    stable_devices[device_id] = port
                else:
                    self._log(f"Device {device_id} on {port} became unstable", "WARNING")
            
            detected_devices = stable_devices
        
        # Update cache
        self._detection_cache = detected_devices
        self._cache_timestamp = current_time
        
        # Summary
        self._log("=== Arduino Detection Complete ===")
        if detected_devices:
            self._log(f"Successfully detected {len(detected_devices)} devices:")
            for device_id, port in detected_devices.items():
                device_name = self.device_catalog[device_id]['name']
                self._log(f"  ✓ {device_name}: {port}")
        else:
            self._log("⚠ No Arduino devices detected!")
        
        return detected_devices.copy()
    
    def get_device_port(self, device_id: str) -> Optional[str]:
        """Get port for a specific device."""
        devices = self.detect_arduino_devices()
        return devices.get(device_id)
    
    def wait_for_device(self, device_id: str, timeout: float = 30.0) -> Optional[str]:
        """Wait for a specific device to be detected."""
        start_time = time.time()
        
        self._log(f"Waiting for {device_id} (timeout: {timeout}s)")
        
        while time.time() - start_time < timeout:
            port = self.get_device_port(device_id)
            if port:
                elapsed = time.time() - start_time
                self._log(f"Device {device_id} found on {port} after {elapsed:.1f}s")
                return port
            
            # Force refresh after first attempt
            self._detection_cache = {}
            time.sleep(2.0)
        
        self._log(f"Timeout waiting for {device_id}", "WARNING")
        return None
    
    def set_environment_variables(self):
        """Set environment variables for detected devices."""
        devices = self.detect_arduino_devices()
        
        # Set specific environment variables
        env_mappings = {
            'HAND_CONTROLLER': 'DETECTED_HAND_PORT',
            'SERVO_CONTROLLER': 'DETECTED_SERVO_PORT', 
            'LIGHTBULB_CONTROLLER': 'DETECTED_LIGHTBULB_PORT',
            'GRBL_CNC': 'DETECTED_GRBL_PORT',
            'UARM_SWIFT': 'DETECTED_UARM_PORT'
        }
        
        for device_id, port in devices.items():
            # Generic environment variable
            generic_env = f"DETECTED_{device_id}_PORT"
            os.environ[generic_env] = port
            
            # Specific environment variable if mapped
            if device_id in env_mappings:
                specific_env = env_mappings[device_id]
                os.environ[specific_env] = port
                self._log(f"Set {specific_env}={port}")
    
    def get_device_info(self, device_id: str) -> Optional[Dict]:
        """Get detailed information about a device."""
        if device_id in self.device_catalog:
            info = self.device_catalog[device_id].copy()
            info['port'] = self.get_device_port(device_id)
            info['connected'] = info['port'] is not None
            return info
        return None
    
    def list_all_devices(self) -> Dict[str, Dict]:
        """List all known devices and their status."""
        devices_info = {}
        for device_id in self.device_catalog:
            devices_info[device_id] = self.get_device_info(device_id)
        return devices_info


# Global detector instance
_global_detector: Optional[ImprovedArduinoDetector] = None

def get_arduino_detector(debug: bool = False) -> ImprovedArduinoDetector:
    """Get or create the global Arduino detector."""
    global _global_detector
    if _global_detector is None:
        _global_detector = ImprovedArduinoDetector(debug=debug)
    return _global_detector

# Convenience functions
def detect_arduinos(debug: bool = False, force_refresh: bool = False) -> Dict[str, str]:
    """Detect Arduino devices."""
    return get_arduino_detector(debug).detect_arduino_devices(force_refresh)

def get_device_port(device_id: str, debug: bool = False) -> Optional[str]:
    """Get port for specific device."""
    return get_arduino_detector(debug).get_device_port(device_id)

def wait_for_device(device_id: str, timeout: float = 30.0, debug: bool = False) -> Optional[str]:
    """Wait for specific device."""
    return get_arduino_detector(debug).wait_for_device(device_id, timeout)


if __name__ == "__main__":
    # Test the improved detection system
    print("Testing Improved Arduino Detection System")
    print("=" * 60)
    
    detector = ImprovedArduinoDetector(debug=True)
    
    # Test detection
    devices = detector.detect_arduino_devices()
    
    if devices:
        print(f"\n✓ Successfully detected {len(devices)} devices:")
        for device_id, port in devices.items():
            info = detector.get_device_info(device_id)
            print(f"  • {info['name']}: {port}")
            print(f"    Type: {info['type']}, Firmware: {info['firmware']}")
    else:
        print("\n✗ No Arduino devices detected")
        print("\nTroubleshooting:")
        print("1. Check USB connections")
        print("2. Verify Arduino firmware has correct DEVICE_ID")
        print("3. Check serial port permissions")
    
    # Test environment variables
    print(f"\n=== Environment Variables ===")
    detector.set_environment_variables()
    
    # List all device status
    print(f"\n=== All Device Status ===")
    all_devices = detector.list_all_devices()
    for device_id, info in all_devices.items():
        status = "✓ Connected" if info['connected'] else "✗ Not found"
        print(f"{info['name']}: {status}")
        if info['port']:
            print(f"  Port: {info['port']}")