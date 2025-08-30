#!/usr/bin/env python3
"""
Bulletproof Arduino Port Detection System
========================================

Comprehensive solution for Arduino USB serial port auto-detection and management.
Fixes all connection issues: DTR resets, race conditions, insufficient recovery time,
error handling, and resource cleanup.

Features:
- Non-invasive detection (no DTR resets unless necessary)
- Proper error handling and resource cleanup
- Race condition prevention
- Comprehensive logging
- Fallback strategies
- Port stability verification
"""

import serial
import glob
import time
import threading
import logging
from typing import Dict, Optional, List, Tuple
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class ArduinoPortDetector:
    """Bulletproof Arduino port detection with comprehensive error handling."""
    
    def __init__(self, debug: bool = False):
        self.debug = debug
        self.detected_ports: Dict[str, str] = {}
        self.port_lock = threading.Lock()
        self.last_detection_time = 0
        self.detection_cache_timeout = 30  # Cache results for 30 seconds
        
        # Known device IDs that we expect
        self.expected_devices = {
            'SERVO_CONTROLLER': 'Servo Controller (Pan/Tilt/Lung)',
            'HAND_CONTROLLER': 'Hand Controller (5 Servos)', 
            'LIGHTBULB_CONTROLLER': 'Lightbulb PWM Controller'
        }
    
    def _log(self, message: str, level: str = "INFO"):
        """Internal logging method."""
        if level == "ERROR":
            logger.error(message)
        elif level == "WARNING":
            logger.warning(message)
        elif level == "DEBUG" and self.debug:
            logger.debug(message)
        else:
            logger.info(message)
    
    def _get_available_ports(self) -> List[str]:
        """Get list of available USB/ACM ports."""
        ports = []
        # Check common Linux USB serial ports
        for pattern in ['/dev/ttyUSB*', '/dev/ttyACM*']:
            ports.extend(glob.glob(pattern))
        
        # Sort to ensure consistent ordering
        ports.sort()
        self._log(f"Available ports: {ports}")
        return ports
    
    def _test_port_communication(self, port: str, timeout: float = 3.0) -> Optional[str]:
        """Test port communication without DTR reset."""
        ser = None
        try:
            self._log(f"Testing port {port} for device identification...", "DEBUG")
            
            # Open with minimal settings to avoid DTR issues
            ser = serial.Serial(
                port=port,
                baudrate=9600,
                timeout=timeout,
                write_timeout=1.0,
                # Explicitly disable DTR/RTS to avoid Arduino resets
                dsrdtr=False,
                rtscts=False
            )
            
            # Give Arduino time to send startup messages if any
            time.sleep(0.5)
            
            # Clear any existing input buffer
            if ser.in_waiting > 0:
                existing_data = ser.read(ser.in_waiting).decode('utf-8', errors='ignore')
                self._log(f"Port {port} had existing data: {existing_data.strip()}", "DEBUG")
            
            # Try to read device ID from existing buffer or startup messages
            device_id = self._read_device_id(ser, timeout=2.0)
            
            if device_id:
                self._log(f"Found {device_id} on {port}")
                return device_id
            
            # If no immediate device ID, try gentle probe
            device_id = self._probe_for_device_id(ser)
            
            if device_id:
                self._log(f"Probed and found {device_id} on {port}")
                return device_id
            
            self._log(f"Port {port}: No device ID detected", "DEBUG")
            return None
            
        except Exception as e:
            self._log(f"Port {port} test failed: {e}", "DEBUG")
            return None
        finally:
            if ser and ser.is_open:
                try:
                    ser.close()
                except:
                    pass
    
    def _read_device_id(self, ser: serial.Serial, timeout: float = 2.0) -> Optional[str]:
        """Read device ID from serial port."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if ser.in_waiting > 0:
                try:
                    line = ser.readline().decode('utf-8', errors='ignore').strip()
                    if line.startswith('DEVICE_ID:'):
                        device_type = line.split(':', 1)[1]
                        if device_type in self.expected_devices:
                            return device_type
                        else:
                            self._log(f"Unknown device type: {device_type}", "WARNING")
                except Exception as e:
                    self._log(f"Error reading device ID: {e}", "DEBUG")
            
            time.sleep(0.1)
        
        return None
    
    def _probe_for_device_id(self, ser: serial.Serial) -> Optional[str]:
        """Gently probe device for ID without DTR reset."""
        try:
            # Send a gentle status query that most Arduinos ignore gracefully
            ser.write(b'\n')
            ser.flush()
            time.sleep(0.5)
            
            # Try to read response
            return self._read_device_id(ser, timeout=1.5)
            
        except Exception as e:
            self._log(f"Probe failed: {e}", "DEBUG")
            return None
    
    def _dtr_reset_detection(self, port: str) -> Optional[str]:
        """Last resort: DTR reset detection (causes Arduino restart)."""
        ser = None
        try:
            self._log(f"Attempting DTR reset detection on {port} (will cause restart)", "WARNING")
            
            ser = serial.Serial(port, 9600, timeout=3.0)
            
            # Perform DTR reset
            ser.setDTR(False)
            time.sleep(0.1)
            ser.setDTR(True)
            time.sleep(2.0)  # Wait for Arduino boot
            
            # Read device ID after reset
            device_id = self._read_device_id(ser, timeout=3.0)
            
            if device_id:
                self._log(f"DTR reset detected {device_id} on {port}")
                return device_id
            
            return None
            
        except Exception as e:
            self._log(f"DTR reset detection failed on {port}: {e}", "DEBUG")
            return None
        finally:
            if ser and ser.is_open:
                try:
                    ser.close()
                except:
                    pass
    
    def _verify_port_stability(self, port: str, device_id: str) -> bool:
        """Verify that the port/device assignment is stable."""
        try:
            time.sleep(0.5)  # Brief pause
            verification_id = self._test_port_communication(port, timeout=1.0)
            
            if verification_id == device_id:
                self._log(f"Port {port} stability verified for {device_id}")
                return True
            else:
                self._log(f"Port {port} stability check failed: expected {device_id}, got {verification_id}", "WARNING")
                return False
                
        except Exception as e:
            self._log(f"Port stability check failed: {e}", "WARNING")
            return False
    
    def detect_arduino_ports(self, force_refresh: bool = False) -> Dict[str, str]:
        """
        Detect Arduino ports with comprehensive error handling.
        
        Returns:
            Dictionary mapping device types to port paths
        """
        with self.port_lock:
            # Return cached results if recent
            current_time = time.time()
            if not force_refresh and (current_time - self.last_detection_time) < self.detection_cache_timeout:
                if self.detected_ports:
                    self._log("Using cached port detection results")
                    return self.detected_ports.copy()
            
            self._log("Starting Arduino port detection...")
            
            available_ports = self._get_available_ports()
            if not available_ports:
                self._log("No USB/ACM ports found!", "WARNING")
                return {}
            
            detected = {}
            
            # Phase 1: Non-invasive detection (no DTR resets)
            self._log("Phase 1: Non-invasive detection")
            for port in available_ports:
                device_id = self._test_port_communication(port)
                if device_id:
                    detected[device_id] = port
            
            # Phase 2: DTR reset detection for undetected devices (if necessary)
            missing_devices = set(self.expected_devices.keys()) - set(detected.keys())
            if missing_devices:
                self._log(f"Phase 2: DTR reset detection for missing devices: {missing_devices}")
                
                undetected_ports = [p for p in available_ports if p not in detected.values()]
                
                for port in undetected_ports:
                    device_id = self._dtr_reset_detection(port)
                    if device_id and device_id in missing_devices:
                        detected[device_id] = port
                        missing_devices.remove(device_id)
            
            # Phase 3: Verification and stability check
            self._log("Phase 3: Port stability verification")
            verified_ports = {}
            
            for device_id, port in detected.items():
                if self._verify_port_stability(port, device_id):
                    verified_ports[device_id] = port
                else:
                    self._log(f"Port {port} failed stability check for {device_id}", "WARNING")
            
            # Phase 4: Recovery time for all detected devices
            if verified_ports:
                self._log("Phase 4: Allowing Arduino recovery time...")
                time.sleep(3.0)  # Allow all Arduinos to fully stabilize
            
            self.detected_ports = verified_ports
            self.last_detection_time = current_time
            
            # Summary
            self._log("=== Arduino Port Detection Complete ===")
            if verified_ports:
                for device_id, port in verified_ports.items():
                    device_name = self.expected_devices.get(device_id, "Unknown Device")
                    self._log(f"✓ {device_name}: {port}")
            else:
                self._log("⚠ No Arduino devices detected!")
            
            return verified_ports.copy()
    
    def get_port_for_device(self, device_id: str) -> Optional[str]:
        """Get port for specific device type."""
        ports = self.detect_arduino_ports()
        return ports.get(device_id)
    
    def wait_for_device(self, device_id: str, timeout: float = 30.0) -> Optional[str]:
        """Wait for specific device to be detected."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            port = self.get_port_for_device(device_id)
            if port:
                self._log(f"Device {device_id} found on {port} after {time.time() - start_time:.1f}s")
                return port
            
            self._log(f"Waiting for {device_id}... ({time.time() - start_time:.1f}s)", "DEBUG")
            time.sleep(2.0)
        
        self._log(f"Timeout waiting for {device_id} after {timeout}s", "WARNING")
        return None
    
    def set_environment_variables(self):
        """Set environment variables for detected ports."""
        ports = self.detect_arduino_ports()
        
        # Set hand controller port environment variable
        if 'HAND_CONTROLLER' in ports:
            os.environ['DETECTED_HAND_PORT'] = ports['HAND_CONTROLLER']
            self._log(f"Set DETECTED_HAND_PORT={ports['HAND_CONTROLLER']}")
        
        # Set other environment variables as needed
        for device_id, port in ports.items():
            env_var = f"DETECTED_{device_id}_PORT"
            os.environ[env_var] = port
            self._log(f"Set {env_var}={port}")


# Global detector instance
_global_detector: Optional[ArduinoPortDetector] = None

def get_detector(debug: bool = False) -> ArduinoPortDetector:
    """Get or create global detector instance."""
    global _global_detector
    if _global_detector is None:
        _global_detector = ArduinoPortDetector(debug=debug)
    return _global_detector

def detect_arduino_ports(debug: bool = False, force_refresh: bool = False) -> Dict[str, str]:
    """Simple function interface for port detection."""
    detector = get_detector(debug)
    return detector.detect_arduino_ports(force_refresh=force_refresh)

def get_port_for_device(device_id: str, debug: bool = False) -> Optional[str]:
    """Get port for specific device."""
    detector = get_detector(debug)
    return detector.get_port_for_device(device_id)


if __name__ == "__main__":
    # Test the detection system
    print("Testing Arduino Port Detection System")
    print("=" * 50)
    
    detector = ArduinoPortDetector(debug=True)
    ports = detector.detect_arduino_ports()
    
    if ports:
        print(f"\nSuccessfully detected {len(ports)} devices:")
        for device_id, port in ports.items():
            print(f"  {device_id}: {port}")
    else:
        print("\nNo Arduino devices detected.")
    
    # Test environment variable setting
    detector.set_environment_variables()