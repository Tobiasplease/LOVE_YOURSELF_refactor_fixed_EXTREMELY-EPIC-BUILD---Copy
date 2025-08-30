#!/usr/bin/env python3
"""
Arduino Connection Diagnostic Tool
=================================
Comprehensive diagnostic tool for troubleshooting Arduino USB connection issues.
Provides detailed analysis and step-by-step troubleshooting guidance.
"""

import sys
import os
import time
import serial
import glob
import subprocess
from typing import Dict, List, Tuple, Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class ArduinoDiagnostic:
    """Comprehensive Arduino connection diagnostic and troubleshooting tool."""
    
    def __init__(self):
        self.detected_devices = {}
        self.connection_tests = {}
        self.environment_check = {}
    
    def check_environment(self) -> Dict[str, any]:
        """Check system environment for Arduino development."""
        env_status = {}
        
        env_status['python_version'] = sys.version
        env_status['platform'] = sys.platform
        
        try:
            import serial
            env_status['pyserial_version'] = serial.__version__
        except ImportError:
            env_status['pyserial_available'] = False
        
        env_status['user_groups'] = self._get_user_groups()
        env_status['udev_rules'] = self._check_udev_rules()
        env_status['detected_hand_port'] = os.environ.get('DETECTED_HAND_PORT', 'Not set')
        
        return env_status
    
    def _get_user_groups(self) -> List[str]:
        """Get user groups that affect serial port access."""
        try:
            result = subprocess.run(['groups'], capture_output=True, text=True)
            return result.stdout.strip().split()
        except:
            return []
    
    def _check_udev_rules(self) -> bool:
        """Check if Arduino udev rules are properly configured."""
        return os.path.exists('/etc/udev/rules.d/99-arduino.rules')
    
    def scan_usb_ports(self) -> List[str]:
        """Scan for available USB serial ports."""
        ports = glob.glob('/dev/ttyUSB*') + glob.glob('/dev/ttyACM*')
        return sorted(ports)
    
    def test_device_id_detection(self, port: str, timeout: float = 3.0) -> Tuple[bool, str, List[str]]:
        """Test device ID detection on a specific port."""
        messages = []
        device_id = None
        
        try:
            with serial.Serial(port, 9600, timeout=1) as ser:
                ser.setDTR(False)
                time.sleep(0.1)
                ser.setDTR(True)
                time.sleep(2)
                
                while ser.in_waiting > 0:
                    line = ser.readline().decode().strip()
                    messages.append(line)
                    if line.startswith('DEVICE_ID:'):
                        device_id = line.split(':', 1)[1]
                
                return device_id is not None, device_id, messages
                
        except Exception as e:
            return False, None, [f"Connection failed: {e}"]
    
    def test_manual_commands(self, port: str, device_type: str) -> Tuple[bool, List[str]]:
        """Test manual commands for specific device types."""
        test_commands = {
            'SERVO_CONTROLLER': ['PAN:90', 'TILT:90', 'LUNG:90'],
            'HAND_CONTROLLER': ['HAND,90,90,90,90'],
            'LIGHTBULB_CONTROLLER': ['B:128', 'F']
        }
        
        if device_type not in test_commands:
            return False, [f"Unknown device type: {device_type}"]
        
        responses = []
        success_count = 0
        
        try:
            with serial.Serial(port, 9600, timeout=1) as ser:
                time.sleep(0.5)
                
                for cmd in test_commands[device_type]:
                    ser.write((cmd + '\n').encode())
                    time.sleep(0.2)
                    
                    while ser.in_waiting > 0:
                        response = ser.readline().decode().strip()
                        responses.append(f"CMD: {cmd} -> {response}")
                        success_count += 1
                
                return success_count > 0, responses
                
        except Exception as e:
            return False, [f"Command test failed: {e}"]
    
    def full_diagnostic(self) -> Dict[str, any]:
        """Run comprehensive diagnostic."""
        print("🔬 ARDUINO CONNECTION DIAGNOSTIC")
        print("=" * 50)
        
        results = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'environment': self.check_environment(),
            'available_ports': [],
            'detected_devices': {},
            'connection_tests': {},
            'recommendations': []
        }
        
        # Environment check
        print("\n📋 ENVIRONMENT CHECK")
        env = results['environment']
        print(f"   Python: {env['python_version'][:20]}...")
        print(f"   PySerial: {env.get('pyserial_version', 'Not available')}")
        print(f"   User groups: {', '.join(env['user_groups'][:5])}")
        print(f"   Arduino udev rules: {'✅' if env['udev_rules'] else '❌'}")
        print(f"   Detected hand port env: {env['detected_hand_port']}")
        
        if 'dialout' not in env['user_groups'] and 'tty' not in env['user_groups']:
            results['recommendations'].append("Add user to 'dialout' or 'tty' group for serial port access")
        
        # Port scanning
        print("\n🔍 USB PORT SCAN")
        available_ports = self.scan_usb_ports()
        results['available_ports'] = available_ports
        print(f"   Found {len(available_ports)} USB serial ports: {available_ports}")
        
        if not available_ports:
            results['recommendations'].append("No USB serial ports found - check Arduino connections")
            return results
        
        # Device detection
        print("\n🤖 DEVICE DETECTION")
        for port in available_ports:
            print(f"   Testing {port}...")
            success, device_id, messages = self.test_device_id_detection(port)
            
            results['detected_devices'][port] = {
                'success': success,
                'device_id': device_id,
                'messages': messages
            }
            
            if success:
                print(f"   ✅ {port}: {device_id}")
                results['connection_tests'][port] = self.test_manual_commands(port, device_id)[1]
            else:
                print(f"   ❌ {port}: No device ID detected")
                print(f"      Messages: {messages}")
        
        # Recommendations
        print("\n💡 RECOMMENDATIONS")
        if not results['detected_devices']:
            results['recommendations'].append("No Arduino devices detected - check firmware and connections")
        
        detected_count = sum(1 for r in results['detected_devices'].values() if r['success'])
        if detected_count < 3:
            results['recommendations'].append(f"Only {detected_count}/3 expected Arduinos detected")
        
        for rec in results['recommendations']:
            print(f"   • {rec}")
        
        if not results['recommendations']:
            print("   ✅ All checks passed!")
        
        return results
    
    def quick_check(self) -> bool:
        """Quick health check - returns True if system looks good."""
        ports = self.scan_usb_ports()
        if len(ports) < 3:
            return False
        
        detected_count = 0
        for port in ports[:3]:
            success, device_id, _ = self.test_device_id_detection(port)
            if success:
                detected_count += 1
        
        return detected_count >= 2  # At least 2/3 Arduinos working


def main():
    """Main diagnostic entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Arduino Connection Diagnostic Tool")
    parser.add_argument("--quick", action="store_true", help="Quick health check only")
    parser.add_argument("--port", help="Test specific port")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    
    args = parser.parse_args()
    
    diagnostic = ArduinoDiagnostic()
    
    if args.quick:
        success = diagnostic.quick_check()
        print("✅ System OK" if success else "❌ Issues detected")
        return 0 if success else 1
    
    if args.port:
        if not os.path.exists(args.port):
            print(f"❌ Port {args.port} does not exist")
            return 1
        
        success, device_id, messages = diagnostic.test_device_id_detection(args.port)
        if success:
            print(f"✅ {args.port}: {device_id}")
            if device_id:
                cmd_success, responses = diagnostic.test_manual_commands(args.port, device_id)
                for response in responses:
                    print(f"   {response}")
        else:
            print(f"❌ {args.port}: No response")
            for msg in messages:
                print(f"   {msg}")
        
        return 0 if success else 1
    
    # Full diagnostic
    results = diagnostic.full_diagnostic()
    
    if args.json:
        import json
        print("\nJSON Results:")
        print(json.dumps(results, indent=2))
    
    return 0


if __name__ == "__main__":
    sys.exit(main())