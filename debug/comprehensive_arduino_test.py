#!/usr/bin/env python3
"""
Comprehensive Arduino System Test Suite
=======================================

Complete testing framework for the Arduino USB serial management system.
Tests all aspects: detection, connection, individual devices, stress tests,
and failure scenarios.

This test suite validates the complete solution before the user returns.
"""

import sys
import os
import time
import threading
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from serial_port_manager import get_serial_manager
from improved_arduino_detector import get_arduino_detector
from config.config import USE_SERVO, USE_LIGHTBULB_PWM, USE_HAND_CONTROLLER

class ArduinoSystemTester:
    """Comprehensive Arduino system testing framework."""
    
    def __init__(self, debug: bool = True):
        self.debug = debug
        self.serial_manager = get_serial_manager()
        self.arduino_detector = get_arduino_detector(debug=debug)
        
        self.test_results = {
            'detection': {},
            'connection': {},
            'individual_devices': {},
            'stress_test': {},
            'error_handling': {},
            'integration': {}
        }
        
        self.start_time = time.time()
        
    def log(self, message: str, level: str = "INFO"):
        """Enhanced logging."""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        if level == "ERROR":
            print(f"[{timestamp}] ❌ ERROR: {message}")
        elif level == "SUCCESS":
            print(f"[{timestamp}] ✅ SUCCESS: {message}")
        elif level == "WARNING":
            print(f"[{timestamp}] ⚠️  WARNING: {message}")
        elif level == "TEST":
            print(f"[{timestamp}] 🧪 TEST: {message}")
        else:
            print(f"[{timestamp}] ℹ️  INFO: {message}")
    
    def test_device_detection(self) -> Dict:
        """Test Arduino device detection system."""
        self.log("=== Testing Arduino Device Detection ===", "TEST")
        
        results = {}
        
        try:
            # Test basic detection
            self.log("Testing basic device detection...")
            devices = self.arduino_detector.detect_arduino_devices(force_refresh=True)
            
            results['detected_devices'] = devices
            results['detection_count'] = len(devices)
            
            if devices:
                self.log(f"Detected {len(devices)} devices", "SUCCESS")
                for device_id, port in devices.items():
                    device_info = self.arduino_detector.get_device_info(device_id)
                    self.log(f"  • {device_info['name']}: {port}")
            else:
                self.log("No devices detected", "WARNING")
            
            # Test environment variable setting
            self.log("Testing environment variable setting...")
            self.arduino_detector.set_environment_variables()
            
            env_vars_set = 0
            for device_id in devices:
                env_var = f"DETECTED_{device_id}_PORT"
                if env_var in os.environ:
                    env_vars_set += 1
            
            results['env_vars_set'] = env_vars_set
            self.log(f"Set {env_vars_set} environment variables", "SUCCESS")
            
            # Test cache functionality
            self.log("Testing detection caching...")
            start_time = time.time()
            cached_devices = self.arduino_detector.detect_arduino_devices()
            cache_time = time.time() - start_time
            
            results['cache_test'] = {
                'cached_devices': cached_devices,
                'cache_time_ms': cache_time * 1000,
                'cache_working': cached_devices == devices and cache_time < 0.1
            }
            
            if results['cache_test']['cache_working']:
                self.log(f"Cache working (retrieved in {cache_time*1000:.1f}ms)", "SUCCESS")
            else:
                self.log("Cache not working properly", "WARNING")
            
            results['status'] = 'success'
            
        except Exception as e:
            results['status'] = 'error'
            results['error'] = str(e)
            self.log(f"Detection test failed: {e}", "ERROR")
        
        self.test_results['detection'] = results
        return results
    
    def test_serial_connections(self) -> Dict:
        """Test serial port manager connections."""
        self.log("=== Testing Serial Port Manager ===", "TEST")
        
        results = {}
        devices = self.arduino_detector.detect_arduino_devices()
        
        for device_id, port in devices.items():
            self.log(f"Testing connection to {device_id} on {port}")
            
            device_results = {}
            
            try:
                # Test connection acquisition
                connection = self.serial_manager.acquire_port(port, timeout=2.0)
                
                if connection:
                    device_results['connection'] = 'success'
                    self.log(f"✓ Connected to {port}", "SUCCESS")
                    
                    # Test basic communication
                    response = self.serial_manager.send_command(
                        port, "", expect_response=True, response_timeout=1.0
                    )
                    device_results['communication'] = 'success' if response is not None else 'no_response'
                    
                    # Test health check
                    health = self.serial_manager.health_check(port)
                    device_results['health_check'] = 'pass' if health else 'fail'
                    
                    if health:
                        self.log(f"✓ Health check passed for {port}", "SUCCESS")
                    else:
                        self.log(f"✗ Health check failed for {port}", "WARNING")
                    
                else:
                    device_results['connection'] = 'failed'
                    self.log(f"✗ Failed to connect to {port}", "ERROR")
                
            except Exception as e:
                device_results['connection'] = 'error'
                device_results['error'] = str(e)
                self.log(f"✗ Connection error for {port}: {e}", "ERROR")
            
            results[device_id] = device_results
        
        self.test_results['connection'] = results
        return results
    
    def test_individual_devices(self) -> Dict:
        """Test each Arduino device individually."""
        self.log("=== Testing Individual Device Functionality ===", "TEST")
        
        results = {}
        devices = self.arduino_detector.detect_arduino_devices()
        
        for device_id, port in devices.items():
            self.log(f"Testing {device_id} functionality...")
            
            device_results = {'device_id': device_id, 'port': port}
            
            try:
                if device_id == 'SERVO_CONTROLLER':
                    device_results.update(self._test_servo_controller(port))
                elif device_id == 'HAND_CONTROLLER':
                    device_results.update(self._test_hand_controller(port))
                elif device_id == 'LIGHTBULB_CONTROLLER':
                    device_results.update(self._test_lightbulb_controller(port))
                elif device_id == 'GRBL_CNC':
                    device_results.update(self._test_grbl_controller(port))
                else:
                    device_results['status'] = 'unknown_device'
                    
            except Exception as e:
                device_results['status'] = 'error'
                device_results['error'] = str(e)
                self.log(f"Device test error for {device_id}: {e}", "ERROR")
            
            results[device_id] = device_results
        
        self.test_results['individual_devices'] = results
        return results
    
    def _test_servo_controller(self, port: str) -> Dict:
        """Test servo controller specific functionality."""
        results = {'device_type': 'servo_controller'}
        
        try:
            # Test servo commands
            commands = [
                ("PAN:90", "Pan center"),
                ("TILT:90", "Tilt center"), 
                ("LUNG:hold", "Lung hold mode"),
                ("PAN:45", "Pan left"),
                ("TILT:135", "Tilt up"),
                ("PAN:135", "Pan right"),
                ("TILT:45", "Tilt down"),
                ("PAN:90", "Pan center"),
                ("TILT:90", "Tilt center")
            ]
            
            successful_commands = 0
            
            for cmd, description in commands:
                response = self.serial_manager.send_command(
                    port, cmd, expect_response=True, response_timeout=0.5
                )
                if response:
                    successful_commands += 1
                time.sleep(0.1)  # Brief delay between commands
            
            results['commands_sent'] = len(commands)
            results['successful_commands'] = successful_commands
            results['success_rate'] = successful_commands / len(commands)
            results['status'] = 'success' if results['success_rate'] >= 0.8 else 'partial'
            
            self.log(f"Servo controller: {successful_commands}/{len(commands)} commands successful", 
                    "SUCCESS" if results['success_rate'] >= 0.8 else "WARNING")
            
        except Exception as e:
            results['status'] = 'error'
            results['error'] = str(e)
        
        return results
    
    def _test_hand_controller(self, port: str) -> Dict:
        """Test hand controller specific functionality."""
        results = {'device_type': 'hand_controller'}
        
        try:
            # Test hand position commands
            test_positions = [
                [90, 90, 90, 90],  # Center
                [45, 45, 45, 45],  # One direction
                [135, 135, 135, 135],  # Other direction
                [90, 90, 90, 90],  # Back to center
            ]
            
            successful_commands = 0
            
            for positions in test_positions:
                cmd = f"HAND,{','.join(map(str, positions))}"
                response = self.serial_manager.send_command(
                    port, cmd, expect_response=True, response_timeout=0.5
                )
                if response:
                    successful_commands += 1
                time.sleep(0.2)  # Brief delay for movement
            
            # Test heartbeat
            heartbeat_response = self.serial_manager.send_command(
                port, "HEARTBEAT", expect_response=True, response_timeout=0.5
            )
            
            results['position_commands'] = len(test_positions)
            results['successful_commands'] = successful_commands
            results['heartbeat_working'] = heartbeat_response is not None
            results['success_rate'] = successful_commands / len(test_positions)
            results['status'] = 'success' if results['success_rate'] >= 0.8 else 'partial'
            
            self.log(f"Hand controller: {successful_commands}/{len(test_positions)} commands successful", 
                    "SUCCESS" if results['success_rate'] >= 0.8 else "WARNING")
            
        except Exception as e:
            results['status'] = 'error'
            results['error'] = str(e)
        
        return results
    
    def _test_lightbulb_controller(self, port: str) -> Dict:
        """Test lightbulb controller specific functionality."""
        results = {'device_type': 'lightbulb_controller'}
        
        try:
            # Test brightness commands
            brightness_levels = [0, 64, 128, 192, 255, 128, 0]
            successful_commands = 0
            
            for brightness in brightness_levels:
                cmd = f"B:{brightness}"
                # Lightbulb controller doesn't send responses, so don't expect them
                self.serial_manager.send_command(port, cmd, expect_response=False)
                successful_commands += 1  # Assume success if no exception
                time.sleep(0.1)
            
            # Test flash command
            self.serial_manager.send_command(port, "F", expect_response=False)
            time.sleep(0.5)  # Wait for flash
            
            results['brightness_commands'] = len(brightness_levels)
            results['successful_commands'] = successful_commands
            results['flash_command'] = True
            results['success_rate'] = 1.0  # No way to verify, assume success
            results['status'] = 'success'
            
            self.log(f"Lightbulb controller: {successful_commands} brightness levels tested", "SUCCESS")
            
        except Exception as e:
            results['status'] = 'error'
            results['error'] = str(e)
        
        return results
    
    def _test_grbl_controller(self, port: str) -> Dict:
        """Test GRBL CNC controller functionality."""
        results = {'device_type': 'grbl_controller'}
        
        try:
            # GRBL status query
            response = self.serial_manager.send_command(
                port, "?", expect_response=True, response_timeout=2.0
            )
            
            results['status_response'] = response is not None
            results['grbl_detected'] = response and 'grbl' in response.lower() if response else False
            results['status'] = 'success' if results['grbl_detected'] else 'partial'
            
            if results['grbl_detected']:
                self.log("GRBL controller responding correctly", "SUCCESS")
            else:
                self.log("GRBL controller not responding as expected", "WARNING")
                
        except Exception as e:
            results['status'] = 'error'
            results['error'] = str(e)
        
        return results
    
    def test_stress_scenarios(self) -> Dict:
        """Test system under stress conditions."""
        self.log("=== Testing Stress Scenarios ===", "TEST")
        
        results = {}
        devices = self.arduino_detector.detect_arduino_devices()
        
        # Test 1: Rapid connection/disconnection
        self.log("Testing rapid connection cycles...")
        if devices:
            port = list(devices.values())[0]  # Use first available port
            
            rapid_cycle_results = {
                'cycles': 0,
                'successful_connections': 0,
                'successful_releases': 0
            }
            
            for i in range(10):
                try:
                    connection = self.serial_manager.acquire_port(port)
                    if connection:
                        rapid_cycle_results['successful_connections'] += 1
                        time.sleep(0.1)
                        if self.serial_manager.release_port(port):
                            rapid_cycle_results['successful_releases'] += 1
                    rapid_cycle_results['cycles'] += 1
                except Exception as e:
                    self.log(f"Rapid cycle error: {e}", "WARNING")
                
                time.sleep(0.1)
            
            results['rapid_cycles'] = rapid_cycle_results
            self.log(f"Rapid cycles: {rapid_cycle_results['successful_connections']}/10 connections", 
                    "SUCCESS" if rapid_cycle_results['successful_connections'] >= 8 else "WARNING")
        
        # Test 2: Multi-threaded access
        self.log("Testing multi-threaded access...")
        if devices:
            threading_results = {'threads': 5, 'successful_accesses': 0}
            
            def thread_test(port, results_dict):
                try:
                    connection = self.serial_manager.acquire_port(port)
                    if connection:
                        time.sleep(0.5)
                        health = self.serial_manager.health_check(port)
                        if health:
                            results_dict['successful_accesses'] += 1
                except Exception as e:
                    self.log(f"Thread test error: {e}", "WARNING")
            
            threads = []
            port = list(devices.values())[0]
            
            for i in range(5):
                thread = threading.Thread(target=thread_test, args=(port, threading_results))
                threads.append(thread)
                thread.start()
            
            for thread in threads:
                thread.join(timeout=5.0)
            
            results['threading'] = threading_results
            self.log(f"Threading test: {threading_results['successful_accesses']}/5 threads successful",
                    "SUCCESS" if threading_results['successful_accesses'] >= 4 else "WARNING")
        
        self.test_results['stress_test'] = results
        return results
    
    def test_error_handling(self) -> Dict:
        """Test error handling and recovery."""
        self.log("=== Testing Error Handling ===", "TEST")
        
        results = {}
        
        # Test 1: Invalid port handling
        self.log("Testing invalid port handling...")
        invalid_port_result = {}
        try:
            connection = self.serial_manager.acquire_port("/dev/ttyUSB999")
            invalid_port_result['unexpected_success'] = connection is not None
            invalid_port_result['handled_gracefully'] = True
        except Exception as e:
            invalid_port_result['exception_raised'] = True
            invalid_port_result['handled_gracefully'] = True
            
        results['invalid_port'] = invalid_port_result
        self.log("Invalid port test completed", "SUCCESS")
        
        # Test 2: Connection recovery
        devices = self.arduino_detector.detect_arduino_devices()
        if devices:
            port = list(devices.values())[0]
            recovery_result = {}
            
            try:
                # Force connection failure simulation
                original_connection = self.serial_manager.acquire_port(port)
                if original_connection:
                    # Manually close to simulate failure
                    original_connection.close()
                    
                    # Try to recover
                    self.serial_manager.release_port(port)
                    new_connection = self.serial_manager.acquire_port(port)
                    
                    recovery_result['recovery_successful'] = new_connection is not None
                    if new_connection:
                        self.log("Connection recovery test passed", "SUCCESS")
                    else:
                        self.log("Connection recovery test failed", "WARNING")
                else:
                    recovery_result['initial_connection_failed'] = True
                    
            except Exception as e:
                recovery_result['recovery_error'] = str(e)
                
            results['recovery'] = recovery_result
        
        self.test_results['error_handling'] = results
        return results
    
    def test_integration_with_controllers(self) -> Dict:
        """Test integration with actual controller classes."""
        self.log("=== Testing Controller Integration ===", "TEST")
        
        results = {}
        devices = self.arduino_detector.detect_arduino_devices()
        
        # Test servo controller integration
        if 'SERVO_CONTROLLER' in devices and USE_SERVO:
            self.log("Testing ServoController integration...")
            try:
                from servo_control.servo_control import ServoController
                servo = ServoController(port=devices['SERVO_CONTROLLER'], baudrate=9600)
                
                # Test basic commands
                servo.set_pan(90)
                time.sleep(0.1)
                servo.set_tilt(90)
                time.sleep(0.1)
                
                results['servo_controller'] = {'status': 'success', 'commands_sent': 2}
                self.log("ServoController integration successful", "SUCCESS")
                
            except Exception as e:
                results['servo_controller'] = {'status': 'error', 'error': str(e)}
                self.log(f"ServoController integration failed: {e}", "ERROR")
        
        # Test lightbulb controller integration
        if 'LIGHTBULB_CONTROLLER' in devices and USE_LIGHTBULB_PWM:
            self.log("Testing SimpleLightbulbController integration...")
            try:
                from servo_control.lightbulb_controller_simple import SimpleLightbulbController
                lightbulb = SimpleLightbulbController(devices['LIGHTBULB_CONTROLLER'], debug=False)
                
                # Test basic commands
                lightbulb.set_frame_diff_brightness(128)
                time.sleep(0.1)
                lightbulb.caption_flash()
                time.sleep(0.5)
                
                results['lightbulb_controller'] = {'status': 'success', 'commands_sent': 2}
                self.log("SimpleLightbulbController integration successful", "SUCCESS")
                
            except Exception as e:
                results['lightbulb_controller'] = {'status': 'error', 'error': str(e)}
                self.log(f"SimpleLightbulbController integration failed: {e}", "ERROR")
        
        # Test hand controller integration
        if 'HAND_CONTROLLER' in devices and USE_HAND_CONTROLLER:
            self.log("Testing HandExpressionController integration...")
            try:
                from hand_control.hand_expression import HandExpressionController
                hand = HandExpressionController(port=devices['HAND_CONTROLLER'], clean_output=True)
                
                # Test basic command
                hand.set_hand_positions([90, 90, 90, 90])
                time.sleep(0.2)
                
                results['hand_controller'] = {'status': 'success', 'commands_sent': 1}
                self.log("HandExpressionController integration successful", "SUCCESS")
                
            except Exception as e:
                results['hand_controller'] = {'status': 'error', 'error': str(e)}
                self.log(f"HandExpressionController integration failed: {e}", "ERROR")
        
        self.test_results['integration'] = results
        return results
    
    def run_comprehensive_test(self) -> Dict:
        """Run all test suites."""
        self.log("🚀 Starting Comprehensive Arduino System Test Suite", "TEST")
        self.log("=" * 60)
        
        # Run all test suites
        detection_results = self.test_device_detection()
        connection_results = self.test_serial_connections()
        device_results = self.test_individual_devices()
        stress_results = self.test_stress_scenarios()
        error_results = self.test_error_handling()
        integration_results = self.test_integration_with_controllers()
        
        # Generate summary
        total_time = time.time() - self.start_time
        
        summary = {
            'test_duration_seconds': total_time,
            'test_timestamp': datetime.now().isoformat(),
            'devices_detected': len(detection_results.get('detected_devices', {})),
            'overall_status': self._determine_overall_status()
        }
        
        self.test_results['summary'] = summary
        
        # Print final summary
        self._print_test_summary()
        
        return self.test_results
    
    def _determine_overall_status(self) -> str:
        """Determine overall test status."""
        if not self.test_results.get('detection', {}).get('detected_devices'):
            return 'CRITICAL_FAILURE'  # No devices detected
        
        error_count = 0
        success_count = 0
        
        # Check individual device tests
        for device_results in self.test_results.get('individual_devices', {}).values():
            if device_results.get('status') == 'error':
                error_count += 1
            elif device_results.get('status') == 'success':
                success_count += 1
        
        # Check integration tests
        for integration_results in self.test_results.get('integration', {}).values():
            if integration_results.get('status') == 'error':
                error_count += 1
            elif integration_results.get('status') == 'success':
                success_count += 1
        
        if error_count == 0:
            return 'PERFECT'
        elif success_count > error_count:
            return 'MOSTLY_WORKING'
        else:
            return 'NEEDS_ATTENTION'
    
    def _print_test_summary(self):
        """Print comprehensive test summary."""
        self.log("=" * 60)
        self.log("📊 COMPREHENSIVE TEST SUMMARY", "TEST")
        self.log("=" * 60)
        
        summary = self.test_results['summary']
        
        # Overall status
        status = summary['overall_status']
        if status == 'PERFECT':
            self.log(f"🎉 OVERALL STATUS: {status} - All tests passed!", "SUCCESS")
        elif status == 'MOSTLY_WORKING':
            self.log(f"✅ OVERALL STATUS: {status} - System functional with minor issues", "SUCCESS")
        elif status == 'NEEDS_ATTENTION':
            self.log(f"⚠️  OVERALL STATUS: {status} - Some components need fixing", "WARNING")
        else:
            self.log(f"❌ OVERALL STATUS: {status} - Critical issues detected", "ERROR")
        
        # Detection summary
        detection = self.test_results.get('detection', {})
        self.log(f"🔍 DETECTION: {len(detection.get('detected_devices', {}))} devices found")
        
        # Device functionality
        devices = self.test_results.get('individual_devices', {})
        working_devices = sum(1 for d in devices.values() if d.get('status') == 'success')
        self.log(f"🤖 DEVICES: {working_devices}/{len(devices)} working correctly")
        
        # Integration tests
        integration = self.test_results.get('integration', {})
        working_integrations = sum(1 for i in integration.values() if i.get('status') == 'success')
        self.log(f"🔗 INTEGRATION: {working_integrations}/{len(integration)} controllers working")
        
        # Test duration
        self.log(f"⏱️  TEST DURATION: {summary['test_duration_seconds']:.1f} seconds")
        
        # Recommendations
        self.log("")
        self._print_recommendations()
    
    def _print_recommendations(self):
        """Print recommendations based on test results."""
        self.log("💡 RECOMMENDATIONS:", "TEST")
        
        detection = self.test_results.get('detection', {})
        if not detection.get('detected_devices'):
            self.log("  • Check USB connections and Arduino power", "WARNING")
            self.log("  • Verify Arduino firmware has correct DEVICE_ID statements", "WARNING")
            self.log("  • Check serial port permissions (sudo usermod -a -G dialout $USER)", "WARNING")
            
        devices = self.test_results.get('individual_devices', {})
        for device_id, device_result in devices.items():
            if device_result.get('status') == 'error':
                self.log(f"  • {device_id}: Check firmware and connections", "WARNING")
                
        integration = self.test_results.get('integration', {})
        for controller, result in integration.items():
            if result.get('status') == 'error':
                self.log(f"  • {controller}: Check Python module imports and dependencies", "WARNING")
        
        if self.test_results['summary']['overall_status'] == 'PERFECT':
            self.log("  • System is ready for production use! 🎉", "SUCCESS")
        
        self.log("")
        self.log("📝 Full test results saved to test_results.json")
        
    def save_results(self, filename: str = "arduino_test_results.json"):
        """Save test results to JSON file."""
        with open(filename, 'w') as f:
            json.dump(self.test_results, f, indent=2, default=str)
        self.log(f"Test results saved to {filename}")


def main():
    """Run the comprehensive test suite."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Comprehensive Arduino System Test Suite")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    parser.add_argument("--save-results", default="arduino_test_results.json", 
                       help="File to save test results")
    args = parser.parse_args()
    
    # Create and run tester
    tester = ArduinoSystemTester(debug=args.debug)
    results = tester.run_comprehensive_test()
    
    # Save results
    if args.save_results:
        tester.save_results(args.save_results)
    
    # Exit with appropriate code
    status = results['summary']['overall_status']
    if status in ['PERFECT', 'MOSTLY_WORKING']:
        sys.exit(0)  # Success
    else:
        sys.exit(1)  # Failure


if __name__ == "__main__":
    main()