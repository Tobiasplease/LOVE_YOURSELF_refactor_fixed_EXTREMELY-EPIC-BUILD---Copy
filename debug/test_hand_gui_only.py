#!/usr/bin/env python3
"""
Test just the hand controller GUI to see connection issues
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set environment variables first
import arduino_port_detector
detector = arduino_port_detector.ArduinoPortDetector()
detector.set_environment_variables()

print("=== HAND CONTROLLER GUI TEST ===")
print(f"DETECTED_HAND_PORT: {os.environ.get('DETECTED_HAND_PORT', 'NOT SET')}")

try:
    from hand_control.hand_control_interface import CleanCursorInterface
    
    print("Starting hand controller GUI...")
    print("Check if the GUI shows the correct port and connects automatically")
    print("Close the GUI window when done testing")
    
    # Create and start the GUI
    controller = CleanCursorInterface()
    controller.run()
    
    print("GUI closed.")
    
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure tkinter is installed")
    
except Exception as e:
    print(f"Error starting GUI: {e}")
    import traceback
    traceback.print_exc()