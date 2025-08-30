#!/usr/bin/env python3
"""
Test ServoController Class Integration
=====================================
Tests the ServoController class used by machine.py
"""

import sys
import os
import time

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import SERIAL_PORT, BAUD_RATE, USE_SERVO
from servo_control.servo_control import ServoController

def test_servo_controller_integration():
    """Test ServoController class integration."""
    print("=" * 50)
    print("SERVO CONTROLLER INTEGRATION TEST")
    print("=" * 50)
    
    print(f"USE_SERVO: {USE_SERVO}")
    print(f"SERIAL_PORT: {SERIAL_PORT}")
    print(f"BAUD_RATE: {BAUD_RATE}")
    
    if not USE_SERVO:
        print("❌ USE_SERVO is False - servos disabled in config!")
        return None
    
    print(f"\n🔗 Creating ServoController...")
    
    try:
        servos = ServoController(port=SERIAL_PORT, baudrate=BAUD_RATE)
        
        if servos.ser is None:
            print("❌ ServoController failed to connect!")
            return None
        
        print(f"✅ ServoController connected successfully!")
        print(f"   Serial object: {servos.ser}")
        print(f"   Port: {servos.ser.port}")
        print(f"   Baudrate: {servos.ser.baudrate}")
        print(f"   Is open: {servos.ser.is_open}")
        
        # Test the ServoController methods
        print("\n🎯 Testing ServoController methods...")
        
        test_movements = [
            ("set_pan", 90),
            ("set_tilt", 90),
            ("set_lung_position", 90),
            ("set_pan", 45),
            ("set_tilt", 135), 
            ("set_lung_position", 60),
            ("set_pan", 135),
            ("set_tilt", 45),
            ("set_lung_position", 120),
            ("set_pan", 90),
            ("set_tilt", 90),
            ("set_lung_position", 90)
        ]
        
        for method_name, value in test_movements:
            print(f"  Calling: {method_name}({value})")
            method = getattr(servos, method_name)
            
            if method_name == "set_lung_position":
                method(value, force=True)  # Force send
            else:
                method(value)
                
            time.sleep(1.5)  # Allow time for movement
        
        # Test lung modes
        print("\n🫁 Testing lung modes...")
        servos.set_lung("hold")
        time.sleep(2)
        servos.set_lung("slow")
        time.sleep(3)
        servos.set_lung("hold")
        
        print("\n✅ ServoController integration test complete!")
        return servos
        
    except Exception as e:
        print(f"❌ Error creating ServoController: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_gaze_and_breathing_functions():
    """Test the actual gaze and breathing calculation functions."""
    print("\n" + "=" * 50)
    print("GAZE AND BREATHING FUNCTION TEST")
    print("=" * 50)
    
    # Test gaze calculation
    try:
        from vision.gaze import update_gaze
        import numpy as np
        
        print("🎯 Testing gaze calculation...")
        
        # Create dummy frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Test with no face
        person_present, pan, tilt = update_gaze(frame, None, 0.0)
        print(f"  No face: person_present={person_present}, pan={pan}, tilt={tilt}")
        
        # Test with face box
        face_box = (200, 150, 400, 350)  # (x1, y1, x2, y2)
        person_present, pan, tilt = update_gaze(frame, face_box, 0.0)
        print(f"  With face: person_present={person_present}, pan={pan}, tilt={tilt}")
        
        print("✅ Gaze calculation working")
        
    except Exception as e:
        print(f"❌ Gaze calculation error: {e}")
        import traceback
        traceback.print_exc()
    
    # Test breathing calculation
    try:
        from breathing.breathing import update_lung_position
        
        print("\n🫁 Testing breathing calculation...")
        
        # Create a dummy servo controller for testing
        class DummyServoController:
            def __init__(self):
                self.commands = []
            def set_lung_position(self, angle, force=False):
                self.commands.append(f"LUNG:{angle} (force={force})")
                print(f"    Dummy servo received: LUNG:{angle}")
        
        dummy_servo = DummyServoController()
        
        # Test breathing calculation
        result = update_lung_position(
            current_mood=0.5,
            person_present=True,
            delta=0.016,  # ~60fps
            lung_angle=0.0,
            breath_speed=2.0,
            breath_paused=False,
            last_breath_direction="down",
            pause_start_time=0,
            pause_duration=1.0,
            servo_controller=dummy_servo
        )
        
        lung_pos = result[0]
        print(f"  Breathing calculation result: lung_pos={lung_pos}")
        print(f"  Commands sent to servo: {len(dummy_servo.commands)}")
        
        print("✅ Breathing calculation working")
        
    except Exception as e:
        print(f"❌ Breathing calculation error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    servos = test_servo_controller_integration()
    test_gaze_and_breathing_functions()
    
    if servos and servos.ser:
        print("\n🔌 Closing servo controller...")
        servos.ser.close()