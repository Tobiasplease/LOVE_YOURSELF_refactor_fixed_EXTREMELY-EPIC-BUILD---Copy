#!/usr/bin/env python3
"""
Test serial interference between multiple controllers
"""
import os
import sys
import time
import threading
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set environment variables
import arduino_port_detector
detector = arduino_port_detector.ArduinoPortDetector()
detector.set_environment_variables()

print("=== SERIAL INTERFERENCE TEST ===")
for k, v in os.environ.items():
    if 'DETECTED' in k:
        print(f"  {k}: {v}")

def test_individual_controllers():
    """Test each controller individually to isolate the problem"""
    
    print("\n🧪 Test 1: Hand Controller ONLY")
    try:
        from hand_control.direct_hand_control import (
            start_hand_controller,
            change_to_emotion,
            stop_hand_controller
        )
        
        print("📋 Starting hand controller...")
        success = start_hand_controller(headless=True)
        
        if success:
            time.sleep(2)
            print("🎭 Testing emotion change...")
            result = change_to_emotion("calm_observant")
            print(f"   Result: {'✅ Success' if result else '❌ Failed'}")
            
            stop_hand_controller()
            print("✅ Hand controller test complete")
        else:
            print("❌ Hand controller failed to start")
            
    except Exception as e:
        print(f"❌ Hand controller error: {e}")
    
    time.sleep(3)  # Wait between tests
    
    print("\n🧪 Test 2: Servo Controller ONLY") 
    try:
        servo_port = os.environ.get('DETECTED_SERVO_CONTROLLER_PORT')
        if servo_port:
            from servo_control.servo_control import ServoController
            
            print(f"📋 Starting servo controller on {servo_port}...")
            servo_controller = ServoController(servo_port)
            
            if servo_controller.ser:
                print("🎯 Testing servo movement...")
                servo_controller.set_servo_position(0, 90)  # Pan to 90 degrees
                time.sleep(1)
                servo_controller.set_servo_position(1, 45)  # Tilt to 45 degrees
                print("✅ Servo controller test complete")
                
                servo_controller.ser.close()
            else:
                print("❌ Servo controller connection failed")
        else:
            print("❌ No servo controller port detected")
            
    except Exception as e:
        print(f"❌ Servo controller error: {e}")
    
    time.sleep(3)  # Wait between tests
        
    print("\n🧪 Test 3: Lightbulb Controller ONLY")
    try:
        lightbulb_port = os.environ.get('DETECTED_LIGHTBULB_CONTROLLER_PORT')
        if lightbulb_port:
            from servo_control.lightbulb_controller_robust import ThreadSafeLightbulbWrapper
            
            print(f"📋 Starting lightbulb controller on {lightbulb_port}...")
            lightbulb = ThreadSafeLightbulbWrapper(lightbulb_port, debug=True)
            
            print("💡 Testing lightbulb brightness...")
            lightbulb.set_brightness(128)
            time.sleep(1)
            lightbulb.set_brightness(0)
            print("✅ Lightbulb controller test complete")
            
            lightbulb.cleanup()
        else:
            print("❌ No lightbulb controller port detected")
            
    except Exception as e:
        print(f"❌ Lightbulb controller error: {e}")

def test_combined_controllers():
    """Test all controllers together like machine.py does"""
    
    print("\n🧪 Test 4: ALL CONTROLLERS TOGETHER (like machine.py)")
    
    controllers = {}
    
    try:
        # Start hand controller
        print("📋 Starting hand controller...")
        from hand_control.direct_hand_control import start_hand_controller, change_to_emotion
        hand_success = start_hand_controller(headless=True)
        controllers['hand'] = hand_success
        print(f"   Hand controller: {'✅' if hand_success else '❌'}")
        
        time.sleep(1)
        
        # Start servo controller  
        print("📋 Starting servo controller...")
        servo_port = os.environ.get('DETECTED_SERVO_CONTROLLER_PORT')
        if servo_port:
            from servo_control.servo_control import ServoController
            servo_controller = ServoController(servo_port)
            controllers['servo'] = servo_controller.ser is not None
            print(f"   Servo controller: {'✅' if controllers['servo'] else '❌'}")
        else:
            controllers['servo'] = False
            print("   Servo controller: ❌ No port")
            
        time.sleep(1)
        
        # Start lightbulb controller
        print("📋 Starting lightbulb controller...")  
        lightbulb_port = os.environ.get('DETECTED_LIGHTBULB_CONTROLLER_PORT')
        if lightbulb_port:
            from servo_control.lightbulb_controller_robust import ThreadSafeLightbulbWrapper
            lightbulb = ThreadSafeLightbulbWrapper(lightbulb_port, debug=False)
            controllers['lightbulb'] = True
            print("   Lightbulb controller: ✅")
        else:
            controllers['lightbulb'] = False
            print("   Lightbulb controller: ❌ No port")
        
        print(f"\n📊 Controller Status: {controllers}")
        
        # Now test if hand controller still works
        if controllers['hand']:
            print("\n🎭 Testing hand controller with all controllers active...")
            for i, emotion in enumerate(['calm_observant', 'energized_engaged', 'alert_curious']):
                print(f"   Test {i+1}: {emotion}")
                result = change_to_emotion(emotion)
                print(f"   Result: {'✅ Success' if result else '❌ Failed'}")
                time.sleep(2)
        
        print("✅ Combined controller test complete")
        
    except Exception as e:
        print(f"❌ Combined controller error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_individual_controllers()
    test_combined_controllers()