#!/usr/bin/env python3
"""
Test the exact initialization sequence from machine.py to identify interference
"""
import os
import sys
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("=== MACHINE.PY INITIALIZATION SEQUENCE TEST ===")

def test_machine_py_sequence():
    """Replicate the exact initialization sequence from machine.py"""
    
    print("🚀 Step 1: Arduino Detection (like machine.py)")
    from arduino_port_detector import ArduinoPortDetector
    
    arduino_detector = ArduinoPortDetector(debug=False)
    detected_ports = arduino_detector.detect_arduino_ports()
    arduino_detector.set_environment_variables()
    
    print(f"   Detected ports: {detected_ports}")
    
    print("\n🚀 Step 2: Lightbulb Controller Init (like machine.py)")
    lightbulb = None
    lightbulb_port = detected_ports.get('LIGHTBULB_CONTROLLER')
    if lightbulb_port:
        print(f"   Initializing lightbulb on {lightbulb_port}...")
        time.sleep(0.5)  # Same as machine.py
        try:
            from servo_control.lightbulb_controller_robust import ThreadSafeLightbulbWrapper
            lightbulb = ThreadSafeLightbulbWrapper(lightbulb_port, debug=False)
            print(f"   ✅ Lightbulb controller initialized on {lightbulb_port}")
        except Exception as e:
            print(f"   ❌ Lightbulb init failed: {e}")
    else:
        print("   ⚠️ No lightbulb controller detected")
    
    print("\n🚀 Step 3: Servo Controller Init (like machine.py)")
    servos = None  
    servo_port = detected_ports.get('SERVO_CONTROLLER')
    if servo_port:
        print(f"   Initializing servo on {servo_port}...")
        time.sleep(0.5)  # Same as machine.py
        try:
            from servo_control.servo_control import ServoController
            servos = ServoController(port=servo_port, baudrate=9600)
            print(f"   ✅ Servo controller initialized on {servo_port}")
        except Exception as e:
            print(f"   ❌ Servo init failed: {e}")
    else:
        print("   ⚠️ No servo controller detected")
    
    print("\n🚀 Step 4: Hand Controller Init (like machine.py)")  
    hand_controller_started = False
    try:
        from hand_control.direct_hand_control import start_hand_controller, start_autonomous_mode
        
        print("   Starting hand controller with UI...")
        hand_controller_started = start_hand_controller(headless=False)  # Same as machine.py
        
        if hand_controller_started:
            print("   ✅ Hand controller started with UI")
            
            # Same autonomous mode sequence as machine.py
            print("   Starting autonomous mode...")
            time.sleep(1)  # Same as machine.py
            if start_autonomous_mode():
                print("   ✅ Autonomous mode activated")
            else:
                print("   ❌ Failed to start autonomous mode")
        else:
            print("   ❌ Hand controller failed to start")
            
    except Exception as e:
        print(f"   ❌ Hand controller error: {e}")
    
    print(f"\n📊 Final Status:")
    print(f"   Lightbulb: {'✅' if lightbulb else '❌'}")
    print(f"   Servo: {'✅' if servos and hasattr(servos, 'ser') and servos.ser else '❌'}")
    print(f"   Hand: {'✅' if hand_controller_started else '❌'}")
    
    # Test hand controller functionality with all systems active
    if hand_controller_started:
        print(f"\n🎭 Testing hand controller with all systems active (like machine.py)...")
        
        try:
            from hand_control.direct_hand_control import change_to_emotion
            
            emotions = ['calm_observant', 'energized_engaged', 'alert_curious']
            
            for i, emotion in enumerate(emotions):
                print(f"   Test {i+1}: {emotion}")
                result = change_to_emotion(emotion)
                
                if result:
                    print(f"   ✅ Success")
                    # Wait and see if hand responds
                    time.sleep(3)
                else:
                    print(f"   ❌ Failed - THIS IS THE INTERFERENCE!")
                    break
                    
        except Exception as e:
            print(f"   ❌ Hand controller test error: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n🔄 Test complete")
    return hand_controller_started

if __name__ == "__main__":
    test_machine_py_sequence()