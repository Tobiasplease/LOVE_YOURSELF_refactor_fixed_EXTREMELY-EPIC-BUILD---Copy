#!/usr/bin/env python3
"""
Test hand controller communication with multiple Arduinos connected
"""
import os
import sys
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set environment variables
import arduino_port_detector
detector = arduino_port_detector.ArduinoPortDetector()
detector.set_environment_variables()

print("=== HAND CONTROLLER MULTI-PORT TEST ===")
print("Available environment variables:")
for k, v in os.environ.items():
    if 'DETECTED' in k:
        print(f"  {k}: {v}")

def test_direct_serial_communication():
    """Test direct serial communication to verify the hand controller responds on its detected port"""
    detected_port = os.environ.get('DETECTED_HAND_PORT')
    if not detected_port:
        print("❌ No DETECTED_HAND_PORT found")
        return False
    
    print(f"\n📡 Testing direct serial communication to hand controller at {detected_port}")
    
    try:
        import serial
        import time
        
        # Connect directly to the detected hand controller port
        ser = serial.Serial(detected_port, 9600, timeout=2)
        print(f"✅ Connected to {detected_port}")
        
        # Wait for Arduino boot
        time.sleep(3)
        
        # Clear buffers
        ser.reset_input_buffer()
        ser.reset_output_buffer()
        
        # Send test commands
        test_commands = [
            "HAND,90,90,90,90",
            "HAND,45,135,45,135", 
            "HEARTBEAT"
        ]
        
        for cmd in test_commands:
            full_cmd = cmd + "\n"
            print(f"📤 Sending: {cmd}")
            ser.write(full_cmd.encode())
            time.sleep(0.5)
            
            # Read response
            responses = []
            while ser.in_waiting > 0:
                try:
                    response = ser.readline().decode().strip()
                    if response:
                        responses.append(response)
                except:
                    break
            
            if responses:
                for resp in responses:
                    print(f"📥 Received: {resp}")
            else:
                print("📥 No response")
            
            time.sleep(1)
        
        ser.close()
        return True
        
    except Exception as e:
        print(f"❌ Error in direct serial test: {e}")
        return False

def test_hand_controller_with_multiple_arduinos():
    """Test the hand controller system with multiple Arduinos connected"""
    try:
        from hand_control.direct_hand_control import (
            start_hand_controller,
            change_to_emotion,
            stop_hand_controller
        )
        
        print(f"\n🎭 Testing hand controller with multiple Arduinos connected...")
        
        # Start hand controller
        print("📋 Starting hand controller...")
        success = start_hand_controller(headless=True)
        
        if not success:
            print("❌ Failed to start hand controller")
            return False
            
        print("✅ Hand controller started")
        time.sleep(3)
        
        # Test emotion changes
        emotions = ['calm_observant', 'energized_engaged', 'alert_curious']
        
        for emotion in emotions:
            print(f"\n🎭 Testing emotion: {emotion}")
            success = change_to_emotion(emotion)
            
            if success:
                print(f"✅ Successfully switched to {emotion}")
                # Wait and observe
                print("⏳ Waiting 3 seconds to observe movement...")
                time.sleep(3)
            else:
                print(f"❌ Failed to switch to {emotion}")
        
        print("\n🔄 Stopping hand controller...")
        stop_hand_controller()
        return True
        
    except Exception as e:
        print(f"❌ Error in multi-Arduino test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Test direct serial communication first
    print("Step 1: Direct serial communication test")
    test_direct_serial_communication()
    
    print("\n" + "="*50)
    print("Step 2: Hand controller system test")
    test_hand_controller_with_multiple_arduinos()