#!/usr/bin/env python3
"""
Test the hand controller communication flow from machine.py perspective
"""
import os
import sys
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set environment variables
import arduino_port_detector
detector = arduino_port_detector.ArduinoPortDetector()
detector.set_environment_variables()

print("=== HAND CONTROLLER COMMUNICATION TEST ===")
print(f"DETECTED_HAND_PORT: {os.environ.get('DETECTED_HAND_PORT')}")

def test_direct_communication():
    """Test the direct communication path that machine.py uses"""
    try:
        from hand_control.direct_hand_control import (
            start_hand_controller,
            change_to_emotion,
            stop_hand_controller
        )
        
        print("📋 Starting hand controller (headless mode)...")
        success = start_hand_controller(headless=False)  # Use GUI mode to see what's happening
        
        if not success:
            print("❌ Failed to start hand controller")
            return False
            
        print("✅ Hand controller started successfully")
        
        # Wait for initialization
        print("⏳ Waiting for initialization (5 seconds)...")
        time.sleep(5)
        
        # Test each emotional state
        emotions = ['energized_engaged', 'alert_curious', 'calm_observant', 'quiet_detached', 'withdrawn_distant']
        
        for emotion in emotions:
            print(f"\n🎭 Testing emotion: {emotion}")
            success = change_to_emotion(emotion)
            if success:
                print(f"✅ Successfully set emotion to {emotion}")
            else:
                print(f"❌ Failed to set emotion to {emotion}")
            
            # Wait to see if hand moves
            print("⏳ Waiting 3 seconds to observe movement...")
            time.sleep(3)
        
        print("\n🔄 Stopping hand controller...")
        stop_hand_controller()
        print("✅ Test complete")
        return True
        
    except Exception as e:
        print(f"❌ Error in direct communication test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_direct_communication()