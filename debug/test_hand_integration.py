#!/usr/bin/env python3
"""
Test the complete hand controller integration the way machine.py uses it
"""
import os
import sys
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set environment variables
import arduino_port_detector
detector = arduino_port_detector.ArduinoPortDetector()
detector.set_environment_variables()

print("=== HAND CONTROLLER INTEGRATION TEST ===")
print(f"DETECTED_HAND_PORT: {os.environ.get('DETECTED_HAND_PORT')}")

def test_machine_py_integration():
    """Test exactly how machine.py uses the hand controller"""
    try:
        from hand_control.direct_hand_control import (
            start_hand_controller,
            change_to_emotion,
            stop_hand_controller,
            start_autonomous_mode
        )
        
        print("📋 Starting hand controller (like machine.py does)...")
        success = start_hand_controller(headless=False)  # machine.py uses headless=False
        
        if not success:
            print("❌ Failed to start hand controller")
            return False
            
        print("✅ Hand controller started successfully")
        
        # Wait for initialization (like machine.py does)
        print("⏳ Waiting for initialization (3 seconds)...")
        time.sleep(3)
        
        # Start autonomous mode (like machine.py does)
        print("🤖 Starting autonomous mode...")
        auto_success = start_autonomous_mode()
        if auto_success:
            print("✅ Autonomous mode started successfully")
        else:
            print("⚠️ Failed to start autonomous mode")
        
        # Test emotion changes (like machine.py does based on mood)
        test_emotions = [
            'calm_observant',    # Default neutral state
            'energized_engaged', # High mood + high arousal
            'alert_curious',     # Anxious and alert
            'quiet_detached',    # Uncertain and confused
            'withdrawn_distant', # Withdrawn and foggy
        ]
        
        for emotion in test_emotions:
            print(f"\n🎭 Setting emotion to: {emotion} (like machine.py would)")
            success = change_to_emotion(emotion)
            
            if success:
                print(f"✅ Successfully changed to {emotion}")
                print("⏳ Observing movement for 5 seconds...")
                time.sleep(5)  # Watch for movement
            else:
                print(f"❌ Failed to change to {emotion}")
        
        print(f"\n📊 Hand controller should now be running autonomously in '{test_emotions[-1]}' mode")
        print("⏳ Letting it run for 10 more seconds...")
        time.sleep(10)
        
        print("\n🔄 Stopping hand controller...")
        stop_hand_controller()
        print("✅ Test complete")
        return True
        
    except Exception as e:
        print(f"❌ Error in integration test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_machine_py_integration()