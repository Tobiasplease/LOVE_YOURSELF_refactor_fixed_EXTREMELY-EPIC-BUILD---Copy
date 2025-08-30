#!/usr/bin/env python3
"""
Test that Markov chain generation is working properly for different emotions
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

print("=== MARKOV GENERATION TEST ===")
print(f"DETECTED_HAND_PORT: {os.environ.get('DETECTED_HAND_PORT')}")

def monitor_hand_movement(duration=10):
    """Monitor the hand movement for a specified duration to see if Markov is working"""
    try:
        from hand_control.direct_hand_control import (
            start_hand_controller,
            change_to_emotion,
            stop_hand_controller,
            get_controller_instance
        )
        
        print("📋 Starting hand controller...")
        success = start_hand_controller(headless=False)
        
        if not success:
            print("❌ Failed to start hand controller")
            return False
            
        print("✅ Hand controller started successfully")
        
        # Wait for initialization
        time.sleep(3)
        
        # Test each emotion with movement monitoring
        emotions = ['calm_observant', 'energized_engaged', 'alert_curious', 'quiet_detached', 'withdrawn_distant']
        
        for emotion in emotions:
            print(f"\n🎭 Testing Markov generation for: {emotion}")
            success = change_to_emotion(emotion)
            
            if success:
                print(f"✅ Successfully switched to {emotion}")
                
                # Monitor movement for this emotion
                controller = get_controller_instance()
                if controller and hasattr(controller, 'hand_controller'):
                    print(f"📊 Monitoring {emotion} movement for {duration} seconds...")
                    
                    # Check if Markov generation is active
                    if hasattr(controller, 'markov_generation_active'):
                        print(f"🔄 Markov generation active: {controller.markov_generation_active}")
                    
                    # Monitor actual movement
                    start_time = time.time()
                    movement_count = 0
                    last_positions = None
                    
                    while time.time() - start_time < duration:
                        if hasattr(controller, 'current_finger_positions'):
                            current_pos = controller.current_finger_positions[:]
                            if last_positions and current_pos != last_positions:
                                movement_count += 1
                                print(f"📍 Movement {movement_count}: {current_pos}")
                            last_positions = current_pos
                        time.sleep(0.5)
                    
                    print(f"📈 {emotion}: {movement_count} position changes detected in {duration}s")
                    
                    # Check dataset availability
                    if hasattr(controller, 'available_datasets'):
                        datasets = controller.available_datasets.get(emotion, [])
                        print(f"📁 {emotion}: {len(datasets)} datasets available")
                    
                    # Check Markov chains
                    if hasattr(controller, 'markov_chains'):
                        if emotion in controller.markov_chains:
                            chain_size = len(controller.markov_chains[emotion])
                            print(f"🔗 {emotion}: Markov chain with {chain_size} states")
                        else:
                            print(f"⚠️ {emotion}: No Markov chain loaded")
                
            else:
                print(f"❌ Failed to switch to {emotion}")
        
        print("\n🔄 Test complete - stopping hand controller...")
        stop_hand_controller()
        return True
        
    except Exception as e:
        print(f"❌ Error in Markov generation test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    monitor_hand_movement(duration=8)  # 8 seconds per emotion