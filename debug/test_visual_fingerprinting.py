#!/usr/bin/env python3
"""
Test visual fingerprinting integration with the captioner system.
"""

import time
import numpy as np
import cv2
from captioner.captioner import Captioner

def create_test_frame(content_type="office"):
    """Create a synthetic test frame for different scene types."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    if content_type == "office":
        # Simulate an office scene
        frame[100:300, 200:400] = [100, 120, 140]  # Desk area
        frame[50:150, 50:200] = [80, 80, 80]      # Computer screen
        frame[200:250, 500:600] = [200, 180, 160] # Window
    elif content_type == "kitchen":
        # Simulate a kitchen scene
        frame[150:350, 100:500] = [180, 160, 140] # Counter
        frame[50:200, 300:450] = [90, 90, 90]     # Appliance
        frame[200:300, 200:300] = [255, 255, 255] # White surface
    elif content_type == "bedroom":
        # Simulate a bedroom scene
        frame[200:400, 150:450] = [120, 100, 80]  # Bed
        frame[50:200, 500:600] = [60, 60, 60]     # Dark corner
        frame[100:150, 100:200] = [200, 200, 180] # Lamp
    
    return frame

def test_visual_stagnation_detection():
    """Test the visual stagnation detection system."""
    print("🧪 Testing Visual Fingerprinting System")
    print("=" * 50)
    
    # Create captioner instance
    captioner = Captioner()
    
    # Test 1: Same scene repeatedly (should detect stagnation)
    print("\n📋 Test 1: Same scene repeatedly")
    office_frame = create_test_frame("office")
    
    for i in range(8):
        hash_val = captioner.calculate_visual_hash(office_frame)
        captioner.update_visual_stagnation(hash_val)
        print(f"Frame {i+1}: Hash={hash_val[:8]}..., Stagnation={captioner.visual_stagnation_score:.2f}")
        time.sleep(0.1)  # Small delay
    
    stagnation_context = captioner.get_visual_stagnation_context()
    print(f"🎯 Stagnation Context: {stagnation_context}")
    
    # Test 2: Scene changes (should reset stagnation)
    print("\n🏠 Test 2: Scene changes")
    
    # Add some different scenes
    kitchen_frame = create_test_frame("kitchen")
    bedroom_frame = create_test_frame("bedroom")
    
    scenes = [kitchen_frame, bedroom_frame, office_frame, kitchen_frame]
    scene_names = ["kitchen", "bedroom", "office", "kitchen"]
    
    for i, (frame, name) in enumerate(zip(scenes, scene_names)):
        hash_val = captioner.calculate_visual_hash(frame)
        captioner.update_visual_stagnation(hash_val)
        print(f"{name.capitalize()}: Hash={hash_val[:8]}..., Stagnation={captioner.visual_stagnation_score:.2f}")
        time.sleep(0.1)
    
    final_context = captioner.get_visual_stagnation_context()
    print(f"🎯 Final Stagnation Context: {final_context}")
    
    # Test 3: Reactivity data integration
    print("\n⚡ Test 3: Reactivity data integration")
    
    # Simulate reactivity data with visual stagnation
    reactivity_data = {
        'visual_stagnation': captioner.visual_stagnation_score,
        'visual_similarity': 1.0,  # Identical frames
        'activity_level': 0.1,    # Low activity
        'sudden_change': 0.0      # No sudden change
    }
    
    novelty_before = captioner.novelty_score
    # Add some dummy text memories for novelty calculation
    captioner.observe("Looking at the same office scene", 0.5, "", reactivity_data=reactivity_data)
    captioner.observe("Still looking at the office", 0.5, "", reactivity_data=reactivity_data)
    
    print(f"Novelty before: {novelty_before:.2f}")
    print(f"Novelty after: {captioner.novelty_score:.2f}")
    print(f"Visual stagnation score: {captioner.visual_stagnation_score:.2f}")
    
    print("\n✅ Visual fingerprinting test complete!")

if __name__ == "__main__":
    test_visual_stagnation_detection()
