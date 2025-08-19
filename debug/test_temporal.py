#!/usr/bin/env python3
"""
Test Temporal Grounding
"""

import tempfile
import os
from PIL import Image
from captioner.model_wrapper import MultimodalModel
from captioner.captioner import Captioner

def test_temporal_transitions():
    print("=== TESTING TEMPORAL GROUNDING ===")
    
    cap = Captioner()
    cap.first_caption_done = True
    cap.current_emotion_state = 'alert_curious'
    model = MultimodalModel(memory_ref=cap)

    # Test sequence: Red -> Blue -> Yellow
    colors = [('red', 'Red'), ('blue', 'Blue'), ('yellow', 'Yellow')]
    
    for i, (color, name) in enumerate(colors):
        img = Image.new('RGB', (100, 100), color=color)
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            img.save(f.name)
            temp_path = f.name

        try:
            result = model.caption_image(temp_path, flowing=True, first_time=False)
            print(f"{i+1}. {name}: \"{result}\"")
            
            # Add to memory for next iteration
            cap.observe(result, 0.6, temp_path)
            
        except Exception as e:
            print(f"{i+1}. {name}: ERROR - {e}")
            
        finally:
            os.unlink(temp_path)
    
    print("\n=== ANALYSIS ===")
    print("Looking for temporal words like 'now', 'after', 'before', 'just'")
    print("Looking for memory connections and curiosity about changes")

if __name__ == "__main__":
    test_temporal_transitions()
