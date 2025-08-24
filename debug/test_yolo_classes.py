#!/usr/bin/env python3
"""
Test to understand YOLO class mappings and investigate person ID leak.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from ultralytics import YOLO
    
    def check_yolo_classes():
        """Check what class IDs YOLO uses for different objects."""
        print("Investigating YOLO class mappings...")
        
        model = YOLO("models/yolov8m.pt")
        
        print(f"Available classes: {len(model.names)}")
        
        # Find person class
        person_id = None
        for class_id, class_name in model.names.items():
            if class_name == "person":
                person_id = class_id
                print(f"PERSON CLASS ID: {class_id} -> '{class_name}'")
                break
        
        if person_id is None:
            print("No 'person' class found!")
        else:
            print(f"The YOLO class ID for 'person' is: {person_id}")
            
            # Check if it's 67
            if person_id == 67:
                print("🚨 FOUND IT! Class ID 67 is 'person' - this explains 'person 67' leak!")
            else:
                print(f"Class ID {person_id} is not 67, so this might not be the source.")
        
        # Show some other class mappings for context
        print("\nOther class mappings:")
        for i in range(min(10, len(model.names))):
            print(f"  {i}: {model.names[i]}")
        
        print(f"\n... and around ID 67:")
        for i in range(65, min(70, len(model.names))):
            print(f"  {i}: {model.names.get(i, 'N/A')}")

    if __name__ == "__main__":
        check_yolo_classes()
        
except ImportError as e:
    print(f"Could not import YOLO: {e}")
    print("This test requires ultralytics to be installed.")
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    print("Make sure models/yolov8m.pt exists.")