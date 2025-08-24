#!/usr/bin/env python3
"""
Test the new visual-based person recognition system.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from captioner.person_recognition import person_recognition

def test_visual_recognition():
    """Test visual person recognition with example captions."""
    print("Testing Visual Person Recognition System")
    print("=" * 50)
    
    # Test scenarios with different people
    test_captions = [
        "I see a young person with long black hair sitting at the desk",
        "The person with short brown hair is typing on a laptop", 
        "A middle-aged individual with glasses is reading",
        "The young person with long black hair is back, now standing",
        "An older person with gray hair walks into the room",
        "The person with short brown hair and glasses is working again",
        "Someone with blonde hair and a blue shirt appears",
        "The middle-aged person with glasses looks up from their book",
    ]
    
    print("Processing captions to build person profiles...\n")
    
    for i, caption in enumerate(test_captions):
        print(f"Caption {i+1}: {caption}")
        
        # Extract visual features
        visual_profile = person_recognition.extract_visual_features(caption)
        if visual_profile:
            print(f"  Extracted features: hair={visual_profile.hair_color} {visual_profile.hair_length}, "
                  f"age={visual_profile.apparent_age}, build={visual_profile.build}")
        
        # Recognize or create person
        person_id = person_recognition.recognize_or_create_person(caption)
        if person_id:
            context = person_recognition.get_person_context(person_id)
            print(f"  Recognized as: {person_id} -> {context}")
            
            # Show similarity scores for debugging
            if visual_profile and len(person_recognition.known_people) > 1:
                print("    Similarity scores:")
                for existing_id, existing_person in person_recognition.known_people.items():
                    if existing_id != person_id:
                        similarity = existing_person.visual_profile.similarity_score(visual_profile)
                        print(f"      vs {existing_id}: {similarity:.2f}")
        else:
            print("  No person detected")
        print()
    
    print("\n" + "=" * 50)
    print("FINAL PERSON REGISTRY:")
    print("=" * 50)
    
    for person_id, person_record in person_recognition.known_people.items():
        print(f"\n{person_id}:")
        print(f"  Description: {person_record.get_description()}")
        print(f"  Encounters: {person_record.total_encounters}")
        print(f"  Familiarity: {person_record.familiarity_level:.2f}")
        print(f"  Visual Profile:")
        print(f"    Hair: {person_record.visual_profile.hair_color} {person_record.visual_profile.hair_length}")
        print(f"    Age: {person_record.visual_profile.apparent_age}")
        print(f"    Build: {person_record.visual_profile.build}")
        if person_record.visual_profile.common_clothing:
            print(f"    Clothing: {person_record.visual_profile.common_clothing}")
    
    print(f"\n{person_recognition.get_recognition_summary()}")
    
    print("\n" + "=" * 50)
    print("KEY IMPROVEMENTS:")
    print("=" * 50)
    print("[+] No more generic 'primary person'")
    print("[+] Visual characteristic-based recognition") 
    print("[+] Multiple people can be tracked simultaneously")
    print("[+] Builds familiarity naturally over encounters")
    print("[+] Descriptive context instead of just 'person I observe'")
    print("[+] Lightweight - only activates when people are present")

if __name__ == "__main__":
    test_visual_recognition()