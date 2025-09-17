#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mood.mood import MoodEngine

def test_caption_to_emotion_flow():
    """Test that captions now properly flow through to emotion states."""

    print("Testing Caption → Mood Analysis → Emotion State Flow")
    print("=" * 60)

    mood_engine = MoodEngine()

    # Test various caption types
    test_captions = [
        ("A person sits quietly in a dimly lit room, looking thoughtful and contemplative", "Neutral/contemplative"),
        ("The individual appears energetic and excited, with bright vibrant lighting and dynamic movement", "Positive/energetic"),
        ("Someone looks tired and withdrawn, slouching sadly in their chair with heavy dull eyes", "Negative/withdrawn"),
        ("A curious person leans forward, examining something with intense focus and interest", "Alert/curious"),
        ("The room is empty and still, with soft shadows creating a peaceful atmosphere", "Calm/detached"),
    ]

    for caption, expected in test_captions:
        print(f"\nCaption: {caption[:80]}...")
        print(f"Expected: {expected}")

        # Process through mood analysis (this was missing!)
        final_mood = mood_engine.analyze_mood(caption, saw_person="person" in caption.lower())

        # Get emotion for hand controller
        emotion_state = mood_engine.get_emotion_for_hand_controller()

        # Show the results including time adjustments
        mood_vector = mood_engine.mood_vector
        novelty = getattr(mood_engine, '_last_novelty', 0.0)

        # Calculate adjusted values like the emotion mapping does
        import time
        import numpy as np
        time_factor = (time.time() - mood_engine.session_start) / 1800.0
        valence_variation = 0.05 * np.sin(time_factor)
        arousal_variation = 0.08 * np.cos(time_factor * 1.3)
        clarity_variation = 0.03 * np.sin(time_factor * 0.7)

        adjusted_valence = mood_vector[0] + valence_variation
        adjusted_arousal = mood_vector[1] + arousal_variation
        adjusted_clarity = mood_vector[2] + clarity_variation

        print(f"Scalar Mood: {final_mood:.3f}")
        print(f"Novelty: {novelty:.3f}")
        print(f"3D Vector: valence={mood_vector[0]:.3f}, arousal={mood_vector[1]:.3f}, clarity={mood_vector[2]:.3f}")
        print(f"Adjusted:  valence={adjusted_valence:.3f}, arousal={adjusted_arousal:.3f}, clarity={adjusted_clarity:.3f}")
        print(f"→ Emotion State: {emotion_state}")
        print("-" * 40)

if __name__ == "__main__":
    test_caption_to_emotion_flow()