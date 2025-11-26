#!/usr/bin/env python3
"""
Test the mood engine mapping to hand controller emotions
"""
import os
import sys
import time

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("=== MOOD TO HAND CONTROLLER MAPPING TEST ===")


def test_mood_mapping():
    """Test how different mood vectors map to hand controller emotions"""
    try:
        from mood.mood import MoodEngine

        print("🧠 Creating mood engine...")
        mood_engine = MoodEngine()

        # Test different mood vectors to see what emotions they produce
        test_moods = [
            # (valence, arousal, clarity), description
            (0.8, 0.2, 0.8, "Happy and calm"),
            (0.8, 0.8, 0.8, "Happy and excited"),
            (-0.5, 0.7, 0.5, "Anxious and alert"),
            (-0.5, 0.2, 0.3, "Sad and withdrawn"),
            (0.0, 0.3, 0.1, "Confused and uncertain"),
            (0.3, 0.5, 0.7, "Neutral alert"),
            (-0.1, 0.2, 0.5, "Slightly negative"),
        ]

        print("\n📊 Testing mood vector → emotion mapping:")
        print("Valence  Arousal  Clarity  → Hand Emotion      Description")
        print("-" * 65)

        for valence, arousal, clarity, description in test_moods:
            # Set the mood vector
            mood_engine.mood_vector = np.array([valence, arousal, clarity])

            # Get the emotion for hand controller
            emotion = mood_engine.get_emotion_for_hand_controller()

            print(f"{valence:>7.1f}  {arousal:>7.1f}  {clarity:>7.1f}  → {emotion:<18} {description}")

        return True

    except Exception as e:
        print(f"❌ Error in mood mapping test: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_full_integration_simulation():
    """Simulate how machine.py would call hand controller based on mood"""
    try:
        from hand_control.direct_hand_control import change_to_emotion, start_hand_controller, stop_hand_controller
        from mood.mood import MoodEngine

        print("\n🎭 Testing full machine.py → hand controller flow...")

        # Start hand controller (like machine.py does)
        print("📋 Starting hand controller...")
        success = start_hand_controller(headless=True)  # Use headless for faster testing

        if not success:
            print("❌ Failed to start hand controller")
            return False

        print("✅ Hand controller started")
        time.sleep(2)

        # Create mood engine (like machine.py does)
        mood_engine = MoodEngine()

        # Simulate different mood changes over time
        mood_scenarios = [
            (0.6, 0.3, 0.8, "Content and observant"),
            (0.8, 0.8, 0.7, "Energized and engaged"),
            (-0.4, 0.6, 0.5, "Alert but anxious"),
            (0.1, 0.1, 0.2, "Quiet and detached"),
            (-0.6, 0.2, 0.3, "Withdrawn and distant"),
        ]

        for valence, arousal, clarity, description in mood_scenarios:
            print(f"\n🧠 Simulating mood change: {description}")
            print(f"   Mood vector: ({valence:.1f}, {arousal:.1f}, {clarity:.1f})")

            # Set mood vector (like machine.py does)
            mood_engine.mood_vector = np.array([valence, arousal, clarity])

            # Get emotion for hand controller (like machine.py does)
            emotion = mood_engine.get_emotion_for_hand_controller()
            print(f"   → Hand emotion: {emotion}")

            # Send to hand controller (like machine.py does)
            success = change_to_emotion(emotion)
            if success:
                print(f"   ✅ Hand controller switched to {emotion}")
            else:
                print(f"   ❌ Failed to switch to {emotion}")

            # Wait between mood changes (like machine.py does)
            time.sleep(3)

        print("\n🔄 Stopping hand controller...")
        stop_hand_controller()
        print("✅ Full integration test complete")
        return True

    except Exception as e:
        print(f"❌ Error in integration test: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Test the mood mapping first
    test_mood_mapping()

    # Then test the full integration
    test_full_integration_simulation()
