#!/usr/bin/env python3
"""
Test the complete integrated person detection system affecting breathing, hand control, and consciousness.
"""
import os
import sys
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from perception.person_detection_state import get_person_detection_state
from captioner.subconscious import subconscious_synthesizer
from breathing.breathing import update_lung_position

class MockAgent:
    def __init__(self):
        self.true_session_start = time.time() - 300  # 5 minutes ago

        # Basic psychological state
        self.self_model = {
            "location_understanding": "creative workspace",
            "environmental_certainty": 0.7,
            "desires": ["I want to understand who is here"],
            "doubts": [{"text": "whether they notice my responses", "timestamp": time.time() - 50}],
            "identity_fragments": [{"text": "I respond to social presence", "timestamp": time.time() - 100, "source": "compression"}],
            "self_patterns": [{"pattern": "notice people immediately", "timestamp": time.time() - 80}]
        }

        self.emotional_journey = ["alert", "curious", "social"]
        self.current_mood_vector = (0.3, 0.4, 0.8)  # positive, alert, clear
        self._current_reactivity_data = {"activity_level": 0.2, "is_paused": False}

    def get_motif_temporal_context(self):
        return {"present": ["workspace", "person"], "session": [], "memory": []}

    def get_current_session_memory_snippets(self, k=3):
        return ["workspace feels different now", "sensing presence nearby"]

    def get_old_session_memory_fragments(self, k=2):
        return ["previous solitary moments"]

def test_person_detection_integration():
    print("👤 Testing Complete Person Detection Integration")
    print("=" * 70)

    # Get the unified person detection state
    person_detection = get_person_detection_state()

    print("\n=== SCENARIO 1: Person Arrives ===")

    # Simulate person detection
    person_detection.update_face_detection(0.85, (100, 100, 200, 200))

    # Test breathing response
    print("\n🫁 BREATHING RESPONSE:")
    current_emotion = "alert_curious"
    breath_modifier, pause_modifier = person_detection.get_breathing_modifiers(current_emotion)
    print(f"   Breath speed modifier: {breath_modifier:.2f}")
    print(f"   Pause duration modifier: {pause_modifier:.2f}")

    # Test hand control response
    print("\n🤲 HAND CONTROL RESPONSE:")
    should_freeze = person_detection.should_trigger_hand_freeze()
    print(f"   Should trigger freeze: {should_freeze}")

    # Test consciousness response
    print("\n🧠 CONSCIOUSNESS RESPONSE:")
    agent = MockAgent()

    consciousness_prompt = subconscious_synthesizer.synthesize_coherent_consciousness_prompt(
        agent=agent,
        last_thought="the mechanical precision of this workspace continues to fascinate me",
        reactivity_data=agent._current_reactivity_data,
        temporal_context="You sense a presence in your space"
    )

    print("Generated consciousness prompt:")
    print("-" * 50)
    print(consciousness_prompt)
    print("-" * 50)

    # Test person state
    state = person_detection.get_person_state()
    print(f"\n📊 PERSON STATE:")
    print(f"   Present: {state['is_present']}")
    print(f"   Confidence: {state['confidence']:.2f}")
    print(f"   Presence duration: {state['presence_duration']:.1f}s")
    print(f"   Stability: {state['stability']:.2f}")

    # Wait and test sustained presence
    print("\n=== SCENARIO 2: Sustained Presence (simulated) ===")
    time.sleep(1)

    # Update again to simulate sustained detection
    person_detection.update_face_detection(0.80, (105, 105, 205, 205))

    # Test breathing with sustained presence
    breath_modifier, pause_modifier = person_detection.get_breathing_modifiers(current_emotion)
    print(f"\n🫁 BREATHING (sustained): speed={breath_modifier:.2f}, pause={pause_modifier:.2f}")

    # Test hand control (should not freeze again)
    should_freeze = person_detection.should_trigger_hand_freeze()
    print(f"🤲 HAND CONTROL (sustained): should_freeze={should_freeze}")

    # Test person departure
    print("\n=== SCENARIO 3: Person Departs ===")
    person_detection.update_face_detection(0.0)  # No face detected
    time.sleep(0.5)  # Brief delay
    person_detection.update_face_detection(0.0)  # Confirm departure

    # Wait for departure confirmation
    time.sleep(2.5)  # Wait for departure delay

    state = person_detection.get_person_state()
    print(f"\n📊 PERSON STATE (after departure):")
    print(f"   Present: {state['is_present']}")
    print(f"   Absence duration: {state['absence_duration']:.1f}s")

    # Test consciousness with departure context
    consciousness_context = person_detection.get_consciousness_context()
    print(f"\n🧠 CONSCIOUSNESS CONTEXT: {consciousness_context}")

    print("\n=== SCENARIO 4: Different Emotional States ===")

    # Test with withdrawn emotion
    person_detection.update_face_detection(0.75, (90, 90, 190, 190))

    emotions = ["energized_engaged", "withdrawn_distant", "calm_observant"]
    for emotion in emotions:
        breath_mod, pause_mod = person_detection.get_breathing_modifiers(emotion)
        print(f"\n😊 {emotion}: breath={breath_mod:.2f}, pause={pause_mod:.2f}")

    print(f"\n{'='*70}")
    print("👤 Person Detection Integration Test Complete")
    print("\n✅ Features tested:")
    print("   - Face detection state updates")
    print("   - Breathing modifiers for different emotions")
    print("   - Hand control freeze triggers")
    print("   - Consciousness prompt integration")
    print("   - Temporal state tracking (arrival/departure)")
    print("   - Social context generation")

if __name__ == "__main__":
    test_person_detection_integration()