#!/usr/bin/env python3
"""
Test script for the new context-rich multi-step drawing analysis system.
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_multi_step_drawing():
    """Test the multi-step drawing analysis system."""
    print("🎨 Testing Context-Rich Multi-Step Drawing Analysis")
    print("=" * 60)

    from captioner.captioner import Captioner
    from config.config import USE_MULTI_STEP_DRAWING_ANALYSIS

    print(f"Multi-step analysis enabled: {USE_MULTI_STEP_DRAWING_ANALYSIS}")

    # Create captioner with some accumulated state
    captioner = Captioner()

    # Simulate some accumulated context
    captioner.last_caption = "I see scattered papers and warm desk lighting creating interesting shadow patterns"
    captioner.current_mood = 0.7

    # Add some beliefs and patterns (simulated)
    if hasattr(captioner, "beliefs"):
        captioner.beliefs = {"lighting_patterns": 0.8, "workspace_activity": 0.6, "shadow_relationships": 0.9}

    # Add some drawing history directly to memory (avoid observe() complexity for test)
    if hasattr(captioner, "memory_ref") and captioner.memory_ref:
        try:
            # Add drawing intents directly to memory
            for i, intent in enumerate(
                [
                    "Captured the geometric interplay of monitor edges against organic paper scatter",
                    "Focused on how tungsten light creates warm pools against blue screen glow",
                    "Explored the rhythm of scattered objects creating visual texture",
                ]
            ):
                entry = {
                    "text": f"Drawing intent: {intent}",
                    "mood": 0.6 + (i * 0.1),
                    "timestamp": __import__("time").time() - (i * 300),  # Spread over time
                    "memory_type": "drawing_intent",
                }
                if hasattr(captioner.memory_ref, "memory_entries"):
                    captioner.memory_ref.memory_entries.append(entry)
        except Exception as e:
            print(f"Note: Could not add test drawing history: {e}")

    # Test with a real image if available
    test_image = "./.venv/lib/python3.12/site-packages/ultralytics/assets/bus.jpg"

    if os.path.exists(test_image):
        print(f"✓ Test image found: {test_image}")
        image_path = test_image
    else:
        print("⚠ Test image not found, proceeding without image")
        image_path = None

    print(f"\n=== Testing Multi-Step Drawing Analysis ===")
    print("This will make 5 separate LLM calls to build up the drawing decision...")

    try:
        # Test the model's drawing prompt generation (uses multi-step if enabled)
        start_time = __import__("time").time()

        # First let's check what kind of data the captioner (memory_ref) actually has
        print(f"\n=== DEBUGGING MEMORY CONTEXT ===")
        print(f"Captioner type: {type(captioner)}")
        print(f"Has beliefs: {hasattr(captioner, 'beliefs')}")
        print(f"Has self_model: {hasattr(captioner, 'self_model')}")
        print(f"Has temporal_prompt_lines: {hasattr(captioner, 'temporal_prompt_lines')}")
        print(f"Has get_memory_entries_by_type: {hasattr(captioner, 'get_memory_entries_by_type')}")
        print(f"Has describe_current_mood: {hasattr(captioner, 'describe_current_mood')}")

        # Show some actual attribute values
        if hasattr(captioner, "current_mood"):
            print(f"Current mood: {captioner.current_mood}")
        if hasattr(captioner, "current_emotion_state"):
            print(f"Current emotion state: {captioner.current_emotion_state}")

        drawing_prompt = captioner.model.generate_drawing_prompt(
            extra="Testing multi-step analysis with accumulated consciousness context", image_path=image_path
        )

        elapsed = __import__("time").time() - start_time

        print(f"✅ Multi-step analysis completed in {elapsed:.1f} seconds")
        print(f"Final prompt length: {len(drawing_prompt)} characters")
        print(f"\n--- FINAL DRAWING PROMPT ---")
        print(drawing_prompt)
        print("--- END PROMPT ---")

        # Check if the prompt shows evidence of deep context integration
        context_indicators = ["accumulated", "identity", "consciousness", "beliefs", "drawing history", "emotional", "communication", "technique"]

        found_indicators = [word for word in context_indicators if word.lower() in drawing_prompt.lower()]

        print(f"\n📊 Context Integration Analysis:")
        print(f"Found context indicators: {', '.join(found_indicators)}")
        print(f"Context richness score: {len(found_indicators)}/{len(context_indicators)}")

        if len(found_indicators) >= 4:
            print("✅ High context integration detected")
        elif len(found_indicators) >= 2:
            print("⚠ Moderate context integration")
        else:
            print("❌ Low context integration - may need debugging")

    except Exception as e:
        print(f"❌ Error during multi-step analysis: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_multi_step_drawing()
    print("\n✅ Multi-step drawing analysis test completed!")
