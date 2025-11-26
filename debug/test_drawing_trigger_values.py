#!/usr/bin/env python3

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from captioner.captioner import Captioner
from drawing.drawing import DrawingController
from mood.mood import MoodEngine


def test_drawing_trigger_values():
    """Test what novelty/boredom values the drawing system actually receives."""

    print("Testing Drawing Trigger Values")
    print("=" * 60)

    # Initialize systems
    mood_engine = MoodEngine()
    captioner = Captioner()
    drawing_controller = DrawingController()

    # Simulate some captions and check values
    test_captions = [
        "A person sits quietly in a dimly lit room, looking thoughtful",
        "The individual appears energetic and excited with bright lighting",
        "Someone looks tired and withdrawn, slouching in their chair",
        "A curious person leans forward, examining something with focus",
        "The room is empty and still, with soft shadows",
    ]

    for i, caption in enumerate(test_captions):
        print(f"\n--- Test {i+1}: {caption[:50]}... ---")

        # Process caption through mood system
        mood = mood_engine.analyze_mood(caption, saw_person="person" in caption.lower())

        # Get pattern data and update captioner (like machine.py does)
        pattern_data = mood_engine.get_pattern_data()
        captioner.set_novelty_score(pattern_data["novelty_score"])
        captioner.current_mood = mood

        # Check what drawing system receives
        novelty = getattr(captioner, "novelty_score", 0.0)
        boredom = getattr(captioner, "boredom", 0.0)

        print(f"Mood: {mood:.3f}")
        print(f"Novelty (from pattern_data): {pattern_data['novelty_score']:.3f}")
        print(f"Novelty (captioner.novelty_score): {novelty:.3f}")
        print(f"Boredom (captioner.boredom): {boredom:.3f}")

        # Test drawing decision
        ready = drawing_controller.ready_to_draw()
        should_draw = drawing_controller.should_draw(mood=mood, novelty=novelty, boredom=boredom)

        # Test logic without cooldown
        logic_only = (novelty > 0.4 or boredom > 0.5 or mood < 0.4) if ready else False
        would_draw_without_cooldown = novelty > 0.4 or boredom > 0.5 or mood < 0.4

        print(f"Ready to draw (cooldown check): {ready}")
        print(f"Drawing thresholds: novelty>0.4 OR boredom>0.5 OR mood<0.4")
        print(f"Should draw: {should_draw} (novelty:{novelty:.3f}, boredom:{boredom:.3f}, mood:{mood:.3f})")
        print(f"Would draw without cooldown: {would_draw_without_cooldown}")

        # Show why it failed/succeeded
        if novelty > 0.4:
            print("  → TRIGGERED by novelty > 0.4")
        elif boredom > 0.5:
            print("  → TRIGGERED by boredom > 0.5")
        elif mood < 0.4:
            print("  → TRIGGERED by mood < 0.4")
        else:
            print("  → NOT TRIGGERED - all thresholds missed")


if __name__ == "__main__":
    test_drawing_trigger_values()
