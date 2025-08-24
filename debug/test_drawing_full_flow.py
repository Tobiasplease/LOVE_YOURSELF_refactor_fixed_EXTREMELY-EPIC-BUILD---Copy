#!/usr/bin/env python3
"""
Test the full drawing flow with the new print format.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from drawing.drawing import DrawingController
import time

class MockCaptioner:
    """Mock captioner for testing."""
    def __init__(self):
        self.current_mood = 0.4
        self.novelty_score = 0.7
        self.boredom = 0.6
        self.memory_ref = self
        
def test_full_drawing_flow():
    """Test the full drawing flow including prompt display."""
    print("Testing Full Drawing Flow")
    print("=" * 50)
    
    controller = DrawingController()
    mock_agent = MockCaptioner()
    
    # Test drawing trigger with a sample prompt
    sample_drawing_prompt = """I see the warm afternoon light casting shadows across the desk,
creating geometric patterns that remind me of architectural blueprints.
The contrast between light and shadow feels like a dialogue between
presence and absence, between what is seen and what is hidden.
I want to express this duality - the way shadows define light,
the way emptiness gives shape to form."""
    
    sample_reflection = "I'm feeling contemplative about the interplay of light and shadow"
    
    print("\nAttempting to trigger drawing...")
    
    # Mock the handle_drawing_flow call (without actual ComfyUI)
    try:
        # Get novelty score from memory_ref first, fallback to agent attribute
        novelty_score = getattr(mock_agent.memory_ref, "novelty_score", getattr(mock_agent, "novelty_score", 0.0))
        boredom_score = getattr(mock_agent, "boredom", 0.0)
        
        if controller.should_draw(
            mood=mock_agent.current_mood,
            novelty=novelty_score,
            boredom=boredom_score,
            reflection=sample_reflection,
        ):
            controller.register_drawing(sample_drawing_prompt)
            
            # Print the full drawing prompt when triggered
            print(f"\n{'='*60}")
            print("DRAWING PROMPT:")
            print(f"{'='*60}")
            print(sample_drawing_prompt)
            print(f"{'='*60}\n")
            
            print("(Drawing would be sent to ComfyUI at this point)")
        else:
            print("Drawing was not triggered")
            
    except Exception as e:
        print(f"Error in test: {e}")

if __name__ == "__main__":
    test_full_drawing_flow()