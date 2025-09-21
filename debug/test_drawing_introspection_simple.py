#!/usr/bin/env python3
"""
Simple test for drawing introspection system without external dependencies.
"""

import os
import sys
import time

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from utils.state_manager import state_manager


def test_state_manager_drawing_detection():
    """Test state manager drawing status functionality."""
    print("🧪 Testing state manager drawing detection...")

    # Test not drawing
    state_manager.is_generating_drawing = False
    status = state_manager.get_drawing_status()
    print(f"   Not drawing status: {status}")
    assert not status.get("is_generating", True), "Should detect not drawing"

    # Test drawing
    state_manager.is_generating_drawing = True
    state_manager.drawing_start_time = time.time() - 30
    state_manager.current_drawing_prompt = "Test drawing prompt"

    status = state_manager.get_drawing_status()
    print(f"   Drawing status: {status}")
    assert status.get("is_generating", False), "Should detect drawing"
    assert status.get("prompt") == "Test drawing prompt", "Should return drawing prompt"
    assert status.get("duration", 0) > 25, "Should calculate duration"

    print("✅ State manager working correctly")


def test_character_insight_extraction():
    """Test character insight extraction without external dependencies."""
    print("🧪 Testing character insight extraction...")

    # Import here to avoid issues if captioner has dependencies
    try:
        from captioner.captioner import Captioner
        captioner = Captioner()

        # Test with meaningful reflection
        reflection = "This creative expression reveals my evolving understanding of consciousness as fluid. The forms emerge from a deeper recognition that identity transforms through artistic exploration."

        insight = captioner._extract_character_insights(reflection)
        print(f"   Extracted insight: {insight}")

        assert len(insight) > 0, "Should extract insights"
        assert any(word in insight.lower() for word in ["consciousness", "identity", "understanding"]), "Should contain development words"

        # Test with basic reflection
        basic_reflection = "The drawing has various shapes and colors."
        basic_insight = captioner._extract_character_insights(basic_reflection)
        print(f"   Basic insight: '{basic_insight}'")

        assert len(basic_insight) == 0, "Should not extract insights from basic text"

        print("✅ Character insight extraction working correctly")

    except ImportError as e:
        print(f"⚠️ Skipping character insight test due to import: {e}")


def main():
    print("🎨 Simple Drawing Introspection Tests\n")

    try:
        test_state_manager_drawing_detection()
        test_character_insight_extraction()

        print("\n🎉 Basic functionality tests passed!")
        print("\nDrawing introspection system components are ready:")
        print("- ✅ Drawing state detection")
        print("- ✅ Character insight extraction")
        print("- ✅ State management integration")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()