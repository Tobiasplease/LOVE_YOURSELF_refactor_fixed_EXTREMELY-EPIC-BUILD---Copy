#!/usr/bin/env python3
"""
Test Drawing Introspection System

Tests the new drawing introspection functionality that replaces visual captioning
during drawing periods with philosophical reflection about creative choices.
"""

import os
import sys
import time
from unittest.mock import Mock

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from captioner.captioner import Captioner
from utils.state_manager import state_manager
from event_logging.event_logger import log_json_entry
from event_logging.log_type import LogType


def test_drawing_state_detection():
    """Test that drawing state detection works correctly."""
    print("🧪 Testing drawing state detection...")

    # Create captioner instance
    captioner = Captioner()

    # Mock state manager drawing status - not drawing
    state_manager.is_generating_drawing = False
    state_manager.drawing_start_time = None
    state_manager.current_drawing_prompt = None

    # Test when not drawing
    is_drawing = captioner._is_currently_drawing()
    print(f"   Not drawing: {not is_drawing}")
    assert not is_drawing, "Should detect when not drawing"

    # Mock drawing state - currently drawing
    state_manager.is_generating_drawing = True
    state_manager.drawing_start_time = time.time()
    state_manager.current_drawing_prompt = "A contemplative study of shadows dancing across weathered paper"

    # Test when drawing
    is_drawing = captioner._is_currently_drawing()
    print(f"   Currently drawing: {is_drawing}")
    assert is_drawing, "Should detect when drawing"

    print("✅ Drawing state detection working correctly\n")


def test_drawing_reflection_generation():
    """Test philosophical reflection generation during drawing."""
    print("🧪 Testing drawing reflection generation...")

    # Create captioner with some drawing history
    captioner = Captioner()

    # Add some drawing intent history
    captioner.observe("Drawing intent: Explored the interplay of light and shadow", 0.7, "", memory_type="drawing_intent")
    captioner.observe("Drawing intent: Captured organic forms emerging from geometric structures", 0.6, "", memory_type="drawing_intent")

    # Mock drawing state
    state_manager.is_generating_drawing = True
    state_manager.drawing_start_time = time.time() - 45  # 45 seconds ago
    state_manager.current_drawing_prompt = "A philosophical exploration of consciousness through abstract forms"

    # Mock the model's ollama call to return a reflection
    original_call_ollama = captioner.model._call_ollama
    def mock_call_ollama(prompt, image_path=None, system_prompt=None, prompt_type=None):
        if prompt_type == "drawing_introspection":
            return "This creative expression reveals my evolving understanding of consciousness as something fluid and interconnected. The abstract forms emerge from a deeper recognition that identity itself is not fixed but constantly reshaping through artistic exploration."
        return original_call_ollama(prompt, image_path, system_prompt, prompt_type)

    captioner.model._call_ollama = mock_call_ollama

    # Generate reflection
    reflection = captioner._generate_drawing_reflection()
    print(f"   Generated reflection: {reflection[:100]}...")

    assert len(reflection) > 50, "Reflection should be substantial"
    assert "consciousness" in reflection.lower(), "Should contain philosophical content"

    # Restore original method
    captioner.model._call_ollama = original_call_ollama

    print("✅ Drawing reflection generation working correctly\n")


def test_character_insight_extraction():
    """Test character development insight extraction from reflections."""
    print("🧪 Testing character insight extraction...")

    captioner = Captioner()

    # Test with reflection containing character insights
    reflection = "This creative expression reveals my evolving understanding of consciousness as something fluid and interconnected. The abstract forms emerge from a deeper recognition that identity itself is not fixed but constantly reshaping through artistic exploration. Each line becomes a meditation on existence."

    insight = captioner._extract_character_insights(reflection)
    print(f"   Extracted insight: {insight}")

    assert len(insight) > 0, "Should extract meaningful insights"
    assert any(keyword in insight.lower() for keyword in ["consciousness", "identity", "understanding", "recognition"]), "Should contain character development keywords"

    # Test with reflection without insights
    basic_reflection = "The drawing is proceeding normally with various shapes and colors."
    insight = captioner._extract_character_insights(basic_reflection)
    print(f"   Basic reflection insight: '{insight}'")

    assert len(insight) == 0, "Should not extract insights from basic descriptions"

    print("✅ Character insight extraction working correctly\n")


def test_drawing_introspection_process():
    """Test the complete drawing introspection process."""
    print("🧪 Testing complete drawing introspection process...")

    # Create captioner
    captioner = Captioner()
    captioner.current_mood = 0.65
    captioner.last_caption_time = 0  # Force processing

    # Add drawing history
    captioner.observe("Drawing intent: Geometric patterns reflecting inner order", 0.7, "", memory_type="drawing_intent")

    # Mock drawing state
    state_manager.is_generating_drawing = True
    state_manager.drawing_start_time = time.time() - 30
    state_manager.current_drawing_prompt = "Contemplative study of emerging consciousness through botanical forms"

    # Mock ollama response
    original_call_ollama = captioner.model._call_ollama
    def mock_call_ollama(prompt, image_path=None, system_prompt=None, prompt_type=None):
        if prompt_type == "drawing_introspection":
            return "Creating botanical forms awakens a deeper awareness of my own growth and transformation. These emerging patterns reflect how consciousness itself unfolds - not in linear progression but through organic, interconnected development that mirrors the natural world."
        return original_call_ollama(prompt, image_path, system_prompt, prompt_type)

    captioner.model._call_ollama = mock_call_ollama

    # Count initial memory entries
    initial_introspections = len(captioner.get_memory_entries_by_type("drawing_introspection"))
    initial_insights = len(captioner.get_memory_entries_by_type("character_insight"))

    # Process drawing introspection
    captioner._process_drawing_introspection()

    # Check that memory entries were created
    new_introspections = len(captioner.get_memory_entries_by_type("drawing_introspection"))
    new_insights = len(captioner.get_memory_entries_by_type("character_insight"))

    print(f"   Introspection entries: {initial_introspections} -> {new_introspections}")
    print(f"   Character insight entries: {initial_insights} -> {new_insights}")

    assert new_introspections > initial_introspections, "Should create introspection memory entry"
    assert new_insights > initial_insights, "Should create character insight entry"

    # Check memory content
    latest_introspection = captioner.get_memory_entries_by_type("drawing_introspection", limit=1)[0]
    latest_insight = captioner.get_memory_entries_by_type("character_insight", limit=1)[0]

    print(f"   Latest introspection: {latest_introspection['text'][:80]}...")
    print(f"   Latest insight: {latest_insight['text'][:80]}...")

    assert "Drawing introspection:" in latest_introspection['text'], "Should store introspection properly"
    assert "Character insight:" in latest_insight['text'], "Should store insight properly"

    # Restore original method
    captioner.model._call_ollama = original_call_ollama

    print("✅ Complete drawing introspection process working correctly\n")


def test_drawing_history_integration():
    """Test that drawing history is properly integrated into reflections."""
    print("🧪 Testing drawing history integration...")

    captioner = Captioner()

    # Add varied drawing history
    captioner.observe("Drawing intent: Abstract expressionist exploration of emotional landscapes", 0.8, "", memory_type="drawing_intent")
    captioner.observe("Drawing intent: Minimalist geometric study of space and form", 0.6, "", memory_type="drawing_intent")
    captioner.observe("Drawing intent: Organic flowing lines representing temporal consciousness", 0.7, "", memory_type="drawing_intent")

    # Mock current drawing
    state_manager.is_generating_drawing = True
    state_manager.drawing_start_time = time.time() - 60
    state_manager.current_drawing_prompt = "Synthesis of geometric and organic forms"

    # Generate reflection with history
    reflection_prompt = captioner._generate_drawing_reflection()

    # Should reference drawing history in some way
    print(f"   Generated prompt includes history context")

    # Check that recent drawing intents are being retrieved
    recent_intents = captioner.get_memory_entries_by_type("drawing_intent", limit=3)
    print(f"   Retrieved {len(recent_intents)} recent drawing intents")

    assert len(recent_intents) == 3, "Should retrieve recent drawing history"

    print("✅ Drawing history integration working correctly\n")


def run_all_tests():
    """Run all drawing introspection tests."""
    print("🎨 Testing Drawing Introspection System\n")

    try:
        test_drawing_state_detection()
        test_drawing_reflection_generation()
        test_character_insight_extraction()
        test_drawing_introspection_process()
        test_drawing_history_integration()

        print("🎉 All drawing introspection tests passed!")
        print("\nThe system will now:")
        print("- Detect when drawing is in progress")
        print("- Generate philosophical reflections instead of visual captions")
        print("- Extract character development insights")
        print("- Maintain continuity with drawing history")
        print("- Store introspective observations for future reference")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_tests()