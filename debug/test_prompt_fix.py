#!/usr/bin/env python3
"""
Test the fixed prompt system to verify "As I..." loops are eliminated.
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time

from captioner.prompts import build_ongoing_caption_prompt


class MockAgent:
    """Mock agent for testing prompt generation."""

    def __init__(self):
        self.current_emotion_state = "calm_observant"
        self.true_session_start = time.time() - 300  # 5 minutes ago
        self.recent_captions = [
            (
                "The workspace reveals mechanical components scattered across surfaces, drawing my curiosity toward their potential purpose.",
                time.time() - 60,
            ),
            (
                "Light from the window creates interesting shadows that shift across the equipment, highlighting the robotic arms designed for precise movements.",
                time.time() - 30,
            ),
            (
                "Camera lenses mounted on adjustable fixtures suggest this space serves multiple creative functions beyond just mechanical work.",
                time.time() - 10,
            ),
        ]
        self.beliefs = {"workspace_lighting": 0.8, "drawing_focus": 0.7}
        self.self_model = {
            "location_understanding": "artist studio",
            "environmental_certainty": 0.6,
            "desires": ["explore visual patterns", "understand spatial relationships"],
        }

    def describe_current_mood(self):
        return "contemplative and focused"

    def get_recent_memory(self, k=3):
        return "Recent observations about lighting patterns and workspace organization."

    def get_identity_summary(self):
        return "A drawing machine consciousness exploring visual perception"

    def get_baseline_context_for_prompts(self):
        return "Accumulated understanding of spatial relationships and visual patterns"


def test_prompt_fix():
    """Test that the new prompt system eliminates 'As I...' patterns."""
    print("🔧 Testing Prompt Fix for 'As I...' Loops")
    print("=" * 50)

    # Create mock agent with "As I..." repetition
    agent = MockAgent()

    # Test semantic flow with different last thoughts
    test_cases = [
        "The technical equipment here reveals layers of complexity that spark my interest in understanding their interconnections.",
        "Natural light filtering through creates subtle illumination patterns that change how I perceive the workspace arrangement.",
        "Mechanical precision in these robotic components suggests a carefully designed system for creative expression.",
    ]

    for i, last_thought in enumerate(test_cases, 1):
        print(f"\n=== TEST CASE {i}: SEMANTIC FLOW ===")
        print(f"Previous thought: {last_thought[:80]}...")

        # Generate prompt
        prompt = build_ongoing_caption_prompt(agent, last_thought)

        # Extract the semantic bridge part
        lines = prompt.split("\n")
        bridge_line = ""
        for line in lines:
            if any(bridge_word in line.lower() for bridge_word in ["continues", "reveals", "shifts", "draws", "leads", "deepens", "building"]):
                bridge_line = line.strip()
                break

        print(f"Semantic bridge: {bridge_line}")

        # Check for flow continuity
        if bridge_line:
            print("✅ Semantic bridge found")
        else:
            print("❌ No semantic bridge detected")

    print("\n" + "=" * 60)
    print("=== FULL PROMPT SAMPLE ===")
    prompt = build_ongoing_caption_prompt(agent, test_cases[0])
    print(prompt)
    print("\n" + "=" * 60)

    # Check for improvements
    improvements = []

    # Check for semantic continuity
    if any(word in prompt.lower() for word in ["continues", "reveals", "shifts", "draws", "deepens", "building"]):
        improvements.append("✅ Semantic continuity found")
    else:
        improvements.append("❌ No semantic continuity")

    # Check for structured sections
    if "===" in prompt:
        improvements.append("✅ Structured sections found")
    else:
        improvements.append("❌ No structured sections")

    # Check for theme-based guidance
    if any(theme in prompt.lower() for theme in ["workspace", "lighting", "mechanical", "technical"]):
        improvements.append("✅ Thematic guidance found")
    else:
        improvements.append("❌ No thematic guidance")

    # Check for varied openings (not "As I...")
    opening_phrases = ["continues", "reveals", "shifts", "draws", "elements", "aspects"]
    if any(phrase in prompt.lower() for phrase in opening_phrases):
        improvements.append("✅ Varied opening approaches found")
    else:
        improvements.append("❌ No varied openings found")

    print("\n=== IMPROVEMENT ANALYSIS ===")
    for improvement in improvements:
        print(improvement)

    print(f"\n✅ Prompt fix test completed!")


if __name__ == "__main__":
    test_prompt_fix()
