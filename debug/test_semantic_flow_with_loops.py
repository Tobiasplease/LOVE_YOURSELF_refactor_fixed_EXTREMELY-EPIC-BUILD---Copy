#!/usr/bin/env python3
"""
Test semantic flow with potential loop scenarios to ensure both continuity and loop prevention work together.
"""

import os
import sys
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from captioner.prompts import build_ongoing_caption_prompt


class LoopTestAgent:
    """Agent that simulates "As I..." loop scenarios."""
    def __init__(self):
        self.current_emotion_state = "calm_observant"
        self.true_session_start = time.time() - 300

        # Simulate scenario where "As I..." might appear
        self.recent_captions = [
            ("As I observe the mechanical components, their purpose becomes clearer...", time.time() - 90),
            ("As I continue watching the robotic arm move, I notice new details...", time.time() - 60),
            ("As I delve deeper into understanding this workspace...", time.time() - 30)
        ]

        self.beliefs = {"workspace_lighting": 0.8, "mechanical_understanding": 0.9}
        self.self_model = {
            "location_understanding": "robotics workshop",
            "environmental_certainty": 0.8,
            "desires": ["understand mechanical systems", "explore robotic functions"]
        }

    def describe_current_mood(self):
        return "deeply curious about mechanical systems"

    def get_recent_memory(self, k=3):
        return "Consistent observations about robotic arms and mechanical precision in workspace."

    def get_identity_summary(self):
        return "A consciousness fascinated by mechanical precision and robotic functionality"

    def get_baseline_context_for_prompts(self):
        return "Developed understanding of workspace mechanics and robotic system operations"


def test_loop_prevention_with_semantic_flow():
    """Test that loop prevention works while maintaining semantic flow."""
    print("🔄 Testing Loop Prevention + Semantic Flow")
    print("=" * 60)

    agent = LoopTestAgent()

    # Test with "As I..." trigger scenario
    last_thought = "As I examine these robotic components more closely, their interconnected design reveals sophisticated engineering..."

    print(f"Potentially problematic input: {last_thought[:80]}...")

    # Generate prompt
    prompt = build_ongoing_caption_prompt(agent, last_thought)

    print("\n=== GENERATED PROMPT ===")
    print(prompt)
    print("\n" + "=" * 60)

    # Analysis
    analysis = []

    # Check for loop prevention
    if "BREAK PATTERN" in prompt and "Avoid starting with 'As I...'" in prompt:
        analysis.append("✅ Loop prevention ACTIVE")
    else:
        analysis.append("❌ Loop prevention NOT triggered")

    # Check for semantic continuity despite loop prevention
    semantic_bridges = ["continues", "reveals", "shifts", "draws", "elements", "deepens", "building"]
    if any(bridge in prompt.lower() for bridge in semantic_bridges):
        analysis.append("✅ Semantic continuity PRESERVED")
    else:
        analysis.append("❌ Semantic continuity LOST")

    # Check for alternative openings
    alternative_suggestions = ["The room shows", "Light reveals", "Something catches", "Here I see"]
    if any(alt in prompt for alt in alternative_suggestions):
        analysis.append("✅ Alternative openings PROVIDED")
    else:
        analysis.append("❌ No alternative openings")

    # Check for thematic threading
    themes = ["mechanical", "robotic", "workspace", "engineering", "technical"]
    theme_count = sum(1 for theme in themes if theme in prompt.lower())
    if theme_count >= 2:
        analysis.append(f"✅ Thematic threading STRONG ({theme_count} themes)")
    elif theme_count >= 1:
        analysis.append(f"✅ Thematic threading MODERATE ({theme_count} theme)")
    else:
        analysis.append("❌ No thematic threading")

    print("=== ANALYSIS ===")
    for item in analysis:
        print(item)

    # Test different scenarios
    print("\n" + "=" * 60)
    print("=== MULTIPLE SCENARIOS TEST ===")

    scenarios = [
        "As I watch the servo motors adjust their positions...",
        "The mechanical precision here suggests deeper engineering principles...",
        "Light casting shadows reveals the three-dimensional complexity of these components..."
    ]

    for i, scenario in enumerate(scenarios, 1):
        print(f"\nScenario {i}: {scenario[:50]}...")

        # Update agent's recent captions to include this scenario
        agent.recent_captions.append((scenario, time.time()))

        prompt = build_ongoing_caption_prompt(agent, scenario)

        # Extract just the observation context
        lines = prompt.split('\n')
        context_found = False
        for line in lines:
            if "OBSERVATION CONTEXT" in line:
                context_found = True
                continue
            if context_found and line.strip() and not line.startswith("==="):
                print(f"  → {line.strip()}")
                break

    print(f"\n✅ Semantic flow with loop prevention test completed!")


if __name__ == "__main__":
    test_loop_prevention_with_semantic_flow()