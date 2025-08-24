#!/usr/bin/env python3
"""
Test the new immediate reaction system to ensure it generates
visceral, immediate responses instead of analytical prose.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from captioner.immediate_response import immediate_reaction_engine


def test_immediate_reactions():
    """Test immediate reaction generation"""
    print("Testing Immediate Reaction System")
    print("=" * 50)
    
    # Test different mood vectors and contexts
    test_scenarios = [
        {
            "name": "High Energy + New Object",
            "objects": {"person", "laptop"},
            "mood": (0.5, 0.8, 0.7),  # positive, high arousal, clear
            "context": {}
        },
        {
            "name": "Tired + Repetitive Objects",
            "objects": {"messy_room", "clutter"},
            "mood": (-0.3, -0.2, 0.5),  # negative, low arousal
            "context": {"high_repetition_motifs": ["messy_room", "clutter"]}
        },
        {
            "name": "Restless + Object Change",
            "objects": {"window", "light"},
            "mood": (0.1, 0.6, 0.4),  # neutral valence, high arousal
            "context": {}
        },
        {
            "name": "Calm + Familiar Space",
            "objects": {"chair", "table"},
            "mood": (0.4, -0.1, 0.8),  # positive, very low arousal, clear
            "context": {}
        }
    ]
    
    print("Testing Programmatic Reactions:")
    print("-" * 30)
    
    for scenario in test_scenarios:
        print(f"\n{scenario['name']}:")
        
        # Test programmatic reactions (multiple samples)
        for i in range(3):
            reaction = immediate_reaction_engine.generate_immediate_reaction(
                objects=set(scenario["objects"]),
                mood_vector=scenario["mood"],
                environmental_context=scenario["context"],
                use_ai_mode=False  # Pure programmatic
            )
            print(f"  Sample {i+1}: '{reaction}'")
    
    print(f"\n" + "=" * 50)
    print("REACTION ANALYSIS:")
    print("-" * 50)
    
    # Analyze if reactions look immediate vs analytical
    sample_reaction = immediate_reaction_engine.generate_immediate_reaction(
        objects={"person"},
        mood_vector=(-0.4, 0.3, 0.6),  # negative, medium arousal
        environmental_context={"high_repetition_motifs": ["person"]},
        use_ai_mode=False
    )
    
    print(f"Sample reaction: '{sample_reaction}'")
    print()
    
    # Check characteristics
    is_short = len(sample_reaction) < 50
    has_fragments = "..." in sample_reaction or not sample_reaction.endswith(".")
    is_immediate = not any(word in sample_reaction.lower() for word in ["i feel", "i think", "i notice", "it seems"])
    
    print("Reaction Characteristics:")
    print(f"  Short and immediate: {'[YES]' if is_short else '[NO]'} ({len(sample_reaction)} chars)")
    print(f"  Uses fragments: {'[YES]' if has_fragments else '[NO]'}")
    print(f"  Avoids analytical language: {'[YES]' if is_immediate else '[NO]'}")
    
    # Test trigger detection
    print(f"\nTrigger Detection Test:")
    print("-" * 30)
    
    # Set up some baseline objects
    immediate_reaction_engine.last_objects = {"chair", "table"}
    
    # Test various trigger conditions
    trigger_tests = [
        {"objects": {"chair", "table", "person"}, "novelty": 0.9, "desc": "High novelty + new object"},
        {"objects": {"chair"}, "novelty": 0.2, "desc": "Object disappeared"},
        {"objects": {"chair", "table"}, "novelty": 0.1, "time_gap": 200, "desc": "Long silence"},
        {"objects": {"chair", "table"}, "novelty": 0.3, "desc": "Normal situation"}
    ]
    
    for test in trigger_tests:
        should_trigger = immediate_reaction_engine.should_use_immediate_mode(
            current_objects=set(test["objects"]),
            novelty_score=test["novelty"],
            time_since_last=test.get("time_gap", 30)
        )
        print(f"  {test['desc']}: {'IMMEDIATE' if should_trigger else 'analytical'}")
    
    print(f"\n" + "=" * 50)
    print("EXPECTED IMPROVEMENTS:")
    print("-" * 50)
    print("[+] 15% of responses will be immediate, visceral reactions")
    print("[+] High novelty/change triggers immediate mode")
    print("[+] Reactions like 'tired of messy places' when appropriate")
    print("[+] Fragments and incomplete thoughts allowed")
    print("[+] Memory accumulation continues in background")
    print("=" * 50)


if __name__ == "__main__":
    test_immediate_reactions()