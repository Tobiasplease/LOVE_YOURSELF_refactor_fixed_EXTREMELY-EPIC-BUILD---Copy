#!/usr/bin/env python3
"""
Test what prompt is actually being generated with current system state.
This will help us see if the accumulated data (beliefs, motifs, desires) is making it into the prompts.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from captioner.captioner import Captioner
from captioner.prompts import build_ongoing_caption_prompt
from config import config
import json

def main():
    # Try to load existing state
    captioner = Captioner()

    print("=" * 80)
    print("CAPTIONER STATE CHECK")
    print("=" * 80)

    # Check if state was loaded
    if hasattr(captioner, 'beliefs') and captioner.beliefs:
        print(f"\n✓ Beliefs loaded: {len(captioner.beliefs)} beliefs")
        print(f"  Top 5 beliefs: {list(captioner.beliefs.keys())[:5]}")
    else:
        print("\n✗ No beliefs found")

    if hasattr(captioner, 'motif_counter') and captioner.motif_counter:
        print(f"\n✓ Motifs tracked: {len(captioner.motif_counter)} motifs")
        top_motifs = captioner.motif_counter.most_common(10)
        print(f"  Top 10 motifs: {top_motifs}")
    else:
        print("\n✗ No motifs found")

    if hasattr(captioner, 'self_model'):
        print(f"\n✓ Self-model exists:")
        print(f"  Location: {captioner.self_model.get('location_understanding', 'unknown')}")
        print(f"  Purpose: {captioner.self_model.get('purpose_understanding', 'unknown')}")
        print(f"  Desires: {captioner.self_model.get('desires', [])}")
        print(f"  Environmental certainty: {captioner.self_model.get('environmental_certainty', 0.0)}")
    else:
        print("\n✗ No self-model found")

    # Check temporal data
    if hasattr(captioner, 'temporal_prompt_lines'):
        temporal_lines = captioner.temporal_prompt_lines()
        print(f"\n✓ Temporal context:")
        for line in temporal_lines:
            print(f"  - {line}")
    else:
        print("\n✗ No temporal_prompt_lines method")

    # Check if baseline context method works
    if hasattr(captioner, 'get_baseline_context_for_prompts'):
        baseline = captioner.get_baseline_context_for_prompts()
        print(f"\n✓ Baseline context method exists")
        print(f"  Output: {baseline[:200] if baseline else '(empty)'}")
    else:
        print("\n✗ No get_baseline_context_for_prompts method")

    # Check emotional self-knowledge
    if hasattr(captioner, 'get_emotional_self_knowledge'):
        emotional = captioner.get_emotional_self_knowledge()
        print(f"\n✓ Emotional self-knowledge method exists")
        print(f"  Output: {emotional[:200] if emotional else '(empty)'}")
    else:
        print("\n✗ No get_emotional_self_knowledge method")

    # Check current self-understanding
    if hasattr(captioner, 'get_current_self_understanding'):
        self_understanding = captioner.get_current_self_understanding()
        print(f"\n✓ Self-understanding method exists")
        print(f"  Output: {self_understanding[:200] if self_understanding else '(empty)'}")
    else:
        print("\n✗ No get_current_self_understanding method")

    print("\n" + "=" * 80)
    print("ACTUAL PROMPT GENERATION TEST")
    print("=" * 80)

    # Generate a prompt with current state
    last_caption = "The room is quiet and still, almost like an empty canvas waiting to be filled with life's stories..."

    try:
        prompt = build_ongoing_caption_prompt(captioner, last_caption=last_caption, person_present=False)
        print("\n✓ Prompt generated successfully")
        print("\nFULL PROMPT:")
        print("-" * 80)
        print(prompt)
        print("-" * 80)

        # Analyze prompt
        print("\nPROMPT ANALYSIS:")
        print(f"  Length: {len(prompt)} characters")
        print(f"  Contains 'beliefs': {'beliefs' in prompt.lower()}")
        print(f"  Contains motif data: {'motif' in prompt.lower() or any(motif in prompt.lower() for motif, _ in captioner.motif_counter.most_common(5))}")
        print(f"  Contains 'desires': {'desires' in prompt.lower() or 'want' in prompt.lower()}")
        print(f"  Contains temporal info: {'awake' in prompt.lower() or 'day' in prompt.lower() or 'hour' in prompt.lower()}")

    except Exception as e:
        print(f"\n✗ Failed to generate prompt: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
