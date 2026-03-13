#!/usr/bin/env python3
"""
Test prompt generation with REAL saved state to see if data is being used.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from captioner.captioner import Captioner
from captioner.prompts import build_ongoing_caption_prompt
from utils.state_manager import state_manager

def main():
    print("=" * 80)
    print("LOADING REAL STATE FROM FILE")
    print("=" * 80)

    # Load actual state
    previous_state = state_manager.load_session_state()
    if not previous_state:
        print("\n✗ No state file found!")
        return

    # Create captioner and apply state (like machine.py does)
    captioner = Captioner()
    state_manager.apply_state_to_captioner(previous_state, captioner)

    print("\n" + "=" * 80)
    print("CAPTIONER STATE AFTER LOADING")
    print("=" * 80)

    print(f"\n✓ Beliefs: {len(captioner.beliefs)} total")
    if captioner.beliefs:
        print(f"  Top 5 beliefs: {list(captioner.beliefs.keys())[:5]}")

    print(f"\n✓ Motifs: {len(captioner.motif_counter)} total")
    if captioner.motif_counter:
        top_motifs = captioner.motif_counter.most_common(10)
        print(f"  Top 10 motifs:")
        for motif, count in top_motifs:
            print(f"    - {motif}: {count}")

    print(f"\n✓ Self-model:")
    print(f"  Location: {captioner.self_model.get('location_understanding', 'unknown')}")
    print(f"  Desires: {captioner.self_model.get('desires', [])[-3:] if captioner.self_model.get('desires') else '(none)'}")

    print(f"\n✓ Temporal data:")
    if hasattr(captioner, 'temporal_prompt_lines'):
        for line in captioner.temporal_prompt_lines():
            print(f"  - {line}")

    print(f"\n✓ Baseline context:")
    if hasattr(captioner, 'get_baseline_context_for_prompts'):
        baseline = captioner.get_baseline_context_for_prompts()
        if baseline:
            print(f"  {baseline[:300]}...")
        else:
            print("  (empty - no reflections yet?)")

    print("\n" + "=" * 80)
    print("GENERATING PROMPT WITH LOADED STATE")
    print("=" * 80)

    # Use a generic last caption (like the repetitive ones in the terminal output)
    last_caption = "The room is quiet and still, almost like an empty canvas waiting to be filled..."

    try:
        prompt = build_ongoing_caption_prompt(captioner, last_caption=last_caption, person_present=False)

        print("\n✓ Prompt generated")
        print("\nFULL PROMPT:")
        print("-" * 80)
        print(prompt)
        print("-" * 80)

        # Analyze what made it into the prompt
        print("\nPROMPT CONTENT ANALYSIS:")
        print(f"  Length: {len(prompt)} chars")

        # Check for specific beliefs (convert to lowercase for matching)
        prompt_lower = prompt.lower()
        beliefs_found = []
        if captioner.beliefs:
            for belief in list(captioner.beliefs.keys())[:10]:
                belief_clean = belief.replace("_", " ").replace("-", " ").lower()
                if belief_clean in prompt_lower:
                    beliefs_found.append(belief_clean)

        if beliefs_found:
            print(f"  ✓ Found {len(beliefs_found)} beliefs in prompt: {beliefs_found}")
        else:
            print(f"  ✗ NO beliefs found in prompt (captioner has {len(captioner.beliefs)} beliefs)")

        # Check for motifs
        motifs_found = []
        if captioner.motif_counter:
            for motif, count in captioner.motif_counter.most_common(10):
                if motif.lower() in prompt_lower:
                    motifs_found.append(f"{motif}({count})")

        if motifs_found:
            print(f"  ✓ Found {len(motifs_found)} motifs in prompt: {motifs_found}")
        else:
            print(f"  ✗ NO motifs found in prompt (captioner has {len(captioner.motif_counter)} motifs)")

        # Check for desires
        if captioner.self_model.get('desires'):
            print(f"  Desires available: {len(captioner.self_model['desires'])}")
            if any(d.lower() in prompt_lower for d in captioner.self_model.get('desires', [])):
                print(f"  ✓ Desires mentioned in prompt")
            else:
                print(f"  ✗ Desires NOT in prompt")

        # Check for temporal info
        temporal_in_prompt = any(word in prompt_lower for word in ['awake', 'day', 'hour', 'minute', 'slept'])
        if temporal_in_prompt:
            print(f"  ✓ Temporal context present")
        else:
            print(f"  ✗ Temporal context missing")

        # Check for the word "believe" or "tend to"
        if "believe" in prompt_lower or "tend to" in prompt_lower:
            print(f"  ✓ 'believe' or 'tend to' found - beliefs structure present")
        else:
            print(f"  ✗ NO 'believe' or 'tend to' - beliefs may not be formatted in")

    except Exception as e:
        print(f"\n✗ Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
