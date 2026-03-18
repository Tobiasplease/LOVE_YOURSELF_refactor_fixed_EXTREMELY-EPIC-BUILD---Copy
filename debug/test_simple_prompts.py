#!/usr/bin/env python3
"""Test the activation-gated simple prompts system."""

import sys
import time
sys.path.insert(0, '/home/impostor/LOVE_YOURSELF_refactor_fixed_EXTREMELY-EPIC-BUILD---Copy')

from collections import Counter

from captioner.prompts import (
    build_simple_caption_prompt,
    _build_simple_system_context,
    build_focused_caption_prompt,
    determine_prompt_mode,
    USE_SIMPLE_PROMPTS,
)
from captioner.activation_memory import (
    get_activation_network,
    should_include_context,
    STATIC_CONCEPTS,
    DYNAMIC_CONCEPTS,
    SOCIAL_CONCEPTS,
)

print(f"USE_SIMPLE_PROMPTS = {USE_SIMPLE_PROMPTS}")
print("=" * 60)


class MockAgent:
    """Mock agent for testing."""
    def __init__(self):
        self.true_session_start = time.time() - 120  # 2 mins ago (not awakening)
        self.recent_captions = []
        self.observation_count = 0
        self.beliefs = {}
        self.last_session_gap = None
        self.prior_session_last_caption = None
        self.motif_counter = Counter()
        self.motif_first_seen = {}

    def get_mood_phrase(self):
        return "calm and observant"

    def get_clean_memory_snippets(self, k=2):
        return [c[0] for c in self.recent_captions[-k:]] if self.recent_captions else []


def test_mode_determination():
    """Test mode determination logic."""
    print("\n=== MODE DETERMINATION ===")
    test_cases = [
        ("tracking", "ahead", 0.3, 0.2, True, "relational"),
        ("idle", "ahead", 0.8, 0.2, False, "observational"),
        ("idle", "ahead", 0.2, 0.7, False, "restless"),
        ("idle", "down", 0.2, 0.3, False, "workspace"),
        ("idle", "ahead", 0.3, 0.3, False, "introspective"),
    ]

    passed = 0
    for gaze_state, gaze_dir, novelty, boredom, person, expected in test_cases:
        mode = determine_prompt_mode(gaze_state, gaze_dir, novelty, boredom, person)
        ok = mode == expected
        passed += ok
        status = "OK" if ok else "FAIL"
        print(f"[{status}] gaze={gaze_state}, dir={gaze_dir}, nov={novelty}, bore={boredom}, person={person}")
        print(f"       Expected: {expected}, Got: {mode}")
    print(f"Passed: {passed}/{len(test_cases)}")


def test_context_gating():
    """Test activation-based context gating."""
    print("\n=== CONTEXT GATING ===")

    # Reset network state
    network = get_activation_network()
    network._last_boredom = 0.3
    network._last_novelty = 0.3

    gating_tests = [
        ("relational", "relational", True, "relational mode includes relational"),
        ("pressure", "restless", True, "restless mode includes pressure"),
        ("curiosity", "observational", True, "observational mode includes curiosity"),
        ("motifs", "introspective", True, "introspective mode includes motifs"),
        ("motifs", "relational", False, "relational mode excludes motifs"),
        ("gaze", "any", True, "gaze always included"),
        ("continuity", "any", True, "continuity always included"),
    ]

    passed = 0
    for context_type, mode, expected, desc in gating_tests:
        result = should_include_context(context_type, mode)
        ok = result == expected
        passed += ok
        status = "OK" if ok else "FAIL"
        print(f"[{status}] {desc}")
        print(f"       should_include_context('{context_type}', '{mode}') = {result}")
    print(f"Passed: {passed}/{len(gating_tests)}")


def test_prompt_building():
    """Test the full prompt building with mode gating."""
    print("\n=== PROMPT BUILDING ===")

    # Set up context_compressor caption_count to avoid awakening
    try:
        from captioner.context_compression import context_compressor
        context_compressor.caption_count = 10
    except Exception as e:
        print(f"Note: Could not set caption_count: {e}")

    agent = MockAgent()
    agent.recent_captions = [
        ("The desk looks familiar.", time.time() - 60, "introspective"),
        ("Papers scattered here.", time.time() - 30, "observational"),
    ]

    # Test 1: Introspective mode (default idle state)
    print("\n--- Introspective Mode ---")
    network = get_activation_network()
    network._last_novelty = 0.3
    network._last_boredom = 0.3

    prompt, mode = build_simple_caption_prompt(agent, "Papers scattered here.", person_present=False)
    print(f"Mode: {mode}")
    print(f"Prompt (~{len(prompt.split())} words):")
    print(prompt)

    # Test 2: Observational mode (high novelty)
    print("\n--- Observational Mode (high novelty) ---")
    network._last_novelty = 0.8
    network._last_boredom = 0.2

    prompt, mode = build_simple_caption_prompt(agent, "Papers scattered here.", person_present=False)
    print(f"Mode: {mode}")
    print(f"Prompt (~{len(prompt.split())} words):")
    print(prompt)

    # Test 3: Restless mode (high boredom)
    print("\n--- Restless Mode (high boredom) ---")
    network._last_novelty = 0.2
    network._last_boredom = 0.8

    prompt, mode = build_simple_caption_prompt(agent, "Papers scattered here.", person_present=False)
    print(f"Mode: {mode}")
    print(f"Prompt (~{len(prompt.split())} words):")
    print(prompt)

    # Test 4: Relational mode (person present)
    print("\n--- Relational Mode (person present) ---")
    network._last_novelty = 0.3
    network._last_boredom = 0.3

    prompt, mode = build_simple_caption_prompt(agent, "Papers scattered here.", person_present=True)
    print(f"Mode: {mode}")
    print(f"Prompt (~{len(prompt.split())} words):")
    print(prompt)

    # Reset
    network._last_novelty = 0.5
    network._last_boredom = 0.0


def test_system_context():
    """Test the minimal system context building."""
    print("\n=== SYSTEM CONTEXT ===")

    agent = MockAgent()

    for mode in ["introspective", "relational", "observational", "restless", "workspace"]:
        system = _build_simple_system_context(agent, mode=mode)
        word_count = len(system.split())
        print(f"\n--- {mode} mode ({word_count} words) ---")
        print(system[:300] + "..." if len(system) > 300 else system)


def test_token_comparison():
    """Compare token counts with old system."""
    print("\n=== TOKEN COMPARISON ===")

    try:
        from captioner.context_compression import context_compressor
        context_compressor.caption_count = 10
    except Exception:
        pass

    agent = MockAgent()
    prompt, mode = build_simple_caption_prompt(agent, "Test caption.", person_present=False)
    system = _build_simple_system_context(agent, mode=mode)

    user_words = len(prompt.split())
    system_words = len(system.split())
    total = user_words + system_words

    print(f"User prompt: ~{user_words} words")
    print(f"System context: ~{system_words} words")
    print(f"Total: ~{total} words")
    print()
    print("Targets:")
    print("  - User prompt: 40-60 words")
    print("  - System context: 30-50 words")
    print("  - Total: 80-120 words")
    print()
    print("Old system average: ~300-400 words")
    print()

    if total <= 150:
        print("PASS: Within acceptable range")
    else:
        print(f"NOTE: {total} words is above target but still much better than old system")


def test_full_pipeline():
    """Test the full pipeline through build_focused_caption_prompt."""
    print("\n=== FULL PIPELINE (build_focused_caption_prompt) ===")

    if not USE_SIMPLE_PROMPTS:
        print("Skipping: USE_SIMPLE_PROMPTS is False")
        return

    try:
        from captioner.context_compression import context_compressor
        context_compressor.caption_count = 10
    except Exception:
        pass

    agent = MockAgent()
    result = build_focused_caption_prompt(agent, "Test caption.", person_present=False)

    print(f"Return type: {type(result)}")
    print(f"Return length: {len(result)} elements")

    user_prompt, system_ctx, dynamic_ctx, mode = result
    print(f"\nMode: {mode}")
    print(f"\nSystem context ({len(system_ctx.split())} words):")
    print(system_ctx[:200] + "..." if len(system_ctx) > 200 else system_ctx)
    print(f"\nUser prompt ({len(user_prompt.split())} words):")
    print(user_prompt)


if __name__ == "__main__":
    print("Testing Activation-Gated Simple Prompts")
    print("=" * 60)

    test_mode_determination()
    test_context_gating()
    test_prompt_building()
    test_system_context()
    test_token_comparison()
    test_full_pipeline()

    print("\n" + "=" * 60)
    print("All tests completed!")
