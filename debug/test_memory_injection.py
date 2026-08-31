#!/usr/bin/env python3
"""
Test script to verify Fix 1: narrative_state injection into user prompt.

This test simulates the prompt building with a mock narrative_state
to confirm it appears in the user prompt (not just system context).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class MockAgent:
    """Mock agent with memory attributes."""

    def __init__(self, session_age_mins=5):
        import time

        # Set session start time to be session_age_mins ago
        self.session_start_time = time.time() - (session_age_mins * 60)
        self.caption_count = 12
        self.last_shutdown_time = None
        self.last_session_gap = None
        self.prior_session_last_caption = None


class MockMemory:
    """Mock memory with baseline understanding."""

    def __init__(self):
        self.baseline_context = "I'm watching a cluttered workspace with shifting patterns of light and shadow."

    def get_baseline_context(self):
        return self.baseline_context

    def get_top_motifs(self, limit=5):
        return ["desk", "papers", "light", "shadow", "ceiling"]

    def get_beliefs(self):
        return ["I often see desk with papers"]

    def get_temporal_context(self):
        return "4 minutes awake, 12 observations"

    def get_emotional_state_summary(self):
        return "calm, attentive"


def test_narrative_injection():
    """Test that narrative_state is injected into user prompt."""
    import captioner.context_compression as context_compression_module
    from captioner import prompts

    # Create mock objects
    agent = MockAgent()
    agent.memory = MockMemory()

    # Simulate a narrative_state (this is what context_compressor returns)
    test_narrative = (
        "I'm watching a cluttered workspace with shifting light. Papers and desk often appear together. The ceiling catches my attention."
    )

    # Mock the context compressor to return our test narrative
    class MockCompressor:
        def __init__(self):
            self.caption_count = 15  # More than 3, so not awakening

        def get_consolidated_understanding(self):
            print(f"[MOCK] get_consolidated_understanding called, returning: {test_narrative[:50]}...")
            return test_narrative

    # Temporarily replace the context compressor in the module
    original_compressor = getattr(context_compression_module, "context_compressor", None)
    mock_compressor = MockCompressor()
    context_compression_module.context_compressor = mock_compressor

    try:
        # Build a prompt (this should include the narrative_state)
        prompt, system_prompt, dynamic_context, prompt_mode = prompts.build_focused_caption_prompt(
            agent=agent,
            last_caption="Papers scattered on the surface",
            person_present=False,
        )

        print("=" * 80)
        print("GENERATED PROMPT:")
        print("=" * 80)
        print(prompt)
        print("=" * 80)

        # Verify the fix: narrative should appear in user prompt
        first_sentence = test_narrative.split(".")[0].strip()[:80]
        expected_injection = f"You know: {first_sentence}."

        if expected_injection in prompt:
            print("\n✅ SUCCESS: narrative_state is injected into user prompt!")
            print(f"   Found: '{expected_injection}'")
            return True
        else:
            print("\n❌ FAILURE: narrative_state NOT found in user prompt")
            print(f"   Expected to find: '{expected_injection}'")
            return False

    finally:
        # Restore original compressor
        if original_compressor:
            prompts.context_compressor = original_compressor


def test_narrative_not_in_system_only():
    """Verify narrative isn't ONLY in system context."""
    import captioner.context_compression as context_compression_module
    from captioner import prompts

    agent = MockAgent()
    agent.memory = MockMemory()

    test_narrative = "I'm watching a cluttered workspace with shifting light"

    class MockCompressor:
        def __init__(self):
            self.caption_count = 15  # More than 3, so not awakening

        def get_consolidated_understanding(self):
            return test_narrative

    original_compressor = getattr(context_compression_module, "context_compressor", None)
    mock_compressor = MockCompressor()
    context_compression_module.context_compressor = mock_compressor

    try:
        prompt, system_prompt, dynamic_context, prompt_mode = prompts.build_focused_caption_prompt(
            agent=agent,
            last_caption="Papers scattered",
            person_present=False,
        )

        # The prompt should contain "You know:" in the USER portion
        # (not just buried in system context as "YOUR STATE:")
        has_you_know = "You know:" in prompt
        has_your_state = "YOUR STATE:" in prompt

        print("\n" + "=" * 80)
        print("INJECTION LOCATION TEST:")
        print("=" * 80)
        print(f"'You know:' in prompt (USER): {has_you_know} ✅" if has_you_know else f"'You know:' in prompt (USER): {has_you_know} ❌")
        print(f"'YOUR STATE:' in prompt (SYSTEM): {has_your_state}")

        if has_you_know:
            print("\n✅ Fix 1 working: narrative in USER prompt (prominent)")
        else:
            print("\n❌ Fix 1 not working: narrative missing from USER prompt")

        return has_you_know

    finally:
        if original_compressor:
            context_compression_module.context_compressor = original_compressor


def test_awakening_mode_excluded():
    """Verify narrative is NOT injected during awakening."""
    import captioner.context_compression as context_compression_module
    from captioner import prompts

    agent = MockAgent()
    agent.memory = MockMemory()

    test_narrative = "I'm watching a cluttered workspace"

    class MockCompressor:
        def __init__(self):
            self.caption_count = 2  # Less than 3 to trigger awakening mode

        def get_consolidated_understanding(self):
            return test_narrative

    original_compressor = getattr(context_compression_module, "context_compressor", None)
    mock_compressor = MockCompressor()
    context_compression_module.context_compressor = mock_compressor

    try:
        # Build prompt - should be in awakening mode due to low caption_count
        prompt, system_prompt, dynamic_context, prompt_mode = prompts.build_focused_caption_prompt(
            agent=agent,
            last_caption="",
            person_present=False,
        )

        has_you_know = "You know:" in prompt

        print("\n" + "=" * 80)
        print("AWAKENING MODE TEST:")
        print("=" * 80)
        print(f"'You know:' during awakening: {has_you_know}")

        if not has_you_know:
            print("✅ Correct: narrative NOT injected during awakening")
            return True
        else:
            print("⚠️  Warning: narrative injected during awakening (may be intentional)")
            return False

    finally:
        if original_compressor:
            context_compression_module.context_compressor = original_compressor


if __name__ == "__main__":
    print("Testing Fix 1: narrative_state injection into user prompt\n")

    test1 = test_narrative_injection()
    test2 = test_narrative_not_in_system_only()
    test3 = test_awakening_mode_excluded()

    print("\n" + "=" * 80)
    print("TEST SUMMARY:")
    print("=" * 80)
    print(f"1. Narrative injected to user prompt: {'✅ PASS' if test1 else '❌ FAIL'}")
    print(f"2. Narrative in prominent location: {'✅ PASS' if test2 else '❌ FAIL'}")
    print(f"3. Awakening mode excluded: {'✅ PASS' if test3 else '⚠️  WARNING'}")

    if test1 and test2:
        print("\n🎉 Fix 1 is working correctly!")
        print("\nNext steps:")
        print("1. Run the main system: python machine.py")
        print("2. Watch for '[MEMORY FIX]' log messages")
        print("3. Observe if captions build on prior knowledge")
        print("4. Check if 'rediscovery' patterns decrease after 10+ minutes")
    else:
        print("\n⚠️  Fix 1 may need adjustment")
