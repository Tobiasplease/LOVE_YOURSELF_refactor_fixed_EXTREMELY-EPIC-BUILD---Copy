#!/usr/bin/env python3
"""Debug script to test the refactored caption prompt building."""

import sys
import time

sys.path.insert(0, "/home/impostor/LOVE_YOURSELF_refactor_fixed_EXTREMELY-EPIC-BUILD---Copy")

import os

from captioner.captioner import Captioner
from captioner.model_wrapper import _is_plantable_prior, build_caption_thread
from captioner.prompts import build_memory_mode_prompt, build_simple_caption_prompt
from config.config import MOOD_SNAPSHOT_FOLDER

# Initialize captioner
print("[TEST] Initializing Captioner...")
captioner = Captioner()

# Simulate some captions in recent_captions
print("\n[TEST] Simulating recent captions...")
captioner.recent_captions = [
    ("The light catches my attention in unexpected ways.", time.time() - 30),
    ("I notice the texture of the desk beneath me.", time.time() - 20),
    ("The room feels familiar yet slightly different.", time.time() - 10),
]

# Test caption thread building
print("\n[TEST] Testing caption thread building...")
thread = build_caption_thread(captioner, max_captions=3)
print(f"Caption thread:\n{thread}")

# Test _is_plantable_prior filter
print("\n[TEST] Testing _is_plantable_prior() filter...")
test_captions = [
    ("The light catches my attention in unexpected ways.", True),
    ("The image appears to show a desk.", False),  # VQA pattern
    ("You notice the texture of the desk.", False),  # Natsumura "You"
    ("*You look at the desk carefully.*", False),  # Natsumura roleplay
    ("I wonder what this space means to me.", True),
]

for caption, expected in test_captions:
    result = _is_plantable_prior(caption)
    status = "✓" if result == expected else "✗"
    print(f"{status} '{caption[:50]}...' → {result} (expected {expected})")

# Test build_simple_caption_prompt
print("\n[TEST] Testing build_simple_caption_prompt()...")
prompt, mode = build_simple_caption_prompt(captioner, person_present=False)
print(f"\nMode: {mode}")
print(f"\nPrompt ({len(prompt)} chars):")
print("=" * 80)
print(prompt)
print("=" * 80)

# Test build_memory_mode_prompt
print("\n[TEST] Testing build_memory_mode_prompt()...")
try:
    mem_prompt, mem_mode = build_memory_mode_prompt(captioner)
    print(f"\nMode: {mem_mode}")
    print(f"\nMemory Mode Prompt ({len(mem_prompt)} chars):")
    print("=" * 80)
    print(mem_prompt)
    print("=" * 80)
except Exception as e:
    print(f"[ERROR] Memory mode failed: {e}")
    import traceback

    traceback.print_exc()

# Check context_compressor
print("\n[TEST] Checking context_compressor...")
try:
    from captioner.context_compression import context_compressor

    baseline = context_compressor.get_baseline_context()
    print(f"Baseline context: '{baseline}'")
    print(f"Baseline length: {len(baseline) if baseline else 0} chars")
except Exception as e:
    print(f"[ERROR] context_compressor failed: {e}")

# Check activation network
print("\n[TEST] Checking activation network...")
try:
    from captioner.activation_memory import get_activation_network

    network = get_activation_network()
    print(f"Novelty: {network._last_novelty:.2f}")
    print(f"Boredom: {network._last_boredom:.2f}")
    print(f"Top activations: {sorted(network.activations.items(), key=lambda x: x[1], reverse=True)[:5]}")
except Exception as e:
    print(f"[ERROR] activation_network failed: {e}")

print("\n[TEST] Complete!")
