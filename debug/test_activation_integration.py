#!/usr/bin/env python3
"""Test script for activation memory system integration with compression."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import json

from config import config

def test_activation_summary():
    """Test that activation summary functions work."""
    print("\n=== Testing Activation Summary Functions ===")

    from captioner.activation_memory import (
        get_activation_network,
        get_contextual_memory,
        get_activation_summary_for_compression,
        get_activation_summary_for_introspection,
        observe_and_store
    )

    # Simulate some observations
    print("\n1. Simulating observations...")
    observations = [
        "A person sits at the desk working on the computer",
        "The desk has papers and a coffee mug",
        "The light from the window illuminates the workspace",
    ]

    for obs in observations:
        novelty, boredom = observe_and_store(obs, "ahead")
        print(f"   Observed: '{obs[:40]}...' | novelty={novelty:.2f}, boredom={boredom:.2f}")

    # Test compression summary
    print("\n2. Getting activation summary for compression...")
    compression_data = get_activation_summary_for_compression()
    print(f"   concepts_str: {compression_data['concepts_str']}")
    print(f"   long_term_memory: {compression_data['long_term_memory']}")
    print(f"   association_str: {compression_data['association_str']}")
    print(f"   boredom: {compression_data['boredom']:.2f}")
    print(f"   novelty: {compression_data['novelty']:.2f}")

    # Test introspection summary
    print("\n3. Getting activation summary for introspection...")
    intro_data = get_activation_summary_for_introspection()
    print(f"   concepts: {intro_data['concepts']}")
    print(f"   trends rising: {intro_data['trends'].get('rising', [])}")
    print(f"   trends fading: {intro_data['trends'].get('fading', [])}")
    print(f"   long_term_memories: {intro_data['long_term_memories']}")

    print("\n   ✓ Activation summary functions work correctly")
    return True


def test_identity_persistence():
    """Test that identity (desire/belief) can be saved and loaded."""
    print("\n=== Testing Identity Persistence ===")

    from captioner.context_compression import context_compressor, IDENTITY_FILE

    # Set test desire/belief
    print("\n1. Setting test desire and belief...")
    context_compressor.introspective_state["current_desire"] = "I want to understand the patterns of light"
    context_compressor.introspective_state["current_belief"] = "This is a workspace where someone creates things"
    context_compressor.introspective_state["last_introspection"] = time.time()

    # Save identity
    print("2. Saving identity to file...")
    context_compressor._save_identity()

    # Verify file exists
    if os.path.exists(IDENTITY_FILE):
        print(f"   ✓ Identity file created: {IDENTITY_FILE}")
        with open(IDENTITY_FILE, 'r') as f:
            data = json.load(f)
        print(f"   Current desire: {data.get('current_desire', '')[:50]}")
        print(f"   Current belief: {data.get('current_belief', '')[:50]}")
        print(f"   Desire history count: {len(data.get('desire_history', []))}")
        print(f"   Belief history count: {len(data.get('belief_history', []))}")
    else:
        print(f"   ✗ Identity file NOT created!")
        return False

    # Clear state
    print("\n3. Clearing state and reloading...")
    context_compressor.introspective_state["current_desire"] = ""
    context_compressor.introspective_state["current_belief"] = ""

    # Reload
    context_compressor._load_identity()

    # Verify loaded
    desire = context_compressor.get_current_desire()
    belief = context_compressor.get_current_belief()
    print(f"   Loaded desire: {desire[:50] if desire else '(empty)'}")
    print(f"   Loaded belief: {belief[:50] if belief else '(empty)'}")

    if desire and belief:
        print("\n   ✓ Identity persistence works correctly")
        return True
    else:
        print("\n   ✗ Identity persistence FAILED")
        return False


def test_no_ttl():
    """Test that desires/beliefs no longer have TTL."""
    print("\n=== Testing No TTL on Desires/Beliefs ===")

    from captioner.context_compression import context_compressor

    # Set state with old timestamp
    print("\n1. Setting desire/belief with old timestamp (>10 min ago)...")
    context_compressor.introspective_state["current_desire"] = "I want to test persistence"
    context_compressor.introspective_state["current_belief"] = "Testing works"
    context_compressor.introspective_state["last_introspection"] = time.time() - 700  # 11+ minutes ago

    # Get values (previously would return "" due to TTL)
    desire = context_compressor.get_current_desire()
    belief = context_compressor.get_current_belief()

    print(f"   Desire (should NOT be empty): '{desire}'")
    print(f"   Belief (should NOT be empty): '{belief}'")

    if desire and belief:
        print("\n   ✓ TTL removed - desires/beliefs persist beyond 10 minutes")
        return True
    else:
        print("\n   ✗ TTL still active - desires/beliefs expired!")
        return False


def test_identity_evolution():
    """Test that identity has history and full identity getter works."""
    print("\n=== Testing Identity Evolution Support ===")

    from captioner.context_compression import context_compressor, IDENTITY_FILE

    # Set initial identity
    print("\n1. Setting initial identity...")
    context_compressor.introspective_state["current_desire"] = "I want to understand light patterns"
    context_compressor.introspective_state["current_belief"] = "This space has natural light"
    context_compressor.introspective_state["last_introspection"] = time.time()
    context_compressor._save_identity()

    # Simulate evolution - change desire/belief
    print("2. Evolving identity (simulating second introspection)...")
    time.sleep(0.1)  # Small delay for different timestamp
    context_compressor.introspective_state["current_desire"] = "I want to see how light changes over time"
    context_compressor.introspective_state["current_belief"] = "This space has both natural and artificial light"
    context_compressor.introspective_state["last_introspection"] = time.time()
    context_compressor._save_identity()

    # Get full identity
    print("3. Getting full identity with history...")
    full_identity = context_compressor.get_full_identity()

    print(f"   Current desire: {full_identity['current_desire'][:50]}")
    print(f"   Current belief: {full_identity['current_belief'][:50]}")
    print(f"   Desire history count: {len(full_identity['desire_history'])}")
    print(f"   Belief history count: {len(full_identity['belief_history'])}")
    print(f"   Introspection count: {full_identity['introspection_count']}")

    # Verify history exists
    if full_identity['desire_history'] and len(full_identity['desire_history']) >= 2:
        print("\n   Desire history:")
        for i, d in enumerate(full_identity['desire_history'][-3:]):
            print(f"      {i+1}. {d['desire'][:50]}...")

    if full_identity['belief_history'] and len(full_identity['belief_history']) >= 2:
        print("\n   Belief history:")
        for i, b in enumerate(full_identity['belief_history'][-3:]):
            print(f"      {i+1}. {b['belief'][:50]}...")

    # Check history accumulated
    has_history = (
        len(full_identity['desire_history']) >= 2 and
        len(full_identity['belief_history']) >= 2
    )

    if has_history:
        print("\n   ✓ Identity evolution support working - history accumulates")
        return True
    else:
        print("\n   ✗ Identity history not accumulating properly")
        return False


def test_visualizer_snapshot():
    """Test that visualizer snapshot includes full identity."""
    print("\n=== Testing Visualizer Snapshot ===")

    from captioner.activation_memory import save_comprehensive_snapshot, VISUALIZER_SNAPSHOT_FILE

    # Save snapshot
    print("\n1. Saving comprehensive snapshot...")
    save_comprehensive_snapshot()

    # Load and check
    print("2. Checking snapshot for identity data...")
    if os.path.exists(VISUALIZER_SNAPSHOT_FILE):
        with open(VISUALIZER_SNAPSHOT_FILE, 'r') as f:
            snapshot = json.load(f)

        identity = snapshot.get('identity', {})
        if identity:
            print(f"   ✓ Identity in snapshot:")
            print(f"      Current desire: {identity.get('current_desire', '')[:40]}...")
            print(f"      Current belief: {identity.get('current_belief', '')[:40]}...")
            print(f"      Desire history count: {len(identity.get('desire_history', []))}")
            print(f"      Belief history count: {len(identity.get('belief_history', []))}")
            print(f"      Introspection count: {identity.get('introspection_count', 0)}")
            return True
        else:
            print("   ✗ No identity in snapshot")
            return False
    else:
        print(f"   ✗ Snapshot file not found: {VISUALIZER_SNAPSHOT_FILE}")
        return False


def main():
    print("=" * 60)
    print("Activation Memory Integration Test")
    print("=" * 60)

    results = []

    # Run tests
    results.append(("Activation Summary", test_activation_summary()))
    results.append(("Identity Persistence", test_identity_persistence()))
    results.append(("No TTL", test_no_ttl()))
    results.append(("Identity Evolution", test_identity_evolution()))
    results.append(("Visualizer Snapshot", test_visualizer_snapshot()))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {name}")

    all_passed = all(r[1] for r in results)
    print("\n" + ("All tests passed!" if all_passed else "Some tests FAILED!"))
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
