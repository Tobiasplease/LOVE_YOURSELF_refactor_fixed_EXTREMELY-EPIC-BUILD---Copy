#!/usr/bin/env python3
"""
Test suite for activation memory system.
Run: python debug/test_activation_memory.py
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from captioner.activation_memory import (
    ActivationNetwork,
    ContextualMemory,
    extract_concepts,
    observe_and_store,
    recall_for_prompt,
    get_beliefs,
)


def test_concept_extraction():
    """Test that concept extraction works correctly."""
    print("\n=== Test: Concept Extraction ===")

    cases = [
        ("A cluttered desk with scattered notebooks", ["desk", "notebook"]),
        ("Someone is sitting at the desk", ["desk"]),
        ("The window lets in bright light", ["window", "light"]),
        ("I feel lonely in this space", []),
        ("The red pen on the wooden table", ["pen", "table"]),
    ]

    for text, expected_subset in cases:
        concepts = extract_concepts(text)
        print(f"  '{text[:40]}...'")
        print(f"    → {concepts}")

        for expected in expected_subset:
            if expected not in concepts:
                print(f"    ⚠ Missing expected: {expected}")

    print("  ✓ Concept extraction working")


def test_activation_boost_and_decay():
    """Test activation boosting and decay."""
    print("\n=== Test: Activation Boost & Decay ===")

    network = ActivationNetwork()

    # Initial observation
    network.observe(["desk", "paper"], "down")
    desk_activation = network.activations.get("desk", 0)
    print(f"  After observe: desk={desk_activation:.2f}")
    assert desk_activation > 0, "Desk should be activated"

    # Second observation (should boost further)
    network.observe(["desk"], "down")
    desk_activation_2 = network.activations.get("desk", 0)
    print(f"  After second observe: desk={desk_activation_2:.2f}")
    assert desk_activation_2 > desk_activation, "Desk activation should increase"

    # Simulate time passing
    network.last_decay_time = time.time() - 60
    network._apply_decay()
    desk_activation_3 = network.activations.get("desk", 0)
    print(f"  After decay: desk={desk_activation_3:.2f}")
    assert desk_activation_3 < desk_activation_2, "Desk activation should decay"

    print("  ✓ Activation boost and decay working")


def test_edge_building():
    """Test that co-occurrence builds edges."""
    print("\n=== Test: Edge Building ===")

    network = ActivationNetwork()

    # Observe concepts together multiple times
    for _ in range(10):
        network.observe(["desk", "paper", "pen"], "down")

    desk_paper_edge = network.edges["desk"]["paper"]
    desk_pen_edge = network.edges["desk"]["pen"]

    print(f"  desk↔paper edge: {desk_paper_edge:.2f}")
    print(f"  desk↔pen edge: {desk_pen_edge:.2f}")

    assert desk_paper_edge > 0.3, "Strong co-occurrence should build edge"
    assert desk_pen_edge > 0.3, "Strong co-occurrence should build edge"

    # Check edge symmetry
    assert network.edges["paper"]["desk"] == desk_paper_edge, "Edges should be symmetric"

    print("  ✓ Edge building working")


def test_spreading_activation():
    """Test that activation spreads through edges."""
    print("\n=== Test: Spreading Activation ===")

    network = ActivationNetwork()

    # Build strong edge between desk and person
    for _ in range(15):
        network.observe(["desk", "person"], "ahead")

    # Let activations decay
    network.last_decay_time = time.time() - 120
    network._apply_decay()

    # Now observe only desk
    initial_person = network.activations.get("person", 0)
    print(f"  Person activation before: {initial_person:.3f}")

    network.observe(["desk"], "down")

    spread_person = network.activations.get("person", 0)
    print(f"  Person activation after desk observed: {spread_person:.3f}")

    assert spread_person > initial_person, "Person should get activation via spread from desk"

    print("  ✓ Spreading activation working")


def test_novelty_calculation():
    """Test novelty scoring."""
    print("\n=== Test: Novelty Calculation ===")

    network = ActivationNetwork()

    # First observation of something new
    novelty_1 = network.observe(["desk"], "down")
    print(f"  First observe 'desk': novelty={novelty_1:.2f}")
    assert novelty_1 > 0.6, "First observation should be novel"

    # Second observation (less novel)
    novelty_2 = network.observe(["desk"], "down")
    print(f"  Second observe 'desk': novelty={novelty_2:.2f}")
    assert novelty_2 < novelty_1, "Repeated observation should be less novel"

    # Observation of something completely new
    novelty_3 = network.observe(["cat"], "ahead")
    print(f"  First observe 'cat': novelty={novelty_3:.2f}")
    assert novelty_3 > 0.6, "New concept should be novel"

    print("  ✓ Novelty calculation working")


def test_boredom_calculation():
    """Test boredom scoring."""
    print("\n=== Test: Boredom Calculation ===")

    network = ActivationNetwork()

    # Fresh scene
    boredom_1 = network.calculate_boredom(["desk", "paper"])
    print(f"  Fresh scene boredom: {boredom_1:.2f}")
    assert boredom_1 < 0.3, "Fresh scene should not be boring"

    # Observe many times
    for _ in range(10):
        network.observe(["desk", "paper"], "down")

    # Same scene now boring
    boredom_2 = network.calculate_boredom(["desk", "paper"])
    print(f"  After many observations: {boredom_2:.2f}")
    assert boredom_2 > boredom_1, "Repeated scene should be more boring"

    print("  ✓ Boredom calculation working")


def test_memory_store_and_recall():
    """Test memory storage and recall."""
    print("\n=== Test: Memory Store & Recall ===")

    network = ActivationNetwork()
    memory = ContextualMemory(network)

    # Store some memories
    memory.store("A person is standing near the desk", ["person", "desk"], "ahead", time.time() - 3600)
    memory.store("The desk is cluttered with papers", ["desk", "paper", "clutter"], "down", time.time() - 1800)
    memory.store("Looking at the window", ["window"], "left", time.time() - 600)

    # Activate desk-related concepts
    network.observe(["desk", "paper"], "down")

    # Recall should find desk-related memories
    recalls = memory.recall(current_gaze="down", mode="workspace", k=2)
    print(f"  Recalled memories:")
    for r in recalls:
        print(f"    {r}")

    assert len(recalls) > 0, "Should recall at least one memory"
    assert any("desk" in r.lower() for r in recalls), "Should recall desk-related memory"

    print("  ✓ Memory store and recall working")


def test_beliefs_from_edges():
    """Test belief generation from strong edges."""
    print("\n=== Test: Beliefs from Edges ===")

    network = ActivationNetwork()

    # Build very strong association
    for _ in range(20):
        network.observe(["desk", "paper"], "down")

    beliefs = network.get_beliefs()
    print(f"  Generated beliefs:")
    for b in beliefs:
        print(f"    {b}")

    assert len(beliefs) > 0, "Should generate beliefs from strong edges"
    assert any("desk" in b.lower() and "paper" in b.lower() for b in beliefs)

    print("  ✓ Beliefs from edges working")


def test_compression_feedback():
    """Test compression feedback loop."""
    print("\n=== Test: Compression Feedback ===")

    network = ActivationNetwork()

    # Low initial activation
    network.observe(["workspace"], "ahead")
    initial = network.activations.get("workspace", 0)
    print(f"  Initial workspace activation: {initial:.2f}")

    # Compression mentions it
    network.boost_from_compression("I keep thinking about my cluttered workspace")

    boosted = network.activations.get("workspace", 0)
    print(f"  After compression boost: {boosted:.2f}")

    assert boosted > initial, "Compression should boost mentioned concepts"

    print("  ✓ Compression feedback working")


def test_drawing_context():
    """Test drawing context generation."""
    print("\n=== Test: Drawing Context ===")

    network = ActivationNetwork()
    memory = ContextualMemory(network)

    # Build up some state
    for _ in range(5):
        network.observe(["desk", "paper"], "down")

    network.observe(["cat"], "ahead")  # Novel
    memory.store("A cat appeared on the desk", ["cat", "desk"], "ahead")

    context = memory.format_drawing_context()
    print(f"  Drawing context: {context}")

    assert len(context) > 0, "Should generate drawing context"

    print("  ✓ Drawing context working")


def test_convenience_functions():
    """Test the convenience API functions."""
    print("\n=== Test: Convenience Functions ===")

    novelty, boredom = observe_and_store("A cluttered desk with scattered papers", "down")
    print(f"  observe_and_store: novelty={novelty:.2f}, boredom={boredom:.2f}")

    recall = recall_for_prompt("down", "workspace")
    print(f"  recall_for_prompt: '{recall[:60]}...'" if recall else "  recall_for_prompt: (empty)")

    beliefs = get_beliefs()
    print(f"  get_beliefs: {beliefs[:2]}...")

    print("  ✓ Convenience functions working")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("ACTIVATION MEMORY SYSTEM TESTS")
    print("=" * 60)

    test_concept_extraction()
    test_activation_boost_and_decay()
    test_edge_building()
    test_spreading_activation()
    test_novelty_calculation()
    test_boredom_calculation()
    test_memory_store_and_recall()
    test_beliefs_from_edges()
    test_compression_feedback()
    test_drawing_context()
    test_convenience_functions()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED ✓")
    print("=" * 60)


def demo_realistic_session():
    """Demonstrate with realistic session data."""
    print("\n" + "=" * 60)
    print("REALISTIC SESSION DEMO")
    print("=" * 60)

    network = ActivationNetwork()
    memory = ContextualMemory(network)

    captions = [
        ("A cluttered desk with papers and notebooks", "down", 0),
        ("The wooden desk surface shows wear", "down", 30),
        ("Someone enters the room", "ahead", 60),
        ("A person stands near the desk", "ahead", 90),
        ("They pick up a notebook", "ahead", 120),
        ("The person leaves", "ahead", 180),
        ("The desk is empty again", "down", 240),
        ("Scattered papers remain", "down", 300),
    ]

    base_time = time.time()

    print("\n--- Processing captions ---")
    for text, zone, offset in captions:
        timestamp = base_time - (300 - offset)
        concepts = extract_concepts(text)
        novelty = network.observe(concepts, zone)
        boredom = network.calculate_boredom(concepts)
        memory.store(text, concepts, zone, timestamp)

        print(f"\n  T+{offset}s [{zone}]: \"{text[:40]}...\"")
        print(f"    Concepts: {concepts}")
        print(f"    Novelty: {novelty:.2f}, Boredom: {boredom:.2f}")

    print("\n--- Final State ---")

    activated = network.get_activated_concepts(threshold=0.2)
    print(f"\n  Activated concepts:")
    for concept, activation in activated[:10]:
        print(f"    {concept}: {activation:.2f}")

    beliefs = network.get_beliefs()
    print(f"\n  Beliefs:")
    for b in beliefs[:5]:
        print(f"    {b}")

    print(f"\n--- Memory Recall (looking down at workspace) ---")
    recalls = memory.recall(current_gaze="down", mode="workspace", k=3)
    for r in recalls:
        print(f"  {r}")

    print(f"\n--- Memory Recall (thinking about person) ---")
    network.observe(["person"], "ahead")
    recalls = memory.recall(current_gaze="ahead", mode="relational", k=2)
    for r in recalls:
        print(f"  {r}")

    print(f"\n--- Drawing Context ---")
    ctx = memory.format_drawing_context()
    print(f"  {ctx}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true", help="Run realistic demo instead of tests")
    args = parser.parse_args()

    if args.demo:
        demo_realistic_session()
    else:
        run_all_tests()
