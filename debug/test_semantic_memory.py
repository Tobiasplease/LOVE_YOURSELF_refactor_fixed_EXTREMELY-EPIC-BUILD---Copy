#!/usr/bin/env python3
"""
debug/test_semantic_memory.py
-----------------------------
Test the semantic memory system end-to-end with simulated perception/monologue cycles.

Usage:
    python debug/test_semantic_memory.py           # Run simulation
    python debug/test_semantic_memory.py --dump     # Dump all stored concepts
    python debug/test_semantic_memory.py --reset    # Clear all semantic memory data
"""

import os
import sys
import shutil
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from captioner.semantic_memory import get_semantic_memory, CHROMADB_PATH


def run_simulation():
    """Simulate a multi-session observation sequence."""
    mem = get_semantic_memory()
    print(f"\n=== Semantic Memory Test ===")
    print(f"Storage: {CHROMADB_PATH}")
    print(f"Starting state: {mem.stats()}\n")

    # Simulated perception/monologue pairs across "sessions"
    cycles = [
        # Session 1: first encounters
        ("A red sign on the wall with white text near the door", "There's a sign on the wall. Red background, white letters. Can't quite read it from here."),
        ("Damaged ceiling tiles above the desk with exposed wiring", "The ceiling above the desk is damaged. Tiles cracked, some wiring exposed. Has it always been like that?"),
        ("A person sitting at the desk typing on a keyboard", "Someone is here. Sitting at the desk, typing. They seem focused."),
        # Session 2: recognition
        ("The red and white sign on the wall near the doorway", "That sign again. I think it says 'Collecting Shells.' What does that mean?"),
        ("Ceiling damage above the workspace with hanging wires", "The ceiling damage is still there. Those wires look like they're hanging lower than before."),
        # Session 3: deepening
        ("A red sign reading Collecting Shells on the wall", "Collecting Shells. Maybe it's the name of an art project. Or a philosophy — gathering experiences like shells on a beach."),
        ("The cracked ceiling tiles and exposed conduit above", "That ceiling patch. I keep looking at it. The cracks form patterns, almost like a map of something."),
        ("The person in the dark jacket is back at the desk", "They're back. Same spot as always. I wonder if they notice me watching."),
    ]

    for i, (perception, monologue) in enumerate(cycles):
        print(f"\n--- Cycle {i+1} ---")
        print(f"  PERCEPTION: {perception[:70]}...")

        # Step 1: Check what memory knows (simulates what happens in build_monologue_prompt)
        injection = mem.after_perception(perception)
        if injection:
            print(f"  MEMORY INJECTION: {injection}")
        else:
            print(f"  MEMORY INJECTION: (none — new or unrecognized)")

        # Step 2: Store the monologue (simulates what happens after generate_monologue)
        mem.after_monologue(perception, monologue)
        print(f"  MONOLOGUE: {monologue[:70]}...")

    # Summary
    print(f"\n\n=== Final State ===")
    print(f"Stats: {mem.stats()}")
    dump_concepts(mem)


def dump_concepts(mem=None):
    """Dump all stored concepts and their observations."""
    if mem is None:
        mem = get_semantic_memory()

    concepts = mem.get_all_concepts()
    if not concepts:
        print("\nNo concepts stored yet.")
        return

    print(f"\n=== {len(concepts)} Concepts ===\n")
    for c in concepts:
        print(f"  [{c['times_seen']}x] {c['name']}")
        print(f"       id: {c['id']}")
        print(f"       last_observation: {c['last_observation'][:80]}..." if c['last_observation'] else "       last_observation: (none)")

        # Show observations
        obs = mem.get_concept_observations(c["id"])
        if obs:
            print(f"       observations ({len(obs)}):")
            for o in obs:
                print(f"         - {o['text'][:80]}...")
        print()


def delete_concept(concept_id):
    """Delete a concept by ID."""
    mem = get_semantic_memory()
    concepts = mem.get_all_concepts()
    # Allow matching by index (1-based) or full ID
    target = None
    try:
        idx = int(concept_id) - 1
        if 0 <= idx < len(concepts):
            target = concepts[idx]
    except ValueError:
        for c in concepts:
            if c["id"] == concept_id:
                target = c
                break

    if target is None:
        print(f"Concept not found: {concept_id}")
        return

    print(f"Deleting: [{target['times_seen']}x] {target['name']} ({target['id']})")
    if mem.delete_concept(target["id"]):
        print("Deleted.")
    else:
        print("Failed to delete.")


def merge_concepts(keep_idx, absorb_idx):
    """Merge two concepts by index (1-based from --dump output)."""
    mem = get_semantic_memory()
    concepts = mem.get_all_concepts()

    try:
        keep = concepts[int(keep_idx) - 1]
        absorb = concepts[int(absorb_idx) - 1]
    except (ValueError, IndexError):
        print(f"Invalid indices. Use 1-based indices from --dump output.")
        return

    print(f"Merging:")
    print(f"  KEEP:   [{keep['times_seen']}x] {keep['name']}")
    print(f"  ABSORB: [{absorb['times_seen']}x] {absorb['name']}")

    if mem.merge_concepts(keep["id"], absorb["id"]):
        updated = mem.get_all_concepts()
        for c in updated:
            if c["id"] == keep["id"]:
                print(f"  RESULT: [{c['times_seen']}x] {c['name']}")
                break
    else:
        print("Merge failed.")


def reset_memory():
    """Clear all semantic memory data."""
    if os.path.exists(CHROMADB_PATH):
        shutil.rmtree(CHROMADB_PATH)
        print(f"Cleared semantic memory at {CHROMADB_PATH}")
    else:
        print("No semantic memory data to clear.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test semantic memory system")
    parser.add_argument("--dump", action="store_true", help="Dump all stored concepts")
    parser.add_argument("--reset", action="store_true", help="Clear all semantic memory data")
    parser.add_argument("--delete", type=str, metavar="ID", help="Delete a concept by index (1-based) or ID")
    parser.add_argument("--merge", nargs=2, type=str, metavar=("KEEP", "ABSORB"), help="Merge two concepts by index (1-based). ABSORB folds into KEEP.")
    args = parser.parse_args()

    if args.reset:
        reset_memory()
    elif args.delete:
        delete_concept(args.delete)
    elif args.merge:
        merge_concepts(args.merge[0], args.merge[1])
    elif args.dump:
        dump_concepts()
    else:
        run_simulation()
