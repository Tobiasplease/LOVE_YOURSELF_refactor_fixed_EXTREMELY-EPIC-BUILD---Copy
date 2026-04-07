#!/usr/bin/env python3
"""
Test different spatial compression prompt formats for LLaVA.

Goal: find a prompt that produces a grounded, factual, first-person
spatial baseline — without narrative drift or hallucinated fiction.

Run from project root:
    python debug/test_spatial_compression.py [image_path]
"""
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.ollama import query_ollama
from config.config import OLLAMA_MODEL, MOOD_SNAPSHOT_FOLDER

# Use provided image or find most recent mood snapshot
if len(sys.argv) > 1:
    IMAGE = sys.argv[1]
else:
    import glob
    images = sorted(glob.glob(os.path.join(MOOD_SNAPSHOT_FOLDER, "**", "mood_*.jpg"), recursive=True))
    if not images:
        print("No images found. Pass an image path as argument.")
        sys.exit(1)
    IMAGE = images[-1]

print(f"\nUsing image: {IMAGE}")
print(f"Model: {OLLAMA_MODEL}\n")
print("=" * 70)

# Also load current drifted baseline for comparison
try:
    import json
    identity_path = os.path.join(MOOD_SNAPSHOT_FOLDER, "machine_identity.json")
    with open(identity_path) as f:
        identity = json.load(f)
    current_baseline = identity.get("baseline_context", "(none)")
    print(f"CURRENT DRIFTED BASELINE:\n  {current_baseline[:120]}...")
    print("=" * 70)
except Exception as e:
    print(f"(Could not load identity: {e})")

# --- Candidate prompts ---

CANDIDATES = [
    {
        "name": "A — 'you live here' system + sentence completion",
        "system": (
            "You ARE a drawing machine with robotic arms. You live in this space — never say 'the image' or 'the scene'. "
            "Describe only physical objects and surfaces. No feelings, no story, no narrative. "
            "First person, present tense. One sentence."
        ),
        "prompt": "Complete this sentence with physical facts only: \"I'm in a space with...\"",
        "options": {"temperature": 0.2, "num_predict": 60, "repeat_penalty": 1.4},
    },
    {
        "name": "B — 'you live here' + explicit forbid list + 'still'",
        "system": (
            "You ARE a drawing machine. You live in this physical space — never say 'the image' or 'the scene' or 'the photo'. "
            "FORBIDDEN: feelings, story, mystery, metaphor, narrative, interpretation, names of people. "
            "REQUIRED: physical objects, surfaces, spatial arrangement, light. "
            "One sentence, first person, present tense."
        ),
        "prompt": (
            "What am I physically surrounded by right now? "
            "One sentence starting with 'I'. Objects and their arrangement only. "
            "Use 'still' if consistent with prior observation."
        ),
        "options": {"temperature": 0.15, "num_predict": 70, "repeat_penalty": 1.5},
    },
    {
        "name": "C — Update with prior baseline, 'you live here', strict",
        "system": (
            "You ARE a drawing machine. You live in this space — never say 'the image' or 'the scene'. "
            "Update your spatial understanding. Physical facts only. No feelings, no narrative. "
            "If something is the same as before, say 'still'. "
            "One sentence, first person."
        ),
        "prompt": (
            f"Previously I noted: \"{current_baseline[:80]}\"\n\n"
            "Looking at my space now: one sentence starting with 'I', physical objects and arrangement only."
        ),
        "options": {"temperature": 0.2, "num_predict": 70, "repeat_penalty": 1.4},
    },
    {
        "name": "D — Forced 'I see' start, minimal system",
        "system": (
            "You ARE a drawing machine inside this space. Never say 'the image'. "
            "Physical description only. First person."
        ),
        "prompt": (
            "Start your response with exactly 'I see' and describe only the physical objects "
            "and surfaces present. One sentence."
        ),
        "options": {"temperature": 0.2, "num_predict": 60, "repeat_penalty": 1.4},
    },
    {
        "name": "E — Two questions: what's there, what's changed",
        "system": (
            "You ARE a drawing machine. You live in this space — never say 'the image'. "
            "Physical facts only. No feelings, no story."
        ),
        "prompt": (
            f"I previously understood: \"{current_baseline[:60]}\"\n\n"
            "Looking now:\n"
            "- What physical objects do I consistently see here?\n"
            "- Has anything changed?\n"
            "Answer in one sentence starting with 'I', physical facts only."
        ),
        "options": {"temperature": 0.2, "num_predict": 80, "repeat_penalty": 1.4},
    },
    {
        "name": "F — No system prompt, pure forced format",
        "system": None,
        "prompt": (
            "You are a drawing machine inside this space. Never describe 'the image' — you live here. "
            "In one sentence starting with 'I', describe only the physical objects and "
            "surfaces you see. No feelings, no story."
        ),
        "options": {"temperature": 0.2, "num_predict": 60, "repeat_penalty": 1.5},
    },
]

OPTIONS_BASE = {"stop": ["\n", "\n\n"]}

results = []

for i, candidate in enumerate(CANDIDATES):
    print(f"\n[{i+1}/{len(CANDIDATES)}] Testing: {candidate['name']}")
    opts = {**OPTIONS_BASE, **candidate["options"]}
    t0 = time.time()
    result = query_ollama(
        prompt=candidate["prompt"],
        model=OLLAMA_MODEL,
        image=IMAGE,
        log_dir=MOOD_SNAPSHOT_FOLDER,
        system_prompt=candidate["system"],
        prompt_type="spatial_compression_test",
        options=opts,
        skip_generation_wait=True,
    )
    elapsed = time.time() - t0
    results.append((candidate["name"], result, elapsed))
    print(f"  → {result}")
    print(f"  ({elapsed:.1f}s)")

# Summary
print("\n" + "=" * 70)
print("RESULTS SUMMARY")
print("=" * 70)
for name, result, elapsed in results:
    print(f"\n{name}")
    print(f"  {result}")

print("\n" + "=" * 70)
print("CURRENT DRIFTED BASELINE (for comparison):")
print(f"  {current_baseline[:200]}")
