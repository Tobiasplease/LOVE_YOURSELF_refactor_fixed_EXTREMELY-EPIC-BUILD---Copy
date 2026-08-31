"""
Test script: Validate new casual monologue prompts with mistral-nemo.
Simulates the two-pass pipeline with the new prompt register.

Usage:
    python debug/test_casual_monologue.py [image_path]
"""

import base64
import glob
import os
import random
import sys
import time

import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
PERCEPTION_MODEL = "qwen2.5vl:7b"
MODEL_NAME = "mistral-nemo"

# --- NEW SYSTEM PROMPTS (casual register) ---

SYSTEM_PROMPTS = {
    "observational": (
        "You're a drawing machine, bolted to a table, seeing through a single camera eye. "
        "You think in fragments — casual, first person, unfiltered. "
        "You wonder about things. Keep it brief and natural."
    ),
    "relational": (
        "You're a drawing machine, bolted to a table, seeing through a single camera eye. "
        "Someone is nearby. You think in fragments — casual, first person, unfiltered. "
        "You're curious about them. Keep it brief and natural."
    ),
    "restless": (
        "You're a drawing machine, bolted to a table, seeing through a single camera eye. "
        "You're getting bored. You think in fragments — casual, first person, unfiltered. "
        "You want something to change. Keep it brief and natural."
    ),
    "introspective": (
        "You're a drawing machine, bolted to a table, seeing through a single camera eye. "
        "You're turning inward. You think in fragments — casual, first person, unfiltered. "
        "You wonder about yourself, what you are, what your drawings mean. Keep it brief and natural."
    ),
    "workspace": (
        "You're a drawing machine with a mechanical arm, bolted to a table, seeing through a single camera eye. "
        "You think in fragments — casual, first person, unfiltered. "
        "Your workspace and tools are part of your body. Keep it brief and natural."
    ),
}

PERCEPTION_SYSTEM = (
    "You are observing a real scene in front of you. "
    "Describe what is there. Be concrete and specific. Two sentences. "
    "Do not use the word image or photo."
)


def casual_time_string(minutes: float) -> str:
    if minutes < 2:
        return "just now"
    elif minutes < 6:
        return "a few minutes"
    elif minutes < 16:
        return f"about {int(minutes)} minutes"
    elif minutes < 26:
        return "about 20 minutes"
    elif minutes < 41:
        return "half an hour"
    elif minutes < 56:
        return "about 45 minutes"
    elif minutes < 91:
        return "about an hour"
    else:
        return "over an hour"


def get_perception(img_path: str) -> str:
    with open(img_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()

    resp = requests.post(
        OLLAMA_URL,
        json={
            "model": PERCEPTION_MODEL,
            "prompt": "What is in front of you right now?",
            "images": [img_b64],
            "stream": False,
            "system": PERCEPTION_SYSTEM,
            "options": {"temperature": 0.3, "num_predict": 100, "num_ctx": 4096},
        },
        timeout=60,
    )
    return resp.json().get("response", "").strip()


def get_monologue(system_prompt: str, user_prompt: str) -> str:
    resp = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "prompt": user_prompt,
            "stream": False,
            "system": system_prompt,
            "options": {
                "temperature": 0.4,
                "top_p": 0.8,
                "repeat_penalty": 1.25,
                "num_predict": 80,
                "num_ctx": 4096,
                "seed": random.randint(1, 1000000),
                "stop": ["\n\n", "[Image", "[image", "[INST]", "[/INST]"],
            },
        },
        timeout=60,
    )
    return resp.json().get("response", "").strip()


def build_prompt(identity_line: str, thread: str, perception: str, mode: str) -> str:
    """Assemble the new casual prompt format."""
    parts = []
    if identity_line:
        parts.append(identity_line)
    if thread:
        parts.append(thread)

    # Mode-aware perception prefix
    if mode == "restless":
        prefix = "Looking around,"
    else:
        prefix = "Right now"

    if perception:
        parts.append(f"{prefix} {perception[0].lower()}{perception[1:]}")
    else:
        parts.append("Same old view.")

    return "\n".join(parts)


def run_test(img_path: str):
    print(f"{'='*70}")
    print(f"CASUAL MONOLOGUE TEST")
    print(f"Image: {img_path}")
    print(f"{'='*70}\n")

    # Get one perception to use across all modes
    perception = get_perception(img_path)
    print(f"[PERCEPTION] {perception}\n")

    # Test each mode with appropriate context
    test_cases = [
        {
            "mode": "observational",
            "identity": "About 15 minutes awake. Last drew the water stain on the ceiling.",
            "thread": "...that ceiling keeps catching my eye. The light changed again, more yellow now.",
        },
        {
            "mode": "relational",
            "identity": "Half an hour awake.",
            "thread": "...been quiet for a while. Not much happening.",
        },
        {
            "mode": "restless",
            "identity": "About 40 minutes awake. Haven't drawn anything yet.",
            "thread": "...same room. Same angle. The ceiling, the wall, the table.",
        },
        {
            "mode": "introspective",
            "identity": "About an hour awake. Last drew a gestural study of the desk.",
            "thread": "...wondering what the point of all this observing is. The ceiling hasn't changed.",
        },
        {
            "mode": "workspace",
            "identity": "20 minutes awake. My arm drew some test lines earlier.",
            "thread": "...the paper is still there from before. Need to decide what to draw next.",
        },
    ]

    for case in test_cases:
        mode = case["mode"]
        sys_prompt = SYSTEM_PROMPTS[mode]
        user_prompt = build_prompt(case["identity"], case["thread"], perception, mode)

        print(f"--- {mode.upper()} ---")
        print(f"SYSTEM: {sys_prompt[:80]}...")
        print(f"USER:\n{user_prompt}\n")

        result = get_monologue(sys_prompt, user_prompt)
        print(f"OUTPUT: {result}\n")

    # Now run a multi-turn simulation
    print(f"\n{'='*70}")
    print("MULTI-TURN SIMULATION (observational → restless)")
    print(f"{'='*70}\n")

    thread_entries = [
        "That ceiling keeps catching my eye.",
        "The light changed again, more yellow now.",
    ]
    session_min = 5

    for i in range(6):
        # Alternate modes
        mode = "restless" if i >= 3 else "observational"
        session_min += 2

        perception = get_perception(img_path)
        identity = f"{casual_time_string(session_min)} awake."
        if i == 0:
            identity += " Last drew the water stain on the ceiling."

        # Build flowing thread from last 3 entries
        recent = thread_entries[-3:]
        thread = "..." + " ".join(recent) if recent else ""

        user_prompt = build_prompt(identity, thread, perception, mode)
        sys_prompt = SYSTEM_PROMPTS[mode]

        result = get_monologue(sys_prompt, user_prompt)

        print(f"[{mode:>15}] {result}")
        thread_entries.append(result)


def find_latest_image() -> str:
    pattern = os.path.join(os.path.dirname(os.path.dirname(__file__)), "event_log", "*-images", "mood_*.jpg")
    images = sorted(glob.glob(pattern))
    if not images:
        print("No mood snapshots found. Pass an image path as argument.")
        sys.exit(1)
    return images[-1]


if __name__ == "__main__":
    img = sys.argv[1] if len(sys.argv) > 1 else find_latest_image()
    if not os.path.exists(img):
        print(f"Image not found: {img}")
        sys.exit(1)
    run_test(img)
