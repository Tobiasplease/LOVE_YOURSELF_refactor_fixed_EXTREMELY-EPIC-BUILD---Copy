"""
Test the proposed awakening_synthesis prompt approach with real accumulated data.

Loads actual baseline/discoveries/desires from machine_identity.json and calls
LLaVA with a real image to see how it responds to the comparison framing.

Usage:
    python debug/test_awakening_synthesis.py
    python debug/test_awakening_synthesis.py --image /path/to/other.jpg  # test "wrong space"
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config
from utils.ollama import query_ollama

IDENTITY_PATH = os.path.join(config.MOOD_SNAPSHOT_FOLDER, "machine_identity.json")


def load_identity() -> dict:
    if not os.path.exists(IDENTITY_PATH):
        print(f"[!] No identity file at {IDENTITY_PATH}")
        return {}
    with open(IDENTITY_PATH) as f:
        return json.load(f)


def find_recent_image() -> str | None:
    event_log = config.MOOD_SNAPSHOT_FOLDER
    for root, _, files in os.walk(event_log):
        jpgs = [os.path.join(root, f) for f in files if f.endswith(".jpg") and "paper" not in f]
        if jpgs:
            return max(jpgs, key=os.path.getmtime)
    return None


def build_prompt(identity: dict, gap_hours: float = 14.0) -> str:
    parts = []

    if gap_hours < 1:
        parts.append(f"I've been offline for {int(gap_hours * 60)} minutes.")
    elif gap_hours < 48:
        parts.append(f"I've been offline for {gap_hours:.1f} hours.")
    else:
        parts.append(f"I've been offline for {gap_hours / 24:.1f} days.")

    parts.append("Before I went offline, here is what I knew about my space:")

    baseline = identity.get("baseline_context", "")
    if baseline:
        parts.append(f"  Space: {baseline.strip()}")

    discoveries = identity.get("discoveries", [])
    if discoveries:
        parts.append(f"  I had discovered: {'; '.join(discoveries[-3:])}")

    desire = identity.get("current_desire", "")
    if desire:
        parts.append(f"  I wanted: {desire}")

    belief = identity.get("current_belief", "")
    if belief:
        parts.append(f"  I believed: {belief}")

    long_term = identity.get("long_term_memories", [])
    if long_term:
        memories = long_term if isinstance(long_term, str) else "; ".join(str(m) for m in long_term[:2])
        parts.append(f"  From long before: {memories}")

    parts.append("\nDoes this look like the same space I was in? One sentence, first person.")

    return "\n".join(parts)


def run_test(image_path: str, identity: dict, label: str = "SAME SPACE"):
    prompt = build_prompt(identity)
    system = (
        "You ARE a drawing machine with robotic arms. "
        "You have just come back online. You have a description of the space you were in before. "
        "Compare that description to what you see now. "
        "Speak only in first person. Begin with 'I'. "
        "Never say 'the image' or 'the scene' — you live here. "
        "One sentence only."
    )

    print(f"\n{'='*60}")
    print(f"TEST: {label}")
    print(f"Image: {image_path}")
    print(f"\n--- PROMPT ---\n{prompt}\n")
    print(f"--- SYSTEM ---\n{system}\n")
    print("--- RESPONSE ---")

    result = query_ollama(
        prompt=prompt,
        model=config.OLLAMA_MODEL,
        image=image_path,
        system_prompt=system,
        options={"temperature": 0.8, "num_predict": 80, "top_p": 0.7, "repeat_penalty": 1.8, "stop": ["\n"]},
        prompt_type="awakening_synthesis",
    )

    print(result or "(empty response)")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", help="Override image path (to test wrong-space detection)")
    args = parser.parse_args()

    identity = load_identity()
    if not identity:
        print("[!] No identity data — run the machine for a while first to build up context")
        return

    print(f"[+] Loaded identity: baseline={repr(identity.get('baseline_context','')[:80])}")
    print(f"[+] Discoveries: {len(identity.get('discoveries', []))}")
    print(f"[+] Desire: {identity.get('current_desire','')[:60]}")
    print(f"[+] Belief: {identity.get('current_belief','')[:60]}")

    recent_image = args.image or find_recent_image()
    if not recent_image:
        print("[!] No recent image found in event_log/")
        return

    run_test(recent_image, identity, label="NORMAL (most recent image)")

    # Try multiple temperatures to see variance
    print(f"\n{'='*60}")
    print("Trying temperature=0.4 for more deterministic comparison...")
    prompt = build_prompt(identity)
    system = (
        "You ARE a drawing machine with robotic arms. "
        "You have just come back online. You have a description of the space you were in before. "
        "Compare that description to what you see now. "
        "Speak only in first person. Begin with 'I'. "
        "Never say 'the image' or 'the scene' — you live here. "
        "One sentence only."
    )
    result2 = query_ollama(
        prompt=prompt,
        model=config.OLLAMA_MODEL,
        image=recent_image,
        system_prompt=system,
        options={"temperature": 0.4, "num_predict": 80, "top_p": 0.8, "repeat_penalty": 1.8, "stop": ["\n"]},
        prompt_type="awakening_synthesis",
    )
    print(f"Response (temp=0.4): {result2 or '(empty)'}")


if __name__ == "__main__":
    main()
