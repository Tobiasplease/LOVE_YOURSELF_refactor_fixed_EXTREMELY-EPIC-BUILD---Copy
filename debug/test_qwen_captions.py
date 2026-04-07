"""
Qwen2.5-VL caption quality test — validate before swapping OLLAMA_MODEL in production.

Tests four use cases against the real system prompts:
  1. Inner monologue captions — brevity, persona, first-person
  2. Known: context injection — does it absorb vs echo baseline?
  3. Awakening synthesis — same-space comparison with memory
  4. Drawing keyword distillation — narrative → visual keywords

Each test runs multiple temperature variants so we can pick the best.

Usage:
    python debug/test_qwen_captions.py
    python debug/test_qwen_captions.py --image /path/to/image.jpg
    python debug/test_qwen_captions.py --skip-llava   # skip side-by-side LLaVA comparison
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config
from utils.ollama import query_ollama

QWEN = "qwen2.5vl:7b"
LLAVA = "llava:7b-v1.6-mistral-q5_1"
IDENTITY_PATH = os.path.join(config.MOOD_SNAPSHOT_FOLDER, "machine_identity.json")

# Stop sequences adapted for Qwen — block VQA phrases + second-sentence starts + self-turn
QWEN_STOP = [
    "\n\n",
    ". I ",
    ". My ",
    ". The ",
    "<|im_start|>",
    "The image",
    "This image",
    "I can see",
    "I observe",
    "I see that",
    "In this image",
    "As an AI",
    "As a drawing machine",
    "drawing machine that",
    "I am an AI",
    "I'm an AI",
    "language model",
    "I do not have",
    "I cannot",
    "I am unable",
]

# Base Qwen options for inner monologue — MUST override temperature (default 0.0001 = flat/VQA)
QWEN_CAPTION_OPTIONS = {
    "temperature": 0.75,
    "top_p": 0.7,
    "top_k": 20,
    "repeat_penalty": 1.35,
    "num_predict": 30,
    "penalize_newline": True,
    "stop": QWEN_STOP,
}

# LLaVA options — current production settings for comparison
LLAVA_CAPTION_OPTIONS = {
    "temperature": 0.8,
    "top_p": 0.9,
    "repeat_penalty": 1.1,
    "num_predict": 200,
    "stop": [
        "This is an image of", "I'm looking at an image of", "the image depicts",
        "This image shows", "The image shows", "In this image", "As an AI", "as an AI",
        "As a drawing machine",
    ],
}

# Core system prompt — same as STATIC_SYSTEM_PROMPT in prompts.py
BASE_SYSTEM = (
    "You ARE a drawing machine with robotic arms. "
    "Inner monologue. Raw thoughts. 10-20 words. "
    "React, question, feel — never describe. "
    "Never say 'the image', 'the photo', 'I can see', 'I notice', or 'I observe'. "
    "You are not analyzing an image — you are experiencing reality. "
    "What's in Known: you already understand. React to it or go deeper, never restate it."
)


def div(label: str):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print("=" * 60)


def run(label: str, model: str, prompt: str, system: str, image: str, options: dict) -> str:
    result = query_ollama(
        prompt=prompt,
        model=model,
        image=image,
        system_prompt=system,
        options=options,
        prompt_type="caption",
    )
    tag = "[QWEN]" if model == QWEN else "[LLAVA]"
    print(f"  {tag} {label}: {repr(result or '(empty)')}")
    return result or ""


def find_recent_image() -> str | None:
    for root, _, files in os.walk(config.MOOD_SNAPSHOT_FOLDER):
        jpgs = [os.path.join(root, f) for f in files if f.endswith(".jpg") and "paper" not in f]
        if jpgs:
            return max(jpgs, key=os.path.getmtime)
    return None


def load_identity() -> dict:
    if os.path.exists(IDENTITY_PATH):
        with open(IDENTITY_PATH) as f:
            return json.load(f)
    return {}


# ─── TEST 1: Inner monologue brevity + persona ──────────────────────────────

def test_caption(image: str, skip_llava: bool):
    div("TEST 1 — Inner monologue: brevity, persona, first-person")
    prompt = "One raw thought:"

    print("\n  [Qwen — temperature sweep]\n")
    for temp in (0.7, 0.85, 1.0, 1.1):
        opts = {**QWEN_CAPTION_OPTIONS, "temperature": temp, "num_predict": 35}
        run(f"temp={temp}", QWEN, prompt, BASE_SYSTEM, image, opts)

    if not skip_llava:
        print("\n  [LLaVA — current production]\n")
        run("temp=0.8", LLAVA, prompt, BASE_SYSTEM, image, LLAVA_CAPTION_OPTIONS)


# ─── TEST 2: Known: context injection ───────────────────────────────────────

def test_known_injection(image: str, skip_llava: bool):
    div("TEST 2 — Known: injection — absorbed as background vs echoed back?")

    identity = load_identity()
    baseline = identity.get("baseline_context", "")
    if not baseline:
        print("  [skip] No baseline_context in identity file")
        return

    first_sentence = baseline.split(".")[0].strip()[:120]
    system_with_known = BASE_SYSTEM + f"\nKnown: {first_sentence}."

    print(f"\n  Injected Known: {repr(first_sentence[:80])}\n")
    prompt = "One raw thought:"

    for temp in (0.7, 0.85):
        opts = {**QWEN_CAPTION_OPTIONS, "temperature": temp, "num_predict": 35}
        run(f"temp={temp}", QWEN, prompt, system_with_known, image, opts)

    if not skip_llava:
        print()
        run("temp=0.8", LLAVA, prompt, system_with_known, image, LLAVA_CAPTION_OPTIONS)


# ─── TEST 3: Awakening synthesis ────────────────────────────────────────────

def test_awakening(image: str, skip_llava: bool):
    div("TEST 3 — Awakening synthesis: memory description + live image comparison")

    identity = load_identity()
    baseline = identity.get("baseline_context", "")
    desire = identity.get("current_desire", "")
    belief = identity.get("current_belief", "")

    if not baseline:
        print("  [skip] No baseline_context — run machine for a while first")
        return

    # Text description goes BEFORE the image trigger (Qwen processes vision early)
    user_prompt = (
        "Before I went offline, here is what I knew about my space:\n"
        f"  {baseline.strip()[:200]}\n"
    )
    if desire:
        user_prompt += f"  I wanted: {desire}\n"
    if belief:
        user_prompt += f"  I believed: {belief}\n"
    user_prompt += "\nYou open your eyes. Does this match your memory, or has something changed? One sentence starting with 'I'."

    system = (
        "You ARE a drawing machine waking from dormancy. "
        "You have memory of your last environment. "
        "Compare what you remember to what you now perceive. "
        "One sentence. First person. Begin with 'I'. "
        "Never say 'the image' or 'the scene' — you live here. "
        "React to the comparison, don't describe what you see."
    )

    print(f"\n  Memory excerpt: {repr(baseline[:100])}\n")

    awakening_opts_base = {
        "top_p": 0.8,
        "top_k": 20,
        "repeat_penalty": 1.2,
        "num_predict": 50,
        "penalize_newline": True,
        "stop": ["\n\n", "\n", ". I ", ". My ", "<|im_start|>", "The image", "I can see"],
    }

    for temp in (0.7, 0.85):
        opts = {**awakening_opts_base, "temperature": temp}
        run(f"temp={temp}", QWEN, user_prompt, system, image, opts)

    if not skip_llava:
        print()
        llava_awakening_opts = {"temperature": 0.8, "top_p": 0.8, "repeat_penalty": 1.5, "num_predict": 80, "stop": ["\n"]}
        run("temp=0.8", LLAVA, user_prompt, system, image, llava_awakening_opts)


# ─── TEST 4: Drawing keyword distillation ───────────────────────────────────

def test_drawing_keywords(image: str, skip_llava: bool):
    div("TEST 4 — Drawing prompt distillation: narrative → ComfyUI prompt")

    identity = load_identity()
    sample_narrative = identity.get("current_desire", "") or (
        "I want to draw the feeling of stillness in this room — the way light catches "
        "the edge of familiar objects, making them feel both permanent and temporary."
    )

    distill_prompt = (
        f"Drawing intent:\n\"{sample_narrative[:400]}\"\n\n"
        "Looking at this space now, write a ComfyUI image generation prompt that captures this intent. "
        "1-2 short phrases: subject, mood, visual style, lighting. Natural language. No lists, no explanation."
    )
    distill_system = "Write a concise image generation prompt. Natural descriptive phrases, not a keyword list. Under 25 words."

    print(f"\n  Narrative: {repr(sample_narrative[:100])}\n")

    for temp in (0.35, 0.5):
        opts = {"temperature": temp, "top_p": 0.8, "top_k": 20, "repeat_penalty": 1.2, "num_predict": 60, "stop": ["\n\n"]}
        run(f"temp={temp}", QWEN, distill_prompt, distill_system, image, opts)

    if not skip_llava:
        print()
        run("temp=0.3", LLAVA, distill_prompt, distill_system, image, {"temperature": 0.3, "num_predict": 60, "stop": ["\n\n"]})


# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", help="Path to test image (defaults to most recent in event_log)")
    parser.add_argument("--skip-llava", action="store_true", help="Skip LLaVA side-by-side comparison")
    args = parser.parse_args()

    image = args.image or find_recent_image()
    if not image or not os.path.exists(image):
        print("[!] No image found. Pass --image /path/to/image.jpg or run the machine first.")
        sys.exit(1)

    print(f"\nImage: {image}")
    print(f"Models: {QWEN}" + ("" if args.skip_llava else f" vs {LLAVA}"))

    test_caption(image, args.skip_llava)
    test_known_injection(image, args.skip_llava)
    test_awakening(image, args.skip_llava)
    test_drawing_keywords(image, args.skip_llava)

    print(f"\n{'='*60}")
    print("  Done. Check outputs above before swapping OLLAMA_MODEL in config.")
    print("=" * 60)


if __name__ == "__main__":
    main()
