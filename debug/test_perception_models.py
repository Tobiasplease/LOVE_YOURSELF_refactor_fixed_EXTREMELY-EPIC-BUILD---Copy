"""
Test script: Compare LLaVA vs Qwen2.5-VL for perception quality.
Runs both models against the same images with the same prompts,
prints results side-by-side for evaluation.

Usage:
    python debug/test_perception_models.py [image_path]

If no image_path given, uses the most recent mood snapshot.
"""

import base64
import glob
import json
import os
import sys
import time

import requests

OLLAMA_URL = "http://localhost:11434/api/generate"

MODELS = {
    "llava": "llava:7b-v1.6-mistral-q5_1",
    "qwen": "qwen2.5vl:7b",
}

# Perception prompts to test — these are the actual prompts used in the pipeline
PERCEPTION_PROMPTS = [
    {
        "name": "default",
        "system": (
            "Describe what you see. Be brief, specific, and factual. "
            "Be concrete, not vague. Name what you actually see, not 'various objects' or 'items'."
            " Never say 'image', 'photo', or 'photograph' — you are looking at a real scene, not an image."
        ),
        "prompt": "What do you see?",
    },
    {
        "name": "person",
        "system": (
            "Describe what you see. Focus on the person — their appearance, posture, "
            "what they are doing. Be brief and specific. Two or three sentences."
            " Be concrete, not vague. Name what you actually see, not 'various objects' or 'items'."
            " Never say 'image', 'photo', or 'photograph' — you are looking at a real scene, not an image."
        ),
        "prompt": "Describe the person — what they look like, what they are doing.",
    },
    {
        "name": "detail",
        "system": (
            "Pick one detail that stands out and describe it closely — its shape, texture, "
            "how light falls on it. Be brief and specific. Two or three sentences."
            " Be concrete, not vague. Name what you actually see, not 'various objects' or 'items'."
            " Never say 'image', 'photo', or 'photograph' — you are looking at a real scene, not an image."
        ),
        "prompt": "What single detail stands out most right now?",
    },
    {
        "name": "change",
        "system": (
            "Describe what you see. Name specific objects, surfaces, and how the space is lit. "
            "Be brief and specific. Two or three sentences."
            " Be concrete, not vague. Name what you actually see, not 'various objects' or 'items'."
            " Never say 'image', 'photo', or 'photograph' — you are looking at a real scene, not an image."
        ),
        "prompt": "What has changed? Describe what looks different now.",
    },
]


def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def query_model(model: str, prompt: str, image_path: str, system_prompt: str, temperature: float = 0.7) -> tuple:
    """Query a model and return (response_text, duration_seconds)."""
    image_b64 = encode_image(image_path)

    payload = {
        "model": model,
        "prompt": prompt,
        "images": [image_b64],
        "stream": False,
        "system": system_prompt,
        "options": {
            "temperature": temperature,
            "top_p": 0.8,
            "num_predict": 150,
            "num_ctx": 4096,
        },
    }

    start = time.time()
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
        resp.raise_for_status()
        result = resp.json().get("response", "").strip()
        duration = time.time() - start
        return result, duration
    except Exception as e:
        duration = time.time() - start
        return f"[ERROR: {e}]", duration


def check_quality(response: str) -> dict:
    """Check response for common issues."""
    lower = response.lower()
    issues = []

    # VQA register
    vqa_phrases = ["the image", "this image", "in this image", "the photo", "photograph", "the picture", "appears to be taken", "provided photo"]
    for phrase in vqa_phrases:
        if phrase in lower:
            issues.append(f"VQA: '{phrase}'")

    # Vague language
    vague_phrases = ["various objects", "various items", "some items", "different items", "miscellaneous", "several things"]
    for phrase in vague_phrases:
        if phrase in lower:
            issues.append(f"VAGUE: '{phrase}'")

    # Hedging
    hedge_phrases = ["appears to be", "seems to be", "looks like it might", "could be", "possibly", "suggesting that"]
    for phrase in hedge_phrases:
        if phrase in lower:
            issues.append(f"HEDGE: '{phrase}'")

    # Lists (numbered)
    if any(f"\n{i}." in response for i in range(1, 10)):
        issues.append("FORMAT: numbered list")

    return {
        "has_vqa": any("VQA" in i for i in issues),
        "has_vague": any("VAGUE" in i for i in issues),
        "has_hedge": any("HEDGE" in i for i in issues),
        "has_list": any("FORMAT" in i for i in issues),
        "issues": issues,
        "word_count": len(response.split()),
    }


def run_test(image_path: str):
    print(f"\n{'='*80}")
    print(f"PERCEPTION MODEL COMPARISON TEST")
    print(f"Image: {image_path}")
    print(f"{'='*80}\n")

    for test in PERCEPTION_PROMPTS:
        print(f"\n{'─'*80}")
        print(f"TEST: {test['name'].upper()}")
        print(f"Prompt: {test['prompt']}")
        print(f"{'─'*80}\n")

        for label, model_name in MODELS.items():
            # Use lower temperature for qwen (its default is near-zero, 0.3 is reasonable for perception)
            temp = 0.3 if "qwen" in model_name else 0.7

            print(f"  [{label.upper()}] ({model_name}, temp={temp})")
            response, duration = query_model(model_name, test["prompt"], image_path, test["system"], temperature=temp)
            quality = check_quality(response)

            # Truncate for display
            display = response[:300]
            if len(response) > 300:
                display += "..."

            print(f"  Time: {duration:.1f}s | Words: {quality['word_count']}")
            if quality["issues"]:
                print(f"  Issues: {', '.join(quality['issues'])}")
            else:
                print(f"  Issues: NONE ✓")
            print(f"  Response: {display}")
            print()


def find_latest_image() -> str:
    """Find the most recent mood snapshot."""
    pattern = os.path.join(os.path.dirname(os.path.dirname(__file__)), "event_log", "*-images", "mood_*.jpg")
    images = sorted(glob.glob(pattern))
    if not images:
        print("No mood snapshots found. Pass an image path as argument.")
        sys.exit(1)
    return images[-1]


if __name__ == "__main__":
    if len(sys.argv) > 1:
        img = sys.argv[1]
    else:
        img = find_latest_image()

    if not os.path.exists(img):
        print(f"Image not found: {img}")
        sys.exit(1)

    run_test(img)
