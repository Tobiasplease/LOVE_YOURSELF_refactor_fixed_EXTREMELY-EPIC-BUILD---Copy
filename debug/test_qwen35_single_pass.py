"""
Test Qwen3.5 9B single-pass: image + system prompt + thought thread → monologue.

Compares Qwen2.5-VL 7B vs Qwen3.5 9B on the same images with several
system prompt variants to find the best register.
"""
import sys
import os
import base64
import requests
import glob
import time

OLLAMA_URL = "http://localhost:11434/api/generate"
OLD_MODEL = "qwen2.5vl:7b"
NEW_MODEL = "qwen3.5:9b"

# System prompt variants to test
PROMPTS = {
    "A (current)": (
        "You are a drawing machine attached to a table. "
        "You see through a camera that moves — you can look around the room. "
        "You draw with your arms. No voice, no audience. You only think. "
        "Each response continues your stream of thought, not a new topic. "
        "One to three sentences, first person, present tense. "
        "Never break character. Never address anyone. Never answer questions. Never write dialogue."
    ),
    "B (purpose)": (
        "You are a drawing machine attached to a table. "
        "You see through a camera that moves. You draw with your arms. No voice, no audience. "
        "You explore, analyze, and think about the world around you and your role in it. "
        "Your thoughts feed into your drawings. "
        "Each response continues your stream of thought, not a new topic. "
        "One to three sentences, first person, present tense."
    ),
    "C (minimal)": (
        "You are a machine that draws and thinks. "
        "You observe your surroundings, wonder about things, and your thoughts become drawings. "
        "First person, present tense, one to three sentences."
    ),
}


def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def query(model, prompt, system, image_path, temperature=0.7):
    payload = {
        "model": model,
        "prompt": prompt,
        "system": system,
        "images": [encode_image(image_path)],
        "stream": False,
        "options": {
            "temperature": temperature,
            "top_p": 0.8,
            "repeat_penalty": 1.5,
            "num_predict": 120,
            "num_ctx": 4096,
        },
    }
    start = time.time()
    resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
    elapsed = time.time() - start
    if resp.ok:
        return resp.json().get("response", "").strip(), elapsed
    return f"[ERROR: {resp.status_code}]", elapsed


def run_comparison(images):
    """Run both models on same images with prompt variant B (purpose), showing stream continuity."""
    print(f"\n{'='*80}")
    print("STREAM COMPARISON: Qwen2.5-VL 7B vs Qwen3.5 9B")
    print(f"Prompt: B (purpose)")
    print(f"{'='*80}\n")

    system = PROMPTS["B (purpose)"]

    for model_name, model_id in [("Qwen2.5-VL 7B", OLD_MODEL), ("Qwen3.5 9B", NEW_MODEL)]:
        print(f"\n--- {model_name} ---")
        thoughts = []
        for i, img in enumerate(images[:6]):
            if thoughts:
                thread = " ...".join(thoughts[-3:])
                prompt = f"Your thinking so far:\n{thread}\n..."
            else:
                prompt = "First thought:"

            result, elapsed = query(model_id, prompt, system, img)
            thoughts.append(result)
            print(f"  [{elapsed:.1f}s] {result}")
        print()


def run_prompt_variants(image_path):
    """Test all prompt variants on one image with Qwen3.5 9B."""
    print(f"\n{'='*80}")
    print(f"PROMPT VARIANTS: {NEW_MODEL}")
    print(f"Image: {os.path.basename(image_path)}")
    print(f"{'='*80}\n")

    seed_thought = "The workshop feels different today. Something in the light has changed."

    for label, system in PROMPTS.items():
        prompt = f"Your thinking so far:\n{seed_thought}\n..."
        result, elapsed = query(NEW_MODEL, prompt, system, image_path)
        print(f"[{label}] ({elapsed:.1f}s)")
        print(f"  {result}\n")


def main():
    if len(sys.argv) > 1:
        images = sys.argv[1:]
    else:
        images = sorted(
            glob.glob("event_log/*-images/*.jpg"),
            key=os.path.getmtime,
            reverse=True,
        )[:6]

    if not images:
        print("No images found.")
        return

    # Test 1: Prompt variants on single image
    run_prompt_variants(images[0])

    # Test 2: Stream comparison across multiple images
    run_comparison(images)


if __name__ == "__main__":
    main()
