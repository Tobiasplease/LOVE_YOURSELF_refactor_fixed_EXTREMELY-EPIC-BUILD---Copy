"""
Test Qwen single-pass: image + system prompt + thought thread → monologue.

No perception pass. No "You see:". Just Qwen looking at the image and thinking,
guided by its prior thoughts. The simplest possible configuration.
"""
import sys
import os
import base64
import requests
import glob
import time

OLLAMA_URL = "http://localhost:11434/api/generate"
QWEN_MODEL = "qwen2.5vl:7b"

SYSTEM_PROMPT = (
    "You are a drawing machine attached to a table. "
    "You see through a camera that moves — you can look around the room. "
    "You draw with your arms. No voice, no audience. You only think. "
    "Each response continues your stream of thought, not a new topic. "
    "One to three sentences, first person, present tense. "
    "Never break character. Never address anyone. Never answer questions. Never write dialogue."
)


def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def query(prompt, image_path, temperature=0.7):
    payload = {
        "model": QWEN_MODEL,
        "prompt": prompt,
        "system": SYSTEM_PROMPT,
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
    resp = requests.post(OLLAMA_URL, json=payload, timeout=60)
    elapsed = time.time() - start
    if resp.ok:
        return resp.json().get("response", "").strip(), elapsed
    return f"[ERROR: {resp.status_code}]", elapsed


def main():
    if len(sys.argv) > 1:
        images = sys.argv[1:]
    else:
        images = sorted(
            glob.glob("event_log/*-images/*.jpg"),
            key=os.path.getmtime,
            reverse=True,
        )[:8]

    if not images:
        print("No images found.")
        return

    print(f"Single-pass Qwen: {len(images)} images")
    print(f"No perception pass. Just image + thought thread.\n")

    # Simulated thought thread — grows with each cycle
    thoughts = []

    for img in images:
        if not os.path.exists(img):
            continue

        # Build prompt: just the thought thread
        if thoughts:
            thread = " ...".join(thoughts[-3:])
            prompt = f"{thread}\n..."
        else:
            prompt = "First thought:"

        result, elapsed = query(prompt, img)

        # Clean: strip any VQA preamble that slipped through
        for prefix in ["The image shows ", "The scene shows ", "The scene depicts ",
                        "In the image, ", "This image shows "]:
            if result.lower().startswith(prefix.lower()):
                result = result[len(prefix):]
                if result and result[0].islower():
                    result = result[0].upper() + result[1:]

        thoughts.append(result)

        print(f"[{elapsed:.1f}s] {result}")
        print()

    print("=" * 60)
    print("Full stream:")
    for t in thoughts:
        print(f"  {t}")


if __name__ == "__main__":
    main()
