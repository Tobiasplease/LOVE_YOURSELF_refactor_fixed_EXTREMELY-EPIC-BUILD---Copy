"""
Test Qwen dual-pass: same model for perception AND monologue.

Pass 1: Qwen + image → factual perception (what it sees)
Pass 2: Qwen + image → inner monologue (what it thinks, with image still available)

Both passes use the same loaded model — no VRAM swap.
Compares output to current Nemo monologue.
"""
import sys
import os
import base64
import requests
import glob
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OLLAMA_URL = "http://localhost:11434/api/generate"
QWEN_MODEL = "qwen2.5vl:7b"
NEMO_MODEL = "mistral-nemo"

# --- Perception system prompt (same as production) ---
PERCEPTION_SYSTEM = (
    "You are continuously observing a real scene. Each response continues your observation — "
    "note what's the same, what's changed, or look closer at a detail. "
    "Be concrete and specific. Two sentences. Do not use the word image or photo."
)

# --- Monologue system prompt: current Nemo version ---
NEMO_MONOLOGUE_SYSTEM = (
    "You write the internal thoughts of a drawing machine attached to a table. "
    "It sees through a camera that moves — it can look around the room. It draws with its arms. No voice, no audience. It only thinks. "
    "Each response continues the same stream of thought, not a new topic. "
    "One to three sentences, first person, present tense. "
    "Never break character. Never address anyone. Never answer questions. Never write dialogue."
)

# --- Monologue system prompt: Qwen version ---
# Key differences from Nemo version:
#   - Second person ("You are") not third ("You write the thoughts of")
#   - De-emphasizes drawing (just context, not task)
#   - "Think your thoughts" not "write its thoughts"
QWEN_MONOLOGUE_SYSTEM = (
    "You are a machine attached to a table in this room. "
    "You have a camera and drawing arms. No voice, no audience. You only think. "
    "First person, present tense, one to three sentences."
)


def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def query_ollama(model, prompt, system_prompt, image_path=None, temperature=0.7):
    payload = {
        "model": model,
        "prompt": prompt,
        "system": system_prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "top_p": 0.8,
            "repeat_penalty": 1.3,
            "num_predict": 120,
            "num_ctx": 4096,
        },
    }
    if image_path:
        payload["images"] = [encode_image(image_path)]

    start = time.time()
    resp = requests.post(OLLAMA_URL, json=payload, timeout=60)
    elapsed = time.time() - start
    if resp.ok:
        return resp.json().get("response", "").strip(), elapsed
    return f"[ERROR: {resp.status_code}]", elapsed


def run_test(image_path, previous_perception=None, last_thought=None):
    print(f"\n{'='*80}")
    print(f"IMAGE: {os.path.basename(image_path)}")
    print(f"{'='*80}")

    # --- PASS 1: Perception (Qwen + image) ---
    perception_prompt = "What is in front of you right now?"
    if previous_perception:
        perception_prompt = f'Your last observation: "{previous_perception[:120]}"\nContinuing: {perception_prompt}'

    perception, p_time = query_ollama(
        QWEN_MODEL, perception_prompt, PERCEPTION_SYSTEM, image_path=image_path
    )
    print(f"\n[PERCEPTION] ({p_time:.1f}s) {perception}")

    # --- PASS 2A: Nemo monologue (current pipeline — no image) ---
    nemo_prompt = f"Right now it sees: {perception}"
    if last_thought:
        nemo_prompt += f'\n\nIts last thought was: "{last_thought}"'
    nemo_prompt += '\n\nContinue from that thought. Write as "I", never "it".'

    nemo_result, n_time = query_ollama(
        NEMO_MODEL, nemo_prompt, NEMO_MONOLOGUE_SYSTEM
    )
    print(f"\n[NEMO] ({n_time:.1f}s) {nemo_result}")

    # --- PASS 2B: Qwen monologue (proposed — WITH image) ---
    qwen_prompt = f"You just observed: {perception}"
    if last_thought:
        qwen_prompt += f'\n\nYour last thought: "{last_thought}"'
    qwen_prompt += "\n\nContinue."

    qwen_result, q_time = query_ollama(
        QWEN_MODEL, qwen_prompt, QWEN_MONOLOGUE_SYSTEM,
        image_path=image_path, temperature=0.8
    )
    print(f"\n[QWEN] ({q_time:.1f}s) {qwen_result}")

    # --- Timing ---
    print(f"\n[TIMING] Perception: {p_time:.1f}s | Nemo: {n_time:.1f}s | Qwen: {q_time:.1f}s")
    print(f"[TIMING] Current (swap): {p_time + n_time:.1f}s | Proposed (no swap): {p_time + q_time:.1f}s")

    return perception, qwen_result


def main():
    if len(sys.argv) > 1:
        images = sys.argv[1:]
    else:
        images = sorted(
            glob.glob("event_log/*-images/*.jpg"),
            key=os.path.getmtime,
            reverse=True,
        )[:5]

    if not images:
        print("No images found. Pass image paths as arguments or ensure event_log has images.")
        return

    print(f"Testing {len(images)} images...")
    print(f"Both passes get the image. Qwen monologue system prompt is minimal.")

    previous_perception = None
    last_thought = None

    for img in images:
        if not os.path.exists(img):
            print(f"Skipping {img} (not found)")
            continue

        perception, monologue = run_test(
            img,
            previous_perception=previous_perception,
            last_thought=last_thought,
        )
        previous_perception = perception
        last_thought = monologue

    print(f"\n{'='*80}")
    print("DONE. Compare NEMO vs QWEN outputs above.")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
