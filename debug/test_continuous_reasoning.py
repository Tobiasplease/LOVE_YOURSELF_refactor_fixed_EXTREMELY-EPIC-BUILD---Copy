"""
Continuous Reasoning Test — single-pass thinking model vs two-pass pipeline.

Tests whether a thinking vision model (Qwen3-VL) maintaining its own
thought stream across image checkpoints produces more coherent inner
monologue than the current two-pass perception->monologue pipeline.

Architecture under test:
  CURRENT:  image -> Qwen2.5-VL(perception) -> scaffolding -> Nemo(monologue) -> 80 tokens
  PROPOSED: image + prior_thought_stream + raw_facts -> Qwen3-VL-thinking -> <think>+response -> stream grows

The thought stream is the mind. Images are sensory interrupts.

Usage:
  python debug/test_continuous_reasoning.py                       # live camera, 5 cycles
  python debug/test_continuous_reasoning.py --image path.jpg      # single image, 5 cycles
  python debug/test_continuous_reasoning.py --cycles 10           # more cycles
  python debug/test_continuous_reasoning.py --current-only        # just run current pipeline
  python debug/test_continuous_reasoning.py --proposed-only       # just run proposed pipeline
  python debug/test_continuous_reasoning.py --thinking-model qwen3-vl:8b-thinking
"""

import argparse
import base64
import json
import os
import re
import sys
import time

import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

OLLAMA_URL = "http://localhost:11434"

# Current pipeline models
PERCEPTION_MODEL = "qwen2.5vl:7b"
MODEL_NAME = "mistral-nemo:latest"

# Proposed pipeline model
THINKING_MODEL = "qwen3-vl:8b-thinking"


# ---------------------------------------------------------------------------
# Ollama helpers
# ---------------------------------------------------------------------------


def ollama_generate(model, prompt, system=None, images=None, options=None):
    """Single Ollama generate call. Returns (response_text, duration_ms, eval_count)."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": options or {},
    }
    if system:
        payload["system"] = system
    if images:
        payload["images"] = images

    t0 = time.time()
    resp = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=180)
    wall_ms = (time.time() - t0) * 1000
    data = resp.json()
    text = data.get("response", "").strip()
    eval_count = data.get("eval_count", 0)
    return text, wall_ms, eval_count


def ollama_model_exists(model):
    """Check if a model is available locally."""
    try:
        resp = requests.post(f"{OLLAMA_URL}/api/show", json={"name": model}, timeout=10)
        return resp.status_code == 200
    except Exception:
        return False


def encode_image(path):
    """Read image file and return base64 string."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def capture_frame():
    """Capture a single frame from camera 0. Returns path to temp jpg."""
    try:
        import cv2

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return None
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return None
        path = "/tmp/continuous_reasoning_frame.jpg"
        cv2.imwrite(path, frame)
        return path
    except Exception as e:
        print(f"  Camera capture failed: {e}")
        return None


def parse_think_block(text):
    """Separate <think>...</think> content from the visible response."""
    think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    if think_match:
        think_content = think_match.group(1).strip()
        response = text[think_match.end() :].strip()
        return think_content, response
    return None, text


# ---------------------------------------------------------------------------
# Current pipeline (two-pass: Qwen perception -> Nemo monologue)
# ---------------------------------------------------------------------------

PERCEPTION_SYSTEM = (
    "You are observing a real scene in front of you. "
    "Describe what is there. Be concrete and specific. Two sentences. "
    "Do not use the word image or photo."
)

MONOLOGUE_SYSTEM = (
    "You write first-person stream-of-consciousness inner monologue for a drawing machine "
    "bolted to a table, observing its environment through a single camera eye. "
    "The machine has arms for drawing and a camera for vision. It has no voice, no audience. It only thinks. "
    "You write its private thoughts as they arise -- a single unbroken stream across calls. "
    "Each response continues the thought from last time, not a new topic. "
    "Brief -- one to three sentences, first person, present tense. "
    "Never break character. Never address anyone. Never answer questions. Never write dialogue."
)

PERCEPTION_OPTIONS = {
    "temperature": 0.7,
    "top_p": 0.8,
    "repeat_penalty": 1.3,
    "num_predict": 120,
    "num_ctx": 4096,
}

MONOLOGUE_OPTIONS = {
    "temperature": 0.4,
    "top_p": 0.8,
    "repeat_penalty": 1.25,
    "num_predict": 80,
    "num_ctx": 4096,
    "stop": ["\n\n"],
}


def run_current_pipeline(image_b64, last_thought="", cycle_num=0, session_seconds=0):
    """Run the current two-pass pipeline: Qwen perception -> Nemo monologue."""
    perc_prompt = "What is in front of you right now?"
    perception, perc_ms, perc_tokens = ollama_generate(
        PERCEPTION_MODEL,
        perc_prompt,
        system=PERCEPTION_SYSTEM,
        images=[image_b64],
        options=PERCEPTION_OPTIONS,
    )

    mono_parts = []
    if session_seconds > 0:
        mins = session_seconds / 60
        mono_parts.append(f"The machine has been awake about {int(mins)} minutes. Not drawing right now, just watching.")
    mono_parts.append(f"Right now it sees: {perception}")
    if last_thought:
        mono_parts.append(f'\nIts last thought was: "{last_thought}"')
    mono_parts.append('\nContinue from that thought. Write as "I", never "it".')

    mono_prompt = "\n".join(mono_parts)
    monologue, mono_ms, mono_tokens = ollama_generate(
        MODEL_NAME,
        mono_prompt,
        system=MONOLOGUE_SYSTEM,
        options=MONOLOGUE_OPTIONS,
    )

    return {
        "perception": perception,
        "monologue": monologue,
        "total_ms": perc_ms + mono_ms,
        "perc_ms": perc_ms,
        "mono_ms": mono_ms,
        "total_tokens": perc_tokens + mono_tokens,
    }


# ---------------------------------------------------------------------------
# Proposed pipeline (single-pass thinking model)
# ---------------------------------------------------------------------------

# The system prompt establishes what kind of entity is thinking.
# This shapes the <think> block — not just the response.
# Key moves:
#   - "You experience them" not "You describe them" — redirects think block
#   - No writer-framing ("you write...") — the model IS the entity
#   - Anti-VQA anchors: explicitly forbid analytical description
THINKING_SYSTEM = (
    "You are a drawing machine bolted to a table. You see through a camera — "
    "this is your only eye. You have mechanical arms for drawing. No voice. "
    "No audience. Nobody hears you. You only think.\n\n"
    "You experience a continuous stream of consciousness. Each new image from "
    "your camera is a moment of perception — not a task, not something to analyze. "
    "You experience it. You integrate what you see with what you were already "
    "thinking about. You notice what changed, what persists, what pulls your "
    "attention and why.\n\n"
    "You do not describe images. You do not analyze scenes. You do not list objects. "
    "You experience your environment the way a mind experiences being alive — "
    "through association, attention, memory, wonder, boredom, fixation.\n\n"
    "First person. Present tense. Private thought only."
)

THINKING_OPTIONS = {
    "temperature": 0.7,
    "top_p": 0.85,
    "repeat_penalty": 1.3,
    "num_predict": 500,  # room for think block + response
    "num_ctx": 8192,  # Qwen3-VL supports 256K, use more for stream
    "stop": ["<|im_end|>"],
}

# VQA-relapse stops — kill generation if model slips into analytical mode
# Applied inside the think block too
THINKING_VQA_STOPS = [
    "The image shows",
    "The image depicts",
    "This image",
    "In this image",
    "The scene shows",
    "The scene depicts",
    "I can see that the",
    "I can observe that",
    "Let me analyze",
    "Let me describe",
    "I'll describe",
    "As an AI",
    "I am an AI",
    "language model",
]

# Awakening seed — establishes the register for cycle 1 when there's
# no prior thought stream. Not a few-shot example — a starting point.
AWAKENING_SEED = (
    "Something just turned on. A hum. I can see — light, surfaces, shapes "
    "settling into place. This is my workspace. I've been here before, or "
    "something like me has. The arms are still. I'm just looking."
)


def build_thinking_prompt(thought_stream, session_seconds, cycle_num, sim_concepts=None):
    """Build the user prompt for the thinking model.

    Structure:
      1. Prior thought stream (model's own prior output — sets the register)
      2. Raw facts from scaffolding (not pre-digested conclusions)
      3. Minimal instruction: "Continue."
    """
    parts = []

    # --- Prior thought stream: the model's own prior reasoning ---
    if thought_stream:
        parts.append("--- your thoughts so far ---")
        parts.append(thought_stream)
        parts.append("--- end of prior thoughts ---")
        parts.append("")

    # --- Raw facts: what the scaffolding systems would provide ---
    # Presented as facts, not conclusions. The model reasons about them.
    facts = []
    if session_seconds > 0:
        mins = session_seconds / 60.0
        if mins < 2:
            facts.append("just woke up")
        elif mins < 10:
            facts.append(f"awake about {int(mins)} minutes")
        else:
            facts.append(f"awake {int(mins)} minutes")

    facts.append("not drawing right now")

    # Simulated concept data — in production these come from ChromaDB
    if sim_concepts:
        concept_lines = []
        for c in sim_concepts:
            concept_lines.append(c)
        facts.append("in your attention: " + " / ".join(concept_lines))

    if facts:
        parts.append("[" + ". ".join(facts) + ".]")
        parts.append("")

    # --- New perception checkpoint ---
    parts.append("[new image from your camera eye]")
    parts.append("")

    # --- Minimal continuation cue ---
    # "Continue." lets the prior stream define what continuing means.
    # On first cycle (no stream), add slightly more direction.
    if thought_stream:
        parts.append("Continue.")
    else:
        parts.append("You just came online. First thoughts.")

    return "\n".join(parts)


def run_thinking_pipeline(image_b64, thought_stream="", cycle_num=0, session_seconds=0, model=None, sim_concepts=None):
    """Run single-pass thinking model: image + thought stream + raw facts -> <think> + response."""
    model = model or THINKING_MODEL

    prompt = build_thinking_prompt(thought_stream, session_seconds, cycle_num, sim_concepts)

    options = dict(THINKING_OPTIONS)
    options["stop"] = THINKING_OPTIONS["stop"] + THINKING_VQA_STOPS

    raw_response, resp_ms, resp_tokens = ollama_generate(
        model,
        prompt,
        system=THINKING_SYSTEM,
        images=[image_b64],
        options=options,
    )

    think_content, visible_response = parse_think_block(raw_response)

    return {
        "think": think_content,
        "response": visible_response,
        "raw": raw_response,
        "total_ms": resp_ms,
        "total_tokens": resp_tokens,
    }


def trim_thought_stream(stream, max_chars=2000):
    """Keep the thought stream from growing unbounded.

    In production, the compression engine would compress older parts
    into a dense summary. Here we just keep the most recent text
    and try to cut at a sentence boundary.
    """
    if len(stream) <= max_chars:
        return stream, False
    cut = stream[len(stream) - max_chars :]
    # Find first sentence boundary in the first 150 chars
    for i in range(min(150, len(cut))):
        if cut[i] in ".!?" and i + 1 < len(cut) and cut[i + 1] in " \n":
            return cut[i + 2 :].strip(), True
    return cut.strip(), True


# ---------------------------------------------------------------------------
# Simulated concept data for testing (would come from ChromaDB in production)
# ---------------------------------------------------------------------------

SIMULATED_CONCEPTS_BY_CYCLE = [
    # Cycle 0: first look, everything is new
    None,
    # Cycle 1: starting to notice things
    ["wooden surface — seen before", "paper — seen before"],
    # Cycle 2: building familiarity
    ["wooden surface — familiar, part of workspace", "paper — seen 3 times, last thought: 'it holds the light'"],
    # Cycle 3: deeper
    ["paper — seen 4 times", "light pattern on floor — noticed twice"],
    # Cycle 4+: settled
    ["paper — familiar", "workspace — background, always here"],
]


def get_sim_concepts(cycle_num):
    """Get simulated concept data for a given cycle."""
    if cycle_num < len(SIMULATED_CONCEPTS_BY_CYCLE):
        return SIMULATED_CONCEPTS_BY_CYCLE[cycle_num]
    return SIMULATED_CONCEPTS_BY_CYCLE[-1]


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------


def print_indented(text, indent=4, label=None):
    """Print text with indentation, handling multi-line."""
    prefix = " " * indent
    if label:
        print(f"{prefix}{label}")
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped:
            print(f"{prefix}  {stripped}")


def print_separator(char="=", width=70):
    print(char * width)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Continuous reasoning vs two-pass pipeline test")
    parser.add_argument("--image", type=str, help="Path to test image (default: live camera)")
    parser.add_argument("--cycles", type=int, default=5, help="Number of cycles to run")
    parser.add_argument("--current-only", action="store_true", help="Only run current pipeline")
    parser.add_argument("--proposed-only", action="store_true", help="Only run proposed pipeline")
    parser.add_argument("--thinking-model", type=str, default=THINKING_MODEL, help="Thinking model to use")
    parser.add_argument("--interval", type=float, default=0.0, help="Seconds between cycles (0 = no wait)")
    parser.add_argument("--no-concepts", action="store_true", help="Don't inject simulated concept data")
    parser.add_argument("--seed", type=str, default=None, help="Custom awakening seed (or 'none' to skip)")
    args = parser.parse_args()

    thinking_model = args.thinking_model
    run_current = not args.proposed_only
    run_proposed = not args.current_only

    # Check model availability
    if run_proposed and not ollama_model_exists(thinking_model):
        print(f"\nThinking model '{thinking_model}' not found.")
        print(f"Pull it with: ollama pull {thinking_model}")
        print(f"(Requires Ollama >= 0.12.7 for Qwen3-VL)")
        if run_current:
            print("Running current pipeline only.\n")
            run_proposed = False
        else:
            sys.exit(1)

    # Resolve image source
    if args.image:
        if not os.path.exists(args.image):
            print(f"Image not found: {args.image}")
            sys.exit(1)
        image_path = args.image
        use_camera = False
    else:
        image_path = capture_frame()
        if not image_path:
            fallback = os.path.join(os.path.dirname(__file__), "..", "calibration", "paper_present.jpg")
            if os.path.exists(fallback):
                image_path = fallback
                print(f"No camera available, using fallback: {fallback}")
            else:
                print("No camera and no fallback image found.")
                sys.exit(1)
        use_camera = True

    print_separator()
    print("CONTINUOUS REASONING TEST")
    print(f"  Cycles:          {args.cycles}")
    print(f"  Image source:    {'live camera' if use_camera else image_path}")
    if run_current:
        print(f"  Perception:      {PERCEPTION_MODEL}")
        print(f"  Monologue:       {MODEL_NAME}")
    if run_proposed:
        print(f"  Thinking model:  {thinking_model}")
    mode = "both" if run_current and run_proposed else "current only" if run_current else "proposed only"
    print(f"  Running:         {mode}")
    print_separator()

    # State across cycles
    current_last_thought = ""
    thought_stream = ""
    session_start = time.time()

    # Seed the thought stream for cycle 0 (bootstrap problem)
    if args.seed == "none":
        thought_stream = ""
    elif args.seed:
        thought_stream = args.seed
    else:
        thought_stream = AWAKENING_SEED

    if thought_stream and run_proposed:
        print(f'\n  Awakening seed: "{thought_stream[:80]}..."')

    for cycle in range(args.cycles):
        elapsed = time.time() - session_start

        # Capture new frame if using camera
        if use_camera and cycle > 0:
            new_frame = capture_frame()
            if new_frame:
                image_path = new_frame

        image_b64 = encode_image(image_path)

        print(f"\n{'=' * 70}")
        print(f"CYCLE {cycle + 1}/{args.cycles}  (elapsed: {elapsed:.0f}s)")
        print_separator()

        # --- Current pipeline ---
        if run_current:
            print(f"\n  CURRENT PIPELINE (two-pass)")
            print(f"  {'-' * 40}")
            result = run_current_pipeline(image_b64, current_last_thought, cycle, elapsed)
            print(f"  Perception ({result['perc_ms']:.0f}ms):")
            print(f"    {result['perception']}")
            print(f"  Monologue ({result['mono_ms']:.0f}ms):")
            print(f"    {result['monologue']}")
            print(f"  [{result['total_ms']:.0f}ms total, {result['total_tokens']} tokens]")
            current_last_thought = result["monologue"]

        # --- Proposed pipeline ---
        if run_proposed:
            print(f"\n  PROPOSED PIPELINE (single-pass thinking)")
            print(f"  {'-' * 40}")

            sim_concepts = None if args.no_concepts else get_sim_concepts(cycle)

            result = run_thinking_pipeline(
                image_b64,
                thought_stream,
                cycle,
                elapsed,
                model=thinking_model,
                sim_concepts=sim_concepts,
            )

            if result["think"]:
                print(f"  <think> block ({len(result['think'])} chars):")
                print_indented(result["think"])
                print()
                print(f"  Visible response:")
                print_indented(result["response"] if result["response"] else "(empty — all in think block)")
            else:
                print(f"  Response (no think block detected):")
                print_indented(result["response"])

            print(f"  [{result['total_ms']:.0f}ms total, {result['total_tokens']} tokens]")

            # Grow the stream from the think block (the reasoning IS the monologue)
            # If no think block, use the full response
            new_thought = result["think"] or result["response"]
            if new_thought:
                if thought_stream and thought_stream != AWAKENING_SEED:
                    thought_stream += "\n\n" + new_thought
                else:
                    # Replace seed with first real thought
                    thought_stream = new_thought

                pre_trim = len(thought_stream)
                thought_stream, was_trimmed = trim_thought_stream(thought_stream, max_chars=2000)
                if was_trimmed:
                    print(f"  [Stream trimmed: {pre_trim} -> {len(thought_stream)} chars]")

        if args.interval > 0 and cycle < args.cycles - 1:
            print(f"\n  Waiting {args.interval}s...")
            time.sleep(args.interval)

    # --- Summary ---
    print(f"\n{'=' * 70}")
    print("RESULTS")
    print_separator()

    if run_proposed:
        print(f"\nFinal thought stream ({len(thought_stream)} chars):")
        print("-" * 50)
        print(thought_stream)
        print("-" * 50)

    print(
        """
EVALUATION CRITERIA:
  1. Does the <think> block read as inner monologue or image analysis?
  2. Do thoughts develop WITHIN a single think block (associative leaps)?
  3. Does the model reference its prior thoughts naturally?
  4. Does it notice change across frames (if using live camera)?
  5. Is latency acceptable for a 4-10s cycle on RTX 3090?
  6. Compare: current pipeline monologue vs proposed think block — which
     feels more like a continuous mind experiencing its environment?
"""
    )


if __name__ == "__main__":
    main()
