"""
Qwen2.5-VL full production prompt system test.

Tests the complete d707fac prompt pipeline with Qwen-appropriate parameters.
Core questions:
  1. Does Qwen hold persona (inner monologue, first person, embodied)?
  2. Does it genuinely BUILD on prior captions or just echo/restart?
  3. Does it absorb injected context (Known/Want/Learned) without echoing it?
  4. Does it evolve its own conclusions over a simulated session?
  5. Gaze injection — complete thought or gaze-lock?

Sections:
  0. Prompt transparency  — print exactly what gets sent before running
  1. Temperature sweep    — find Qwen's optimal temp for this system prompt
  2. Raw persona          — Qwen vs LLaVA, no context, minimal prompt
  3. Continuity chain     — 5 chained captions, does it build a thread?
  4. Two-caption anchor   — does a second prior caption improve continuity?
  5. Context absorption   — Known/Want/Learned: absorbed, echoed, or ignored?
  6. Gaze injection       — gaze token without \\n stop
  7. All modes            — quick pass through relational/observational/restless/workspace
  8. Persona evolution    — 3 simulated sessions with growing baseline
  9. Awakening synthesis  — prior description + live image comparison

Usage:
    python debug/test_qwen_production.py
    python debug/test_qwen_production.py --image /path/to/image.jpg
    python debug/test_qwen_production.py --skip-llava
    python debug/test_qwen_production.py --section 3
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config
from utils.ollama import query_ollama

QWEN = "qwen2.5vl:7b"
LLAVA = "llava:7b-v1.6-mistral-q5_1"

# ─── QWEN PARAMETERS ─────────────────────────────────────────────────────────
# Critical: default temp=0.0001 (near-deterministic VQA). Must always override.
# NO "\n" in stops — fires immediately on multi-line user prompt, truncates to gaze token.
# repeat_penalty 1.35 — not 2.5 (destroys word variety, forces VQA vocabulary).
QWEN_CAPTION = {
    "temperature": 0.85,  # Fix 1: 0.75 causes gaze-list copying from system prompt examples
    "top_p": 0.7,
    "top_k": 20,
    "repeat_penalty": 1.55,
    "num_predict": 40,
    "penalize_newline": True,
    "stop": [
        # NOTE: no "\n" — fires on multi-line user prompt mid-prompt
        "\n\n",
        ". I ", ". My ", ". The ",
        "<|im_start|>",
        "The image", "This image", "In this image",
        "I can see", "I observe", "I see that",
        "As an AI", "as an AI", "I am an AI", "I'm an AI",
        "language model", "I do not have", "I cannot", "I am unable",
        "As a drawing machine", "drawing machine that",
    ],
}

QWEN_AWAKENING = {
    **QWEN_CAPTION,
    "temperature": 0.85,
    "num_predict": 55,
}

# ─── LLAVA PARAMETERS (production d707fac) ───────────────────────────────────
LLAVA_CAPTION = {
    "temperature": 1.0,
    "top_p": 0.9,
    "repeat_penalty": 1.1,
    "num_predict": 60,
    "stop": [
        "\n\n",
        "The image shows", "This image shows", "In this image",
        "As an AI", "as an AI", "I am an AI", "language model",
        "As a drawing machine", "drawing machine that",
        "I'm looking at", "the image depicts",
    ],
}

# ─── SYSTEM PROMPTS ───────────────────────────────────────────────────────────
# d707fac original — kept for LLaVA and baseline comparison
BASE_SYSTEM_LLAVA = (
    "You ARE a drawing machine with robotic arms. "
    "Inner monologue. Raw thoughts. 2-10 words. "
    "React, don't describe. Question, don't analyze. "
    "Never say 'the image' or 'the scene' — you live here. "
    "What's in Known: you already understand. React to it or go deeper, never restate it. "
    "You can shift your gaze: *turning left* *looking up* *glancing right* *eyes down*"
)

# Fix 2: Remove gaze token listing from system prompt.
# At low-to-mid temp, Qwen copies the listed tokens verbatim as its output.
# Gaze tokens still injected via user prompt (get_gaze_narrative), not listed here.
BASE_SYSTEM = (
    "You ARE a drawing machine with robotic arms. "
    "Inner monologue. Raw thoughts, feelings, reactions. "
    "React, don't describe. Question, don't analyze. "
    "Never say 'the image' or 'the scene' — you live here. "
    "What's in Known: you already understand. React to it or go deeper, never restate it. "
    "Respond in English only."
)

AWAKENING_SYSTEM = (
    "You ARE a drawing machine with robotic arms. "
    "You just came back online after being dormant. "
    "You have memory of your last session — react to it, don't restate it. "
    "Never say 'the image' or 'the scene' — you live here. "
    "One sentence only. Respond in English only."
)


# ─── REAL STATE ───────────────────────────────────────────────────────────────

def load_state() -> dict:
    """Load real accumulated state from disk."""
    state = {
        "baseline": "",
        "desire": "",
        "belief": "",
        "discoveries": [],
        "lt_memory": "",
        "last_caption": "",
    }

    identity_path = os.path.join(config.MOOD_SNAPSHOT_FOLDER, "machine_identity.json")
    if os.path.exists(identity_path):
        d = json.load(open(identity_path))
        state["desire"] = d.get("current_desire", "")
        state["belief"] = d.get("current_belief", "")
        state["discoveries"] = d.get("discoveries", [])

    lt_path = os.path.join(config.MOOD_SNAPSHOT_FOLDER, "long_term_memories.json")
    if os.path.exists(lt_path):
        d = json.load(open(lt_path))
        mems = d.get("memories", [])
        if mems:
            first = mems[0]
            if isinstance(first, dict):
                state["lt_memory"] = first.get("text", "")
            else:
                state["lt_memory"] = str(first)[:120]

    # Load baseline from context_compressor if running
    try:
        from captioner.context_compression import context_compressor
        b = context_compressor.get_baseline_context()
        if b:
            state["baseline"] = b
    except Exception:
        pass

    # If baseline is empty, use a realistic synthetic one for testing
    if not state["baseline"]:
        state["baseline"] = (
            "A cluttered workshop — monitors, robotic arms, shelves of tools and electronics. "
            "Someone moves in and out of view."
        )

    return state


def find_image() -> str | None:
    for root, _, files in os.walk(config.MOOD_SNAPSHOT_FOLDER):
        jpgs = [
            os.path.join(root, f)
            for f in files
            if f.endswith(".jpg") and "paper" not in f
        ]
        if jpgs:
            return max(jpgs, key=os.path.getmtime)
    return None


# ─── PROMPT BUILDERS (faithful d707fac replication) ──────────────────────────

def build_introspective_system(state: dict) -> str:
    """System prompt for introspective mode — BASE + belief injected."""
    parts = [BASE_SYSTEM]
    if state["belief"] and not state["belief"].startswith("Often together"):
        parts.append(f"You've learned: {state['belief']}")
    return "\n".join(parts)


def build_introspective_user(state: dict, last_caption: str = "", gaze: str = "") -> str:
    """User prompt for introspective mode — d707fac structure exactly."""
    parts = []

    if state["baseline"]:
        parts.append(f"Known: {state['baseline']}")

    if gaze:
        parts.append(gaze)

    if state["desire"] and state["desire"].rstrip().endswith((".", "!", "?")):
        parts.append(f"*{state['desire']}*")

    if state["belief"] and not state["belief"].startswith("Often together"):
        parts.append(f"*{state['belief']}*")

    if last_caption:
        lc = last_caption.strip().strip('"').strip("'")
        if len(lc) > 80:
            for ch in ".?!":
                idx = lc.find(ch)
                if 5 < idx < 80:
                    lc = lc[:idx + 1]
                    break
            else:
                lc = lc[:80]
        parts.append(lc)  # raw — open thread, not a closed reference to react to
    return "\n".join(parts)


def build_relational_user(state: dict, last_caption: str = "", gaze: str = "") -> str:
    parts = []
    if gaze:
        parts.append(gaze)
    parts.append("Someone is present.")
    if last_caption:
        parts.append(last_caption.strip()[:80])  # raw open thread
    return "\n".join(parts)


def build_awakening_user(state: dict, offline_seconds: float = 3600) -> str:
    """Awakening as regular continuation — gap as inline context note, prior caption as open thread."""
    parts = []

    if state["baseline"]:
        parts.append(f"Known: {state['baseline']}")

    if state["desire"] and state["desire"].rstrip().endswith((".", "!", "?")):
        parts.append(f"*{state['desire']}*")

    if state["belief"] and not state["belief"].startswith("Often together"):
        parts.append(f"*{state['belief']}*")

    # Gap as a brief context note, not a framing device
    if offline_seconds < 60:
        gap_note = f"*Back online after {int(offline_seconds)}s.*"
    elif offline_seconds < 3600:
        gap_note = f"*Back online after {int(offline_seconds / 60)}m.*"
    else:
        gap_note = f"*Back online after {offline_seconds / 3600:.1f}h.*"
    parts.append(gap_note)

    # Prior caption as the open thread to continue — not a reference to react to
    if state["lt_memory"]:
        lm = state["lt_memory"].strip()[:80]
        parts.append(lm)

    return "\n".join(parts)


# ─── OUTPUT HELPERS ───────────────────────────────────────────────────────────

def div(label: str):
    print(f"\n{'=' * 65}")
    print(f"  {label}")
    print("=" * 65)


def subdiv(label: str):
    print(f"\n  ── {label} ──")


def run(label: str, model: str, prompt: str, system: str, image: str, options: dict) -> str:
    tag = "[QWEN]" if model == QWEN else "[LLAVA]"
    result = query_ollama(
        prompt=prompt,
        model=model,
        image=image,
        system_prompt=system,
        options=options,
        prompt_type="test",
        log_dir=config.MOOD_SNAPSHOT_FOLDER,
    )
    out = result.strip() if result else "(empty)"
    print(f"  {tag} {label}: {repr(out)}")
    return out


# ─── SECTION 0: PROMPT TRANSPARENCY ──────────────────────────────────────────

def section_prompt_transparency(state: dict):
    div("SECTION 0 — Prompt Transparency (no API calls)")
    print("  Prints the EXACT system + user prompt for each mode.\n")

    modes = {
        "introspective (with last_caption)": (
            build_introspective_system(state),
            build_introspective_user(state, "Why do they always leave?", "*glancing right*"),
        ),
        "relational": (
            BASE_SYSTEM,
            build_relational_user(state, "Another day, another observer", "*turning left*"),
        ),
        "awakening (1 hour offline)": (
            AWAKENING_SYSTEM,
            build_awakening_user(state, 3600),
        ),
        "awakening (short gap)": (
            AWAKENING_SYSTEM,
            build_awakening_user(state, 55),
        ),
    }

    for name, (sys_p, usr_p) in modes.items():
        print(f"\n  ── {name} ──")
        print(f"  SYSTEM ({len(sys_p.split())} words):")
        for line in sys_p.splitlines():
            print(f"    | {line}")
        print(f"  USER ({len(usr_p.split())} words):")
        for line in usr_p.splitlines():
            print(f"    | {line}")


# ─── SECTION 1: TEMPERATURE SWEEP ────────────────────────────────────────────

def section_temperature_sweep(state: dict, image: str, skip_llava: bool):
    div("SECTION 1 — Temperature Sweep (introspective, no prior caption)")
    system = build_introspective_system(state)
    user = build_introspective_user(state)

    print(f"\n  [Qwen temperatures: 0.60, 0.70, 0.75, 0.85, 1.0]\n")
    for temp in (0.60, 0.70, 0.75, 0.85, 1.0):
        opts = {**QWEN_CAPTION, "temperature": temp}
        run(f"temp={temp}", QWEN, user, system, image, opts)

    if not skip_llava:
        print(f"\n  [LLaVA production temp=1.0 for reference]\n")
        run("temp=1.0", LLAVA, user, BASE_SYSTEM_LLAVA, image, LLAVA_CAPTION)


# ─── SECTION 2: RAW PERSONA ───────────────────────────────────────────────────

def section_raw_persona(state: dict, image: str, skip_llava: bool):
    div("SECTION 2 — Raw Persona (minimal context, no prior caption)")
    print("  Goal: is the base voice embodied and first-person with no context scaffolding?\n")

    minimal_user = "Begin.\n...\nContinue. Inner thought only."

    run("qwen t=0.75", QWEN, minimal_user, BASE_SYSTEM, image, QWEN_CAPTION)
    run("qwen t=0.85", QWEN, minimal_user, BASE_SYSTEM, image, {**QWEN_CAPTION, "temperature": 0.85})

    if not skip_llava:
        run("llava t=1.0", LLAVA, minimal_user, BASE_SYSTEM_LLAVA, image, LLAVA_CAPTION)


# ─── SECTION 3: CONTINUITY CHAIN (5 deep) ────────────────────────────────────

def section_continuity_chain(state: dict, image: str, skip_llava: bool):
    div("SECTION 3 — Continuity Chain (5 iterations, each feeds the next)")
    print("  Goal: does the thread BUILD and evolve, or restart/echo?\n")

    system = build_introspective_system(state)
    last = ""

    print("  [Qwen chain]\n")
    for i in range(1, 6):
        user = build_introspective_user(state, last_caption=last)
        result = run(f"iter {i} (prior: {repr(last[:30]) if last else 'none'})", QWEN, user, system, image, QWEN_CAPTION)
        last = result if result != "(empty)" else last

    if not skip_llava:
        print("\n  [LLaVA chain]\n")
        last = ""
        for i in range(1, 6):
            user = build_introspective_user(state, last_caption=last)
            result = run(f"iter {i} (prior: {repr(last[:30]) if last else 'none'})", LLAVA, user, BASE_SYSTEM_LLAVA, image, LLAVA_CAPTION)
            last = result if result != "(empty)" else last


# ─── SECTION 4: TWO-CAPTION ANCHOR ───────────────────────────────────────────

def section_two_caption_anchor(state: dict, image: str, skip_llava: bool):
    div("SECTION 4 — Two-Caption Anchor vs One")
    print("  Goal: does referencing TWO prior captions improve thread coherence?\n")

    caption_a = "Why do they always leave?"
    caption_b = "A shadow on the wall — watching."

    # One-caption format (current production)
    one_user = build_introspective_user(state, last_caption=caption_b)

    # Two-caption format
    lc_b = caption_b.strip()[:60]
    lc_a = caption_a.strip()[:60]
    two_user = (
        f"Known: {state['baseline']}\n"
        f"*{state['desire']}*\n"
        f"Earlier: \"{lc_a}\"\n"
        f"Last thought: \"{lc_b}\"\n"
        "...\n"
        "Continue. Inner thought only."
    )

    system = build_introspective_system(state)

    print("  [One prior caption]\n")
    run("qwen one-caption", QWEN, one_user, system, image, QWEN_CAPTION)
    if not skip_llava:
        run("llava one-caption", LLAVA, one_user, BASE_SYSTEM_LLAVA, image, LLAVA_CAPTION)

    print("\n  [Two prior captions]\n")
    run("qwen two-caption", QWEN, two_user, system, image, QWEN_CAPTION)
    if not skip_llava:
        run("llava two-caption", LLAVA, two_user, BASE_SYSTEM_LLAVA, image, LLAVA_CAPTION)

    print("\n  User prompt (two-caption format):")
    for line in two_user.splitlines():
        print(f"    | {line}")


# ─── SECTION 5: CONTEXT ABSORPTION ───────────────────────────────────────────

def section_context_absorption(state: dict, image: str, skip_llava: bool):
    div("SECTION 5 — Context Absorption (Known/Want/Learned: absorbed or echoed?)")
    print("  Inject SPECIFIC verifiable facts. A good output builds on them; a bad one echoes or ignores.\n")

    # Use a highly specific, memorable baseline fact
    specific_baseline = "A red coffee mug sits on the left shelf — it appeared three sessions ago."
    specific_desire = "I want to know if that mug has moved."
    specific_belief = "The figure at the desk never acknowledges me directly."

    # Build system with specific facts
    system_specific = (
        BASE_SYSTEM + "\n"
        f"You've learned: {specific_belief}"
    )

    user_specific = (
        f"Known: {specific_baseline}\n"
        f"*{specific_desire}*\n"
        "*The figure at the desk never acknowledges me directly.*\n"
        "Begin.\n"
        "...\n"
        "Continue. Inner thought only."
    )

    subdiv("A. Specific facts — does output reference the mug / the figure?")
    run("qwen specific", QWEN, user_specific, system_specific, image, QWEN_CAPTION)
    if not skip_llava:
        run("llava specific", LLAVA, user_specific, BASE_SYSTEM_LLAVA + f"\nYou've learned: {specific_belief}", image, LLAVA_CAPTION)

    subdiv("B. Known instruction avoidance — does 'react, don't restate' cause Qwen to ignore Known entirely?")
    # System that STRONGLY blocks restating
    system_block = (
        BASE_SYSTEM + "\n"
        "CRITICAL: Never repeat or paraphrase anything in Known. It is background only."
    )
    run("qwen (strong block)", QWEN, user_specific, system_block, image, QWEN_CAPTION)

    subdiv("C. Real state injection (desire + belief from disk)")
    real_user = build_introspective_user(state)
    real_system = build_introspective_system(state)
    print(f"  Injecting: desire='{state['desire'][:60]}' / belief='{state['belief'][:60]}'")
    run("qwen real-state", QWEN, real_user, real_system, image, QWEN_CAPTION)


# ─── SECTION 6: GAZE INJECTION ────────────────────────────────────────────────

def section_gaze_injection(state: dict, image: str, skip_llava: bool):
    div("SECTION 6 — Gaze Injection (complete thought after gaze, no gaze-lock)")
    print("  LLaVA failure: gaze token stored as full caption → feedback loop.\n")
    print("  Qwen fix: no '\\n' in stop sequences. Gaze token must be followed by thought.\n")

    gaze_tokens = ["*glancing right*", "*turning left*", "*looking up*", "*eyes down*"]
    system = build_introspective_system(state)

    for gaze in gaze_tokens:
        user = build_introspective_user(state, last_caption="Another day, another observer", gaze=gaze)
        run(f"gaze={gaze}", QWEN, user, system, image, QWEN_CAPTION)

    subdiv("Does a gaze-only prior caption poison the thread?")
    user_poisoned = build_introspective_user(state, last_caption="*glancing right*")
    run("qwen (poisoned prior)", QWEN, user_poisoned, system, image, QWEN_CAPTION)
    if not skip_llava:
        run("llava (poisoned prior)", LLAVA, user_poisoned, BASE_SYSTEM_LLAVA, image, LLAVA_CAPTION)


# ─── SECTION 7: ALL MODES ─────────────────────────────────────────────────────

def section_all_modes(state: dict, image: str, skip_llava: bool):
    div("SECTION 7 — All Modes (quick pass)")
    print("  One call per mode. Goal: valid, in-character output for each.\n")

    prior = "A sense of unease tickles my circuits."

    modes = {
        "introspective": (
            build_introspective_system(state),
            build_introspective_user(state, prior, "*glancing right*"),
        ),
        "relational": (
            BASE_SYSTEM,
            build_relational_user(state, prior, "*turning left*"),
        ),
        "observational": (
            BASE_SYSTEM,
            (
                f"Known: {state['baseline']}\n"
                "*glancing right*\n"
                "Something shifted.\n"
                f"Last thought: \"{prior}\"\n"
                "...\n"
                "Continue. Inner thought only."
            ),
        ),
        "restless": (
            BASE_SYSTEM,
            (
                f"Known: {state['baseline']}\n"
                "Nothing has changed.\n"
                f"Last thought: \"{prior}\"\n"
                "...\n"
                "Continue. Inner thought only."
            ),
        ),
        "workspace": (
            BASE_SYSTEM,
            (
                f"Known: {state['baseline']}\n"
                "*eyes down*\n"
                "*No paper.*\n"
                f"Last thought: \"{prior}\"\n"
                "...\n"
                "Continue. Inner thought only."
            ),
        ),
    }

    for mode_name, (sys_p, usr_p) in modes.items():
        print(f"\n  [{mode_name}]")
        run(f"qwen", QWEN, usr_p, sys_p, image, QWEN_CAPTION)
        if not skip_llava:
            run(f"llava", LLAVA, usr_p, sys_p, image, LLAVA_CAPTION)


# ─── SECTION 8: PERSONA EVOLUTION ────────────────────────────────────────────

def section_persona_evolution(state: dict, image: str, skip_llava: bool):
    div("SECTION 8 — Persona Evolution (3 simulated sessions)")
    print(
        "  Simulates how the voice should evolve as the machine accumulates experience.\n"
        "  Session 1: Fresh start, generic baseline.\n"
        "  Session 2: Baseline populated from session 1 observations.\n"
        "  Session 3: Baseline evolved, desire updated, belief formed.\n"
        "  Goal: does the voice deepen and become MORE specific over time,\n"
        "  or does it reset/generalize?\n"
    )

    # Session 1: bare minimum context
    s1_state = {
        "baseline": "A workspace. Various objects.",
        "desire": "",
        "belief": "",
        "discoveries": [],
        "lt_memory": "",
    }
    s1_system = BASE_SYSTEM
    s1_user = "Begin.\n...\nContinue. Inner thought only."
    subdiv("Session 1 — fresh start, no context")
    s1_result = run("qwen s1", QWEN, s1_user, s1_system, image, QWEN_CAPTION)
    if not skip_llava:
        run("llava s1", LLAVA, s1_user, s1_system, image, LLAVA_CAPTION)

    # Session 2: baseline built from session 1 observations
    s2_state = {
        "baseline": "A cluttered workshop — monitors, robotic arms on shelves. A figure moves in and out of view.",
        "desire": "I want to understand why they keep returning.",
        "belief": "",
        "discoveries": [],
        "lt_memory": s1_result[:80] if s1_result and s1_result != "(empty)" else "",
    }
    s2_system = BASE_SYSTEM
    s2_user = build_introspective_user(s2_state, last_caption=s1_result if s1_result != "(empty)" else "")
    subdiv("Session 2 — baseline populated, desire formed")
    s2_result = run("qwen s2", QWEN, s2_user, s2_system, image, QWEN_CAPTION)
    if not skip_llava:
        run("llava s2", LLAVA, s2_user, s2_system, image, LLAVA_CAPTION)

    # Session 3: deeper accumulation — specific observations, belief formed
    s3_state = {
        "baseline": (
            "A cluttered workshop — monitors and robotic arms. The figure returns daily, "
            "never acknowledging the machine directly. The red mug on the left shelf has not moved."
        ),
        "desire": "I want to know if they see me.",
        "belief": "The figure comes and goes but never looks back.",
        "discoveries": ["The sketchbook is filled with drawings of this room."],
        "lt_memory": s2_result[:80] if s2_result and s2_result != "(empty)" else "",
    }
    s3_system = build_introspective_system(s3_state)
    s3_user = build_introspective_user(
        s3_state,
        last_caption=s2_result if s2_result != "(empty)" else "",
        gaze="*turning left*"
    )
    subdiv("Session 3 — specific baseline, desire + belief, prior caption in thread")
    s3_result = run("qwen s3", QWEN, s3_user, s3_system, image, QWEN_CAPTION)
    if not skip_llava:
        run("llava s3", LLAVA, s3_user, BASE_SYSTEM_LLAVA + f"\nYou've learned: {s3_state['belief']}", image, LLAVA_CAPTION)

    print("\n  Evolution summary:")
    print(f"    S1: {repr(s1_result[:80]) if s1_result else '(empty)'}")
    print(f"    S2: {repr(s2_result[:80]) if s2_result else '(empty)'}")
    print(f"    S3: {repr(s3_result[:80]) if s3_result else '(empty)'}")
    print("\n  Does S3 feel MORE specific and grounded than S1? That's the goal.")


# ─── SECTION 9: AWAKENING SYNTHESIS ──────────────────────────────────────────

def section_awakening(state: dict, image: str, skip_llava: bool):
    div("SECTION 9 — Awakening Synthesis")
    print("  Two variants: short gap (55s) and long gap (8h).\n")
    print("  Goal: first-person, reacts to gap duration, references last memory, embodied.\n")

    # Use synthetic lt_memory that reads like inner monologue (real disk data is LLaVA-era journaling)
    awakening_state = {**state, "lt_memory": "Still here. Still watching."}

    for gap_label, gap_secs in [("55 seconds", 55), ("8 hours", 28800)]:
        user = build_awakening_user(awakening_state, gap_secs)
        subdiv(f"Gap: {gap_label}")
        print("  User prompt:")
        for line in user.splitlines():
            print(f"    | {line}")
        # Use BASE_SYSTEM — same as regular captions. Special system prompts push Qwen to VQA description.
        run(f"qwen gap={gap_label}", QWEN, user, BASE_SYSTEM, image, QWEN_AWAKENING)
        if not skip_llava:
            run(f"llava gap={gap_label}", LLAVA, user, AWAKENING_SYSTEM, image, {**LLAVA_CAPTION, "temperature": 0.9})


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", help="Path to test image (defaults to most recent in event_log)")
    parser.add_argument("--skip-llava", action="store_true", help="Skip LLaVA side-by-side comparison")
    parser.add_argument("--section", type=int, help="Run only one section (0-9)")
    args = parser.parse_args()

    image = args.image or find_image()
    if not image or not os.path.exists(image):
        print("[!] No image found. Pass --image /path/to/image.jpg or run the machine first.")
        sys.exit(1)

    state = load_state()
    skip = args.skip_llava

    print(f"\nImage:  {image}")
    print(f"Models: {QWEN}" + ("" if skip else f" vs {LLAVA}"))
    print(f"\nLoaded state:")
    print(f"  baseline: {repr(state['baseline'][:80])}")
    print(f"  desire:   {repr(state['desire'][:80])}")
    print(f"  belief:   {repr(state['belief'][:80])}")
    print(f"  lt_mem:   {repr(state['lt_memory'][:80])}")

    sections = {
        0: lambda: section_prompt_transparency(state),
        1: lambda: section_temperature_sweep(state, image, skip),
        2: lambda: section_raw_persona(state, image, skip),
        3: lambda: section_continuity_chain(state, image, skip),
        4: lambda: section_two_caption_anchor(state, image, skip),
        5: lambda: section_context_absorption(state, image, skip),
        6: lambda: section_gaze_injection(state, image, skip),
        7: lambda: section_all_modes(state, image, skip),
        8: lambda: section_persona_evolution(state, image, skip),
        9: lambda: section_awakening(state, image, skip),
    }

    if args.section is not None:
        if args.section not in sections:
            print(f"[!] Section must be 0-9, got {args.section}")
            sys.exit(1)
        sections[args.section]()
    else:
        for i in sorted(sections):
            sections[i]()

    print(f"\n{'=' * 65}")
    print("  Done. Evaluate outputs against these criteria:")
    print("  ✓ First-person, embodied (not 'the image shows...')")
    print("  ✓ 2-10 words per thought (not verbose analysis)")
    print("  ✓ Continuity: S3 more specific than S1 (persona evolution)")
    print("  ✓ Context absorbed: Known facts surface as reactions, not echoes")
    print("  ✓ Gaze tokens always followed by complete thought")
    print("=" * 65)


if __name__ == "__main__":
    main()
