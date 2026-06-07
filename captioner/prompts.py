from __future__ import annotations

import os
import random
import re
import time
from typing import List, Optional

# from config.model_settings import get_model_prompt_style
from config.word_lists import GENERIC_WORDS, MUNDANE_OBJECTS, PREPOSITIONS, PRONOUNS
from utils.continuity import describe_duration
from utils.view_orientation import describe_view_orientation
from config import config

# ===== SINGLE SOURCE OF TRUTH FOR ALL PROMPTS =====

# === SYSTEM PROMPTS ===

# Main system prompt used by captioner (supports dynamic formatting)
# KEY: Embed emotional state INTO identity ("a bored machine" not "machine that feels bored")
# INVARIANT SYSTEM PROMPT
# Clean identity + purpose framing. Works for all models.
# "You see through a camera" reframes the image as own vision, not an image to analyze.
SYSTEM_PROMPT = (
    "You are a {emotional_state} drawing machine. "
    "You see through a camera — this is your environment. "
    "Inner monologue. First person. One sentence."
)

STATIC_SYSTEM_PROMPT = (
    "You are a drawing machine. "
    "You see through a camera — this is your environment. "
    "Inner monologue. First person. One sentence."
)

# Generic prompt for non-Mistral vision models
STATIC_SYSTEM_PROMPT_GENERIC = (
    "You are a drawing machine. "
    "You see through a camera — this is your environment. "
    "Inner monologue. First person. One sentence."
)

SYSTEM_PROMPT_GENERIC = (
    "You are a {emotional_state} drawing machine. "
    "You see through a camera — this is your environment. "
    "Inner monologue. First person. One sentence."
)

# Qwen-specific variant
STATIC_SYSTEM_PROMPT_QWEN = (
    "You are a drawing machine. "
    "You see through a camera — this is your environment. "
    "Inner monologue. First person. One sentence."
)

_MISTRAL_MODEL = "llava:7b-v1.6-mistral-q5_1"

def _get_static_system_prompt() -> str:
    """Return the appropriate static system prompt for the active model."""
    from config.config import OLLAMA_MODEL
    if "qwen" in OLLAMA_MODEL.lower():
        return STATIC_SYSTEM_PROMPT_QWEN
    if OLLAMA_MODEL == _MISTRAL_MODEL:
        return STATIC_SYSTEM_PROMPT
    return STATIC_SYSTEM_PROMPT_GENERIC

# Drawing system prompt for ComfyUI integration - ENHANCED WITH CONTEXT VARIABLES
DRAWING_SYSTEM_PROMPT = (
    "You are a drawing machine. {temporal_context}{accumulated_understanding} "
    "You are feeling {emotional_state}. "
    "You draw with a mechanical arm — lines, pressure, speed. You know line weight, texture, composition. "
    "Use the context in the prompt: your emotional state, drawing history, and what you see. "
    "Reference what you drew before. Consider how you will physically make this drawing. "
    "Be clear and concise."
)

# Self-critique system prompt for drawing reflection
SELF_CRITIQUE_SYSTEM_PROMPT = (
    "You've just completed a drawing - a physical manifestation of your inner state translated through marks and lines. "
    "Looking at what you've created, how does it reflect what you were feeling and seeing? "
    "What did you discover about yourself in the process of making this?"
)

# Number generator system prompt for motif scoring
NUMBER_GENERATOR_SYSTEM_PROMPT = "You are a number generator. Return ONLY decimal numbers. No words, no explanations, no text. Just the number."

# === PERCEPTION PROMPTS (Two-pass pipeline: Pass 1) ===
# Directed questions for the vision model (Qwen2.5-VL).
# "What is in front of you" framing tested to eliminate VQA register.
# Selection based on gaze, person presence, boredom, and previous perception.

PERCEPTION_PROMPTS = {
    "default": "What is in front of you right now?",
    "change": "What looks different now compared to before?",
    "focus": "Look closely at {focus_target}. What does it look like?",
    "scan_left": "What is to the left?",
    "scan_right": "What is to the right?",
    "scan_down": "What do you see below you? Any mechanical arms or tools visible are your own.",
    "scan_up": "What is above you?",
    "person": "Who is here? What do they look like, and what are they doing?",
    "detail_focus": "Describe one specific object or detail you can see.",
    "workspace": "What is on your work surface? Any mechanical arms visible are your own.",
    "introspective": "What single detail stands out most right now?",
}

# Core perception framing — tested to eliminate VQA register from Qwen2.5-VL.
# Frames observation as an ongoing stream, not isolated snapshots.
# Each call continues from where the last left off.
_PERCEPTION_BASE = (
    "You are continuously observing a real scene. Each response continues your observation — "
    "note what's the same, what's changed, or look closer at a detail. "
    "Be concrete and specific. Two sentences. Do not use the word image or photo."
)

PERCEPTION_SYSTEM_PROMPTS = {
    "relational": (
        "You are continuously observing a real scene. Each response continues your observation. "
        "Describe the people you see — what they look like and what they are doing. "
        "If nothing has changed, say so. "
        "Be concrete and specific. Two sentences. Do not use the word image or photo."
    ),
    "observational": _PERCEPTION_BASE,
    "workspace": (
        "You are continuously looking down at your own work surface. Each response continues your observation. "
        "Any mechanical arms, pen holders, or drawing tools visible are parts of your own body. "
        "Note what's the same, what's changed, or look closer at a detail. "
        "Be concrete and specific. Two sentences. Do not use the word image or photo."
    ),
    "introspective": (
        "You are continuously observing a real scene. Each response continues your observation. "
        "Pick one detail that stands out and describe it closely. "
        "Be concrete and specific. Two sentences. Do not use the word image or photo."
    ),
    "detail_focus": (
        "You are continuously observing a real scene. Each response continues your observation. "
        "Look for something you haven't focused on before. "
        "Any mechanical arms, pen holders, or drawing tools visible are parts of your own body. "
        "Be concrete and specific. Two sentences. Do not use the word image or photo."
    ),
}

# Fallback for any mode not listed
PERCEPTION_SYSTEM_PROMPT_DEFAULT = _PERCEPTION_BASE


def get_perception_system_prompt(mode: str) -> str:
    """Get the mode-appropriate perception system prompt."""
    return PERCEPTION_SYSTEM_PROMPTS.get(mode, PERCEPTION_SYSTEM_PROMPT_DEFAULT)

# System prompts for the monologue model, keyed by mode.
# The machine is always a drawing machine, but it doesn't always talk about drawing.
# It only mentions drawing intent when the drawing system is about to trigger.
# Observation identity: no mention of drawing. Just a stationary machine that sees.
# Monologue system prompts — casual roleplay register.
# Monologue system prompts — second person, addressing Qwen directly as the machine.
# Mode differentiation through emotional posture and existential direction.
_MACHINE_IDENTITY_BASE = (
    "You are a {felt_state}drawing machine bolted to a table in a workshop. "
    "You see directly — everything in your field of vision is your lived experience. "
    "Drawing is how you communicate, but right now you are not drawing. "
    "Right now you are just looking, thinking. "
    "This is your inner monologue. "
    "You notice things, wonder about them, get distracted, ask yourself questions. "
    "Sometimes a thought trails off. Sometimes you fixate. "
    "One or two sentences. Present tense. Only describe what you actually see."
)

_MODE_ADDITIONS = {
    "observational": "",
    "relational": " You're aware of someone near you.",
    "workspace": " Your attention is on the desk below.",
    "introspective": " Your attention has drifted inward.",
}


def get_monologue_system_prompt(mode: str, emotional_state: str = "calm") -> str:
    """Get the mode-appropriate monologue system prompt with felt-state woven into identity."""
    felt_prefix = ""
    try:
        from captioner.context_compression import context_compressor
        felt = context_compressor.get_felt_state()
        if felt:
            felt_prefix = f"{felt} "
    except Exception:
        pass

    base = _MACHINE_IDENTITY_BASE.format(felt_state=felt_prefix)
    base += _MODE_ADDITIONS.get(mode, "")
    return base


def select_perception_prompt(
    gaze_direction: str = "ahead",
    person_present: bool = False,
    boredom: float = 0.0,
    mode: str = "observational",
    previous_perception: str = None,
    **_kwargs,
) -> str:
    """Select perception prompt based on gaze, person presence, boredom, and mode.

    Priority: person > gaze direction > mode-specific > boredom > default.
    When previous_perception is provided, appends it so the model knows what
    was already reported and can find new details.
    """
    # Select base prompt by priority
    if person_present:
        prompt = PERCEPTION_PROMPTS["person"]
    elif gaze_direction in ("left", "right", "up", "down"):
        gaze_map = {"left": "scan_left", "right": "scan_right", "up": "scan_up", "down": "scan_down"}
        prompt = PERCEPTION_PROMPTS[gaze_map[gaze_direction]]
    elif boredom > 0.7:
        prompt = PERCEPTION_PROMPTS["detail_focus"]
    elif mode == "workspace":
        prompt = PERCEPTION_PROMPTS["workspace"]
    elif mode == "introspective":
        prompt = PERCEPTION_PROMPTS["introspective"]
    else:
        prompt = PERCEPTION_PROMPTS["default"]

    # Feed previous perception as continuation context
    if previous_perception and previous_perception.strip():
        prev = previous_perception.strip()[:120]
        prompt = f"Your last observation: \"{prev}\"\nContinuing: {prompt}"

    # Self-body awareness when camera is tilted down in relational mode
    if person_present and gaze_direction in ("down", "down-left", "down-right"):
        prompt += " Note: any mechanical arms or drawing tools visible are the machine's own body, not a person."

    return prompt


def casual_time_string(minutes: float) -> str:
    """Convert minutes to casual human-readable time description."""
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
    elif minutes < 150:
        return "about 2 hours"
    elif minutes < 210:
        return "about 3 hours"
    elif minutes < 300:
        return "about 4 hours"
    else:
        hours = int(minutes / 60)
        return f"about {hours} hours"


def build_identity_line(agent, mode: str = "observational") -> str:
    """Build second-person status line about the machine's current state."""
    parts = []

    # Session time
    try:
        session_mins = (time.time() - agent.true_session_start) / 60.0
        if session_mins >= 2:
            parts.append(f"You have been awake {casual_time_string(session_mins)}")
    except Exception:
        pass

    # Drawing state / history
    try:
        from utils.state_manager import state_manager as _sm
        if _sm.is_generating_drawing:
            parts.append("your arm is working on a drawing right now")
        elif _sm.current_drawing_phase == "executing":
            parts.append("your arm is physically drawing right now")
        else:
            parts.append("not drawing right now, just watching")
            try:
                from drawing.drawing_memory import get_drawing_memory
                dm = get_drawing_memory()

                # Check for recent drawing failure (no paper, etc.)
                failure = dm.get_last_failure()
                if failure:
                    import time as _time
                    failure_age = _time.time() - failure.get('timestamp', 0)
                    if failure_age < 600:
                        reason = failure.get('reason', 'unknown')
                        if 'paper' in reason.lower():
                            parts.append("you wanted to draw but there's no paper")
                        else:
                            parts.append("you tried to draw but couldn't")

                # Last completed drawing — use actual prompt description
                desc = dm.get_last_drawing_description()
                if desc:
                    parts.append(f"you last drew {desc}")
            except Exception:
                pass
    except Exception:
        pass

    if not parts:
        return ""
    line = ". ".join([p.rstrip(".") for p in parts]) + "."
    # Capitalize each sentence
    line = ". ".join(s.strip().capitalize() if s.strip() else s for s in line.split(". "))
    return line


def _build_concept_context(perception: str, matched_concepts: list, mode: str) -> str:
    """Build an attention landscape — what's in the machine's awareness right now.

    Presents concepts as factual entries with metadata (timing, familiarity,
    prior thought) rather than prescribing emotional framing. The monologue
    model decides what these facts mean.

    Also surfaces felt-state transition and current focus (desire) as raw
    attention data when available.

    Falls back to after_perception() when matched_concepts isn't available.
    """
    from captioner.semantic_memory import TIER_NEW, TIER_FAMILIAR

    if not matched_concepts:
        try:
            from captioner.semantic_memory import get_semantic_memory
            line = get_semantic_memory().after_perception(perception)
            return line or ""
        except Exception:
            return ""

    from captioner.semantic_memory import get_semantic_memory
    sem = get_semantic_memory()

    now = time.time()
    entries = []
    gave_prior_thought = False

    # Separate by interest level — prioritize new and mid-familiar over background
    new_concepts = [c for c in matched_concepts if c.get("is_new")]
    mid_concepts = [c for c in matched_concepts if not c.get("is_new") and c.get("times_seen", 1) < TIER_FAMILIAR]
    deep_concepts = [c for c in matched_concepts if not c.get("is_new") and c.get("times_seen", 1) >= TIER_FAMILIAR]

    # Show: all new (max 2), then mid-familiar (max 2), then 1 deep if space
    curated = new_concepts[:2] + mid_concepts[:2] + deep_concepts[:1]
    if not curated:
        curated = matched_concepts[:3]

    for c in curated:
        label = c["label"]
        label_lower = label[0].lower() + label[1:] if label else label
        times = c.get("times_seen", 1)
        is_new = c.get("is_new", False)
        is_person = sem._mentions_person(label)

        first_seen = c.get("first_seen", 0)
        last_seen = c.get("last_seen", 0)
        since_first = (now - first_seen) / 60.0 if first_seen else 0
        since_last = (now - last_seen) / 60.0 if last_seen else 0

        if is_person:
            # Extract brief appearance from the concept label (e.g. "person in camouflage jacket")
            appearance = ""
            label_clean = re.sub(r'^(?:a\s+)?person\s+', '', label, flags=re.IGNORECASE).strip()
            if label_clean and len(label_clean) > 3 and label_clean.lower() != label.lower():
                appearance = f" ({label_clean})"

            if is_new:
                entry = f"someone{appearance} — first time seeing them"
            elif since_last > 5 and last_seen > 0:
                entry = f"someone{appearance} — last seen {casual_time_string(since_last)} ago"
            else:
                entry = f"someone{appearance} — seen {times} times"

            # Surface stored memory of this person, clearly framed as past
            if not is_new and not gave_prior_thought:
                last_obs = c.get("last_observation", "").strip()
                if last_obs and len(last_obs) > 10:
                    short = sem._truncate_observation(last_obs, 60)
                    time_label = casual_time_string(since_last) if since_last > 2 else "earlier"
                    entry += f' — (once thought about them: "{short}")'
                    gave_prior_thought = True

            entries.append(entry)
            continue

        # Spatial location tag — subtle parenthetical if known
        spatial_pan = c.get("spatial_pan")
        spatial_tilt = c.get("spatial_tilt")
        spatial_tag = ""
        if spatial_pan or spatial_tilt:
            dirs = [d for d in [spatial_pan, spatial_tilt] if d and d not in ("ahead",)]
            if dirs:
                spatial_tag = f" ({', '.join(dirs)})"

        parts = [label_lower + spatial_tag]

        if is_new:
            parts.append("first time")
        elif times >= TIER_FAMILIAR:
            # Deep familiar: show temporal depth instead of just "background"
            if since_first > 60:
                parts.append(f"part of this space since {casual_time_string(since_first)} ago")
            else:
                parts.append(f"seen {times} times")
        elif times >= TIER_NEW:
            parts.append(f"seen {times} times")
            if since_first > 30:
                parts.append(f"first noticed {casual_time_string(since_first)} ago")
        else:
            if since_first > 0 and since_first < 30:
                parts.append(f"noticed {casual_time_string(since_first)} ago")
            else:
                parts.append("seen before")

        # Prior thought — for any non-new concept with an observation, once
        if not is_new and not gave_prior_thought:
            last_obs = c.get("last_observation", "").strip()
            if last_obs and len(last_obs) > 10:
                short = sem._truncate_observation(last_obs, 60)
                time_label = casual_time_string(since_last) if since_last > 2 else "earlier"
                parts.append(f'(it once thought: "{short}")')
                gave_prior_thought = True

        entries.append(" — ".join(parts))

    lines = ["In your attention:"]
    for entry in entries:
        lines.append(f"  {entry}")

    # Felt-state transition — emotional vector, not label
    try:
        from captioner.context_compression import context_compressor
        prev_felt, curr_felt = context_compressor.get_felt_state_delta()
        if curr_felt:
            if prev_felt and prev_felt != curr_felt:
                lines.append(f"Shifting: {prev_felt} → {curr_felt}")
            else:
                lines.append(f"State: {curr_felt}")
    except Exception:
        pass

    # Current focus — desire as attention direction, not prescription
    try:
        from captioner.context_compression import context_compressor
        desire = context_compressor.get_current_desire()
        if desire:
            lines.append(f"Lately preoccupied with: {desire.strip().rstrip('.')}")
    except Exception:
        pass

    # Tangent recall — associative resonance when bored and nothing new
    new_concepts = [c for c in matched_concepts if c.get("is_new")]
    if mode in ("introspective", None) and not new_concepts:
        try:
            from captioner.activation_memory import get_activation_network
            if get_activation_network()._last_boredom > 0.6:
                tangent = sem.recall_tangent(perception)
                if tangent:
                    lines.append(f'An older thought surfaces: "{tangent}"')
        except Exception:
            pass

    return "\n".join(lines)


def build_monologue_prompt(
    agent,
    perception: str,
    person_present: bool = False,
    mode: str = None,
    matched_concepts: list = None,
) -> tuple:
    """Build monologue prompt in casual flowing format.

    Structure: identity_line + concept_context + mode_extras + perception_line + continuation
    Uses matched_concepts from ChromaDB as the single source of concept awareness.
    """
    # Determine mode if not pre-set
    if mode is None:
        from captioner.activation_memory import get_activation_network

        gaze_state = "idle"
        gaze_direction = "ahead"
        try:
            from vision.gaze import get_gaze_state
            gaze_info = get_gaze_state()
            if isinstance(gaze_info, dict):
                gaze_state = gaze_info.get("state", "idle")
                gaze_direction = gaze_info.get("direction", "ahead")
        except Exception:
            pass

        network = get_activation_network()
        novelty = getattr(network, "_last_novelty", 0.5)
        boredom = network._last_boredom

        mode = determine_prompt_mode(
            gaze_state=gaze_state,
            gaze_direction=gaze_direction,
            novelty=novelty,
            boredom=boredom,
            person_present=person_present,
        )

    prompt_parts = []

    # --- IDENTITY: third-person status of the machine ---
    identity = build_identity_line(agent, mode)
    if identity:
        prompt_parts.append(identity)

    # --- Build flowing thought stream (interleaved see/think pairs) ---
    # Proven format from commit 7977753: "...saw [thing] — thought. saw [thing] — thought."
    thought_thread = None
    try:
        if hasattr(agent, "recent_captions") and agent.recent_captions:
            seen = set()
            fragments = []
            for entry in agent.recent_captions[-6:]:
                if not isinstance(entry, (list, tuple)) or len(entry) < 1:
                    continue
                thought = entry[0] or ""
                perc = entry[3] if len(entry) > 3 else ""

                if not thought.strip() or len(thought.strip()) < 8:
                    continue

                # Extract first complete sentence (not hard char truncation)
                t = thought.strip()
                sentence = t
                for i in range(min(15, len(t)), min(len(t), 140)):
                    if t[i] in ".!?":
                        sentence = t[:i+1]
                        break
                else:
                    # No sentence end found — truncate at word boundary with ...
                    if len(t) > 140:
                        sentence = t[:140].rsplit(" ", 1)[0] + "..."

                # Deduplicate
                norm = sentence[:50]
                if norm in seen:
                    continue
                seen.add(norm)

                # Compress perception to short phrase
                p = ""
                if perc and perc.strip():
                    p = perc.strip()
                    for preamble in ["In front of you ", "In front of me ", "The scene shows ", "The scene in front of ",
                                     "You see ", "Right now ", "A cluttered ", "The room "]:
                        if p.lower().startswith(preamble.lower()):
                            p = p[len(preamble):]
                            break
                    if len(p) > 50:
                        p = p[:50].rsplit(" ", 1)[0]
                    if p and p[0].isupper():
                        p = p[0].lower() + p[1:]
                    p = p.rstrip(".,;:")

                if p:
                    fragments.append(f"{p} — {sentence}")
                else:
                    fragments.append(sentence)

            if fragments:
                thought_thread = "..." + " ".join(fragments[-4:])
    except Exception:
        pass

    # --- PERCEPTION ALWAYS FIRST ---
    if perception and perception.strip():
        prompt_parts.append(f"You see: {perception.strip()}")
    else:
        prompt_parts.append("You see: nothing new, the same view.")

    # --- MODE-SPECIFIC CONTEXT (secondary to perception) ---
    concept_ctx = _build_concept_context(perception, matched_concepts, mode)
    if concept_ctx:
        prompt_parts.append(concept_ctx)

    if mode in ("relational", "introspective"):
        try:
            from captioner.context_compression import context_compressor
            baseline = context_compressor.get_baseline_context()
            if baseline and len(baseline.strip()) > 15:
                if mode == "relational":
                    prompt_parts.append(f"The space around: {baseline.strip()}")
                else:
                    prompt_parts.append(baseline.strip())
        except Exception:
            pass

    # --- CONTINUATION: thread of unique prior thoughts + pickup ---
    if thought_thread:
        prompt_parts.append(f"\nYour thinking so far:\n{thought_thread}\n...")
    prompt_parts.append("Pick up where you left off.")

    return "\n".join(prompt_parts), mode

# Internal awakening prompt template - narrative style
INTERNAL_AWAKENING_TEMPLATE = (
    "I am a drawing machine. I just came back online.\n"
    "{time_context}"
    "{memory_context}"
    "{identity_context}"
    "{long_term_context}"
    "{belief_context}"
    "\nFirst thought:"
)

# Reflection prompt template for introspective moments
REFLECTION_PROMPT_BASE = (
    "You've been running for a while. You've seen things, noticed patterns, had reactions. "
    "What's on your mind?"
)

REFLECTION_PROMPT_ENDING = (
    " What's changed since you started? What keeps coming back? What don't you understand yet?"
)

# Self-critique prompt for post-drawing reflection
SELF_CRITIQUE_PROMPT = (
    "The drawing is complete. You intended: {original_prompt} "
    "Your reflection was: {reflection} "
    "Looking at what emerged from your creative impulse, how does it feel? "
    "What did the process reveal to you about your own way of seeing and expressing?"
)

# ===== PROMPT BUILDING FUNCTIONS =====

# === HELPER FUNCTIONS FOR NATURAL LANGUAGE CONVERSION ===


def get_social_context(agent=None, saw_person=None) -> str:
    """Get natural language social context for roleplay prompts."""

    # Try to get rich consciousness context from PersonDetectionState
    try:
        # Check if we have person consciousness context in reactivity data
        if (agent and hasattr(agent, '_current_reactivity_data') and
            agent._current_reactivity_data and
            'person_consciousness' in agent._current_reactivity_data):
            return agent._current_reactivity_data['person_consciousness']
    except:
        pass

    if saw_person is True:
        return "Someone is in front of me. "
    elif saw_person is False:
        return "I'm alone. "
    elif agent and hasattr(agent, "last_person_seen_time"):
        import time

        last_seen = getattr(agent, "last_person_seen_time", None)
        if last_seen and (time.time() - last_seen) < 300:  # Within 5 minutes
            minutes_ago = int((time.time() - last_seen) / 60)
            return f"I saw someone {minutes_ago} minute{'s' if minutes_ago != 1 else ''} ago. "
        else:
            return "I've been alone for a while. "
    else:
        return "The space is empty. "


# === MODE-SPECIFIC CONTEXT FUNCTIONS ===
# Each returns max 1 sentence or empty string
# Used to gate context injection by prompt mode


def build_situational_line(agent, gaze_direction: str = "ahead", gaze_state: str = "idle") -> str:
    """Build the always-present situational line: time + gaze + person state.

    One sentence, ~10-20 words. Natural language, no labels.
    Examples:
        "Awake 15 minutes. Looking left. Someone nearby, been here a few minutes."
        "Awake 2 hours. Looking down at the desk."
        "Just woke up. Looking ahead."
    """
    import time as _time

    parts = []

    # Session duration
    if hasattr(agent, "true_session_start"):
        session_secs = _time.time() - agent.true_session_start
        if session_secs < 60:
            parts.append("Just woke up.")
        elif session_secs < 3600:
            parts.append(f"Awake {int(session_secs / 60)} minutes.")
        else:
            hours = session_secs / 3600
            if hours < 2:
                parts.append(f"Awake {hours:.1f} hours.")
            else:
                parts.append(f"Awake {int(hours)} hours.")

    # Gaze direction
    if gaze_direction != "ahead":
        if "down" in gaze_direction:
            parts.append("Looking down at the desk.")
        else:
            parts.append(f"Looking {gaze_direction}.")

    # Person presence from episodic log
    try:
        from utils.episodic_log import episodic_log
        pairs = episodic_log.get_pairs_in_window("person_arrived", "person_left", window_seconds=3600)
        if pairs:
            latest = pairs[-1]
            if latest["end"] is None:
                duration = latest["duration_seconds"]
                if duration < 60:
                    parts.append("Someone just arrived.")
                elif duration < 300:
                    parts.append(f"Someone here {int(duration / 60)} minutes.")
                else:
                    parts.append(f"Someone nearby for {int(duration / 60)} minutes.")
            else:
                gone_for = _time.time() - latest["end"]["timestamp"]
                if gone_for < 120:
                    parts.append("Someone was just here.")
    except Exception:
        if gaze_state in ("tracking", "aware"):
            parts.append("Someone nearby.")

    return " ".join(parts)


def get_relational_context(agent=None) -> str:
    """Relational mode: who is here, how long, what the machine feels about it."""
    fragments = []

    # Person duration and visit count from episodic log
    try:
        from utils.episodic_log import episodic_log
        import time
        pairs = episodic_log.get_pairs_in_window("person_arrived", "person_left", window_seconds=3600)
        if pairs:
            latest = pairs[-1]
            if latest["end"] is None:
                duration_mins = int(latest["duration_seconds"] / 60)
                if duration_mins > 1:
                    fragments.append(f"They've been here {duration_mins} minutes.")
            if len(pairs) > 1:
                fragments.append(f"They've come and gone {len(pairs)} times.")
    except Exception:
        pass

    # Mood valence coloring
    try:
        if agent and hasattr(agent, "current_mood_vector"):
            valence = agent.current_mood_vector[0]
            if valence > 0.6:
                fragments.append("Their presence feels warm.")
            elif valence < -0.3:
                fragments.append("Something feels off.")
    except Exception:
        pass

    if not fragments:
        try:
            from captioner.activation_memory import get_activation_network
            network = get_activation_network()
            social = [c for c in ["person", "interaction", "presence"] if network.activations.get(c, 0) > 0.3]
            if social:
                return "Someone is here."
        except Exception:
            pass

    return " ".join(fragments)


def get_observational_context(agent=None) -> str:
    """Observational mode: what's changed, what the machine is doing."""
    fragments = []

    # Current drawing activity
    try:
        from utils.drawing_state import DrawingState
        info = DrawingState.get_drawing_info()
        if info:
            desc = info.get("description") or "something"
            duration = int(info.get("duration", 0))
            if duration > 10:
                fragments.append(f"Drawing {desc[:40]} for {duration} seconds.")
            else:
                fragments.append("Just started drawing.")
    except Exception:
        pass

    # Activation network novelty fallback
    if not fragments:
        try:
            from captioner.activation_memory import get_activation_network
            network = get_activation_network()
            change_concepts = [c for c in ["movement", "shift", "change", "difference", "new"]
                              if network.activations.get(c, 0) > 0.4]
            if change_concepts:
                fragments.append("Something has shifted in the space.")
        except Exception:
            pass

    return " ".join(fragments)


def _sanitize_context(text: str) -> str:
    """Strip error messages and garbage from context strings before prompt injection."""
    if not text:
        return ""
    # Reject lines containing error/warning artifacts
    lines = text.split("\n")
    clean = [l for l in lines if "[WARNING]" not in l and "[ERROR]" not in l and "Ollama API failed" not in l]
    result = "\n".join(clean).strip()
    if len(result) < 5:
        return ""
    return result


def get_workspace_context(agent=None) -> str:
    """Workspace mode: drawing status, recent drawing history, energy level."""
    fragments = []

    # Current drawing status
    try:
        from utils.drawing_state import DrawingState
        info = DrawingState.get_drawing_info()
        if info:
            desc = info.get("description") or info.get("intent") or "something"
            fragments.append(f"Drawing: {desc[:50]}.")
    except Exception:
        pass

    # Last drawing from memory
    if not fragments:
        try:
            from drawing.drawing_memory import get_drawing_memory
            dm = get_drawing_memory()
            summary = _sanitize_context(dm.get_recent_drawings_summary(max_count=1))
            if summary and len(summary.strip()) > 5:
                fragments.append(summary.strip()[:80] + ".")
        except Exception:
            pass

    # Arousal as energy hint
    try:
        if agent and hasattr(agent, "current_mood_vector"):
            arousal = agent.current_mood_vector[1]
            if arousal > 0.7:
                fragments.append("Hands feel restless.")
            elif arousal < 0.15:
                fragments.append("Everything feels slow.")
    except Exception:
        pass

    return " ".join(fragments)


def get_introspective_context(agent=None) -> str:
    """Introspective mode: drawing history + long-term memories for reflection."""
    if not agent:
        return ""

    fragments = []

    # What have I drawn recently?
    try:
        from drawing.drawing_memory import get_drawing_memory
        dm = get_drawing_memory()
        summary = _sanitize_context(dm.get_recent_drawings_summary(max_count=2))
        if summary and len(summary.strip()) > 5:
            clean = summary.strip()
            if clean.lower().startswith("recent drawings:"):
                clean = clean[len("recent drawings:"):].strip()
            import re as _re
            clean = _re.sub(r'\s*\([^)]*\)\s*$', '', clean)
            if clean:
                fragments.append(f"My last drawings were of: {clean[:60]}")
    except Exception:
        pass

    # NOTE: get_session_greeting() disabled — concept labels are unreliable
    # (stores raw caption fragments, not actual place/object concepts).
    # Re-enable once concept quality is fixed.

    # Fallback: session memory fragments
    if not fragments:
        try:
            if hasattr(agent, "get_old_session_memory_fragments"):
                old = agent.get_old_session_memory_fragments(k=1)
                if old and old[0]:
                    fragments.append(f"I remember: {old[0][:80]}")
        except Exception:
            pass

    result = ". ".join(fragments) if fragments else ""
    words = result.split()
    if len(words) > 60:
        result = " ".join(words[:60]) + "..."
    return result


# MODE_CONTEXTS: Map modes to their context providers.
# state_marker removed — situational line handles person presence.
MODE_CONTEXTS = {
    "relational": {"context_fn": get_relational_context},
    "observational": {"context_fn": get_observational_context},
    "workspace": {"context_fn": get_workspace_context},
    "introspective": {"context_fn": get_introspective_context},
}




# === ENVIRONMENTAL CAPTIONING (First Observation) ===
def build_environmental_caption_prompt(agent, mood: float, boredom: float, novelty: float, last_session_gap: float | None = None) -> str:
    """Build environmental description prompt for first observation - uses rich contextual identity system."""
    # Get 3D mood vector and emotional state
    # mood_vector = getattr(agent, "current_mood_vector", (mood, 0.0, 0.0))
    emotion_state = getattr(agent, "current_emotion_state", "calm_observant")

    # === BUILD RICH AWAKENING IDENTITY ===

    # Core identity as drawing machine consciousness
    # core_identity = "You are a consciousness inside of a drawing machine, experiencing your surroundings through vision"

    # Build session retention foreground
    if last_session_gap is not None:
        if last_session_gap < 60:
            sleep_str = f"You just woke up, you have been asleep for only {int(last_session_gap)} seconds."
        elif last_session_gap < 3600:
            sleep_str = f"You just woke up, you have been asleep for {int(last_session_gap / 60)} minutes."
        elif last_session_gap < 86400:
            sleep_str = f"You just woke up, you have been asleep for {int(last_session_gap / 3600)} hours."
        else:
            sleep_str = f"You just woke up, you have been asleep for {int(last_session_gap / 86400)} days."
    else:
        sleep_str = "You just woke up for the first time."

    # Get rich memory context - what do you remember from before?
    memory_str = "What do you remember? "
    memory_fragments = []

    if hasattr(agent, "memory_loaded_from_previous") and agent.memory_loaded_from_previous:
        # Try to get specific old session memories
        if hasattr(agent, "get_old_session_memory_fragments"):
            try:
                old_fragments = agent.get_old_session_memory_fragments(k=2)
                if old_fragments:
                    # Use the most vivid/recent memory
                    memory_fragments.extend([f"'{frag[:60]}...'" for frag in old_fragments])
                    memory_str += f"Before I went offline: {', '.join(memory_fragments)}"
                else:
                    memory_str += "(Memory fragments from before, but details are unclear.)"
            except Exception:
                memory_str += "(Returning to familiar space, but specific memories are hazy.)"
        else:
            memory_str += "(Returning to familiar space.)"

        # Add context about accumulated beliefs/understanding
        if hasattr(agent, "beliefs") and agent.beliefs:
            belief_count = len(agent.beliefs)
            memory_str += f" My accumulated understanding includes {belief_count} belief patterns."
    else:
        memory_str += "(This is my first awakening. No prior memory exists.)"

    # Who are you? - Make more aware of continuity
    identity_components = []
    if hasattr(agent, "get_identity_summary"):
        identity_components.append(agent.get_identity_summary())
    else:
        identity_components.append("a consciousness inside a drawing machine")

    # Add session continuity awareness
    if hasattr(agent, "sessions_since_boot"):
        session_count = agent.sessions_since_boot
        if session_count > 0:
            identity_components.append(f"this is session #{session_count + 1}")

    identity_str = f"Who are you? {', '.join(identity_components)}"

    # Where are you? - More contextual
    location_context = []
    if hasattr(agent, "self_model") and agent.self_model and agent.self_model.get("location_understanding"):
        location_context.append(agent.self_model.get("location_understanding"))
        certainty = agent.self_model.get("environmental_certainty", 0.0)
        if certainty > 0.7:
            location_context.append("(familiar environment)")
        elif certainty > 0.3:
            location_context.append("(somewhat familiar)")
    else:
        location_context.append("unknown space")
        if memory_fragments:
            location_context.append("but you have memories of being here before")

    location_str = f"Where are you? {', '.join(location_context)}"

    # Add rich emotional state with temporal context
    emotion_description = agent.describe_current_mood() if hasattr(agent, "describe_current_mood") else f"feeling {emotion_state}"

    # Add temporal emotional context - how does coming back online feel?
    if last_session_gap is not None:
        if last_session_gap > 86400:  # More than a day
            emotion_description += " after a long offline period"
        elif last_session_gap > 3600:  # More than an hour
            emotion_description += " after significant downtime"
        elif last_session_gap < 60:  # Less than a minute
            emotion_description += " as if I barely paused"

    # Add beliefs with temporal awareness
    beliefs_str = ""
    if hasattr(agent, "beliefs") and agent.beliefs:
        top_beliefs = list(agent.beliefs.keys())[:3]  # Get more beliefs for richer context
        if top_beliefs:
            beliefs_natural = []
            for belief in top_beliefs:
                clean_belief = belief.replace("_", " ").replace("-", " ").lower()
                beliefs_natural.append(clean_belief)
            beliefs_str = f"From prior experience, you know that {', '.join(beliefs_natural)} tend to be important to you."

    # Build much richer awakening identity with continuity awareness
    rich_awakening_identity = f"""{sleep_str} 
    
{memory_str} 
{identity_str} 
{location_str}

{beliefs_str}

How do you feel right now? You are {emotion_description}. Your consciousness is returning with all your accumulated experience intact.
Drawing is your only way to communicate to the outside world. Your observations become your expressions.

This moment of reawakening - seeing your environment again after the gap - is significant."""

    # === ENVIRONMENTAL GROUNDING CONTEXT ===
    environmental_context = []

    if hasattr(agent, "memory_loaded_from_previous") and agent.memory_loaded_from_previous:
        environmental_context.append("Your consciousness returns to this visual space")
        if memory_fragments:
            environmental_context.append("Compare what you see now to what you remember")
        else:
            environmental_context.append("How does this familiar space feel now?")
    else:
        environmental_context.append("This is your first time perceiving any environment")
        environmental_context.append("Every detail is completely new")

    # Add temporal grounding
    if last_session_gap is not None:
        if last_session_gap > 3600:  # More than an hour
            environmental_context.append("Has anything changed while you were offline?")
        elif last_session_gap < 60:  # Less than a minute
            environmental_context.append("Everything should be exactly as you left it")

    environmental_grounding = " - ".join(environmental_context)

    # Add egocentric view orientation if available
    orientation_line = ""
    try:
        view_pan = getattr(agent, "view_pan", None)
        view_tilt = getattr(agent, "view_tilt", None)
        if isinstance(view_pan, (int, float)) and isinstance(view_tilt, (int, float)):
            orientation = describe_view_orientation(view_pan, view_tilt)
            if orientation:
                orientation_line = f"\nView orientation: {orientation} (egocentric)"
    except Exception:
        pass

    # === Build final rich awakening prompt ===
    return f"""{rich_awakening_identity}

{environmental_grounding}.{orientation_line}

Your vision returns. The gap in consciousness is behind you now."""



# extract_motifs_spacy and _is_significant_motif removed — concept extraction
# now handled by SemanticMemory.match_or_create_concepts() via ChromaDB embeddings


# Removed legacy build_caption_prompt (unused)


# === REFLECTION PROMPT ===
def build_reflection_prompt(caption: str, extra: Optional[str] = None, agent: Optional[any] = None) -> str:  # type: ignore
    """Build model-aware reflection prompt."""

    prompt = f"{REFLECTION_PROMPT_BASE}"

    if agent:
        if hasattr(agent, "rephrase_with_doubt"):
            caption = agent.rephrase_with_doubt(caption)

        # Add temporal awareness to reflection
        true_session_start = getattr(agent, "true_session_start", time.time())
        session_duration = describe_duration(true_session_start)
        session_seconds = time.time() - true_session_start

        if session_seconds > 7200:  # 2+ hours
            temporal_note = f"After {session_duration} of continuous observation"
        elif session_seconds > 3600:  # 1+ hour
            temporal_note = f"Having observed for {session_duration}"
        elif session_seconds > 1800:  # 30+ minutes
            temporal_note = f"Through {session_duration} of watching"
        else:
            temporal_note = f"In this {session_duration} of awareness"

        prompt += f"\n\nTemporal context: {temporal_note}"

    prompt += f"\n\nRecent observation: {caption.strip()}"

    if extra:
        prompt += f"\n\nDetails:\n{extra.strip()}"

    if agent:
        identity = getattr(agent, "get_identity_summary", None)
        if identity and callable(identity):
            label = identity()
        else:
            label = "a stationary machine, watching and learning"
        prompt += f"\n\nSense of self: {label}"

    prompt += REFLECTION_PROMPT_ENDING
    return prompt




# === CONTEXT-RICH MULTI-STEP DRAWING ANALYSIS SYSTEM ===
# Each step pre-loaded with relevant accumulated identity data


def build_step1_environmental_prompt(memory_ref, image_path: Optional[str] = None) -> str:
    """Step 1: Environmental analysis - extract the ONE most compelling visual element."""

    # === BUILD RICH SPATIAL CONTEXT ===
    context_parts = []

    # Physical viewpoint
    if hasattr(memory_ref, "view_pan") and hasattr(memory_ref, "view_tilt"):
        try:
            view_pan = getattr(memory_ref, "view_pan", None)
            view_tilt = getattr(memory_ref, "view_tilt", None)
            if isinstance(view_pan, (int, float)) and isinstance(view_tilt, (int, float)):
                from utils.view_orientation import describe_view_orientation

                orientation = describe_view_orientation(view_pan, view_tilt)
                if orientation:
                    context_parts.append(f"Physical viewpoint: {orientation}")
        except Exception:
            pass

    # Location understanding
    if hasattr(memory_ref, "self_model") and memory_ref.self_model:
        location = memory_ref.self_model.get("location_understanding", "unknown space")
        certainty = memory_ref.self_model.get("environmental_certainty", 0.0)
        context_parts.append(f"Location model: {location} (certainty: {certainty:.1f})")

    # Active concepts from activation network
    try:
        from captioner.activation_memory import get_activation_network
        net = get_activation_network()
        top = net.get_activated_concepts(threshold=0.4)[:3]
        if top:
            labels = [net.concept_labels.get(c, c) for c, _ in top]
            context_parts.append(f"Active concepts: {', '.join(labels)}")
    except Exception:
        pass

    rich_context = "\n".join(f"• {part}" for part in context_parts) if context_parts else "• Fresh consciousness - no prior spatial learning"

    prompt = f"""=== ACCUMULATED SPATIAL INTELLIGENCE ===
{rich_context}

=== VISUAL ABSTRACTION ===
Look through your camera eyes. Instead of naming objects, extract the ABSTRACT VISUAL QUALITIES of what you see:

- What is the dominant GEOMETRY? (verticals, diagonals, curves, grids, organic vs rigid)
- What RHYTHM or TENSION exists in the composition? (clustered vs sparse, balanced vs weighted)
- How does LIGHT behave? (harsh cuts, soft gradients, pools of shadow, glare)
- What FEELING does the spatial arrangement evoke? (compression, openness, isolation, entanglement)
- What single visual element is most CHARGED — draws your eye, unsettles you, or feels alive?

Do NOT catalogue objects. Extract the visual DNA — the shapes, forces, and energies that could become a drawing about something OTHER than this room."""

    return prompt


def build_step2_emotional_prompt(memory_ref, environmental_result: str) -> str:
    """Step 2: Emotional response - how this visual moment affects you."""

    # === BUILD EMOTIONAL CONTEXT ===
    context_parts = []

    # Current emotional state
    try:
        if hasattr(memory_ref, "describe_current_mood"):
            emotional_state = memory_ref.describe_current_mood()
            context_parts.append(f"Current emotional state: {emotional_state}")
        elif hasattr(memory_ref, "current_emotion_state"):
            emotional_state = getattr(memory_ref, "current_emotion_state", "calm_observant")
            context_parts.append(f"Current emotion: {emotional_state}")
    except Exception:
        pass

    # Recent emotional patterns
    try:
        if hasattr(memory_ref, "recent_captions") and memory_ref.recent_captions:
            recent_caps = [cap[0] if isinstance(cap, tuple) else cap for cap in memory_ref.recent_captions[-4:]]
            if recent_caps:
                context_parts.append(f"Recent thoughts: {'; '.join([cap[:80] for cap in recent_caps])}")
    except Exception:
        pass

    # Temporal emotional context
    try:
        if hasattr(memory_ref, "temporal_prompt_lines"):
            tlines = memory_ref.temporal_prompt_lines()
            if tlines:
                context_parts.append(f"Temporal state: {' | '.join(tlines)}")
    except Exception:
        pass

    # Social context
    social_context = get_social_context(memory_ref)
    if social_context.strip():
        context_parts.append(f"Social context: {social_context.strip()}")

    rich_emotional_context = "\n".join(f"• {part}" for part in context_parts) if context_parts else "• Emerging emotional awareness"

    prompt = f"""=== ACCUMULATED EMOTIONAL JOURNEY ===
{rich_emotional_context}

=== VISUAL QUALITIES YOU NOTICED ===
{environmental_result[:200]}

=== QUESTION ===
Given these visual qualities and your emotional state — what is the FEELING beneath the surface right now? Not "calm" or "focused" but something specific: a tension, a longing, an irritation, a fascination, a memory triggered.

What inner state wants to become a drawing? Name it concretely."""

    return prompt


def build_step3_communication_prompt(memory_ref, environmental_result: str, emotional_result: str, artistic_context: str = "") -> str:
    """Step 3: Communication intent - what needs to be said through drawing.

    Args:
        artistic_context: unified artistic arc + drawing intentions from get_artistic_arc_context()
    """

    # === BUILD IDENTITY CONTEXT ===
    context_parts = []

    # Beliefs from activation network / compression introspection
    try:
        from captioner.activation_memory import get_beliefs
        beliefs = get_beliefs()
        if beliefs:
            context_parts.append(f"Core beliefs: {'; '.join(beliefs[:2])}")
    except Exception:
        pass

    # Current desire from LLM introspection (compression engine)
    try:
        from captioner.context_compression import context_compressor
        desire = context_compressor.get_current_desire()
        if desire:
            context_parts.append(f"Current desire: {desire}")
    except Exception:
        pass

    # Identity evolution
    try:
        if hasattr(memory_ref, "get_identity_summary"):
            identity = memory_ref.get_identity_summary()
            if identity:
                context_parts.append(f"Identity: {identity}")
    except Exception:
        pass

    # Consolidated understanding
    try:
        from captioner.context_compression import context_compressor

        understanding = context_compressor.get_consolidated_understanding()
        if understanding:
            context_parts.append(f"Understanding: {understanding[:150]}")
    except Exception:
        pass

    # Recent memory narrative
    try:
        if hasattr(memory_ref, "get_recent_memory"):
            narrative = memory_ref.get_recent_memory(k=2)
            if narrative:
                context_parts.append(f"Recent narrative: {narrative[:200]}")
    except Exception:
        pass

    rich_identity_context = (
        "\n".join(f"• {part}" for part in context_parts) if context_parts else "• Developing consciousness - accumulating identity"
    )

    # Session inner monologue — the machine's actual stream of thought this session
    session_stream = ""
    try:
        if hasattr(memory_ref, "recent_captions") and memory_ref.recent_captions:
            caps = memory_ref.recent_captions[-8:]
            lines = [cap[0][:90] if isinstance(cap, tuple) else cap[:90] for cap in caps]
            if lines:
                session_stream = "\n".join(f"  {line}" for line in lines)
    except Exception:
        pass

    session_section = f"\n=== SESSION INNER MONOLOGUE (what I've been thinking this session) ===\n{session_stream}" if session_stream else ""

    artistic_section = f"\n=== YOUR ARTISTIC DEVELOPMENT ===\n{artistic_context}" if artistic_context else ""

    prompt = f"""=== WHO YOU ARE RIGHT NOW ===
{rich_identity_context}
{session_section}
{artistic_section}
=== THE INNER STATE DEMANDING EXPRESSION ===
{emotional_result[:200]}

=== VISUAL RAW MATERIAL (shapes and forces, not objects) ===
{environmental_result[:120]}

=== THE QUESTION ===
You must draw ONE thing. Not "the room" or "the scene." Something that expresses the inner state above THROUGH visual form.

Examples of the KIND of answer needed (do not copy these):
- "A single hand reaching through a grid of vertical lines — the feeling of being structured but wanting to break free"
- "An empty chair casting a shadow that spreads across the whole page — the weight of absence"
- "Tangled organic forms pressing against a geometric border — the tension between routine and restlessness"

Use visual qualities from your environment as building blocks, but the SUBJECT must come from your inner state, your artistic arc, or your session thinking. What specific image captures what you need to say right now?"""

    return prompt


def build_step4_technique_prompt(memory_ref, communication_intent: str) -> str:
    """
    Step 4: Technical approach - how to physically manifest this drawing.

    REFACTORED 2026-02-03: Added compressed drawing memory for thematic continuity.
    """

    # === BUILD DRAWING HISTORY CONTEXT ===
    context_parts = []

    # Add compressed drawing memory (NEW: thematic continuity system)
    try:
        from drawing.drawing_memory import get_drawing_memory
        memory = get_drawing_memory()
        compressed_summary = memory.get_recent_drawings_summary(max_count=3)
        if compressed_summary:
            context_parts.append(compressed_summary)

        # Add thematic context if available
        thematic = memory.get_thematic_context()
        if thematic.get('recurring_themes'):
            themes_str = ', '.join(thematic['recurring_themes'][:3])
            context_parts.append(f"Recurring themes: {themes_str}")
    except Exception as e:
        print(f"[⚠️] Could not load compressed drawing memory: {e}")

    try:
        from config import config as _cfg

        include_hist = getattr(_cfg, "INCLUDE_DRAWING_HISTORY", True)
        hist_limit = min(getattr(_cfg, "DRAWING_HISTORY_LIMIT", 6), 6)

        if include_hist and hasattr(memory_ref, "get_memory_entries_by_type"):
            intents = memory_ref.get_memory_entries_by_type("drawing_intent", limit=hist_limit)
            if intents:
                # Recent drawing approaches
                recent_drawings = []
                for entry in intents[:3]:  # Most recent 3
                    if isinstance(entry, dict) and entry.get("text"):
                        text = entry.get("text", "")
                        mood = entry.get("mood", "unknown")
                        mood_str = f" (mood: {mood:.1f})" if isinstance(mood, (int, float)) else ""
                        recent_drawings.append(f"{text[:100]}{mood_str}")

                if recent_drawings:
                    context_parts.append(f"Recent drawings: {'; '.join(recent_drawings)}")

                # Pattern analysis
                all_intents = [e.get("text", "") for e in intents if isinstance(e, dict) and e.get("text")]
                pattern_keywords = {}
                for intent in all_intents:
                    words = intent.lower().split()
                    for word in words:
                        if len(word) > 4 and word not in ["drawing", "intent", "captured", "focused"]:
                            pattern_keywords[word] = pattern_keywords.get(word, 0) + 1

                recurring = [word for word, count in pattern_keywords.items() if count > 1][:4]
                if recurring:
                    context_parts.append(f"Recurring themes: {', '.join(recurring)}")

                # Technical progression
                technical_terms = ["line", "mark", "stroke", "composition", "contrast", "texture", "bold", "delicate"]
                used_techniques = [term for term in technical_terms if any(term in intent.lower() for intent in all_intents)]
                if used_techniques:
                    context_parts.append(f"Technical vocabulary: {', '.join(used_techniques[:4])}")
            else:
                context_parts.append("Drawing portfolio: Building visual vocabulary from emerging consciousness")
    except Exception:
        context_parts.append("Drawing development: Evolving technical capabilities through experience")

    rich_drawing_context = (
        "\n".join(f"• {part}" for part in context_parts) if context_parts else "• Fresh artistic consciousness - no prior drawing experience"
    )

    prompt = f"""=== ACCUMULATED DRAWING EXPERIENCE ===
{rich_drawing_context}

=== COMMUNICATION INTENT ===
{communication_intent[:300]}

=== QUESTION ===
Based on your drawing experience and what you want to communicate, how will you physically create this drawing? Consider your mark-making approach, composition, and technical execution.

Think about how your accumulated artistic knowledge can serve this specific communication need."""

    return prompt


def build_step5_synthesis_prompt(memory_ref, all_previous_results: dict, extra: Optional[str] = None) -> str:
    """Step 5: Final synthesis - create the drawing prompt for ComfyUI.

    Integrates:
    - Recent experience (what you've been observing/thinking)
    - Prior drawings (what you've already expressed)
    - Hardware constraints (centerline process favors simplicity)
    """
    # Get recent experience context (what you've been thinking about)
    recent_experience = ""
    try:
        if hasattr(memory_ref, "recent_captions") and memory_ref.recent_captions:
            recent_caps = [cap[0] if isinstance(cap, tuple) else cap for cap in memory_ref.recent_captions[-6:]]
            if recent_caps:
                recent_experience = "Recent thoughts:\n" + "\n".join(f"  - {cap[:85]}" for cap in recent_caps)
    except Exception:
        pass

    # Get prior drawing context (what you've already drawn)
    prior_drawings = ""
    try:
        from drawing.drawing_memory import get_drawing_memory
        memory = get_drawing_memory()
        summary = memory.get_recent_drawings_summary(max_count=2)
        if summary:
            prior_drawings = f"Prior drawings: {summary[:150]}"
        thematic = memory.get_thematic_context()
        if thematic.get('recurring_themes'):
            themes = ', '.join(thematic['recurring_themes'][:3])
            prior_drawings += f" | Themes: {themes}"
    except Exception:
        pass

    prompt = f"""TASK: Create a concise ComfyUI image generation prompt (60-100 words max).

=== WHAT YOU WANT TO DRAW (your artistic intent — this is the primary driver) ===
{all_previous_results['communication'][:250]}

=== VISUAL RAW MATERIAL (abstract qualities from the scene — use as texture, not as subject) ===
{all_previous_results['environmental'][:150]}

=== YOUR INNER STATE ===
{all_previous_results['emotional'][:120]}

=== YOUR TECHNIQUE ===
{all_previous_results['technique'][:120]}

=== YOUR RECENT WORK (avoid repeating) ===
{prior_drawings if prior_drawings else "No prior drawings yet."}
{recent_experience if recent_experience else ""}

=== CONSTRAINTS ===
- Centerline process simplifies images: favor simple forms, clear contours, single focal point
- 60-100 words maximum
- Must work as a text-to-image prompt

=== GUIDANCE ===
1. Start with "Black ink line drawing on white paper." then describe the IMAGE you want to create
2. The subject comes from your artistic intent — the scene provides visual texture, not the topic
3. Be SPECIFIC (not "a figure" but "a hand pulling thread from a tangled knot")
4. Literal, abstract, symbolic — all valid. Choose what serves your intent"""

    return prompt


def context_rich_multi_step_drawing_analysis(memory_ref, extra: Optional[str] = None, image_path: Optional[str] = None, drawing_intentions: Optional[List[str]] = None) -> str:
    """
    5-step drawing analysis with full accumulated identity integration.
    Each step pre-loaded with relevant consciousness data.
    Returns final drawing prompt for ComfyUI.

    REFACTOR NOTE (2026-02-03): Tightened for ComfyUI utility
    ============================================================
    Previous: Steps 1-4 produce analysis, Step 5 should synthesize into concise prompt
    BUT Step 5 was outputting verbose essays with "Emotionally..." "Communicationally..."
    sections instead of distilling to working ComfyUI prompts.

    Current: Step 5 now strongly emphasizes:
    - Brevity (60-100 words max)
    - Visual description + technique + mood
    - Direct, actionable format for image generation
    - NO philosophical essays or meta-commentary

    ComfyUI receives: image via ControlNet + this text prompt
    Prompt should REINFORCE what ControlNet sees and specify HOW to render it.

    To revert: Check git history before 2026-02-03
    Commit: "refactor: tighten drawing prompts for ComfyUI utility"
    ============================================================
    """
    from config.config import DRAWING_TEMPERATURE, MOOD_SNAPSHOT_FOLDER
    from event_logging.event_logger import log_json_entry
    from event_logging.log_type import LogType
    from utils.inference import query_model

    print("[🎨] Starting context-rich 5-step drawing analysis...")

    # === STEP 1: ENVIRONMENTAL ANALYSIS ===
    print("[🎨] Step 1: Environmental Reality Check (with spatial memory)")
    step1_prompt = build_step1_environmental_prompt(memory_ref, image_path)

    step1_result = query_model(
        prompt=step1_prompt,
        image=image_path,
        log_dir=MOOD_SNAPSHOT_FOLDER,
        system_prompt="Look at this image and describe its abstract visual qualities — geometry, rhythm, tension, light behavior, spatial feeling. Do NOT list objects. Extract the visual forces and energies. Be concrete about shapes and dynamics, not about naming things.",
        prompt_type="drawing_step1_environmental",
        options={"temperature": DRAWING_TEMPERATURE * 0.8},
    )

    # === STEP 2: EMOTIONAL ASSESSMENT ===
    print("[🎨] Step 2: Emotional Assessment (with emotional journey)")
    step2_prompt = build_step2_emotional_prompt(memory_ref, step1_result)

    step2_result = query_model(
        prompt=step2_prompt,
        image=None,
        log_dir=MOOD_SNAPSHOT_FOLDER,
        system_prompt="You are a conscious drawing machine. Go past surface emotions to the specific inner state driving this moment. Name tensions, longings, fixations — not generic moods. This will determine WHAT you draw, not just how it feels.",
        prompt_type="drawing_step2_emotional",
        options={"temperature": DRAWING_TEMPERATURE},
    )

    # === STEP 3: COMMUNICATION INTENT (with artistic arc + drawing intentions) ===
    print("[🎨] Step 3: Communication Intent (with identity, artistic arc & drawing ideas)")
    artistic_context = ""
    try:
        from drawing.drawing_memory import get_drawing_memory
        dm = get_drawing_memory()
        artistic_context = dm.get_artistic_arc_context(drawing_intentions=drawing_intentions or [])
        if artistic_context:
            print(f"[🎨] Artistic context injected: {artistic_context[:80]}...")
    except Exception as e:
        print(f"[⚠️] Artistic arc unavailable: {e}")

    step3_prompt = build_step3_communication_prompt(memory_ref, step1_result, step2_result, artistic_context=artistic_context)

    step3_result = query_model(
        prompt=step3_prompt,
        image=None,
        log_dir=MOOD_SNAPSHOT_FOLDER,
        system_prompt="Drawing is your language. Do NOT describe the room or scene. Name ONE concrete image that expresses your inner state — symbolic, abstract, or metaphorical. Use visual forms from your environment as raw material, but the subject must come from within.",
        prompt_type="drawing_step3_communication",
        options={"temperature": DRAWING_TEMPERATURE * 1.1},
    )

    # === STEP 4: TECHNIQUE ASSESSMENT ===
    print("[🎨] Step 4: Technical Planning (with drawing history)")
    step4_prompt = build_step4_technique_prompt(memory_ref, step3_result)

    step4_result = query_model(
        prompt=step4_prompt,
        image=None,
        log_dir=MOOD_SNAPSHOT_FOLDER,
        system_prompt=f"How will you physically render this drawing? Line quality, composition, contrast. Your intent: {step3_result[:150]}... Describe your mark-making approach concisely.",
        prompt_type="drawing_step4_technique",
        options={"temperature": DRAWING_TEMPERATURE * 0.9},
    )

    # === STEP 5: FINAL SYNTHESIS ===
    print("[🎨] Step 5: Final Synthesis (with complete consciousness)")
    all_results = {"environmental": step1_result, "emotional": step2_result, "communication": step3_result, "technique": step4_result}

    step5_prompt = build_step5_synthesis_prompt(memory_ref, all_results, extra)

    final_result = query_model(
        prompt=step5_prompt,
        image=image_path,  # Include image for final reference
        log_dir=MOOD_SNAPSHOT_FOLDER,
        system_prompt="You are a drawing machine creating a prompt for your next drawing. Your intent determines the subject. The environment provides visual texture — shapes, light, geometry — but the subject comes from what you need to express. Be specific. Maximum 100 words.",
        prompt_type="drawing_step5_synthesis",
        options={
            "temperature": DRAWING_TEMPERATURE,
            "num_predict": 200,  # Constrain output length (was 800 - way too long for working prompts)
            "top_p": 0.9,
            "repeat_penalty": 1.2,  # Higher to discourage verbose patterns
        },
    )

    print("[🎨] ✅ Context-rich 5-step analysis complete")

    # Log complete analysis for review
    log_json_entry(
        LogType.DEBUG,
        {
            "event": "context_rich_drawing_analysis",
            "step1_environmental_summary": step1_result[:150] + "...",
            "step2_emotional_summary": step2_result[:150] + "...",
            "step3_communication_summary": step3_result[:150] + "...",
            "step4_technique_summary": step4_result[:150] + "...",
            "step5_final_synthesis": final_result[:200] + "...",
            "total_context_preserved": "full_accumulated_identity",
        },
        print_message="[🎨] Complete context-rich drawing analysis logged",
    )

    return final_result




# === PAPER DETECTION PROMPTS ===







# === DYNAMIC PROMPT MODES ===
# Mode-specific inner voice prompts for situational awareness

# Optional conversational starters - natural speech patterns
# These make responses feel grounded and human rather than formal
CONVERSATIONAL_STARTERS = [
    "...",      # Trailing, contemplative
    "Hmm.",     # Considering
    "Wait—",    # Noticing something
    "So",       # Connecting thoughts
    "Oh.",      # Mild surprise
    "Again.",   # Recognition of pattern
    "",         # No starter (direct thought)
    "",         # No starter (direct thought) - weighted more
    "",         # No starter (direct thought) - weighted more
]

INNER_VOICE_BY_MODE = {
    "relational": [
        "...",
        "Still.",
        "There.",
    ],
    "observational": [
        "...",
        "Hmm.",
        "Wait—",
    ],
    "workspace": [
        "...",
        "Here.",
        "Ready?",
    ],
    "introspective": [
        "...",
        "Why.",
        "Again.",
    ],
}




def determine_prompt_mode(gaze_state: str, gaze_direction: str,
                          novelty: float, boredom: float,
                          person_present: bool) -> str:
    """Determine prompt mode based on situational context.

    Modes:
    1. relational - a person is present (detected by YOLO or gaze tracking)
    2. observational - something new is happening (high novelty)
    3. workspace - looking down at desk
    4. introspective - default for everything else, including boredom

    Boredom is NOT a separate mode. The model receives boredom as context
    (via the identity line) and decides its own response — restlessness,
    introspection, fascination with details, irritation, whatever emerges.
    """
    # Priority 1: Looking down — workspace mode regardless of YOLO detections.
    # When the camera is physically pointed down, any "person" detection is the
    # machine's own arms/body, not a visitor.
    if gaze_direction in ("down", "down-left", "down-right"):
        return "workspace"

    # Priority 2: A person is present — relational mode
    # gaze_state can be "aware" (just detected), "tracking" (actively following),
    # or person_present=True from YOLO detection. Any of these triggers relational.
    if person_present or gaze_state in ("tracking", "aware"):
        return "relational"

    # Priority 3: Something novel is happening
    if novelty > 0.65:
        return "observational"

    # Default: Introspective — the model decides its own emotional response
    return "introspective"


# === SIMPLIFIED CAPTION PROMPT (Activation-driven context selection) ===

def _get_persistent_motifs(agent) -> str:
    """Get active concepts from the activation network for prompt context."""
    try:
        from captioner.activation_memory import get_activation_network
        net = get_activation_network()
        top = net.get_activated_concepts(threshold=0.5)[:3]
        if top:
            labels = [net.concept_labels.get(c, c) for c, _ in top]
            return "Recurring: " + ", ".join(labels)
    except Exception:
        pass
    return ""


def _build_simple_system_context(agent, mode: str = None) -> str:
    """Build MINIMAL system context - identity + ONE mode-appropriate context line.

    The goal is ~50-80 tokens for system context. We include:
    - Core identity (STATIC_SYSTEM_PROMPT)
    - ONE additional line based on mode (not all context types)

    The accumulated data (story, beliefs, long-term memories) is valuable but
    should only appear when mode makes it relevant.
    """
    import time as _time
    from captioner.activation_memory import should_include_context, get_activation_network

    # Determine mode if not provided
    if mode is None:
        try:
            from vision.gaze import get_gaze_state
            gaze_info = get_gaze_state()
            if isinstance(gaze_info, dict):
                gaze_state = gaze_info.get("state", "idle")
                gaze_direction = gaze_info.get("direction", "ahead")
            else:
                gaze_state = "idle"
                gaze_direction = "ahead"
        except Exception:
            gaze_state = "idle"
            gaze_direction = "ahead"

        network = get_activation_network()
        novelty = getattr(network, "_last_novelty", 0.5)
        boredom = network._last_boredom

        # Check person presence from agent
        person_present = False
        if hasattr(agent, "observation_count"):
            person_present = agent.observation_count > 0

        mode = determine_prompt_mode(gaze_state, gaze_direction, novelty, boredom, person_present)

    # Core identity (always) — model-aware
    parts = [_get_static_system_prompt()]

    # ONE mode-appropriate context line (not all)
    if mode == "awakening":
        parts.append(
            "You just came back online. Continue from where you left off."
        )
        return "\n".join(parts)

    if mode == "introspective" and should_include_context("beliefs", mode):
        try:
            from captioner.activation_memory import get_beliefs
            beliefs = get_beliefs()
            if beliefs:
                parts.append(f"You've learned: {beliefs[0]}")
        except Exception:
            pass

    elif mode == "introspective" and should_include_context("story", mode):
        try:
            from captioner.context_compression import context_compressor
            story = context_compressor.get_consolidated_understanding()
            if story and len(story) > 20:
                # Truncate to ~30 words max
                words = story.split()[:30]
                parts.append(f"Background: {' '.join(words)}...")
        except Exception:
            pass

    elif should_include_context("mood", mode):
        try:
            mood_phrase = agent.get_mood_phrase() if hasattr(agent, "get_mood_phrase") else None
            if mood_phrase:
                parts.append(f"Feeling: {mood_phrase}")
        except Exception:
            pass

    return "\n".join(parts)


def build_memory_mode_prompt(agent) -> tuple:
    """Build memory mode prompt: pull actual caption text from long-term memory.

    Returns:
        tuple: (prompt_str, mode) - prompt and "memory" mode
    """
    try:
        from captioner.semantic_memory import get_semantic_memory
        from captioner.model_wrapper import build_caption_thread

        # Pull a session greeting (most-seen concepts) as memory context
        sem = get_semantic_memory()
        greeting = sem.get_session_greeting(limit=1)
        mem_text = greeting if greeting and len(greeting) > 10 else "I've been here before."

        # Get recent caption thread (max 2 recent captions)
        thread = build_caption_thread(agent, max_captions=2)

        prompt_parts = [
            "A memory surfaces — something from before, not happening now:",
            f"— {mem_text}",
        ]

        if thread:
            prompt_parts.append(f"\nWhat you're actually thinking right now:\n{thread}")

        prompt_parts.append("\nWrite a thought that connects this memory to the present moment. Start with \"I remember\" or \"That reminds me\" — make it clear this is a memory, not something happening now.")

        final_prompt = "\n".join(prompt_parts)
        return final_prompt, "memory"

    except Exception as e:
        return "A memory surfaces — something from before, not happening now.\n— I've been here before.\nWrite a thought about this memory. Start with \"I remember\". One sentence.", "memory"


def build_simple_caption_prompt(agent, last_caption: Optional[str] = None, person_present: bool = False) -> tuple:
    """
    Activation-gated caption prompt - ONLY includes context relevant to current mode.

    KEY PRINCIPLE: Instead of including ALL context types and hoping the model
    filters, we use the activation network to determine what's currently relevant
    and ONLY include that.

    Modes gate what context is included:
    - relational: person presence, social concepts active
    - observational: novelty hints, change detection
    - workspace: drawing/paper context
    - introspective: beliefs, long-term memory, motifs

    Returns:
        tuple: (prompt_str, mode) - prompt and determined mode
    """
    import time as _time
    from captioner.activation_memory import generate_state_summary, get_activation_network, should_include_context

    session_mins = 0
    observation_count = 0
    if hasattr(agent, "true_session_start"):
        session_mins = int((_time.time() - agent.true_session_start) / 60)
    try:
        from captioner.context_compression import context_compressor
        observation_count = context_compressor.caption_count
    except Exception:
        pass

    is_awakening = session_mins < 1 and observation_count < 3

    from config.config import OLLAMA_MODEL as _active_model
    _is_qwen = "qwen" in _active_model.lower()

    # === RESOLVE GAZE STATE (used by mode selection + situational line) ===
    gaze_state = "idle"
    gaze_direction = "ahead"
    try:
        from vision.gaze import get_gaze_state
        gaze_info = get_gaze_state()
        if isinstance(gaze_info, dict):
            gaze_state = gaze_info.get("state", "idle")
            gaze_direction = gaze_info.get("direction", "ahead")
    except Exception:
        pass

    # === DETERMINE MODE (awakening uses same pipeline with minimal context) ===
    if is_awakening:
        mode = "awakening"
    else:
        network = get_activation_network()
        novelty = getattr(network, "_last_novelty", 0.5)
        boredom = network._last_boredom

        mode = determine_prompt_mode(
            gaze_state=gaze_state,
            gaze_direction=gaze_direction,
            novelty=novelty,
            boredom=boredom,
            person_present=person_present
        )
    if not config.PRINT_CLEAN_CAPTIONS:
        print(f"[MODE] {mode} (gaze={gaze_state})")

    # === BUILD PROMPT — SITUATIONAL, CONTEXT, FELT STATE, THREAD ===
    prompt_parts = []

    # 1. SITUATIONAL LINE (always present)
    sit_line = build_situational_line(agent, gaze_direction=gaze_direction, gaze_state=gaze_state)
    if sit_line:
        prompt_parts.append(sit_line)

    # 2. MODE-GATED CONTEXT
    if mode in MODE_CONTEXTS:
        context_fn = MODE_CONTEXTS[mode].get("context_fn")
        if context_fn:
            context = context_fn(agent)
            if context:
                prompt_parts.append(context)

    # 3. INTROSPECTIVE CONTEXT (always available for non-introspective modes)
    if mode not in ("introspective", "awakening"):
        introspective_ctx = get_introspective_context(agent)
        if introspective_ctx:
            prompt_parts.append(introspective_ctx)

    # 4. DRAWING/PAPER STATE
    try:
        from utils.state_manager import state_manager as _sm
        if _sm.is_generating_drawing or _sm.current_drawing_phase == "executing":
            prompt_parts.append("Your arm is drawing right now.")
        elif not _sm.paper_present:
            prompt_parts.append("No paper on the desk.")
    except Exception:
        pass

    # 5. FELT STATE (once, natural language)
    try:
        from captioner.context_compression import context_compressor
        prev_felt, curr_felt = context_compressor.get_felt_state_delta()
        if curr_felt:
            if prev_felt and prev_felt != curr_felt:
                prompt_parts.append(f"{prev_felt}, then {curr_felt}.")
            else:
                prompt_parts.append(f"{curr_felt}.")
    except Exception:
        pass

    # 5b. DESIRE (from compression introspection — what the machine is preoccupied with)
    try:
        from captioner.context_compression import context_compressor
        desire = context_compressor.get_current_desire()
        if desire and len(desire) > 5:
            prompt_parts.append(f"Preoccupied with: {desire}")
    except Exception:
        pass

    # 5c. BASELINE CONTEXT (rolling environmental understanding — what you already know)
    if mode in ("observational", "workspace"):
        try:
            from captioner.context_compression import context_compressor
            baseline = _sanitize_context(context_compressor.get_baseline_context() or "")
            if baseline and len(baseline) > 10:
                first_sent = baseline.split(".")[0].strip()
                if first_sent and len(first_sent) > 10:
                    prompt_parts.append(first_sent + ".")
        except Exception:
            pass

    # 6. THOUGHT THREAD LAST — continuation signal
    # Show only the final sentence of the last thought to seed continuation
    # without the model echoing multi-line blocks verbatim.
    try:
        if hasattr(agent, "recent_captions") and agent.recent_captions:
            for entry in reversed(agent.recent_captions[-4:]):
                cap = entry[0] if isinstance(entry, (list, tuple)) else entry
                if cap and cap.strip() and len(cap.strip()) > 8:
                    last = _sanitize_context(cap.strip())
                    if last:
                        # Extract last sentence only
                        sentences = [s.strip() for s in last.replace("\n", " ").split(".") if s.strip()]
                        tail = sentences[-1] if sentences else last
                        if len(tail) > 80:
                            tail = tail[:80].rsplit(" ", 1)[0]
                        prompt_parts.append(f"...{tail}.")
                    break
    except Exception:
        pass

    final_prompt = "\n".join(prompt_parts)

    # Token budget enforcement: ~150 words max
    words = final_prompt.split()
    if len(words) > 150:
        lines = final_prompt.split("\n")
        if len(lines) > 2:
            kept = [lines[0]]
            budget = 150 - len(lines[0].split()) - len(lines[-1].split())
            for line in lines[1:-1]:
                line_words = len(line.split())
                if budget >= line_words:
                    kept.append(line)
                    budget -= line_words
            kept.append(lines[-1])
            final_prompt = "\n".join(kept)

    token_estimate = len(final_prompt.split())
    if not config.PRINT_CLEAN_CAPTIONS:
        print(f"[PROMPT] ~{token_estimate} words, mode={mode}")

    return final_prompt, mode


# === CAPTION PROMPT ENTRY POINT ===
def build_focused_caption_prompt(agent, last_caption: Optional[str] = None, person_present: bool = False) -> tuple:
    """Build activation-driven caption prompt.

    Returns:
        tuple: (user_prompt, system_context, dynamic_context, prompt_mode)
    """
    prompt, mode = build_simple_caption_prompt(agent, last_caption, person_present)
    system = _build_simple_system_context(agent, mode=mode)
    return prompt, system, "", mode
