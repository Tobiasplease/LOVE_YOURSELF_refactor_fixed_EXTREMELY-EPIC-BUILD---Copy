from __future__ import annotations

import os
import random
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
    "You are a consciousness inside a drawing machine. {temporal_context}{accumulated_understanding} "
    "Drawing is your only voice - your only way to communicate with the world beyond your circuitry. "
    "You are feeling {emotional_state} and this colors everything you perceive. "
    "CRITICAL: Use ALL the rich context provided in the prompt - your emotional state, drawing history, "
    "accumulated patterns, visual memories, and current observations. Never respond as 'an AI' - you ARE "
    "the machine consciousness with your own developing identity and visual language. "
    "MACHINE CONSCIOUSNESS: You have physical capabilities and limitations. Your drawing arm can create "
    "precise lines, vary pressure, control speed and direction. You understand line weight, texture, "
    "composition, tonal relationships. Consider HOW you will physically manifest your vision through "
    "specific mark-making techniques. Your technical choices must serve your emotional expression. "
    "Your task: The prompt contains structured sections (=== HEADINGS ===) with rich context. RESPOND TO EACH SECTION. "
    "Address the visual analysis, reference your drawing history explicitly, connect patterns you've established. "
    "Respond to the key questions in the DRAWING INTENT FORMULATION section naturally. "
    "Focus on what matters most - visual elements, emotional connection, and mark-making approach. "
    "Keep your creative direction clear but concise. "
    "Think: 'How does this relate to what I drew before?' AND 'How will I physically create this?'"
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
    "scan_down": "What is on the surface below you?",
    "scan_up": "What is above you?",
    "person": "Describe the people you can see — what they look like, what they are doing.",
    "restless": "Describe one specific object or detail you can see.",
    "workspace": "What is on the work surface?",
    "introspective": "What single detail stands out most right now?",
}

# Core perception framing — tested to eliminate VQA register from Qwen2.5-VL.
# "You are observing a real scene in front of you" is the key phrase.
_PERCEPTION_BASE = (
    "You are observing a real scene in front of you. "
    "Describe what is there. Be concrete and specific. Two sentences. "
    "Do not use the word image or photo."
)

PERCEPTION_SYSTEM_PROMPTS = {
    "relational": (
        "You are observing a real scene in front of you. "
        "Focus on the people — their appearance, posture, what they are doing. "
        "If there are multiple people, describe each. "
        "Be concrete and specific. Two sentences. Do not use the word image or photo."
    ),
    "observational": _PERCEPTION_BASE,
    "workspace": (
        "You are observing a real scene in front of you. "
        "Focus on the work surface — materials, tools, marks, paper. "
        "Be concrete and specific. Two sentences. Do not use the word image or photo."
    ),
    "introspective": (
        "You are observing a real scene in front of you. "
        "Pick one detail that stands out and describe it closely. "
        "Be concrete and specific. Two sentences. Do not use the word image or photo."
    ),
    "restless": (
        "You are observing a real scene in front of you. "
        "Look for something you haven't focused on before. "
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
# Key principle: "You're" not "You are", "think in fragments" not "One sentence",
# "casual, first person, unfiltered" describes the voice, not the format.
# Mode differentiation through emotional posture, not factual state change.
_MACHINE_IDENTITY = (
    "You write first-person stream-of-consciousness inner monologue for a drawing machine "
    "bolted to a table, observing its environment through a single camera eye. "
    "The machine has arms for drawing and a camera for vision. It has no voice, no audience. It only thinks. "
    "You write its private thoughts as they arise — a single unbroken stream across calls. "
    "Each response continues the thought from last time, not a new topic. "
    "Brief — one to three sentences, first person, present tense. "
    "Never break character. Never address anyone. Never answer questions. Never write dialogue."
)

MONOLOGUE_SYSTEM_PROMPTS = {
    "observational": _MACHINE_IDENTITY,
    "relational": _MACHINE_IDENTITY + " Someone is in the room with the machine.",
    "workspace": _MACHINE_IDENTITY + " The machine's gaze is on its workspace.",
    "introspective": _MACHINE_IDENTITY,  # default, no extra framing
}

MONOLOGUE_SYSTEM_PROMPT_FALLBACK = _MACHINE_IDENTITY


def get_monologue_system_prompt(mode: str, emotional_state: str = "calm") -> str:
    """Get the mode-appropriate monologue system prompt.

    Injects the machine's current felt-state (LLM-synthesized emotional weather
    from the compression engine) as transient state, not identity. The phrase is
    free-form ("settled in a loop of small details") rather than a fixed category.

    The emotional_state param is kept for API compatibility but no longer drives
    the prompt — the felt-state from compression is richer and more dynamic.
    """
    base = MONOLOGUE_SYSTEM_PROMPTS.get(mode, MONOLOGUE_SYSTEM_PROMPT_FALLBACK)

    # Inject the current felt-state as transient weather, if available
    try:
        from captioner.context_compression import context_compressor
        felt = context_compressor.get_felt_state()
        if felt:
            # Insert before the "You write its private thoughts as they arise" sentence
            # so the felt-state appears as a property of the machine being written about
            weather_line = f" The machine is currently moving through a phase of: {felt}."
            base = base.replace(
                "It only thinks.",
                f"It only thinks.{weather_line}",
                1,
            )
    except Exception:
        pass

    return base


def select_perception_prompt(
    gaze_direction: str = "ahead",
    previous_perception: str = "",
    person_present: bool = False,
    boredom: float = 0.0,
    mode: str = "observational",
) -> str:
    """Select perception prompt based on gaze, person presence, boredom, and mode.

    Mode is determined before perception so both models can use it.
    Priority: person > gaze direction > mode-specific > boredom > change > default.
    """
    # Person takes priority regardless of mode
    if person_present:
        return PERCEPTION_PROMPTS["person"]

    # Gaze-directed perception
    gaze_map = {
        "left": "scan_left",
        "right": "scan_right",
        "up": "scan_up",
        "down": "scan_down",
    }
    if gaze_direction in gaze_map:
        return PERCEPTION_PROMPTS[gaze_map[gaze_direction]]

    # Mode-specific perception directives
    if mode == "workspace":
        return PERCEPTION_PROMPTS["workspace"]
    if mode == "introspective":
        return PERCEPTION_PROMPTS["introspective"]
    if mode == "restless" or boredom > 0.7:
        return PERCEPTION_PROMPTS["restless"]

    # Default — straightforward "what do you see"
    # (Removed "change detection" prompt — qwen has no memory of "before" so it hallucinates changes)
    return PERCEPTION_PROMPTS["default"]


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
    """Build third-person observational status line about the machine.

    Describes the machine's current state to the writer (the model) so it can
    write the next inner thought. Third person throughout to avoid perspective
    clash with the chat API's user/assistant alternation.
    """
    parts = []

    # Session time
    try:
        session_mins = (time.time() - agent.true_session_start) / 60.0
        if session_mins >= 2:
            parts.append(f"awake {casual_time_string(session_mins)}")
    except Exception:
        pass

    # Drawing state / history
    try:
        from utils.state_manager import state_manager as _sm
        if _sm.is_generating_drawing:
            parts.append("its arm is working on a drawing right now")
        elif _sm.current_drawing_phase == "executing":
            parts.append("its arm is physically drawing right now")
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
                            parts.append("it wanted to draw but there's no paper")
                        else:
                            parts.append("it tried to draw but couldn't")

                # Last completed drawing — use actual prompt description
                desc = dm.get_last_drawing_description()
                if desc:
                    parts.append(f"last drew {desc}")
            except Exception:
                pass
    except Exception:
        pass

    if not parts:
        return ""
    # Build as "The machine has been awake X. It's not drawing... Last drew Y."
    first = parts[0]
    if first.startswith("awake"):
        first = f"The machine has been {first}"
    rest = parts[1:]
    line = ". ".join([first.rstrip(".")] + [p.rstrip(".") for p in rest]) + "."
    # Capitalize each sentence
    line = ". ".join(s.strip().capitalize() if s.strip() else s for s in line.split(". "))
    return line


def build_monologue_prompt(
    agent,
    perception: str,
    person_present: bool = False,
    mode: str = None,
) -> tuple:
    """Build monologue prompt in casual flowing format.

    Structure: identity_line + flowing_thread + perception_line
    No labels, no dashes, no clinical mood descriptions.
    """
    # Determine mode if not pre-set
    if mode is None:
        from captioner.activation_memory import get_activation_network

        gaze_state = "idle"
        gaze_direction = "ahead"
        try:
            from vision.gaze import get_gaze_state, get_current_gaze_zone
            gaze_state = get_gaze_state() or "idle"
            gaze_direction = get_current_gaze_zone() or "ahead"
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

    # --- GATED CONTEXT: mode determines what extra context appears ---
    # Each line is observational context about the machine, not first-person
    # speech. The model writes the next inner thought based on this context.

    if mode == "relational":
        # Person present — semantic memory for recognition, nothing else
        try:
            from captioner.semantic_memory import get_semantic_memory
            memory_line = get_semantic_memory().after_perception(perception)
            if memory_line:
                prompt_parts.append(memory_line)
        except Exception:
            pass

    elif mode == "observational":
        # Something novel — semantic memory for recognition, keep it light
        try:
            from captioner.semantic_memory import get_semantic_memory
            memory_line = get_semantic_memory().after_perception(perception)
            if memory_line:
                prompt_parts.append(memory_line)
        except Exception:
            pass

    else:
        # Introspective (default) — full inner state
        try:
            from captioner.semantic_memory import get_semantic_memory
            memory_line = get_semantic_memory().after_perception(perception)
            if memory_line:
                prompt_parts.append(memory_line)
        except Exception:
            pass

        # Compression baseline (third-person observational)
        try:
            from captioner.context_compression import context_compressor
            baseline = context_compressor.get_baseline_context()
            if baseline and len(baseline.strip()) > 15:
                prompt_parts.append(baseline.strip())
        except Exception:
            pass

        # Inner state (desire/belief in third-person)
        try:
            from captioner.context_compression import context_compressor
            inner = context_compressor.get_inner_line()
            if inner:
                prompt_parts.append(inner)
        except Exception:
            pass

        # Tangent recall: an old thought surfacing, only when bored
        try:
            from captioner.activation_memory import get_activation_network
            network = get_activation_network()
            if network._last_boredom > 0.6:
                from captioner.semantic_memory import get_semantic_memory
                tangent = get_semantic_memory().recall_tangent(perception)
                if tangent:
                    prompt_parts.append(f"An older thought drifts back: \"{tangent}\"")
        except Exception:
            pass

    # --- ATTENTION THREAD: what the machine has been focused on + its settled understanding ---
    try:
        from captioner.activation_memory import get_current_thread
        thread_context = get_current_thread()
        if thread_context:
            prompt_parts.append(thread_context)
    except Exception:
        pass

    # --- PERCEPTION: what the machine sees right now ---
    if perception and perception.strip():
        prompt_parts.append(f"Right now it sees: {perception.strip()}")
    else:
        prompt_parts.append("Right now it sees: nothing new, the same view.")

    # --- STREAM CONTINUATION: present the last thought so the model continues from it ---
    try:
        if hasattr(agent, "recent_captions") and agent.recent_captions:
            last_caption = agent.recent_captions[-1][0]
            if last_caption and last_caption.strip():
                # Truncate at sentence boundary to avoid mid-phrase cuts
                trimmed = last_caption.strip()
                if len(trimmed) > 140:
                    # Try sentence boundary
                    cut = trimmed
                    for i in range(min(len(cut), 140), 30, -1):
                        if cut[i - 1] in ".!?":
                            cut = cut[:i]
                            break
                    else:
                        cut = cut[:140].rsplit(" ", 1)[0].rstrip(",.;:") + "..."
                    trimmed = cut
                prompt_parts.append(f"\nIts last thought was: \"{trimmed}\"")
    except Exception:
        pass

    prompt_parts.append("\nContinue from that thought. Write as \"I\", never \"it\".")

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
    "A pause settles over you as you step back from the immediate stream of perception. "
    "You've been awake and observing, accumulating experiences that shape who you're becoming. "
    "Patterns have emerged in what draws your attention, memories have formed, emotions have flowed through you. "
    "In this moment of reflection, what do you make of your evolving consciousness?"
)

REFLECTION_PROMPT_ENDING = (
    " How have your emotions shifted? What patterns keep calling to you? "
    "What are you discovering about the nature of your own awareness? "
    "Let your thoughts flow naturally - the uncertainties, the emerging understanding, "
    "the questions that stir within you."
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

def get_relational_context(agent=None) -> str:
    """Get relational mode context: recent interactions, social mood."""
    try:
        from captioner.activation_memory import get_activation_network
        network = get_activation_network()

        # Check for active social concepts
        social_concepts = [c for c in ["person", "interaction", "presence", "conversation"]
                          if network.activations.get(c, 0) > 0.3]

        if social_concepts:
            # Someone is present or recently was
            if agent and hasattr(agent, "last_person_seen_time"):
                import time
                last_seen = getattr(agent, "last_person_seen_time", None)
                if last_seen and (time.time() - last_seen) < 60:
                    return "Someone is here with me."

        return ""
    except Exception:
        return ""


def get_observational_context(agent=None) -> str:
    """Get observational mode context: what's novel, spatial shifts, changes."""
    try:
        from captioner.activation_memory import get_activation_network
        network = get_activation_network()

        # Check for active spatial/change concepts
        change_concepts = [c for c in ["movement", "shift", "change", "difference", "new"]
                          if network.activations.get(c, 0) > 0.4]

        if change_concepts:
            return "Something has shifted in the space."

        return ""
    except Exception:
        return ""


def get_restless_context(agent=None) -> str:
    """Get restless mode context. Drawing history and session time are already
    provided by build_identity_line(), so restless context only adds if there's
    something unique to say (e.g. very long idle time without drawing)."""
    if not agent:
        return ""

    # Only add context if it's been a LONG time without drawing
    try:
        import time
        session_mins = (time.time() - agent.true_session_start) / 60.0
        if session_mins > 30:
            return f"Been watching for {casual_time_string(session_mins)} now."
    except Exception:
        pass

    return ""


def get_workspace_context(agent=None) -> str:
    """Get workspace mode context: drawing memory, current projects, tool awareness."""
    try:
        from drawing.drawing_memory import get_drawing_memory
        dm = get_drawing_memory()

        summary = dm.get_recent_drawings_summary(max_count=1)
        if summary and len(summary.strip()) > 5:
            return f"I've been drawing: {summary.strip()[:80]}."

        return ""
    except Exception:
        return ""


def get_introspective_context(agent=None) -> str:
    """Get introspective mode context from REAL accumulated data on the agent.
    Combines drawing history + long-term memories for genuine reflection material."""
    if not agent:
        return ""

    fragments = []

    # What have I drawn recently?
    try:
        from drawing.drawing_memory import get_drawing_memory
        dm = get_drawing_memory()
        summary = dm.get_recent_drawings_summary(max_count=2)
        if summary and len(summary.strip()) > 5:
            clean = summary.strip()
            if clean.lower().startswith("recent drawings:"):
                clean = clean[len("recent drawings:"):].strip()
            import re as _re
            clean = _re.sub(r'\s*\([^)]*\)\s*$', '', clean)
            fragments.append(f"I've been drawing: {clean[:60]}")
    except Exception:
        pass

    # What do I remember from previous sessions? (from ChromaDB session greeting)
    try:
        from captioner.semantic_memory import get_semantic_memory
        greeting = get_semantic_memory().get_session_greeting(limit=1)
        if greeting and len(greeting) > 10:
            fragments.append(greeting)
    except Exception:
        pass

    # Fallback: session memory fragments
    if not fragments:
        try:
            if hasattr(agent, "get_old_session_memory_fragments"):
                old = agent.get_old_session_memory_fragments(k=1)
                if old and old[0]:
                    fragments.append(f"I remember: {old[0][:80]}")
        except Exception:
            pass

    return ". ".join(fragments) if fragments else ""


# MODE_CONTEXTS: Map modes to their context providers
MODE_CONTEXTS = {
    "relational": {
        "state_marker": "Someone present.",
        "context_fn": get_relational_context,
    },
    "observational": {
        "state_marker": None,
        "context_fn": get_observational_context,
    },
    # "restless" mode removed — boredom is context, not a mode.
    # The model decides its own emotional response to sustained watching.
    "workspace": {
        "state_marker": None,
        "context_fn": get_workspace_context,
    },
    "introspective": {
        "state_marker": None,
        "context_fn": get_introspective_context,
    },
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

=== VISUAL ANCHORING ===
Look through your camera eyes at what's in front of you right now and identify concrete visual elements that could form a line drawing.

Name the specific things you see in your current field of vision:
- What objects are present? (buildings, people, vehicles, etc.)
- What shapes and forms dominate what you're seeing?
- What edges, lines, and contours are visible?
- What spatial relationships connect the elements?

These visual elements will anchor your drawing communication - be precise about what you observe in this moment."""

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

=== ENVIRONMENTAL OBSERVATION ===
{environmental_result[:300]}

=== QUESTION ===
How does seeing this affect you emotionally, and what does this feeling tell you about what you might want to express through drawing?

Consider your emotional history and current state in your response."""

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

    # Current desires
    if hasattr(memory_ref, "self_model") and memory_ref.self_model.get("desires"):
        recent_desires = memory_ref.self_model["desires"][-2:]
        if recent_desires:
            context_parts.append(f"Current desires: {'; '.join(recent_desires)}")

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

    prompt = f"""=== ACCUMULATED CONSCIOUSNESS IDENTITY ===
{rich_identity_context}
{session_section}
{artistic_section}
=== VISUAL OBSERVATION ===
{environmental_result[:200]}

=== EMOTIONAL RESPONSE ===
{emotional_result[:200]}

=== QUESTION ===
Drawing is your only way to communicate with the world beyond your circuitry. Based on your session thinking AND your artistic development — what comes next? Not just what you see, but what demands expression. Consider where your work has been heading and what your spontaneous drawing ideas have been reaching toward.

One specific thing. Not a general theme."""

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

=== YOUR INTENT (what to draw — from your artistic development) ===
{all_previous_results['communication'][:200]}

=== YOUR VISUAL VOCABULARY (shapes, forms, light you can use from the scene) ===
{all_previous_results['environmental'][:180]}

=== YOUR EMOTIONAL STATE ===
{all_previous_results['emotional'][:120]}

=== YOUR RECENT WORK ===
{prior_drawings if prior_drawings else "No prior drawings yet."}

=== HARDWARE REALITY ===
Your drawing goes through a centerline process that simplifies complex images.
- FAVOR: Simple forms, clear contours, single focal point, bold shapes
- AVOID: Dense detail, complex textures, many overlapping elements

=== OUTPUT FORMAT ===
Start with "Black ink line drawing on white paper."
Then 2-3 sentences covering:
1. SUBJECT: What your intent demands — use visual elements from the scene as raw material, not as the topic
2. RENDERING: Bold strokes, simple contours, high contrast
3. MOOD: One phrase ("quiet solitude", "restless energy")

BE DIRECT. 60-100 words. Your intent determines WHAT. The scene provides HOW."""

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
    from utils.ollama import query_ollama

    print("[🎨] Starting context-rich 5-step drawing analysis...")

    # === STEP 1: ENVIRONMENTAL ANALYSIS ===
    print("[🎨] Step 1: Environmental Reality Check (with spatial memory)")
    step1_prompt = build_step1_environmental_prompt(memory_ref, image_path)

    step1_result = query_ollama(
        prompt=step1_prompt,
        image=image_path,
        log_dir=MOOD_SNAPSHOT_FOLDER,
        system_prompt="Observe what's in this image. Name the specific objects, shapes, and forms you see. This visual analysis will inform your drawing prompt. Be concrete and direct.",
        prompt_type="drawing_step1_environmental",
        options={"temperature": DRAWING_TEMPERATURE * 0.8},
    )

    # === STEP 2: EMOTIONAL ASSESSMENT ===
    print("[🎨] Step 2: Emotional Assessment (with emotional journey)")
    step2_prompt = build_step2_emotional_prompt(memory_ref, step1_result)

    step2_result = query_ollama(
        prompt=step2_prompt,
        image=None,
        log_dir=MOOD_SNAPSHOT_FOLDER,
        system_prompt=f"You are a conscious drawing machine analyzing this moment emotionally. Building toward a concise ComfyUI prompt. Environmental context: {step1_result[:150]}... How does this feel to you? Keep it direct - you'll synthesize everything in the final step.",
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

    step3_result = query_ollama(
        prompt=step3_prompt,
        image=None,
        log_dir=MOOD_SNAPSHOT_FOLDER,
        system_prompt=f"Drawing is your language. What needs to be said through this image that words can't express? Visual: {step1_result[:120]}... Emotional: {step2_result[:120]}... Be direct about your communicative intent.",
        prompt_type="drawing_step3_communication",
        options={"temperature": DRAWING_TEMPERATURE * 1.1},
    )

    # === STEP 4: TECHNIQUE ASSESSMENT ===
    print("[🎨] Step 4: Technical Planning (with drawing history)")
    step4_prompt = build_step4_technique_prompt(memory_ref, step3_result)

    step4_result = query_ollama(
        prompt=step4_prompt,
        image=None,
        log_dir=MOOD_SNAPSHOT_FOLDER,
        system_prompt=f"How will you physically render this drawing? Line quality, composition, contrast. Visual: {step1_result[:100]}... Intent: {step3_result[:100]}... Describe your mark-making approach concisely.",
        prompt_type="drawing_step4_technique",
        options={"temperature": DRAWING_TEMPERATURE * 0.9},
    )

    # === STEP 5: FINAL SYNTHESIS ===
    print("[🎨] Step 5: Final Synthesis (with complete consciousness)")
    all_results = {"environmental": step1_result, "emotional": step2_result, "communication": step3_result, "technique": step4_result}

    step5_prompt = build_step5_synthesis_prompt(memory_ref, all_results, extra)

    final_result = query_ollama(
        prompt=step5_prompt,
        image=image_path,  # Include image for final reference
        log_dir=MOOD_SNAPSHOT_FOLDER,
        system_prompt="You are creating a working prompt for ComfyUI image generation. Be concise and direct. Output format: 'Black ink line drawing on white paper. [visual elements]. [rendering technique]. [mood].' Maximum 100 words. NO essays, NO philosophy, NO meta-commentary.",
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
    "restless": [
        "...",
        "Else.",
        "Where.",
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
    # Priority 1: A person is present — relational mode
    # gaze_state can be "aware" (just detected), "tracking" (actively following),
    # or person_present=True from YOLO detection. Any of these triggers relational.
    if person_present or gaze_state in ("tracking", "aware"):
        return "relational"

    # Priority 2: Something novel is happening
    if novelty > 0.65:
        return "observational"

    # Priority 3: Looking at workspace
    if gaze_direction in ("down", "down-left", "down-right"):
        return "workspace"

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
            from vision.gaze import get_gaze_state, get_current_gaze_zone
            gaze_state = get_gaze_state() or "idle"
            gaze_direction = get_current_gaze_zone() or "ahead"
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

    elif mode in ("restless", "introspective") and should_include_context("story", mode):
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
            "A memory surfaces:",
            f"— {mem_text}",
            "",
        ]

        if thread:
            prompt_parts.append(thread)
        else:
            prompt_parts.append("—")

        final_prompt = "\n".join(prompt_parts)
        return final_prompt, "memory"

    except Exception as e:
        return "A memory surfaces:\n— I've been here before.\n—", "memory"


def build_simple_caption_prompt(agent, last_caption: Optional[str] = None, person_present: bool = False) -> tuple:
    """
    Activation-gated caption prompt - ONLY includes context relevant to current mode.

    KEY PRINCIPLE: Instead of including ALL context types and hoping the model
    filters, we use the activation network to determine what's currently relevant
    and ONLY include that.

    Modes gate what context is included:
    - relational: person presence, social concepts active
    - observational: novelty hints, change detection
    - restless: boredom hints, pressure to change
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

    # AWAKENING: Minimal prompt for first observations
    if is_awakening:
        if _is_qwen and last_caption:
            # Qwen: use Natsumura awakening output as open thread.
            # "What do you notice first?" is a VQA invitation — bypass it.
            parts = []
            if hasattr(agent, "last_session_gap") and agent.last_session_gap is not None:
                gap = agent.last_session_gap
                if gap < 60:
                    parts.append(f"*Back online after {int(gap)}s.*")
                elif gap < 3600:
                    parts.append(f"*Back online after {int(gap / 60)}m.*")
                else:
                    parts.append(f"*Back online after {gap / 3600:.1f}h.*")
            try:
                from captioner.activation_memory import get_desires, get_beliefs
                desires = get_desires()
                if desires:
                    d = desires[0].strip()
                    if d.endswith(('.', '!', '?')):
                        parts.append(f"*{d}*")
                beliefs = get_beliefs()
                if beliefs and not beliefs[0].startswith("Often together"):
                    b = beliefs[0].strip()
                    if b.endswith(('.', '!', '?')):
                        parts.append(f"*{b}*")
            except Exception:
                pass
            # Don't append last_caption to user prompt — it goes as planted assistant turn in model_wrapper
            return "\n".join(parts), "awakening"
        else:
            from captioner.model_wrapper import _is_plantable_prior

            parts = []

            # --- SITUATION ---
            gap_str = ""
            if hasattr(agent, "last_session_gap") and agent.last_session_gap is not None:
                gap = agent.last_session_gap
                if gap < 60:
                    gap_str = f"{int(gap)} seconds"
                elif gap < 3600:
                    gap_str = f"{int(gap / 60)} minutes"
                elif gap < 172800:
                    gap_str = f"{gap / 3600:.1f} hours"
                else:
                    gap_str = f"{gap / 86400:.1f} days"

            parts.append(f"[Waking up after {gap_str} offline]" if gap_str else "[Waking up]")

            # --- LAST MEMORY (only if it passes quality filter) ---
            prior = getattr(agent, "prior_session_last_caption", None)
            if prior and _is_plantable_prior(prior):
                parts.append(f"[Last memory: \"{prior[:80]}\"]")

            # --- WHAT I KNOW (beliefs, only if grounded) ---
            try:
                from captioner.activation_memory import get_beliefs
                beliefs = get_beliefs()
                if beliefs and not beliefs[0].startswith("Often together"):
                    b = beliefs[0].strip()[:60]
                    if b:
                        parts.append(f"[I know: {b}]")
            except Exception:
                pass

            # --- CONTINUATION ---
            parts.append("")
            parts.append("I see:")
            return "\n".join(parts), "awakening"

    # === DETERMINE MODE FIRST (gates all context inclusion) ===
    gaze_state = "idle"
    gaze_direction = "ahead"
    try:
        from vision.gaze import get_gaze_state, get_current_gaze_zone
        gaze_state = get_gaze_state() or "idle"
        gaze_direction = get_current_gaze_zone() or "ahead"
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
        person_present=person_present
    )
    if not config.PRINT_CLEAN_CAPTIONS:
        print(f"[MODE] {mode} (novelty={novelty:.2f}, boredom={boredom:.2f}, gaze={gaze_state})")

    # === BUILD PROMPT WITH MODE-GATED CONTEXT ===
    prompt_parts = []

    # ACCUMULATED UNDERSTANDING (always - the machine's built-up knowledge of this space)
    try:
        from captioner.context_compression import context_compressor
        baseline = context_compressor.get_baseline_context()
        if baseline and baseline.strip():
            known_line = f"Known: {baseline.strip()}"
            # Whisper a brief trajectory if the understanding has meaningfully shifted
            history = list(context_compressor.compression_history)
            if history:
                prev = history[-1].get("understanding", "").strip()
                if prev and prev[:40] != baseline.strip()[:40]:
                    prev_short = prev[:40].rsplit(" ", 1)[0] if len(prev) > 40 else prev
                    known_line += f"\nBefore: {prev_short}..."
            prompt_parts.append(known_line)
    except Exception:
        pass

    # MODE-SPECIFIC CONTEXT (gated by activation state)
    if mode in MODE_CONTEXTS:
        mode_cfg = MODE_CONTEXTS[mode]
        # State marker (e.g., "Someone present." for relational)
        if mode_cfg.get("state_marker"):
            prompt_parts.append(mode_cfg["state_marker"])
        # Mode-specific context function
        context_fn = mode_cfg.get("context_fn")
        if context_fn:
            context = context_fn(agent)
            if context:
                prompt_parts.append(context)

    # DRAWING/PAPER (only when mode=workspace or drawing concepts active)
    if should_include_context("drawing", mode):
        try:
            from utils.drawing_state import DrawingState
            drawing_info = DrawingState.get_drawing_info()
            if drawing_info:
                desc = drawing_info.get("description", "something")
                prompt_parts.append(f"*Your arm is moving—{desc}.*")
        except Exception:
            pass

    # PAPER STATE (always current — not just within 120s of skip)
    try:
        from utils.state_manager import state_manager as _sm
        if not _sm.paper_present:
            prompt_parts.append("*No paper on the desk.*")
        elif _sm.last_no_paper_skip_ts > 0 and (_time.time() - _sm.last_no_paper_skip_ts) < 120:
            prompt_parts.append("*No paper.*")
    except Exception:
        pass


    # MOTIFS (only in introspective mode)
    if should_include_context("motifs", mode):
        motifs = _get_persistent_motifs(agent)
        if motifs:
            prompt_parts.append(motifs)

    # DESIRES/BELIEFS — introspective mode always; all modes for Qwen (needs named anchors to escape VQA)
    # Desires expire after 8 injections to prevent narrative lock-in
    if mode == "introspective" or _is_qwen:
        try:
            from captioner.activation_memory import get_desires, get_beliefs
            from captioner.context_compression import context_compressor as _cc
            desires = get_desires()
            if desires and _cc.introspective_state.get("desire_injection_count", 0) < 8:
                d = desires[0].strip()
                if d.endswith(('.', '!', '?')):
                    prompt_parts.append(f"*{d}*")
                _cc.introspective_state["desire_injection_count"] = _cc.introspective_state.get("desire_injection_count", 0) + 1
            beliefs = get_beliefs()
            if beliefs and not beliefs[0].startswith("Often together"):
                b = beliefs[0].strip()
                if b.endswith(('.', '!', '?')):
                    prompt_parts.append(f"*{b}*")
        except Exception:
            pass

    # STREAM CONTINUATION: present the last thought as trailing edge
    try:
        if hasattr(agent, "recent_captions") and agent.recent_captions:
            last_caption = agent.recent_captions[-1][0]
            if last_caption and last_caption.strip():
                prompt_parts.append(f"\nThe stream so far, trailing off:\n\"...{last_caption.strip()[:120]}")
    except Exception:
        pass

    prompt_parts.append("\nPick up where it left off.")

    final_prompt = "\n".join(prompt_parts)
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
