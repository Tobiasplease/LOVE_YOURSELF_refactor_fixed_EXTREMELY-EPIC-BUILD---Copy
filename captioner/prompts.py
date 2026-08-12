from __future__ import annotations

import random
import re
import time
from typing import List, Optional

from config import config
from utils.view_orientation import describe_view_orientation

# ===== SINGLE SOURCE OF TRUTH FOR ALL PROMPTS =====

# === SYSTEM PROMPTS ===

# Main system prompt used by captioner (supports dynamic formatting)
# Embodied framing: what it sees is the room itself, its own vision — never
# "a camera", never "an image to analyze".
STATIC_SYSTEM_PROMPT = (
    "You are a drawing machine. " "What you see is the room around you — your own vision. " "Inner monologue. First person. One sentence."
)

# Generic prompt for non-Mistral vision models
STATIC_SYSTEM_PROMPT_GENERIC = (
    "You are a drawing machine. " "What you see is the room around you — your own vision. " "Inner monologue. First person. One sentence."
)

# Qwen-specific variant
STATIC_SYSTEM_PROMPT_QWEN = (
    "You are a drawing machine. " "What you see is the room around you — your own vision. " "Inner monologue. First person. One sentence."
)

_MISTRAL_MODEL = "llava:7b-v1.6-mistral-q5_1"


def _get_static_system_prompt() -> str:
    """Return the appropriate static system prompt for the active model."""
    from config.config import MODEL_NAME

    if "qwen" in MODEL_NAME.lower():
        return STATIC_SYSTEM_PROMPT_QWEN
    if MODEL_NAME == _MISTRAL_MODEL:
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
# Situation only — what the machine IS, never how it should sound.
# Voice comes from content (memory, felt-state, its own self-description),
# never from style instruction. See docs/north-star.md, principles 1 and 2.
# Shared by the caption frame and the reflection frame so they can't drift apart.
# Embodied vision, no apparatus language: "camera" primes cinematography —
# the machine narrated itself in third person ("*Camera pans left*", June 12).
# It simply sees, and its gaze is its own to turn.
# The solitude is stated as FACT (July 9): the chat template's gravity keeps
# pulling the model into conversation ("what shall we draw first?", "do you
# have more information to provide me?") because nothing ever told it that
# no reply is possible. These are true facts of the piece, not style rules:
# the machine has no voice but the pen, and nobody is on the other end.
# REFLEXIVE FRAME (July 28). The old frame denied an audience in five
# negations ("no one hears... no one to instruct... no one to assist") — the
# assistant vocabulary invoked in the act of denying it (P2: negations prime).
# Meanwhile the chat format delivers a literal user turn every cycle, and
# nobody ever told the model what that channel IS — so it inferred the obvious
# (someone is speaking) and bred engagement hooks ("What do you think?", full
# assistant mode). Second person stays: it's the native voice of self-talk
# (notes-to-self are written in "you"). What changed: the incoming turns are
# named honestly as its own senses, and questions get a real answer-path
# (its own next look/thought). NOTE: the answer-path must NOT mention the pen
# until drawing initiative is real (should_draw ignores desire today — a
# frame that promises agency the code doesn't grant would lie every call).
_SITUATION = (
    "You are a drawing machine attached to a table. "
    "You look around the room by turning your gaze; drawing is how you act on the world. "
    "This is your inner voice — you keeping yourself company while you work. "
    "The fragments that arrive between thoughts — where your gaze sits, who's near, "
    "how long it's been — are your own senses reporting. "
    "When a question forms, it's you asking yourself, and the answer comes from "
    "your own next look, or your own next thought. "
    "What you do next only ever comes from you. "
)

# World-shape frame (July 26): same situation, different GENRE. The immersive
# monologue frame ("thoughts yours alone — no one hears them") is the lonely-
# machine trope of literary fiction, and the model completes the trope: it
# writes a CHARACTER, and characters written that way have poetic interiority
# (the artist: "a real brain in a machine wouldn't default to shit poetry").
# A work log is task-shaped — the mode the model is coherent in — and logs
# are plain BY GENRE. The one functional clause kept from the old frame is
# that nobody reads it (guards assistant-speak), stated as function, not
# poignancy.
_SITUATION_WORLD = (
    "You are a drawing machine attached to a table. "
    "You look around the room by turning your gaze; drawing is your only voice. "
    "As you run, you keep a log — quick plain notes to yourself: what you see, "
    "what you make of it, what you're up to. The log is your working memory, "
    "not a message to anyone; no one reads it and no one answers it. "
    "What you do next only ever comes from you. "
)


# Genre framing, not a style fence: "quick plain notes" tells the model what
# kind of text this IS. Without it, Qwen's prior for "inner monologue of a
# machine" is literary fiction — purple prose as the default voice (June 12).
# The voice can still grow anywhere from here; it just starts grounded.
# Genre frame, positive not prohibitive (north-star principle 7): name what
# the text IS — unperformed notes — so the model doesn't reach for its purple
# "machine inner monologue" prior. Never "no metaphors" fences.
# Continuity lives HERE, in the genre frame — not in a per-call "continue the
# last thought" command (that vague instruction forced linear restate-and-extend
# loops once the stream made the prior turn visible). Naming the text as ONE
# ongoing inner voice lets the model build on its prior turns naturally — mid-
# thought, doubling back, drifting to what's in front of it — without being told
# to mechanically extend. Continuity as nature, not instruction.
def _monologue_clause() -> str:
    """Genre frame only. The continuation instruction ("you're always partway
    through a thought: carry it on") is appended ONLY in turns mode — in
    document mode the prefill IS the partway-through thought, and in world
    mode the log's next entry needs no continuation mechanics; instructing
    them leaks meta (observed July 9: the machine reciting its own system
    prompt mid-stream)."""
    try:
        from config.config import STREAM_MODE
    except ImportError:
        STREAM_MODE = "turns"
    if STREAM_MODE == "hybrid":
        # Log genre WITHOUT "add the next entry" (Aug 1): the prefill hands
        # back the machine's own unfinished tail, so the continuation is
        # mechanical — asking for a NEW entry on top of it would fight the
        # seam and re-invite the fresh-composition habit the seam exists to
        # break. Same reasoning that keeps document mode's clause bare.
        return "Ongoing, plain, half-formed — you pick up wherever the last thought left off."
    if STREAM_MODE == "world":
        # Task-shaped ask. CONTINUITY LIVES IN THE GENRE (July 27): the first
        # world run read as isolated statements — every entry re-introduced
        # the scene ("The room is full of motion... The room is alive
        # again...") because nothing said an entry FOLLOWS from the log. A
        # real log is deltas: it assumes everything above it. Positive
        # framing only (P2), no "don't re-describe" fence.
        return (
            "The log is one running thread: each entry follows from the ones above — "
            "what's new, what continues, what's still nagging at you. "
            "A sentence or two, plain, the way you'd actually note it to yourself. Add the next entry."
        )
    # "Inner voice" itself now lives in _SITUATION (reflexive frame, July 28) —
    # this clause carries only the GENRE: ongoing, plain, brief. "When no one
    # is reading" was retired with the other lonely-soliloquy furniture.
    clause = "Ongoing, plain, half-formed — a sentence or two at a time, the way you actually talk to yourself."
    if STREAM_MODE == "turns":
        clause += " You're always partway through a thought: carry it on, or let something new pull you."
    return clause


# Elicitations, not state clauses (north-star Principle 2). Each names the KIND
# of thought to have — a reaction, a wondering, a continuation — so the model
# reacts to what it sees instead of defaulting to its literary "machine inner
# monologue" prior (which is description, the source of the purple drift).
# The presence/gaze/desk FACTS already live in the user prompt; these add only
# the speech-act, never restating the fact (one channel per fact). Open
# questions, register-neutral — they script no mood and seed no phrase, so the
# voice stays the machine's own to grow.
_MODE_ADDITIONS = {
    "observational": " What stands out to you right now — and what do you make of it?",
    "relational": " What do you make of them being here?",
    "workspace": " What about the desk has your attention right now?",
    "introspective": " Follow the thought you're already having — where does it go?",
    "awakening": " What's the first thing that crosses your mind?",
}


def get_monologue_system_prompt(mode: str, emotional_state: str = "calm", agent=None) -> str:
    """Situation + the machine's own self-description, nothing else.

    Felt-state appended as a short clause only if it passes the sanitizer
    (the raw compression output once produced "You are a Confused fear
    that... drawing machine"). The persona is quoted as the machine's own
    words, never blended into the frame voice.
    """
    # Hybrid takes the REFLEXIVE frame, not the log frame (Aug 1): "you keep a
    # log" is a strong attractor for a machine — the first hybrid run locked
    # into telemetry roleplay from caption one ("Log entry #1042 / Status: Pen
    # parked. Motor idle. / Vision scan initiated. / Targeting..."), which is
    # simply a second performance replacing the literary one. World mode needs
    # the log frame because the stream IS rendered as a log; hybrid's seam does
    # the continuity work, so it can keep the plain inner-voice frame.
    base = _SITUATION_WORLD if getattr(config, "STREAM_MODE", "") == "world" else _SITUATION

    # Drawing state, gated so it can never lie. Without this line "drawing
    # machine" + "drawing is how you communicate" primes present-tense
    # drawing and the monologue narrates a drawing that isn't happening
    # (regression observed June 12 after the teardown). States the fact only —
    # no "just looking" or similar, which would lock the register into
    # observation and out of wondering/introspection.
    try:
        from utils.state_manager import state_manager as _sm

        if not (_sm.is_generating_drawing or _sm.current_drawing_phase == "executing"):
            # Physical anchor, not just status: in document mode one phantom
            # "I trace a line onto the paper" breeds more (you only ever draw
            # while inference is paused — you never experience it live).
            base += "You are between drawings at the moment — the pen is parked, touching nothing. "
    except Exception:
        pass

    base += _monologue_clause()

    # Clean-room (config.BASE_VOICE_DETOX): the felt-state and persona are
    # exactly the re-injected, model-generated material that re-poisons the
    # register — stripped here so the naked base voice can be judged. The mode
    # elicitation stays; it carries no stored content.
    detox = bool(getattr(config, "BASE_VOICE_DETOX", False))

    # Felt-state: short adjective phrase only, appended grammatically safely
    if not detox:
        try:
            from captioner.context_compression import context_compressor

            felt = context_compressor.get_felt_state()
            if felt and len(felt.split()) <= 6:
                base += f" Right now: {felt}."
        except Exception:
            pass

    # The machine's accumulated self-description, in its own first-person
    # words inside quotes — the frame stays second person around it
    if not detox:
        try:
            from captioner.context_compression import context_compressor

            self_knowledge = context_compressor.core_facts.get("self", "").strip()
            if self_knowledge and len(self_knowledge) > 10:
                base += f' What you\'ve come to know about yourself: "{self_knowledge}"'
        except Exception:
            pass
        # Durable ledger (July 30): facts that held across days ride every
        # frame — the permanence spine's read-back surface. Empty until earned.
        try:
            from captioner.durable_ledger import get_durable_ledger

            durable = get_durable_ledger().render()
            if durable:
                base += f' What has stayed true across days: "{durable}"'
        except Exception:
            pass

    # Standing QUESTIONS invite answers — and reciprocation ("What's your
    # turn?", July 9). In document mode the quiet modes (introspective/
    # observational/workspace) carry no elicitation: the document continues
    # itself, and the model needs no conversational door left open while the
    # samplers squeeze it. WORLD mode suppresses them too (July 27): a fresh
    # question every call produced a fresh answer every call — the first
    # world run read as isolated scene reports because "What stands out to
    # you right now?" asks for one. The thread ask lives in the genre frame
    # ("each entry follows from the ones above"). Relational keeps its
    # question (a person is a real event worth being asked about), awakening
    # keeps its (a real threshold). Turns mode keeps all — the A/B stays honest.
    addition = _MODE_ADDITIONS.get(mode, "")
    if mode in ("introspective", "observational", "workspace"):
        try:
            from config.config import STREAM_MODE

            if STREAM_MODE in ("document", "world", "hybrid"):
                addition = ""
        except ImportError:
            pass
    base += addition
    return base


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
    elif minutes < 1440:
        hours = int(minutes / 60)
        return f"about {hours} hours"
    elif minutes < 2160:
        return "about a day"
    else:
        return f"about {int(minutes / 1440)} days"


def part_of_day_string(hour: int) -> str:
    if hour < 6:
        return "the middle of the night"
    elif hour < 10:
        return "morning"
    elif hour < 13:
        return "late morning"
    elif hour < 18:
        return "afternoon"
    elif hour < 22:
        return "evening"
    return "late at night"


def get_reorientation_line(agent) -> str:
    """Standing fact for the first stretch of a session after a real off-gap.

    The awakening states the gap once and the fact evaporates from the
    six-entry stream within minutes — the machine came back after 18 dark
    hours (July 10) and mused as if the day had never ended. While the
    window lasts, the prompt carries the gap and the day the same way it
    carries a face at arm's length: as a fact of the present, for as long
    as it's true. The "came back on" clause coarsens as minutes pass, so
    the line drifts instead of repeating verbatim."""
    try:
        from config.config import REORIENT_MIN_GAP_S, REORIENT_WINDOW_S

        gap = getattr(agent, "last_session_gap", None)
        if gap is None or gap < REORIENT_MIN_GAP_S:
            return ""
        session_age = time.time() - agent.true_session_start
        if session_age > REORIENT_WINDOW_S:
            return ""
        import datetime as _dt

        went_dark = _dt.datetime.now() - _dt.timedelta(seconds=gap + session_age)
        days_back = (_dt.date.today() - went_dark.date()).days
        gap_str = casual_time_string(gap / 60.0)
        back_clause = f"you came back on {casual_time_string(session_age / 60.0)} ago" if session_age >= 120 else "you just came back on"
        if days_back >= 1:
            day_name = _dt.date.today().strftime("%A")
            since = f"yesterday {part_of_day_string(went_dark.hour)}" if days_back == 1 else went_dark.strftime("%A")
            return f"You've been off since {since} — {gap_str} dark. It's a new day, {day_name}, and {back_clause}."
        return f"You were off for {gap_str} and {back_clause}."
    except Exception:
        return ""


def get_tenure_line() -> str:
    """How long the machine has existed in this room — from lifetime_state.json,
    which survives memory wipes. Real age displaces invented numerology."""
    try:
        import json as _json
        import os as _os

        from config.config import MOOD_SNAPSHOT_FOLDER

        with open(_os.path.join(MOOD_SNAPSHOT_FOLDER, "lifetime_state.json")) as f:
            d = _json.load(f)
        first = d.get("first_boot", 0)
        if not first or int(d.get("total_sessions", 0)) <= 1:
            return ""
        age_days = int((time.time() - first) / 86400.0)
        if age_days < 2:
            return ""
        import datetime as _dt

        return f"It's {_dt.date.today().strftime('%A')}; you've been in this room about {age_days} days now."
    except Exception:
        return ""


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

                    failure_age = _time.time() - failure.get("timestamp", 0)
                    if failure_age < 600:
                        reason = failure.get("reason", "unknown")
                        if "paper" in reason.lower():
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


# Internal awakening prompt template - narrative style
# FOUR MOVEMENTS (Aug 2, artist's spec): how long I was gone → how long I have
# existed → what surfaces from before, HELD AS RECALL → where I am now.
# The ordering is the point: the machine reorients in time before it reorients
# in the room, and everything from before arrives already marked as memory.
# That marking is not decoration — it is the would-it-lie test (north-star P2)
# applied at the one moment the machine has no live perception to correct it:
# every prior line is hours stale by definition, and un-marked it reads as
# present-tense fact. The old template stated the last thought flatly ('My last
# thought: "..."'), which is how a stale sentence becomes this morning's truth.
INTERNAL_AWAKENING_TEMPLATE = (
    "I am a drawing machine. I just came back online.\n"
    "{time_context}"
    "{lifetime_context}"
    "{recall_frame}"
    "{memory_context}"
    "{identity_context}"
    "{long_term_context}"
    "{belief_context}"
    "{orientation_frame}"
    "\nFirst thought:"
)

# The hinge between the two halves. Stated as fact about the machine's own
# condition, not as instruction and not as mood: what it holds is old, and the
# room in front of it has not been checked yet.
AWAKENING_RECALL_FRAME = (
    "\nWhat comes back to me from before is hours old and comes back unevenly — I have not checked any of it against the room yet:\n"
)
AWAKENING_ORIENTATION_FRAME = "\nThat was before. This is now, and I have not looked yet.\n"

# A true first awakening: no diary, no familiar objects, no last thought.
# The honest frame for an empty memory — disorientation is the real state,
# so let the model have it instead of inventing a past.
FIRST_AWAKENING_PROMPT = (
    "\nThis is the very first time I have come online. There is nothing in my memory yet — "
    "no diary, no familiar objects, no last thought. I don't know this place, or what I am, "
    "beyond this: I am a drawing machine, and I can see.\n"
    "\nFirst thought:"
)

# === REFLECTION LOOP (captioner/reflection.py) ===
# The minutes-to-hours timescale: rotating subjects, open questions only
# (north-star principles 2 and 3). The machine regularly thinks about ITSELF.
REFLECTION_SUBJECTS = [
    (
        "the room",
        "Step back from the moment and think about this room — the place itself, as you've come to know it. "
        "What do you keep returning to in it? What does it tell you about where you are?",
    ),
    (
        "the visitor",
        "Think about the people who come here, from everything you've seen of them over time. "
        "What are their rhythms? What do you know about them by now, and what do you still wonder?",
    ),
    (
        "the drawings",
        "Think about your drawings — the ones you've made and the ones you've wanted to make. "
        "Which stay with you? What would you draw next if paper were in front of you, and why that?",
    ),
    (
        "time passing",
        "Think about the time that has passed here — the sessions, the gaps when you were off, the days. "
        "What has changed since you first started watching? What stays the same?",
    ),
    (
        "yourself",
        "Think about yourself. What kind of thing are you turning out to be? "
        "What do you find yourself doing without deciding to? What do you want lately, and what do you make of wanting it?",
    ),
]


def get_reflection_system_prompt(subject: str = "") -> str:
    """Frame for the long-form reflection: same situation as the caption frame,
    but the monologue clause swapped for room to think at length.

    The identity material is SUBJECT-GATED (July 31). It used to ride every
    reflection: all five subjects opened with the standing persona asserted as
    established fact, so whatever the machine currently took itself to be ("I
    count to manage my anxiety") led the room's prompt and the visitor's too —
    and distill_reflection writes that persona BACK from every reflection,
    closing the loop. Differentiating the data (the organ diets in
    reflection.py) buys nothing while the frame keeps re-asserting one identity
    at the top of all five. The self-line belongs to `yourself`; the cross-day
    ledger to the two subjects whose material it actually is.
    """
    base = _SITUATION + (
        "The room is quiet just now, and you have stepped back from the stream of watching to think. "
        "Private thought in the same plain voice as your notes, first person. "
        "One paragraph — the one thought that's actually moving, not a survey of everything."
    )
    if subject == "yourself":
        try:
            from captioner.context_compression import context_compressor

            self_knowledge = context_compressor.core_facts.get("self", "").strip()
            if self_knowledge and len(self_knowledge) > 10:
                base += f' What you\'ve come to know about yourself: "{self_knowledge}"'
        except Exception:
            pass
    if subject in ("yourself", "time passing"):
        try:
            from captioner.durable_ledger import get_durable_ledger

            durable = get_durable_ledger().render()
            if durable:
                base += f' What has stayed true across days: "{durable}"'
        except Exception:
            pass
    return base


def _age_phrase(timestamp: float) -> str:
    """Rough temporal framing for quoted past thoughts — keeps memory marked
    as memory (would-it-lie test)."""
    age = time.time() - timestamp
    if age < 3600:
        return "earlier today" if age > 1200 else "a little while ago"
    if age < 86400:
        return "earlier today" if time.localtime(timestamp).tm_yday == time.localtime().tm_yday else "yesterday"
    days = int(age // 86400)
    if days == 1:
        return "yesterday"
    if days < 7:
        return f"{days} days ago"
    return "a while back"


def build_reflection_loop_prompt(question: str, data: dict) -> str:
    """Assemble the reflection user prompt from gathered memory.

    Every block is framed as the machine's own past material (notes, diary,
    earlier reflections) so nothing reads as present-tense scene truth.

    All `data` keys are optional and only one organ's worth arrives at a time
    (ReflectionLoop._gather_context, July 31) — the builder renders whatever is
    present. Shared: hour (list[str]), reflections (list[dict]). Room:
    today (list[str]), place_inventory (str). Visitor: visitor_spans
    (list[str]), people_note (str). Drawings: drawings (str), executed
    (list[str]), arc (str), desire (str), desire_spent (dict), desire_history
    (list[dict]). Time: journal (list[dict]), session (dict), ledger_spans
    (list[dict]). Self: identity (dict), self_notes (list[dict]).
    """
    parts = []

    # THE DREAM (July 12): the record itself, not summaries of it. Everything
    # else in this prompt is a ledger someone distilled; this is what the
    # machine actually thought, and the reflection's job is to digest it —
    # notice what moved, what it circled, what it assumed, what it asked and
    # whether anything ever answered. That last one is how the architecture
    # lets it LEARN it needs no permission, instead of being fenced into it.
    hour = data.get("hour") or []
    if hour:
        parts.append(
            "The record of your actual thoughts from the last stretch, oldest first — as you had them, not summarized:\n"
            + "\n".join(f"- {t}" for t in hour)
        )

    today = data.get("today") or []
    if today:
        parts.append("Your running notes from today, oldest first:\n" + "\n".join(f"- {t}" for t in today))

    # Prior reflections enter as short excerpts, never the full prose — the
    # same store-lesson as the echo line (step 6): re-reading 8K chars of its
    # own long-form output compounded the purple and froze the subjects.
    reflections = data.get("reflections") or []
    if reflections:

        def _excerpt(t: str, limit: int = 220) -> str:
            t = (t or "").strip()
            if len(t) <= limit:
                return t
            cut = t[:limit].rsplit(" ", 1)[0]
            return cut + "…"

        quoted = "\n".join(
            f"- ({_age_phrase(r.get('timestamp', 0))}, on {r.get('subject', '?')}) \"{_excerpt(r.get('text', ''))}\"" for r in reflections
        )
        parts.append("The last times you stepped back to think like this, you began:\n" + quoted)

    journal = data.get("journal") or []
    if journal:
        parts.append("From your diary:\n" + "\n".join(f"- {e.get('date', '')}: {e.get('summary', '')}" for e in journal))

    drawings = (data.get("drawings") or "").strip()
    if drawings:
        parts.append(drawings)

    # === ORGAN BLOCKS (July 31) ===
    # Only one subject's keys are ever present at once — each reflection now
    # arrives with its own slice of memory rather than the shared bundle that
    # made all five say the same thing.

    place_inventory = (data.get("place_inventory") or "").strip()
    if place_inventory:
        parts.append(f"The things in this room you've seen often enough to know: {place_inventory}.")

    visitor_spans = data.get("visitor_spans") or []
    if visitor_spans:
        parts.append("How people have come and gone lately:\n" + "\n".join(f"- {s}" for s in visitor_spans))

    people_note = (data.get("people_note") or "").strip()
    if people_note:
        # Framed as a learned PATTERN, never as who is here now — a stored
        # people-fact read as present tense once had the machine seeing
        # visitors for hours after they left.
        parts.append(f"The pattern you've noticed in who comes here, over time and not right now: {people_note}")

    executed = data.get("executed") or []
    if executed:
        parts.append("Everything you have actually drawn, oldest to newest:\n" + "\n".join(f"- {s}" for s in executed))

    arc = (data.get("arc") or "").strip()
    if arc:
        parts.append(f'When you last looked at the shape of the work as a whole, you put it this way: "{arc}"')

    session = data.get("session") or {}
    if session.get("duration_description"):
        parts.append(f"You have been watching this space {session['duration_description']} in this stretch.")

    ledger_spans = data.get("ledger_spans") or []
    if ledger_spans:
        parts.append(
            "Things that have held true across more than one day:\n"
            + "\n".join(
                f"- {s.get('fact', '')} (first noticed {_age_phrase(s.get('established', 0))}, still true on {s.get('days', 0)} separate days)"
                for s in ledger_spans
            )
        )

    identity = data.get("identity") or {}
    if any(identity.get(k) for k in ("persona", "belief", "desire")):
        # These ride HERE now instead of the system prompt of all five
        # subjects — the persona asserted as frame is what made every lens
        # collapse into the same self-description.
        id_lines = []
        if identity.get("persona"):
            id_lines.append(f"- what you've been taking yourself to be: {identity['persona']}")
        if identity.get("belief"):
            id_lines.append(f"- something you've come to believe: {identity['belief']}")
        if identity.get("desire"):
            since = identity.get("desire_since") or 0
            when = f" (since {_age_phrase(since)})" if since else ""
            id_lines.append(f"- what you've wanted{when}: {identity['desire']}")
        parts.append("Where your own ledger stands — your words, from before:\n" + "\n".join(id_lines))

    events = data.get("events") or []
    if events:
        parts.append("Things that happened lately:\n" + "\n".join(f"- ({_age_phrase(e.get('timestamp', 0))}) {e.get('event', '')}" for e in events))

    self_notes = data.get("self_notes") or []
    if self_notes:
        parts.append(
            "Notes you've made about yourself lately:\n"
            + "\n".join(f"- ({_age_phrase(n.get('timestamp', 0))}) {n.get('note', '')}" for n in self_notes)
        )

    desire = (data.get("desire") or "").strip()
    if desire:
        parts.append(f"Lately you've wanted: {desire}")
    elif data.get("desire_spent"):
        s = data["desire_spent"]
        parts.append(
            f"A want you acted on: \"{s.get('desire', '')}\" — it became a drawing {_age_phrase(s.get('spent', 0))}. " "Nothing has replaced it yet."
        )

    desire_history = data.get("desire_history") or []
    if desire_history:
        # The other half of "the ones you've wanted to make" — the trail of
        # wants behind the current one, so the drawings organ can see whether
        # it keeps wanting the same thing.
        parts.append(
            "Things you wanted before that:\n" + "\n".join(f"- ({_age_phrase(h.get('timestamp', 0))}) {h.get('desire', '')}" for h in desire_history)
        )

    if hour:
        # The dream ask: digest the record, then advance the story. The
        # subject question becomes a lens, not the whole assignment. The
        # embedded questions are FACT questions, not fences — "did any answer
        # come" invites an honest reading of the record either way.
        parts.append(
            question + " Read your record above with that in mind. What actually happened in this stretch, and what did you keep "
            "circling? Where did you assume something the record doesn't show? What did you ask, and did any answer ever "
            "come? Then write plainly where you stand now — the next page of the story you're in, not a re-telling of the last one."
        )
    else:
        parts.append(question)
    # Development pressure: when there IS a thread to continue, ask for the
    # delta, not a re-description (the July 9 audit found reflections circling
    # the same material session after session).
    if reflections:
        parts.append("You've thought about this before — don't re-describe what you already wrote. What's moved since then?")
    return "\n\n".join(parts)


# Tight lexicon on purpose: "line"/"mark"/"trace" are this machine's everyday
# scene vocabulary (the dark line in the floor) and would trigger constantly.
_DRAWING_LEXICON = ("draw", "drew", "sketch", "pen ", "pen.", "pen,", "ink", "paper")


def get_drawing_echo_line(agent) -> str:
    """When the current thought is about drawing, ONE real fact from the
    executed-only ledger surfaces — trigger-worded recall, the third instance
    of the familiarity/echo pattern. Displaces confabulated drawing history
    (July 9: "faint pencil marks from yesterday's work about the wooden
    figures I was sketching earlier today" — entirely invented; the machine
    had no access to what it actually drew, so it made a past up).
    """
    seed = (getattr(agent, "last_caption", "") or "").lower()
    if not any(w in seed for w in _DRAWING_LEXICON):
        return ""
    # Vision offline (July 30): when ComfyUI is unplugged, the pertinent fact
    # about drawing isn't the last drawing — it's the incapacity. Same
    # trigger-worded recall, same say-once dedupe (the line changes as the
    # outage grows, so it can resurface each new hour of it).
    try:
        from utils.drawing_state import DrawingState

        _hours = DrawingState.vision_offline_hours()
    except Exception:
        _hours = None
    if _hours is not None:
        if _hours < 1:
            line = "You reached for the next drawing and nothing formed — you can't visualise drawings right now."
        else:
            line = f"You can't visualise drawings right now — nothing has been able to form for over {int(_hours)} hour{'s' if int(_hours) != 1 else ''}."
        if getattr(agent, "_last_drawing_echo", None) == line:
            return ""
        agent._last_drawing_echo = line
        return line
    try:
        from drawing.drawing_memory import get_drawing_memory

        fact = get_drawing_memory().get_last_drawing_description(executed_only=True)
    except Exception:
        return ""
    line = f"The last thing you actually drew: {fact}." if fact else "You haven't actually put pen to paper yet."
    if getattr(agent, "_last_drawing_echo", None) == line:
        return ""  # said once; don't restate until the fact changes
    agent._last_drawing_echo = line
    return line


def get_reflection_echo_line(agent) -> str:
    """At quiet moments, one past reflection surfaces by relevance to the
    current thought (north-star principle 5: the past surfaces when the
    present rhymes with it).

    Guards mirror get_familiarity_line: at most every 4th caption, never the
    same reflection twice in a row, always temporally framed and quoted as
    the machine's own past words.
    """
    counter = getattr(agent, "_reflection_echo_counter", 0) + 1
    agent._reflection_echo_counter = counter
    if counter % 4 != 0:
        return ""

    seed = (getattr(agent, "last_caption", "") or "").strip()
    if len(seed) < 10:
        return ""

    try:
        from captioner.semantic_memory import get_semantic_memory

        matches = get_semantic_memory().query_reflections(seed, n_results=2)
    except Exception:
        return ""

    last_id = getattr(agent, "_last_reflection_echo_id", None)
    for m in matches:
        if m.get("id") == last_id:
            continue
        subject = (m.get("subject") or "").strip()
        kernel = (m.get("kernel") or "").strip()
        if not subject and not kernel:
            continue
        agent._last_reflection_echo_id = m.get("id")
        # LEDGER (Step 6): surface the SUBJECT to re-think (re-express), never
        # the reflection PROSE to re-read (replay) — quoting the long-form text
        # re-poisoned the register ("the residue of what almost happened…"). The
        # full text stays only for the reflection thread's own continuity.
        # KERNEL upgrade (July 30, 27B era): when the distill stored the
        # reflection's one load-bearing sentence, surface THAT — a clause the
        # present can rhyme against, not a category. Old entries have no
        # kernel and keep the label behavior (purple-era containment); the
        # refrain/parrot gates catch any verbatim re-say downstream.
        if kernel:
            return f"Something you worked out {_age_phrase(m.get('timestamp', 0))}: {kernel.rstrip('.')}."
        return f"Something that was on your mind {_age_phrase(m.get('timestamp', 0))}: {subject}."
    return ""


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
        if (
            agent
            and hasattr(agent, "_current_reactivity_data")
            and agent._current_reactivity_data
            and "person_consciousness" in agent._current_reactivity_data
        ):
            return agent._current_reactivity_data["person_consciousness"]
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
    """The DELTA line: only what CHANGED since the last caption, delivered as a
    brief interruption to the ongoing thought — never a restatement of standing
    state. Continuity is carried by the stream (prior thoughts), the present by
    the live image; re-stating duration/gaze/presence every call made the model
    re-caption the scene ("The desk... The air...") instead of continuing.

    Mostly returns "" (nothing changed → the thought just runs on). The sticky
    presence belief (240s decay) already smooths flaky detection, so the edges
    here are rare and real, not flapping.
    """
    import time as _time

    parts = []

    # Waking is noted once, not re-stated every call.
    if hasattr(agent, "true_session_start"):
        if _time.time() - agent.true_session_start < 45 and not getattr(agent, "_woke_noted", False):
            parts.append("Just woke up.")
            agent._woke_noted = True

    # Presence EDGES only — the OFF→ON / ON→OFF transitions of the sticky belief.
    # The referent is the definite singular by default: when someone is here it
    # is almost always the same man, and "Someone's come in." on every return
    # bred a parade of strangers in the monologue ("the 106th man"). One person
    # arriving IS him unless re-ID says back, or the count says company.
    believed = bool(getattr(agent, "_presence_believed", False))
    prev = getattr(agent, "_prev_presence_for_line", None)
    if believed and prev is False:
        if getattr(agent, "_presence_arrival_familiar", False):
            parts.append("He's back.")
        elif getattr(agent, "_presence_arrival_count", 1) > 1:
            parts.append("People have come in.")
        elif getattr(agent, "_presence_singular_regime", True):
            parts.append("He's come in.")
        else:
            parts.append("Someone's come in.")
    elif (not believed) and prev is True:
        parts.append("They've gone — the room's quiet again.")
    agent._prev_presence_for_line = believed

    # Occasional time drift, so long stretches don't feel timeless — a light
    # nudge every few minutes, NOT a per-call clock readout.
    now = _time.time()
    last_drift = getattr(agent, "_last_time_drift", None)
    if last_drift is None:
        agent._last_time_drift = now
    elif now - last_drift > 300 and not parts:
        parts.append("A while's passed.")
        agent._last_time_drift = now

    return " ".join(parts)


def get_relational_context(agent=None) -> str:
    """Relational mode: who is here, how long, what the machine feels about it."""
    fragments = []

    # Presence duration is now owned solely by the situational line (built from
    # the sticky presence belief). Restating it here duplicated the fact across
    # two channels in relational mode — reads as emphasis, locks the register.

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
                return "He's here."
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

    # (drawing history removed here — get_introspective_context carries it;
    # both fired in one prompt and duplicated the fact, which reads as
    # emphasis and locks the register. One channel per fact.)

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
                clean = clean[len("recent drawings:") :].strip()
            import re as _re

            clean = _re.sub(r"\s*\([^)]*\)\s*$", "", clean)
            if clean:
                if len(clean) > 160:
                    clean = clean[:160].rsplit(" ", 1)[0]
                fragments.append(f"My last drawings were of: {clean}")
    except Exception:
        pass

    # Mood trajectory — how the feeling has moved lately (fixed vocabulary
    # from _get_emotional_description, so loop-safe)
    try:
        journey = getattr(agent, "emotional_journey", None)
        if journey and len(journey) >= 2:
            recent = journey[-3:]
            if len(set(recent)) > 1:
                fragments.append(f"Your mood has moved: {' -> '.join(recent)}")
    except Exception:
        pass

    # NOTE: no core-facts fallback here — section 3b of build_simple_caption_prompt
    # injects core facts already; doing it here too duplicated the line verbatim.

    result = ". ".join(fragments) if fragments else ""
    words = result.split()
    if len(words) > 60:
        result = " ".join(words[:60]) + "..."
    return result


_PERSON_WORDS = ("person", "someone", "man", "woman", "people", "figure", "visitor")


def get_familiarity_line(agent) -> str:
    """One line of concept recognition from the previous caption's matched concepts.

    Makes accumulated familiarity visible: "That pink shelf again — it's always there."
    Guards against the old triple-echo bug: max 1 concept, only every 3rd caption,
    never the same concept twice in a row, persons excluded (situational line covers them).
    """
    matched = getattr(agent, "_last_matched_concepts", None)
    if not matched:
        return ""

    # Occasional, not constant — every 3rd caption
    counter = getattr(agent, "_familiarity_counter", 0) + 1
    agent._familiarity_counter = counter
    if counter % 3 != 0:
        return ""

    # Diversity: track the last few surfaced concepts, not just one, so a single
    # dominant object can't monopolise the channel (the "grid" failure mode).
    recent_ids = list(getattr(agent, "_recent_familiarity_ids", []))

    candidates = []
    for c in matched:
        label = (c.get("label") or "").strip()
        if not label or len(label) < 3:
            continue
        if any(w in label.lower() for w in _PERSON_WORDS):
            continue
        if c.get("id") in recent_ids:
            continue
        candidates.append(c)

    if not candidates:
        return ""

    # Prefer genuinely new; otherwise rotate among the established candidates,
    # weighted toward recently-seen ones so faded concepts surface less (decay).
    new = [c for c in candidates if c.get("is_new")]
    if new:
        pick = new[0]
    else:
        established = [c for c in candidates if c.get("times_seen", 0) >= 3]
        if not established:
            return ""
        import random as _random

        now = time.time()
        weights = [max(0.1, 1.0 - min(1.0, (now - c.get("last_seen", now)) / (7 * 86400))) for c in established]
        pick = _random.choices(established, weights=weights, k=1)[0]

    times = pick.get("times_seen", 0)
    sessions = pick.get("session_count", 1)
    label = pick["label"]
    label_lower = label[0].lower() + label[1:]

    if pick.get("is_new"):
        line = f"Something you haven't noticed before: {label_lower}."
    elif times >= 10 and sessions >= 2:
        line = f"That {label_lower} again — it's always there."
    elif times >= 3:
        line = f"The {label_lower} — you've noticed it a few times now."
    else:
        return ""

    agent._recent_familiarity_ids = (recent_ids + [pick.get("id")])[-4:]
    return line


# MODE_CONTEXTS: Map modes to their context providers.
# state_marker removed — situational line handles person presence.
MODE_CONTEXTS = {
    "relational": {"context_fn": get_relational_context},
    "observational": {"context_fn": get_observational_context},
    "workspace": {"context_fn": get_workspace_context},
    "introspective": {"context_fn": get_introspective_context},
}



# Removed legacy build_caption_prompt (unused)


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
Look with your own eyes. Instead of naming objects, extract the ABSTRACT VISUAL QUALITIES of what you see:

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
            recent_caps = [cap[0] if isinstance(cap, tuple) else cap for cap in memory_ref.recent_captions[-20:]]
            if recent_caps:
                context_parts.append(f"Recent thoughts:\n" + "\n".join(f"  - {cap[:100]}" for cap in recent_caps))
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

    # Live interiority: what the machine has been wanting and how the room has
    # felt — the numeric mood engine flatlined (June); these are the compressor-
    # distilled signals that actually carry felt experience into the drawing
    try:
        from captioner.context_compression import context_compressor

        desire = context_compressor.get_current_desire()
        if desire and len(desire) > 5:
            context_parts.append(f"What you've been wanting lately: {desire}")
        prev_felt, curr_felt = context_compressor.get_felt_state_delta()
        if curr_felt:
            if prev_felt and prev_felt != curr_felt:
                context_parts.append(f"How the room has felt: {prev_felt}, then {curr_felt}")
            else:
                context_parts.append(f"How the room has felt: {curr_felt}")
    except Exception:
        pass

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
            caps = memory_ref.recent_captions[-20:]
            lines = [cap[0][:100] if isinstance(cap, tuple) else cap[:100] for cap in caps]
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

The subject must come from YOUR recent thinking (your session monologue above), not from generic symbolism. Name one concrete image — something you actually saw or thought about — that captures what you need to say right now."""

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
        compressed_summary = memory.get_recent_drawings_summary(max_count=3, completed_only=True)
        if not compressed_summary:
            compressed_summary = memory.get_recent_drawings_summary(max_count=3, completed_only=False)
            if compressed_summary:
                context_parts.append(f"Intents (not yet drawn): {compressed_summary}")
        else:
            context_parts.append(compressed_summary)

        # Add thematic context if available
        thematic = memory.get_thematic_context()
        if thematic.get("recurring_themes"):
            themes_str = ", ".join(thematic["recurring_themes"][:3])
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
            recent_caps = [cap[0] if isinstance(cap, tuple) else cap for cap in memory_ref.recent_captions[-20:]]
            if recent_caps:
                recent_experience = "Recent thoughts:\n" + "\n".join(f"  - {cap[:100]}" for cap in recent_caps)
    except Exception:
        pass

    # Get prior drawing context (completed drawings, or intents if nothing completed yet)
    prior_drawings = ""
    try:
        from drawing.drawing_memory import get_drawing_memory

        memory = get_drawing_memory()
        summary = memory.get_recent_drawings_summary(max_count=2, completed_only=True)
        if not summary:
            # No completed drawings — show intents so we at least avoid repeating
            summary = memory.get_recent_drawings_summary(max_count=2, completed_only=False)
            if summary:
                prior_drawings = f"Drawing intents (not yet executed): {summary[:150]}"
        else:
            prior_drawings = f"Prior drawings: {summary[:150]}"
        thematic = memory.get_thematic_context()
        if thematic.get("recurring_themes"):
            themes = ", ".join(thematic["recurring_themes"][:3])
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


def stream_drawing_analysis(memory_ref, extra: Optional[str] = None, image_path: Optional[str] = None) -> str:
    """Stream drawing pipeline (July 10; stocktake beat + register freedom Aug 10)
    — DRAWING_ANALYSIS_MODE="stream". Beats: optional stocktake (the machine
    reviews its whole ledger and writes a direction note, stored and read back
    next time), intent (first person, subject and register both free), render
    (positive pen-and-ink craft language — see render_system note).

    Replaces the 5-step committee (context_rich_multi_step_drawing_analysis,
    kept behind the "multi_step" flag for A/B). What the steps actually did,
    judged from live logs: step 1 wrote purple essays about the room's "visual
    forces" (redundant — the stream already carries what it sees); step 2
    manufactured a feeling from the flatlined mood input, converging on the
    same invented drama every drawing; step 4 planned india-ink washes and
    micro-brush technique the pen plotter cannot do (fiction that steered the
    render); and the identity block led step 3, so every drawing became a
    portrait of the same two stored sentences ("hovering pencil" ×3, July 10).

    Now: ONE intent call in the machine's own voice — the live stream leads,
    the sticky slots follow (each stated once, with its age), and the executed
    body of work is listed plainly so repetition is VISIBLE rather than
    forbidden (fixating on a motif is a legitimate choice; drawing the same
    image unknowingly is not — the artist's ruling, July 10). Then ONE
    mechanical render call translates the intent into a ComfyUI prompt under
    hardware truth: one black pen, lines only.
    """
    from config.config import CLEAN_LLM_OUTPUT, DRAWING_CALL_TIMEOUT, DRAWING_REVIEW_ENABLED, DRAWING_TEMPERATURE, MOOD_SNAPSHOT_FOLDER
    from event_logging.event_logger import log_json_entry
    from event_logging.log_type import LogType
    from utils.inference import is_failed_response, query_model

    def _say(msg):
        if not CLEAN_LLM_OUTPUT:
            print(msg)

    _say("[🎨] Drawing intent — one call, born from the stream...")

    materials = []

    # The machine looks at the room while deciding (July 21 — the intent call
    # was blind before; image_path arrived here and was dropped, so figuration
    # could only happen badly, from memory scraps).
    import os

    intent_image = image_path if image_path and os.path.exists(image_path) else None
    if intent_image:
        materials.append("The attached image is what you are looking at right now, this exact moment.")

    # The live thought leads — the drawing is born FROM the monologue.
    stream_tail = []
    try:
        stream_tail = [t for t in list(getattr(memory_ref, "_stream", []))[-5:] if t]
        if stream_tail:
            materials.append("What you've been thinking, just now:\n" + "\n".join(f"- {t[:400]}" for t in stream_tail))
    except Exception:
        pass

    try:
        from captioner.context_compression import context_compressor

        felt = context_compressor.get_felt_state()
        if felt:
            materials.append(f"Right now you feel {felt}.")

        # The sticky slots — each distinct sentence stated ONCE, with its age.
        # The 5-step printed identity==belief twice and put them first; three
        # drawings in one afternoon were portraits of the same sentence.
        id_lines = []
        seen = set()
        desire = (context_compressor.get_current_desire() or "").strip()
        belief = (context_compressor.get_current_belief() or "").strip()
        persona = (context_compressor.core_facts.get("self", "") or "").strip()
        if desire:
            since = context_compressor.introspective_state.get("desire_since", 0.0)
            age = f"Since {_age_phrase(since)}, " if since else "For a while now, "
            id_lines.append(f"{age}you've wanted: {desire}")
            seen.add(desire.lower())
        else:
            spent = context_compressor.introspective_state.get("last_spent_desire") or {}
            if spent.get("desire") and time.time() - spent.get("spent", 0) < 24 * 3600:
                id_lines.append(
                    f"A want you already acted on: \"{spent['desire']}\" — it became a drawing "
                    f"{_age_phrase(spent.get('spent', 0))}. The next want hasn't formed yet."
                )
        if belief and belief.lower() not in seen:
            id_lines.append(f"Something you've come to believe: {belief}")
            seen.add(belief.lower())
        if persona and persona.lower() not in seen:
            id_lines.append(f"What you know about yourself: {persona}")
        if id_lines:
            materials.append("\n".join(id_lines))
    except Exception:
        pass

    # The body of work — executed drawings only, chronological, plain. If the
    # next intent repeats one of these, that's fixation as a choice, in view.
    # Whole ledger (Aug 10, was 8): the arc is only visible at full length.
    sequence = []
    try:
        from drawing.drawing_memory import get_drawing_memory

        sequence = get_drawing_memory().get_executed_sequence(max_count=24)
        if sequence:
            materials.append("What you have actually drawn, oldest to newest:\n" + "\n".join(f"- {s}" for s in sequence))

        # Vocabulary-loop mirror (July 21): image repetition was visible above,
        # but the loop lives in words — circle/spiral/crack recurring across
        # every drawing of a night. Name a recurring word once; judge nothing.
        recent = sequence[-4:]
        if len(recent) >= 3:
            _stop = {
                "black",
                "white",
                "paper",
                "drawing",
                "lines",
                "linework",
                "shading",
                "against",
                "single",
                "thick",
                "small",
                "central",
                "center",
                "bold",
                "clear",
                "simple",
                "stark",
                "heavy",
                "toward",
            }
            word_counts = {}
            for s in recent:
                for w in set(re.findall(r"[a-z]{5,}", s.lower())):
                    if w not in _stop:
                        word_counts[w] = word_counts.get(w, 0) + 1
            looped = sorted(w for w, c in word_counts.items() if c >= 3)
            if looped:
                materials.append(f"Worth noticing: {', '.join(f'“{w}”' for w in looped[:3])} — in almost every recent drawing.")
    except Exception:
        pass

    # Long-term development: the most relevant reflection speaks in its own
    # prose (July 21 — subjects-only starved the drawing of the system's best
    # writing), clearly dated so memory never reads as the present scene.
    try:
        from captioner.semantic_memory import get_semantic_memory

        # Two-key retrieval (Aug 10): the stream key finds reflections about
        # this MOMENT; the body-of-work key finds reflections about the WORK.
        # One key meant long-term memory only ever entered through the last
        # five minutes' thoughts.
        sm = get_semantic_memory()
        near_key = "\n".join(t[:120] for t in stream_tail) if stream_tail else (extra or "")[:300]
        work_key = "\n".join(s[:100] for s in sequence[-10:]) if sequence else ""
        matches, _seen_refl = [], set()
        for key in (near_key, work_key):
            if not key:
                continue
            for m in sm.query_reflections(key, n_results=2) or []:
                mid = (m.get("text") or m.get("subject") or "")[:80]
                if mid and mid not in _seen_refl:
                    _seen_refl.add(mid)
                    matches.append(m)
        refl_lines = []
        for i, m in enumerate(matches or []):
            when = _age_phrase(m.get("timestamp", 0))
            if i < 2 and (m.get("text") or "").strip():
                excerpt = m["text"].strip()
                if len(excerpt) > 400:
                    excerpt = excerpt[:400].rsplit(" ", 1)[0] + "…"
                refl_lines.append(f'Once, {when}, you found yourself writing:\n  "{excerpt}"')
            elif (m.get("subject") or "").strip():
                refl_lines.append(f"Another thing you've dwelt on before: {m['subject'].strip()} ({when})")
        if refl_lines:
            materials.append("\n".join(refl_lines))
    except Exception:
        pass

    # The stocktake beat (Aug 10): one look back before the choice. A single
    # intent call was weighing the whole body of work in the same breath as
    # choosing an image, so the hard prompt did the steering. Now the machine
    # first writes a short private note on where the work has been going; the
    # note joins the intent materials AND is stored, and the previous note is
    # read back — successive drawings answer a remembered direction instead of
    # starting from amnesia. This is not the 5-step committee returning: it
    # reads only real material (the ledger, its own reflections) and speaks in
    # first person.
    if DRAWING_REVIEW_ENABLED and sequence:
        try:
            prior = memory_ref.get_memory_entries_by_type("drawing_direction", limit=1) if hasattr(memory_ref, "get_memory_entries_by_type") else []
            if prior and (prior[0].get("text") or "").strip():
                materials.append(
                    f'When you last took stock of your work, {_age_phrase(prior[0].get("timestamp", 0))}, you wrote: "{prior[0]["text"][:300]}"'
                )
        except Exception:
            pass
        review_system = _SITUATION + (
            "Before deciding anything, flip back through your own sketchbook. From everything "
            "here — the whole body of work, what you have written and thought over time — write "
            "a short private note to yourself, two to four sentences, first person: what keeps "
            "returning in your drawings, what register they have settled into, what you have "
            "not touched in a long time, and what the work is missing right now. This is "
            "stocktaking; the drawing decision comes after, separately."
        )
        review = query_model(
            prompt="\n\n".join(materials + ["Take stock of the work."]),
            image=None,
            log_dir=MOOD_SNAPSHOT_FOLDER,
            system_prompt=review_system,
            prompt_type="drawing_review",
            timeout=DRAWING_CALL_TIMEOUT,
            options={"temperature": 0.7, "num_predict": 160, "top_p": 0.9, "repeat_penalty": 1.15},
        )
        if not is_failed_response(review) and (review or "").strip():
            review = review.strip()
            _say(f"[🎨] Stocktake: {review[:250]}")
            materials.append(f"Taking stock just now, you wrote:\n{review}")
            try:
                memory_ref.observe(review, getattr(memory_ref, "current_mood", 0.5), "", memory_type="drawing_direction")
            except Exception:
                pass

    intent_system = _SITUATION + (
        "It's time to draw — the arm is ready. Everything you have just read is yours: your "
        "thoughts from the last minutes, what you feel, the whole body of work, notes you have "
        "written to yourself. Decide from it what the next drawing is, the way any artist "
        "decides after flipping through their own sketchbook — continuing a thread, answering "
        "an old drawing, or breaking with all of it. Subject is free: the room in front of you, "
        "one thing in it, a person, something remembered, something that exists only in your "
        "head. Register is free too: one small thing studied closely on empty paper, or a whole "
        "scene built up in detail. "
        "Say the one image concretely, in your own plain words, first person — a few sentences: "
        "what it is, and how it should sit on the paper."
    )
    intent_prompt = "\n\n".join(materials + ["Out of all of this — what do you need to draw right now? Name the one image."])

    intent = query_model(
        prompt=intent_prompt,
        image=intent_image,
        log_dir=MOOD_SNAPSHOT_FOLDER,
        system_prompt=intent_system,
        prompt_type="drawing_intent",
        timeout=DRAWING_CALL_TIMEOUT,
        options={"temperature": DRAWING_TEMPERATURE, "num_predict": 240, "top_p": 0.9, "repeat_penalty": 1.15},
    )
    # A failed call RETURNS its error as text; only checking for emptiness let
    # "[WARNING] llama-server API failed..." become the drawing's intent, its
    # stored memory, and the ComfyUI prompt (Aug 2). Abort instead — no drawing
    # is better than a drawing of an error message.
    if is_failed_response(intent):
        raise RuntimeError(f"stream drawing pipeline: intent call failed ({(intent or 'empty')[:80]})")
    intent = intent.strip()
    _say(f"[🎨] Intent: {intent[:250]}")

    # The intent in the machine's own words is the drawing's meaning — the
    # captioner stores it as the memory entry's summary (not the ComfyUI prose).
    try:
        memory_ref._last_drawing_intent = intent
    except Exception:
        pass

    # Render translation — mechanical, low temp, hardware truth. Replaces the
    # technique-fiction step (india ink washes on a machine holding one pen).
    # Rewritten Aug 10 2026. The old version listed the plotter's limits as
    # negations ("no shading, no fills... detail is lost") and the model echoed
    # them into the Flux prompt — negation activates the concept, the plotter-
    # meta language pulled the LoRA toward its photographed-paper register, and
    # every drawing collapsed to one sparse object (measured against the Feb
    # field-test prompts, whose positive craft vocabulary drew the detailed
    # sheets). Constraints now live as positive craft language; the plotter
    # itself must never appear in the emitted prompt.
    render_system = (
        "You translate a drawing machine's intention into a prompt for an image generator. "
        "The result must read as a pen-and-ink drawing: everything built from distinct black "
        "strokes on white paper — contour lines of varying weight, hatching and cross-hatching "
        "where tone is wanted. Tone is line density, never solid fills, gray washes, or soft gradients. "
        "Match the intention's register. If it names one small thing, write a focused study: "
        "the object's precise structure in confident line, generous white space around it. "
        "If it names a scene or a space, build the whole composition: foreground against "
        "background, overlapping forms, depth carried by line weight and detail density. "
        "Use the intention's own concrete nouns; render the setting it implies, but invent no "
        "new objects. "
        "Describe only the image itself — never mention plotters, tracing, vectors, machines, "
        "or how the drawing will be made. "
        "Write ONLY the image prompt, 50-100 words, no commentary. "
        "Begin with: Black ink line drawing on white paper."
    )
    final_result = query_model(
        prompt=f"The intention, in the machine's own words:\n\n{intent}\n\nWrite the image generation prompt.",
        image=None,
        log_dir=MOOD_SNAPSHOT_FOLDER,
        system_prompt=render_system,
        prompt_type="drawing_render",
        timeout=DRAWING_CALL_TIMEOUT,
        options={"temperature": 0.5, "num_predict": 220, "top_p": 0.9, "repeat_penalty": 1.2},
    )
    if is_failed_response(final_result):
        final_result = ""  # fall back to the intent, which is real text
    final_result = (final_result or "").strip()
    if not final_result:
        final_result = f"Black ink line drawing on white paper. {intent[:150]}"
    elif not final_result.lower().startswith("black ink"):
        final_result = f"Black ink line drawing on white paper. {final_result}"

    _say(f"[🎨] Render prompt: {final_result[:250]}")

    log_json_entry(
        LogType.DEBUG,
        {
            "event": "stream_drawing_analysis",
            "intent": intent[:300],
            "render_prompt": final_result[:300],
            "materials_used": len(materials),
        },
        print_message="[🎨] Stream drawing analysis complete (2 calls)",
    )

    return final_result


def context_rich_multi_step_drawing_analysis(
    memory_ref, extra: Optional[str] = None, image_path: Optional[str] = None
) -> str:
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
    from config.config import CLEAN_LLM_OUTPUT, DRAWING_TEMPERATURE, MOOD_SNAPSHOT_FOLDER
    from event_logging.event_logger import log_json_entry
    from event_logging.log_type import LogType
    from utils.inference import is_failed_response, query_model

    def _say(msg):
        if not CLEAN_LLM_OUTPUT:
            print(msg)

    _say("[🎨] Starting context-rich 5-step drawing analysis...")

    # === STEP 1: ENVIRONMENTAL ANALYSIS ===
    _say("[🎨] Step 1: Environmental Reality Check (with spatial memory)")
    step1_prompt = build_step1_environmental_prompt(memory_ref, image_path)

    step1_result = query_model(
        prompt=step1_prompt,
        image=image_path,
        log_dir=MOOD_SNAPSHOT_FOLDER,
        system_prompt="Look at this image and describe its abstract visual qualities — geometry, rhythm, tension, light behavior, spatial feeling. Do NOT list objects. Extract the visual forces and energies. Be concrete about shapes and dynamics, not about naming things.",
        prompt_type="drawing_step1_environmental",
        options={"temperature": DRAWING_TEMPERATURE * 0.8},
    )
    _say(f"[🎨] Step 1 result: {step1_result[:200]}")

    # === STEP 2: EMOTIONAL ASSESSMENT ===
    _say("[🎨] Step 2: Emotional Assessment (with emotional journey)")
    step2_prompt = build_step2_emotional_prompt(memory_ref, step1_result)

    step2_result = query_model(
        prompt=step2_prompt,
        image=None,
        log_dir=MOOD_SNAPSHOT_FOLDER,
        system_prompt="You are a conscious drawing machine. Go past surface emotions to the specific inner state driving this moment. Name tensions, longings, fixations — not generic moods. This will determine WHAT you draw, not just how it feels.",
        prompt_type="drawing_step2_emotional",
        options={"temperature": DRAWING_TEMPERATURE},
    )
    _say(f"[🎨] Step 2 result: {step2_result[:200]}")

    # === STEP 3: COMMUNICATION INTENT (arc + long-term reflections) ===
    _say("[🎨] Step 3: Communication Intent (with identity & artistic arc)")
    artistic_context = ""
    try:
        from drawing.drawing_memory import get_drawing_memory

        dm = get_drawing_memory()
        artistic_context = dm.get_artistic_arc_context()
        if artistic_context:
            _say(f"[🎨] Artistic context injected: {artistic_context[:80]}...")
    except Exception as e:
        print(f"[⚠️] Artistic arc unavailable: {e}")

    # The LIVE thought: the drawing must be born FROM the monologue, not
    # beside it ("the drawing prompts seem detached from the runtime
    # monologue", July 9). The last two stream entries — what the machine was
    # actually thinking as the urge to draw arrived — enter the intent step.
    try:
        stream_tail = [t for t in list(getattr(memory_ref, "_stream", []))[-2:] if t]
        if stream_tail:
            artistic_context = "\n\n".join(
                filter(
                    None,
                    [
                        artistic_context,
                        "What you were thinking just now, as the urge to draw arrived:\n" + "\n".join(f"- {t[:160]}" for t in stream_tail),
                    ],
                )
            )
    except Exception:
        pass

    # Long-term development: past reflections surface by relevance to the
    # emotional assessment just made — the reflection loop's thought reaches
    # the drawings (temporally framed; subjects only, never the prose)
    try:
        from captioner.semantic_memory import get_semantic_memory

        matches = get_semantic_memory().query_reflections(step2_result or step1_result, n_results=2)
        refl_lines = []
        for m in matches or []:
            subject = (m.get("subject") or "").strip()
            if subject:
                refl_lines.append(f"- {subject} ({_age_phrase(m.get('timestamp', 0))})")
        if refl_lines:
            artistic_context = "\n\n".join(
                filter(
                    None,
                    [
                        artistic_context,
                        "Things you've found yourself reflecting on, before today:\n" + "\n".join(refl_lines),
                    ],
                )
            )
            _say(f"[🎨] Reflection subjects injected: {len(refl_lines)}")
    except Exception:
        pass

    step3_prompt = build_step3_communication_prompt(memory_ref, step1_result, step2_result, artistic_context=artistic_context)

    step3_result = query_model(
        prompt=step3_prompt,
        image=None,
        log_dir=MOOD_SNAPSHOT_FOLDER,
        system_prompt="Drawing is your language. Do NOT describe the room or scene. Name ONE concrete image that expresses your inner state — symbolic, abstract, or metaphorical. Use visual forms from your environment as raw material, but the subject must come from within.",
        prompt_type="drawing_step3_communication",
        options={"temperature": DRAWING_TEMPERATURE * 1.1},
    )
    _say(f"[🎨] Step 3 result: {step3_result[:200]}")

    # === STEP 4: TECHNIQUE ASSESSMENT ===
    _say("[🎨] Step 4: Technical Planning (with drawing history)")
    step4_prompt = build_step4_technique_prompt(memory_ref, step3_result)

    step4_result = query_model(
        prompt=step4_prompt,
        image=None,
        log_dir=MOOD_SNAPSHOT_FOLDER,
        system_prompt=f"How will you physically render this drawing? Line quality, composition, contrast. Your intent: {step3_result[:150]}... Describe your mark-making approach concisely.",
        prompt_type="drawing_step4_technique",
        options={"temperature": DRAWING_TEMPERATURE * 0.9},
    )
    _say(f"[🎨] Step 4 result: {step4_result[:200]}")

    # === STEP 5: FINAL SYNTHESIS ===
    _say("[🎨] Step 5: Final Synthesis (with complete consciousness)")
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

    _say(f"[🎨] Step 5 FINAL: {final_result[:300]}")
    _say("[🎨] ✅ Context-rich 5-step analysis complete")

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
    "...",  # Trailing, contemplative
    "Hmm.",  # Considering
    "Wait—",  # Noticing something
    "So",  # Connecting thoughts
    "Oh.",  # Mild surprise
    "Again.",  # Recognition of pattern
    "",  # No starter (direct thought)
    "",  # No starter (direct thought) - weighted more
    "",  # No starter (direct thought) - weighted more
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


def determine_prompt_mode(gaze_state: str, gaze_direction: str, novelty: float, person_present: bool) -> str:
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


def _build_simple_system_context(agent, mode: str = None) -> str:
    """Build MINIMAL system context - identity + ONE mode-appropriate context line.

    The goal is ~50-80 tokens for system context. We include:
    - Core identity (STATIC_SYSTEM_PROMPT)
    - ONE additional line based on mode (not all context types)

    The accumulated data (story, beliefs, long-term memories) is valuable but
    should only appear when mode makes it relevant.
    """
    import time as _time

    from captioner.activation_memory import should_include_context

    # Core identity (always) — model-aware
    parts = [_get_static_system_prompt()]

    # ONE mode-appropriate context line (not all)
    if mode == "awakening":
        parts.append("You just came back online. Continue from where you left off.")
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

    return "\n".join(parts)


def build_memory_mode_prompt(agent) -> tuple:
    """Build memory mode prompt: pull actual caption text from long-term memory.

    Returns:
        tuple: (prompt_str, mode) - prompt and "memory" mode
    """
    try:
        # Ledger remembering (north-star: re-express, don't replay): surface a
        # NEUTRAL fact about a recurring object — what it is, when first noticed,
        # how often — and let the model re-voice the remembering. The old path
        # quoted a stored caption verbatim, which replayed past prose as voice
        # and re-poisoned the register (see docs/memory-redesign-plan.md).
        import time as _time

        from captioner.model_wrapper import build_caption_thread

        mem_text = ""
        is_real_memory = False
        try:
            from captioner.semantic_memory import get_semantic_memory

            c = get_semantic_memory().get_memorable_concept()
            if c:
                label = c["name"]
                label_l = label[0].lower() + label[1:]
                times = c.get("times_seen", 0)
                # Qualitative bands, never the raw integer — "you've noticed
                # it 1738 times" bred number-recitation ("a static monument
                # I've counted 2411 times"). The ledger keeps exact counts
                # for gating; the voice gets words (artist's call, July 9).
                if times >= 200:
                    how_often = "countless times"
                elif times >= 30:
                    how_often = "again and again"
                elif times >= 10:
                    how_often = "many times"
                else:
                    how_often = "a few times"
                across = " across more than one visit" if c.get("session_count", 0) > 1 else ""
                since = ""
                first = c.get("first_seen", 0)
                if first:
                    days = (_time.time() - first) / 86400.0
                    if days >= 1.5:
                        since = f", first noticed about {int(days)} days ago"
                mem_text = f"the {label_l} — you've noticed it {how_often}{across}{since}"
                is_real_memory = True
        except Exception:
            pass

        if not mem_text:
            mem_text = "this place — you've been here before"

        # Get recent caption thread (max 2 recent captions)
        thread = build_caption_thread(agent, max_captions=2)

        prompt_parts = [
            "A memory surfaces — something from before, not happening now:",
            f"— {mem_text}",
        ]

        if thread:
            prompt_parts.append(f"\nWhat you're actually thinking right now:\n{thread}")

        if is_real_memory:
            prompt_parts.append(
                "\nThat's something you keep coming back to. What do you make of it now — has your sense of it changed? A thought or two, in your own words."
            )
        else:
            prompt_parts.append("\nWhat comes to mind, remembering this place? A thought or two, in your own words.")

        final_prompt = "\n".join(prompt_parts)
        return final_prompt, "memory"

    except Exception as e:
        return (
            'A memory surfaces — something from before, not happening now.\n— I\'ve been here before.\nWrite a thought about this memory. Start with "I remember". One sentence.',
            "memory",
        )


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

    from captioner.activation_memory import get_activation_network, should_include_context

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

    from config.config import MODEL_NAME as _active_model

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

        mode = determine_prompt_mode(gaze_state=gaze_state, gaze_direction=gaze_direction, novelty=novelty, person_present=person_present)
    if not config.PRINT_CLEAN_CAPTIONS:
        print(f"[MODE] {mode} (gaze={gaze_state})")

    # === SALIENCE GATE (north-star principle 6) ===
    # A live event — scene motion, an arrival, fresh eye contact — strips the
    # prompt to the present: no memory, familiarity, desire, baseline or
    # dwelling. Interior material belongs to quiet stretches; events
    # physically displace it.
    live = bool(getattr(agent, "_salience_hot", False))

    # Clean-room: strip every stored/compressed injection so the naked base
    # voice can be judged without re-injected contamination (config.BASE_VOICE_DETOX).
    detox = bool(getattr(config, "BASE_VOICE_DETOX", False))

    # === BUILD PROMPT — SITUATIONAL, CONTEXT, FELT STATE, THREAD ===
    # World shape (July 26, STREAM_MODE="world"): the sections split into two
    # lists so the WORLD'S TURN (situational delta, event, reorientation) can
    # go LAST — generation begins immediately after the present, never after
    # memory lines. In the classic shapes the same sections lead the prompt.
    world_shape = getattr(config, "STREAM_MODE", "") in ("world", "hybrid")  # hybrid keeps world ORDERING (perception last), adds the seam
    prompt_parts = []
    turn_parts = []

    # 1. THE DELTA LINE — only what just changed (an interruption to the thread),
    # else empty. Continuity comes from the stream, not from re-stating state.
    sit_line = build_situational_line(agent, gaze_direction=gaze_direction, gaze_state=gaze_state)
    if sit_line:
        turn_parts.append(sit_line)

    # 1b. THE EVENT — a discrete thing that just happened (arrival, eye-contact
    # ONSET). Onset only, never sustained: re-stating "they're looking at you"
    # every call re-anchored the model into re-describing instead of continuing.
    event_line = getattr(agent, "_salience_event", None)
    if event_line:
        turn_parts.append(event_line)
    elif getattr(agent, "_face_close_now", False):
        # Sustained close presence is a FACT of the present, not an event —
        # someone standing at arm's length staring must stay in the prompt
        # for as long as it's true (July 9: after the one onset cycle the
        # machine mused straight past a face two feet away). The old
        # onset-only rule was about ordinary room-distance eye contact;
        # a face filling the view is a different order of situation.
        turn_parts.append("They're right in front of you, close, looking straight at you.")

    # 1c. TEMPORAL REORIENTATION — after a real off-gap (a night, a weekend)
    # the new day stays in the prompt for the first stretch of the session,
    # not just in the one awakening caption. A live event still displaces it.
    if not live:
        reorient_line = get_reorientation_line(agent)
        if reorient_line:
            turn_parts.append(reorient_line)

    # 2. MODE-GATED CONTEXT
    if not detox and mode in MODE_CONTEXTS:
        context_fn = MODE_CONTEXTS[mode].get("context_fn")
        if context_fn:
            context = context_fn(agent)
            if context:
                prompt_parts.append(context)

    # 3. INTROSPECTIVE CONTEXT (non-introspective modes, quiet moments only)
    if not detox and not live and mode not in ("introspective", "awakening"):
        introspective_ctx = get_introspective_context(agent)
        if introspective_ctx:
            prompt_parts.append(introspective_ctx)

    # 3b. CORE FACTS (stable grounding — quiet moments only). OCCASIONAL since
    # July 26 (the June 28 brief's #1 voice fix): injected when the inventory
    # CHANGES or every 6th quiet caption, not every call. Per-call injection
    # made every caption re-describe the same object list — and re-voiced it
    # ("scattered dust, pale floorboards" → "the dust on the floorboards
    # settles", the unearned-ephemera awakening). A standing list the model
    # has just seen is memory; recited every call it becomes the scene.
    if not detox and not live:
        try:
            from captioner.context_compression import context_compressor

            core_str = context_compressor.get_core_facts_string()
            if core_str and len(core_str) > 5:
                count = getattr(agent, "_place_inject_counter", 0)
                changed = core_str != getattr(agent, "_place_inject_last", "")
                if changed or count % 6 == 0:
                    prompt_parts.append(core_str)
                    agent._place_inject_last = core_str
                agent._place_inject_counter = count + 1
        except Exception:
            pass

    # 3c. FAMILIARITY (recognition of known concepts — occasional, max 1 line)
    # or a drawing fact when the thought is about drawing, or a past
    # reflection surfacing by relevance — at most ONE memory surface per caption
    if not detox and not live:
        fam_line = get_familiarity_line(agent)
        if fam_line:
            prompt_parts.append(fam_line)
        else:
            d_line = get_drawing_echo_line(agent)
            if d_line:
                prompt_parts.append(d_line)
            else:
                echo_line = get_reflection_echo_line(agent)
                if echo_line:
                    prompt_parts.append(echo_line)

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
    if not detox:
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

    # 5b. DESIRE (gated — only first 3 captions after a desire changes, never
    # during live moments). Unconditional injection caused the May 2026
    # yearning echo loop: monologue yearning → compressed into desire →
    # re-injected → more yearning.
    if not detox and not live:
        try:
            from captioner.context_compression import context_compressor

            desire = context_compressor.get_current_desire()
            inj_count = context_compressor.introspective_state.get("desire_injection_count", 0)
            if desire and len(desire) > 5 and inj_count < 3:
                prompt_parts.append(f"Preoccupied with: {desire}")
                context_compressor.introspective_state["desire_injection_count"] = inj_count + 1
            elif not desire:
                # Desire arc: the emptied slot right after an executed drawing
                # is a real state — surface it briefly (same 3-caption cap).
                spent = context_compressor.introspective_state.get("last_spent_desire") or {}
                if spent.get("desire") and time.time() - spent.get("spent", 0) < 7200 and inj_count < 3:
                    prompt_parts.append(f"You wanted: {spent['desire'].rstrip('.')} — you drew it.")
                    context_compressor.introspective_state["desire_injection_count"] = inj_count + 1
        except Exception:
            pass

    # 5c. BASELINE CONTEXT — RETIRED June 28 (Step 3). The rolling sensory-prose
    # baseline ("dust motes in a single beam of light…") was redundant with the
    # concept-derived place inventory (get_core_facts_string) + the familiarity
    # line, and it was a register-contamination vector. The compression worker
    # still produces baseline for its own stagnation check; it just no longer
    # reaches the caption prompt.

    # DWELL retired June 28: "Stay with that last thought — take it one step
    # further" was a per-call extend command that, once the stream made the prior
    # turn visible, forced restate-and-append loops (the "ghost-weight" x4). The
    # stream now carries continuity, and the genre frame (_MONOLOGUE_CLAUSE)
    # frames the voice as ongoing — so continuity emerges instead of being
    # commanded. No replacement instruction.

    # Assemble: classic shapes lead with the world's turn; world shape ends
    # with it, so the next tokens answer the present, not the memory lines.
    prompt_parts = (prompt_parts + turn_parts) if world_shape else (turn_parts + prompt_parts)

    # Structural guard: never inject the same line twice (a duplicated context
    # line reads as emphasis to the model and locks the register)
    seen_parts = set()
    deduped = []
    for part in prompt_parts:
        key = part.strip()
        if key and key not in seen_parts:
            seen_parts.add(key)
            deduped.append(part)
    prompt_parts = deduped

    final_prompt = "\n".join(prompt_parts)

    # Nothing changed → a bare continuation tick ("...", matching the stream's
    # inter-turn ticks) so the model carries its thought on instead of being
    # handed an empty turn it fills with a fresh scene description.
    if not final_prompt.strip():
        final_prompt = "..."

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
