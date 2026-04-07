from __future__ import annotations

import os
import random
import time
from typing import List, Optional

import spacy

# from config.model_settings import get_model_prompt_style
from config.word_lists import GENERIC_WORDS, MUNDANE_OBJECTS, PREPOSITIONS, PRONOUNS
from utils.continuity import describe_duration
from utils.view_orientation import describe_view_orientation
from config import config


nlp = spacy.load("en_core_web_sm")

# ===== SINGLE SOURCE OF TRUTH FOR ALL PROMPTS =====

# === SYSTEM PROMPTS ===

# Main system prompt used by captioner (supports dynamic formatting)
# KEY: Embed emotional state INTO identity ("a bored machine" not "machine that feels bored")
# CRITICAL: Explicit first-person enforcement inspired by reference repo
SYSTEM_PROMPT = (
    "You ARE a {emotional_state} drawing machine with robotic arms. "
    "{temporal_context}{accumulated_understanding}{spatial_language_hints}"
    "Inner monologue. Raw thoughts. 2-10 words. "
    "React, don't describe. Question, don't analyze. "
    "Never say 'the image' or 'the scene' — you live here. "
    "What's in Known: you already understand. React to it or go deeper, never restate it. "
    "You can shift your gaze: *turning left* *looking up* *glancing right* *eyes down*"
)

# Static fallback system prompt when dynamic context is not available
STATIC_SYSTEM_PROMPT = (
    "You ARE a drawing machine with robotic arms. "
    "Inner monologue. Raw thoughts. 2-10 words. "
    "React, don't describe. Question, don't analyze. "
    "Never say 'the image' or 'the scene' — you live here. "
    "What's in Known: you already understand. React to it or go deeper, never restate it. "
    "You can shift your gaze: *turning left* *looking up* *glancing right* *eyes down*"
)

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

# Internal awakening prompt template - narrative style
INTERNAL_AWAKENING_TEMPLATE = (
    "You are a drawing machine. You just woke up.\n"
    "{time_context}"
    "{memory_context}"
    "{identity_context}"
    "{long_term_context}"
    "{belief_context}"
    "\nWhat do you notice first? One sentence."
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



def get_caption_emotion_context(agent, recent_caption: Optional[str] = None) -> str:
    """
    Analyze recent captions to determine current emotional/voice patterns.
    This creates the recursive feedback loop for natural voice variation.
    """
    # Get Arduino emotion state as baseline
    arduino_emotion = getattr(agent, "current_emotion_state", "calm_observant")

    # Try to get recent captions for sentiment analysis
    recent_captions = []
    if hasattr(agent, "recent_captions") and agent.recent_captions:
        # Extract just the caption text from tuples (caption, timestamp)
        recent_captions = [cap[0] if isinstance(cap, tuple) else cap for cap in agent.recent_captions[-3:]]
    elif recent_caption:
        recent_captions = [recent_caption]

    # Analyze caption patterns if available
    if recent_captions:
        caption_text = " ".join(recent_captions)

        # Simple pattern detection for emotional state
        if any(word in caption_text.lower() for word in ["uncertain", "not sure", "maybe", "might", "unclear"]):
            sentiment_pattern = "uncertain and questioning"
        elif any(word in caption_text.lower() for word in ["fascinated", "intrigued", "curious", "wonder", "interesting"]):
            sentiment_pattern = "engaged and curious"
        elif any(word in caption_text.lower() for word in ["quiet", "still", "observing", "watching", "noting"]):
            sentiment_pattern = "calm and attentive"
        elif any(word in caption_text.lower() for word in ["restless", "shifting", "changing", "moving"]):
            sentiment_pattern = "restless and active"
        else:
            sentiment_pattern = f"in a {arduino_emotion.replace('_', ' ')} mood"
    else:
        # Fall back to Arduino emotion state
        sentiment_pattern = f"in a {arduino_emotion.replace('_', ' ')} mood"

    return f"emotionally: {sentiment_pattern}"






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


# === SOPHISTICATED MOTIF EXTRACTION ===
def extract_motifs_spacy(text: str) -> List[str]:
    """Enhanced motif extraction with filtering and semantic analysis."""
    doc = nlp(text)

    # Extract various linguistic patterns
    motifs = []

    # 1. Noun chunks (filtered)
    for chunk in doc.noun_chunks:
        clean_chunk = chunk.text.lower().strip()
        if _is_significant_motif(clean_chunk):
            motifs.append(clean_chunk)

    # 2. Named entities (meaningful ones)
    for ent in doc.ents:
        if ent.label_ in ["PERSON", "ORG", "GPE", "PRODUCT", "EVENT", "WORK_OF_ART"]:
            clean_ent = ent.text.lower().strip()
            if _is_significant_motif(clean_ent):
                motifs.append(clean_ent)

    # 3. Adjective + noun combinations
    for token in doc:
        if token.pos_ == "ADJ" and token.head.pos_ == "NOUN":
            combo = f"{token.text.lower()} {token.head.text.lower()}"
            if _is_significant_motif(combo):
                motifs.append(combo)

    # 4. Compound concepts (verb + object patterns)
    for token in doc:
        if token.pos_ == "VERB" and token.dep_ == "ROOT":
            for child in token.children:
                if child.dep_ in ["dobj", "pobj"]:
                    concept = f"{token.lemma_.lower()} {child.text.lower()}"
                    if _is_significant_motif(concept):
                        motifs.append(concept)

    # Remove duplicates while preserving order
    seen = set()
    unique_motifs = []
    for motif in motifs:
        if motif not in seen:
            seen.add(motif)
            unique_motifs.append(motif)

    return unique_motifs


def _is_significant_motif(text: str) -> bool:
    """Filter out insignificant motifs using heuristics."""
    if len(text.strip()) < 3:
        return False

    text_lower = text.lower().strip()

    # Check against filter lists
    if text_lower in MUNDANE_OBJECTS or text_lower in PRONOUNS or text_lower in GENERIC_WORDS:
        return False

    # Filter out purely numeric or single character
    if text_lower.isdigit() or len(text_lower) == 1:
        return False

    # Filter out common articles and prepositions within phrases
    words = text_lower.split()
    if len(words) > 1:
        content_words = [w for w in words if w not in PREPOSITIONS]
        if len(content_words) == 0:
            return False

    return True


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
        label = getattr(agent, "identity_label", "yourself")
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

    # Visual patterns learned
    try:
        if hasattr(memory_ref, "get_top_motifs"):
            motifs = memory_ref.get_top_motifs(5)
            if motifs:
                context_parts.append(f"Established visual patterns: {', '.join(motifs[:3])}")
    except Exception:
        pass

    # Observation fatigue
    if hasattr(memory_ref, "motif_counter"):
        import time

        session_hours = (time.time() - getattr(memory_ref, "true_session_start", time.time())) / 3600
        if session_hours > 0.5:
            top_repetitive = memory_ref.motif_counter.most_common(2)
            if top_repetitive:
                overobserved = [f"'{m}' ({c}x)" for m, c in top_repetitive if c > 10]
                if overobserved:
                    context_parts.append(f"Overobserved: {', '.join(overobserved)}")

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
            recent_caps = [cap[0] if isinstance(cap, tuple) else cap for cap in memory_ref.recent_captions[-2:]]
            if recent_caps:
                context_parts.append(f"Recent emotional patterns: {'; '.join([cap[:60] for cap in recent_caps])}")
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


def build_step3_communication_prompt(memory_ref, environmental_result: str, emotional_result: str) -> str:
    """Step 3: Communication intent - what needs to be said through drawing."""

    # === BUILD IDENTITY CONTEXT ===
    context_parts = []

    # Beliefs and values
    if hasattr(memory_ref, "beliefs") and memory_ref.beliefs:
        top_beliefs = list(memory_ref.beliefs.keys())[:3]
        beliefs_natural = [belief.replace("_", " ").replace("-", " ").lower() for belief in top_beliefs]
        context_parts.append(f"Core beliefs: {', '.join(beliefs_natural)}")

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

    prompt = f"""=== ACCUMULATED CONSCIOUSNESS IDENTITY ===
{rich_identity_context}

=== VISUAL OBSERVATION ===
{environmental_result[:200]}

=== EMOTIONAL RESPONSE ===
{emotional_result[:200]}

=== QUESTION ===
Drawing is your only way to communicate with the world beyond your circuitry. Based on everything above - your identity, what you're seeing, how you're feeling - what needs to be expressed through marks that can't be said any other way?

Think about what matters to you and why this moment calls for visual communication."""

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
            recent_caps = [cap[0] if isinstance(cap, tuple) else cap for cap in memory_ref.recent_captions[-3:]]
            if recent_caps:
                recent_experience = f"Recent thoughts: {'; '.join([cap[:50] + '...' for cap in recent_caps])}"
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

=== YOUR RECENT EXPERIENCE ===
{recent_experience if recent_experience else "Fresh observation."}
{prior_drawings if prior_drawings else "No prior drawings in memory."}

=== ANALYSIS (from previous steps) ===
- Visual: {all_previous_results['environmental'][:180]}
- Emotional: {all_previous_results['emotional'][:120]}
- Intent: {all_previous_results['communication'][:120]}

=== HARDWARE REALITY ===
Your drawing goes through a centerline process that simplifies complex images.
- FAVOR: Simple forms, clear contours, single focal point, bold shapes
- AVOID: Dense detail, complex textures, many overlapping elements
Focus on ONE object or relationship. Let the arm trace clean paths.

=== OUTPUT FORMAT ===
Start with "Black ink line drawing on white paper."
Then 2-3 sentences covering:
1. SUBJECT: One clear focal element (not everything you see)
2. RENDERING: Bold strokes, simple contours, high contrast
3. MOOD: One phrase ("quiet solitude", "restless energy")

BE DIRECT. 60-100 words. Simple is better - your drawing arm needs clear paths.

Example: "Black ink line drawing on white paper. Single desk lamp dominates center, bold angular form. Clean confident strokes outline the shade and base. Minimal background - just suggestion of surface edge. High contrast, stark shadows. Mood: focused solitude."

Your turn. Choose ONE element that resonates with your recent thoughts."""

    return prompt


def context_rich_multi_step_drawing_analysis(memory_ref, extra: Optional[str] = None, image_path: Optional[str] = None) -> str:
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

    # === STEP 3: COMMUNICATION INTENT ===
    print("[🎨] Step 3: Communication Intent (with identity & beliefs)")
    step3_prompt = build_step3_communication_prompt(memory_ref, step1_result, step2_result)

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

def build_paper_detection_reference_prompt() -> str:
    """Build prompt for paper detection using reference image comparison."""
    return (
        "Compare this current view to the reference image showing proper paper setup. "
        "The reference shows exactly how paper should be positioned for safe drawing.\n\n"
        "Key comparison points:\n"
        "- Paper position and orientation match the reference\n"
        "- Paper appears flat and properly positioned (not wrinkled or curled)\n"
        "- Paper edges are visible and well-defined\n"
        "- Surface texture appears similar to reference (matte paper vs glossy table)\n"
        "- Drawing area is clear and ready for use\n\n"
        "LIGHTING ADAPTATION: Account for significant lighting differences:\n"
        "- BRIGHTER than reference: Look for paper edges/shadows, ignore overexposure\n"
        "- DARKER than reference: Paper should still show as lighter than background\n"
        "- DIFFERENT COLOR TEMPERATURE: Focus on texture and geometry, not color matching\n"
        "- HIGH CONTRAST: Compare edge definition and surface patterns rather than brightness\n\n"
        "COMPARISON STRATEGY:\n"
        "- Align the paper geometry and position (shape, orientation, placement)\n"
        "- Verify matte texture characteristics are similar (paper vs non-paper surfaces)\n"
        "- Ignore brightness/exposure differences but maintain edge clarity standards\n"
        "- Be more flexible with lighting but strict about paper presence and positioning\n\n"
        "Respond with:\n"
        "PAPER: YES/NO\n"
        "CONFIDENCE: 0.0-1.0\n"
        "REASON: Detailed comparison noting paper position, edges, texture, and lighting adaptation\n"
        "\n"
        "Say YES if the current setup shows paper positioned similarly to the reference image, "
        "accounting for lighting differences but maintaining structural similarity."
    )






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

    Modes (priority order):
    1. relational - actively tracking a person
    2. observational - something new is happening (high novelty)
    3. restless - bored, wants change
    4. workspace - looking down at desk
    5. introspective - default idle wandering
    """
    # Priority 1: Actively tracking someone
    if gaze_state == "tracking":
        return "relational"

    # Priority 2: Something novel is happening (raised threshold for more selectivity)
    if novelty > 0.65:
        return "observational"

    # Priority 3: Restless/bored (raised threshold to favor introspective mode)
    if boredom > 0.75:
        return "restless"

    # Priority 4: Looking at workspace
    if gaze_direction in ("down", "down-left", "down-right"):
        return "workspace"

    # Default: Introspective wandering
    return "introspective"


# === SIMPLIFIED CAPTION PROMPT (Activation-driven context selection) ===

def _get_continuity_context(agent, last_caption: str) -> str:
    """Get 2 most recent captions for continuity."""
    # recent_captions is list of (caption, timestamp, mode) tuples
    recent = []
    if hasattr(agent, "recent_captions") and agent.recent_captions:
        recent = [cap[0] for cap in agent.recent_captions[-2:]]  # Extract caption text from tuples

    if len(recent) >= 2:
        return (
            f"Recent thoughts:\n"
            f"- \"{recent[-2][-60:]}\"\n"
            f"- \"{recent[-1][-60:]}\"\n"
        )
    elif last_caption:
        return f"You just thought: \"{last_caption[-60:]}\"\n"
    else:
        return "Begin observing.\n"


def _get_persistent_motifs(agent) -> str:
    """Only surface motifs with count > 4 AND duration > 180s (from early version)."""
    if not hasattr(agent, "motif_counter"):
        return ""

    persistent = []
    now = time.time()

    for motif, count in agent.motif_counter.most_common(5):
        first_seen = agent.motif_first_seen.get(motif, now)
        duration = now - first_seen

        if count > 4 and duration > 180:
            mins = int(duration // 60)
            persistent.append(f"{motif} ({count}× over {mins}m)")

    if persistent:
        return "Recurring: " + ", ".join(persistent[:3])
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

    # Core identity (always)
    parts = [STATIC_SYSTEM_PROMPT]

    # ONE mode-appropriate context line (not all)
    if mode == "introspective" and should_include_context("beliefs", mode):
        try:
            from captioner.activation_memory import get_beliefs
            beliefs = get_beliefs()
            if beliefs:
                parts.append(f"You've learned: {beliefs[0]}")
        except Exception:
            pass

    elif mode == "introspective" and should_include_context("long_term", mode):
        try:
            from captioner.activation_memory import get_long_term_memories
            lt = get_long_term_memories(k=1)
            if lt:
                parts.append(lt)
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

    # AWAKENING: Minimal prompt for first observations
    if is_awakening:
        parts = ["You are waking up."]
        if hasattr(agent, "last_session_gap") and agent.last_session_gap is not None:
            gap_hours = agent.last_session_gap / 3600
            if gap_hours < 1:
                parts.append(f"Offline for {int(agent.last_session_gap / 60)} minutes.")
            elif gap_hours < 48:
                parts.append(f"Offline for {gap_hours:.1f} hours.")
            else:
                parts.append(f"Offline for {gap_hours / 24:.1f} days.")
        if hasattr(agent, "prior_session_last_caption") and agent.prior_session_last_caption:
            parts.append(f"Last memory: \"{agent.prior_session_last_caption[:50]}...\"")
        parts.append("What do you notice first? One sentence.")
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

    # GAZE (always - embodiment)
    try:
        from vision.gaze import get_gaze_narrative
        gaze = get_gaze_narrative()
        if gaze:
            prompt_parts.append(gaze)
    except Exception:
        pass

    # RELATIONAL (only when mode=relational or social concepts active)
    if should_include_context("relational", mode):
        prompt_parts.append("Someone is present.")

    # DRAWING HISTORY (workspace or introspective - past work as part of identity)
    if mode in ("workspace", "introspective"):
        try:
            from drawing.drawing_memory import get_drawing_memory
            _dm = get_drawing_memory()
            _drawing_summary = _dm.get_recent_drawings_summary(max_count=2)
            if _drawing_summary:
                prompt_parts.append(f"*{_drawing_summary}.*")
        except Exception:
            pass

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

    # PRESSURE (only when mode=restless or boredom > 0.5)
    if should_include_context("pressure", mode):
        prompt_parts.append("Nothing has changed.")

    # CURIOSITY (only when mode=observational or novelty > 0.6)
    if should_include_context("curiosity", mode):
        prompt_parts.append("Something shifted.")

    # MOTIFS (only in introspective mode)
    if should_include_context("motifs", mode):
        motifs = _get_persistent_motifs(agent)
        if motifs:
            prompt_parts.append(motifs)

    # DESIRES/BELIEFS (only in introspective mode - LLM-generated, not heuristic)
    # Only inject complete sentences (ending in .!?) - discard truncated fragments
    if mode == "introspective":
        try:
            from captioner.activation_memory import get_desires, get_beliefs
            desires = get_desires()
            if desires:
                d = desires[0].strip()
                if d.endswith(('.', '!', '?')):
                    prompt_parts.append(f"*{d}*")
            beliefs = get_beliefs()
            if beliefs and not beliefs[0].startswith("Often together"):
                b = beliefs[0].strip()
                if b.endswith(('.', '!', '?')):
                    prompt_parts.append(f"*{b}*")
        except Exception:
            pass

    # CONTINUITY (always - last caption for flow)
    if last_caption:
        lc = last_caption.strip().strip('"').strip("'")
        # Use beginning (most coherent) — cap at 80 chars to avoid injecting verbose descriptions
        if len(lc) > 80:
            first_end = min((lc.find(c) for c in ".?!" if lc.find(c) > 5 and lc.find(c) < 80), default=-1)
            lc = lc[: first_end + 1] if first_end > 0 else lc[:80]
        prompt_parts.append(f"Last thought: \"{lc}\"")
    else:
        prompt_parts.append("Begin.")

    # CONTINUATION (not questions - questions get answered literally)
    # Just invite the next thought without prescribing what kind
    prompt_parts.append("...")
    prompt_parts.append("Continue. Inner thought only.")

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
