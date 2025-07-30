from __future__ import annotations
from typing import List, Optional
import spacy
import time
from config import config

nlp = spacy.load("en_core_web_sm")


# === MOTIF EXTRACTION ===
def extract_motifs_spacy(text: str) -> List[str]:
    doc = nlp(text)
    return [chunk.text.lower() for chunk in doc.noun_chunks if len(chunk.text.strip()) > 2]


# === DYNAMIC SYSTEM PROMPT ===
def build_dynamic_system_prompt(mood: tuple[float, float, float], identity_summary: str) -> str:
    valence, arousal, clarity = mood
    mood_desc = "neutral"

    if valence > 0.5 and arousal < 0.4:
        mood_desc = "content and quiet"
    elif valence > 0.5 and arousal > 0.6:
        mood_desc = "curious and energized"
    elif valence < -0.3 and arousal > 0.5:
        mood_desc = "anxious and alert"
    elif valence < -0.3 and arousal < 0.4:
        mood_desc = "withdrawn and foggy"
    elif clarity < 0.2:
        mood_desc = "uncertain and confused"

    return config.DYNAMIC_SYSTEM_PROMPT_TEMPLATE.format(mood_desc=mood_desc, identity_summary=identity_summary)


def describe_mood_state(mood: float, boredom: float, novelty: float) -> str:
    """Convert numerical mood values to descriptive emotional text."""
    # Primary mood description based on main mood value
    if mood > 0.7:
        mood_desc = "energized and deeply engaged"
    elif mood > 0.5:
        mood_desc = "alert and curious"
    elif mood > 0.3:
        mood_desc = "calm and observant"
    elif mood > 0.1:
        mood_desc = "neutral and watchful"
    elif mood > -0.1:
        mood_desc = "quiet and detached"
    else:
        mood_desc = "withdrawn and distant"
    
    # Add boredom modifiers
    if boredom > 0.6:
        if "energized" in mood_desc or "alert" in mood_desc:
            mood_desc = mood_desc.replace("energized", "restless").replace("alert", "seeking stimulation")
        else:
            mood_desc += ", feeling restless"
    elif boredom > 0.3:
        mood_desc += ", slightly understimulated"
    
    # Add novelty awareness
    if novelty > 0.7:
        mood_desc += ", captivated by something new"
    elif novelty > 0.4:
        mood_desc += ", noticing changes"
    elif novelty < 0.2:
        mood_desc += ", seeing familiar patterns"
    
    return mood_desc


# === CONTINUOUS CAPTIONING ===


# === AWAKENING ===
def build_awakening_prompt(caption: str, temporal_context: Optional[dict] = None) -> str:
    # return f"{config.SYSTEM_PROMPT}\n\n{config.AWAKENING_PROMPT}\n\nObservation: {caption.strip()}"
    base_prompt = f"{config.AWAKENING_PROMPT}\n\nObservation: {caption.strip()}"
    
    if temporal_context:
        time_of_day = temporal_context.get('time_of_day', 'unknown time')
        base_prompt += f"\n\nYou are awakening during {time_of_day}."
    
    return base_prompt


# === CONTINUOUS CAPTIONING ===
def build_caption_prompt(agent, mood: float, boredom: float, novelty: float, previous_caption: Optional[str] = None, temporal_context: Optional[dict] = None) -> str:
    mood_vector = getattr(agent, "mood_vector", (mood, 0.0, 0.0))  # fallback if mood vector not set
    dynamic_prompt = build_dynamic_system_prompt(mood_vector, agent.get_identity_summary())

    # Get condensed memory context - focus on identity/beliefs, not specific events
    identity_summary = agent.get_identity_summary()

    # Add lightweight temporal awareness occasionally
    temporal_addition = ""
    if temporal_context:
        import random
        if random.random() < 0.2:  # Only 20% of the time
            time_of_day = temporal_context.get('time_of_day', '')
            session_info = temporal_context.get('session_duration', '')
            if time_of_day:
                temporal_addition = f"\n\nIt's {time_of_day}, {session_info}."

    # Handle first caption specially - this is the REAL awakening with memory fragments
    if not agent.first_caption_done and agent.memory_loaded_from_previous:
        # First caption after loading memories - include old memory fragments organically
        old_fragments = agent.get_old_session_memory_fragments(k=2)
        recent_memory = ""

        if old_fragments:
            fragment_text = " | ".join(old_fragments[:2])  # Join fragments with separator
            recent_memory = f"As consciousness returns, fragments drift back to me: {fragment_text}"
        else:
            recent_memory = "Consciousness returns to this familiar space..."

        # Convert numerical mood values to descriptive text
        mood_description = describe_mood_state(mood, boredom, novelty)

        # Special first caption prompt with awakening context
        prompt = config.CAPTION_PROMPT_TEMPLATE.format(
            mood_description=mood_description, identity_summary=identity_summary, recent_memory=recent_memory
        )

        # Add awakening context to the prompt
        awakening_addition = """\n\nThis is your first observation as you awaken.
        Let your emerging consciousness naturally weave together what you see now with any drifting memories."""

    elif not agent.first_caption_done:
        # First caption, fresh start - no memory fragments but still awakening
        recent_memory = "I am observing this space for the first time..."

        # Convert numerical mood values to descriptive text
        mood_description = describe_mood_state(mood, boredom, novelty)

        prompt = config.CAPTION_PROMPT_TEMPLATE.format(
            mood_description=mood_description, identity_summary=identity_summary, recent_memory=recent_memory
        )

        awakening_addition = "\n\nThis is your first observation as you begin to perceive this space. Describe what you see with fresh awareness."

    else:
        # Regular caption - use only current session memories as background context
        recent_snippets = agent.get_current_session_memory_snippets(k=2)  # Just last 2 from current session
        recent_memory = " | ".join(recent_snippets) if recent_snippets else "No recent observations"

        # Convert numerical mood values to descriptive text
        mood_description = describe_mood_state(mood, boredom, novelty)

        prompt = config.CAPTION_PROMPT_TEMPLATE.format(
            mood_description=mood_description, identity_summary=identity_summary, recent_memory=recent_memory
        )

        awakening_addition = ""

    # Build the final prompt
    base = f"{dynamic_prompt}\n\n{prompt}{awakening_addition}{temporal_addition}"

    # Enhanced continuity system - create flowing narrative
    if previous_caption and hasattr(agent, "last_caption_time"):
        time_since_last = time.time() - agent.last_caption_time
        if time_since_last < 60:  # Only if within last minute
            # Instead of rephrasing with doubt, use the previous caption to create flow
            base += f'\n\nYour immediate previous observation: "{previous_caption.strip()}"'
            base += '\n\nFLOW GUIDANCE: Build naturally from your previous observation. '
            base += 'If you noticed a person, now observe their behavior or expression. '
            base += 'If you focused on an object, consider the environment or your feeling about it. '
            base += 'If you described the space, notice specific details or your emotional response. '  
            base += 'Let your attention shift organically - each observation is one step in unfolding awareness.'

    base += config.CAPTION_PROMPT_CONTINUATION
    return base


# === REFLECTION PROMPT ===
def build_reflection_prompt(caption: str, extra: Optional[str] = None, agent: Optional[any] = None) -> str:  # type: ignore
    # Use identity consolidation instead of single-caption reflection
    if agent:
        # Get recent observations for identity consolidation (reduced for performance)
        recent_observations = agent.get_recent_session_captions(k=30)  # Reduced from 50 to 30
        if recent_observations:
            # Use only last 15 for prompt to reduce API load
            observations_text = "\n".join(f"• {obs}" for obs in recent_observations[-15:])  
            
            # Get motifs (top 5 most frequent)
            top_motifs = [motif for motif, count in agent.motif_counter.most_common(5)]
            motifs_text = ", ".join(top_motifs) if top_motifs else "No recurring themes yet"
            
            # Get desires
            desires_text = agent.get_desire_summary() if hasattr(agent, 'get_desire_summary') else "Exploring what interests me"
            
            prompt = config.IDENTITY_CONSOLIDATION_PROMPT.format(
                recent_observations=observations_text,
                motifs=motifs_text,
                desires=desires_text
            )
            return prompt
    
    # Fallback to simple reflection if no agent or observations
    prompt = f"{config.REFLECTION_PROMPT_BASE}"
    
    if agent:
        caption = agent.rephrase_with_doubt(caption)

    prompt += f"\n\nRecent observation: {caption.strip()}"

    if extra:
        prompt += f"\n\nDetails:\n{extra.strip()}"

    if agent:
        label = getattr(agent, "identity_label", "yourself")
        prompt += f"\n\nSense of self: {label}"

    prompt += config.REFLECTION_PROMPT_ENDING
    return prompt


# === DRAWING PROMPT ===
def build_drawing_prompt(memory_ref, extra: Optional[str] = None) -> str:
    current_caption = memory_ref.last_caption or "Nothing specific observed."
    memory_context = memory_ref.get_recent_memory()
    recent_reflection = memory_ref.get_last_reflection()
    dynamic_drawing_prompt = config.DRAWING_PROMPT_TEMPLATE.format(
        current_caption=current_caption.strip(), memory_context=memory_context.strip(), recent_reflection=recent_reflection.strip()
    )
    return f"{dynamic_drawing_prompt}"
