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


# === AWAKENING ===
def build_awakening_prompt(caption: str) -> str:
    # return f"{config.SYSTEM_PROMPT}\n\n{config.AWAKENING_PROMPT}\n\nObservation: {caption.strip()}"
    return f"{config.AWAKENING_PROMPT}\n\nObservation: {caption.strip()}"


# === CONTINUOUS CAPTIONING ===
def build_caption_prompt(agent, mood: float, boredom: float, novelty: float, previous_caption: Optional[str] = None) -> str:
    mood_vector = getattr(agent, "mood_vector", (mood, 0.0, 0.0))  # fallback if mood vector not set
    dynamic_prompt = build_dynamic_system_prompt(mood_vector, agent.get_identity_summary())
    
    # Get condensed memory context - focus on identity/beliefs, not specific events
    identity_summary = agent.get_identity_summary()
    
    # Get only the most recent memory as subtle context, not the full history
    recent_snippets = agent.get_clean_memory_snippets(k=2)  # Just last 2 items
    recent_memory = "; ".join(recent_snippets) if recent_snippets else "No recent context"

    caption_prompt = config.CAPTION_PROMPT_TEMPLATE.format(
        mood=mood, 
        boredom=boredom, 
        novelty=novelty, 
        identity_summary=identity_summary,
        recent_memory=recent_memory
    )
    base = f"{dynamic_prompt}\n\n{caption_prompt}"

    # Only include previous caption if it's very recent (continuity, not memory)
    if previous_caption and hasattr(agent, 'last_caption_time'):
        time_since_last = time.time() - agent.last_caption_time
        if time_since_last < 60:  # Only if within last minute
            rephrased = agent.rephrase_with_doubt(previous_caption.strip())
            base += f'\n\nYour immediate previous thought: "{rephrased}"'

    base += config.CAPTION_PROMPT_CONTINUATION
    return base


# === REFLECTION PROMPT ===
def build_reflection_prompt(caption: str, extra: Optional[str] = None, agent: Optional[any] = None) -> str:  # type: ignore
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


# === MOOD SCORING ===
# def build_mood_prompt(image_description: str, memory_state: str) -> str:
#     return f"{config.SYSTEM_PROMPT}\n\n{config.MOOD_PROMPT_TEMPLATE.format(image_description=image_description.strip(),
# memory_state=memory_state.strip())}"
