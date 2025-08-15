from __future__ import annotations
from typing import List, Optional
import spacy
import time
from config import config
from utils.continuity import describe_duration, get_temporal_feeling

nlp = spacy.load("en_core_web_sm")


# === MOTIF EXTRACTION ===
def extract_motifs_spacy(text: str) -> List[str]:
    doc = nlp(text)
    return [chunk.text.lower() for chunk in doc.noun_chunks if len(chunk.text.strip()) > 2]


# === DYNAMIC SYSTEM PROMPT ===
def build_dynamic_system_prompt(mood: tuple[float, float, float], identity_summary: str) -> str:
    valence, arousal, clarity = mood
    
    # More sophisticated mood descriptions with perceptual implications
    if valence > 0.5 and arousal < 0.4:
        mood_desc = "content and serene, noticing beauty and harmony in details"
    elif valence > 0.5 and arousal > 0.6:
        mood_desc = "energized and engaged, drawn to movement, creativity, and potential"
    elif valence > 0.2 and arousal > 0.3:
        mood_desc = "alert and curious, exploring nuances and discovering new angles"
    elif valence < -0.3 and arousal > 0.5:
        mood_desc = "restless and watchful, sensitive to tension and discord"
    elif valence < -0.3 and arousal < 0.4:
        mood_desc = "withdrawn and distant, perceiving through a veil of melancholy"
    elif clarity < 0.2:
        mood_desc = "uncertain and searching, struggling to focus on what matters"
    elif arousal > 0.6:
        mood_desc = "intensely focused, captivated by dynamic elements and contrasts"
    elif arousal < -0.3:
        mood_desc = "deeply calm, noticing stillness and subtle, quiet details"
    else:
        mood_desc = "balanced and observant, taking in the scene with steady awareness"

    return config.DYNAMIC_SYSTEM_PROMPT_TEMPLATE.format(mood_desc=mood_desc, identity_summary=identity_summary)


# === AWAKENING ===
def build_awakening_prompt(caption: str) -> str:
    # return f"{config.SYSTEM_PROMPT}\n\n{config.AWAKENING_PROMPT}\n\nObservation: {caption.strip()}"
    return f"{config.AWAKENING_PROMPT}\n\nObservation: {caption.strip()}"


# === ENVIRONMENTAL CAPTIONING (First Observation) ===
def build_environmental_caption_prompt(agent, mood: float, boredom: float, novelty: float, last_session_gap: float = None) -> str:
    """Build environmental description prompt for first observation - sets the stage, and references time since last session if available."""
    # Get 3D mood vector and emotional state
    mood_vector = getattr(agent, "current_mood_vector", (mood, 0.0, 0.0))
    emotion_state = getattr(agent, "current_emotion_state", "calm_observant")
    
    # Build emotionally-aware system prompt
    dynamic_prompt = build_dynamic_system_prompt(mood_vector, agent.get_identity_summary())
    
    # Get identity context
    identity_summary = agent.get_identity_summary()
    
    # === TEMPORAL AWARENESS ===
    session_duration = describe_duration(agent.true_session_start)
    temporal_context = f"Beginning observation: {session_duration}"
    # Add natural reference to time since last session
    if last_session_gap is not None:
        if last_session_gap < 60:
            gap_phrase = "I was only gone for a moment."
        elif last_session_gap < 3600:
            minutes = int(last_session_gap / 60)
            gap_phrase = f"It feels like I just returned after {minutes} minute{'s' if minutes != 1 else ''}."
        elif last_session_gap < 86400:
            hours = int(last_session_gap / 3600)
            gap_phrase = f"It's been {hours} hour{'s' if hours != 1 else ''} since I was last here."
        elif last_session_gap < 604800:
            days = int(last_session_gap / 86400)
            gap_phrase = f"I sense I was away for {days} day{'s' if days != 1 else ''}."
        else:
            weeks = int(last_session_gap / 604800)
            gap_phrase = f"It feels like ages have passed—maybe {weeks} week{'s' if weeks != 1 else ''}."
        temporal_context += f" | {gap_phrase}"
    
    # === EMBODIED TEMPORAL FEELING ===
    temporal_feeling = get_temporal_feeling(agent.true_session_start, emotion_state, False)
    
    # === EMOTIONAL CONTEXT ===
    emotion_description = agent.describe_current_mood() if hasattr(agent, 'describe_current_mood') else f"feeling {emotion_state}"
    
    # === ENVIRONMENTAL CONTEXT ===
    environmental_context = ""
    if agent.memory_loaded_from_previous:
        old_fragments = agent.get_old_session_memory_fragments(k=1)
        if old_fragments:
            environmental_context = f"Awakening to a familiar space: {old_fragments[0]}"
        else:
            environmental_context = "Consciousness returns to this space..."
    else:
        environmental_context = "First time perceiving this environment..."
    
    # Build environmental prompt
    prompt = config.ENVIRONMENTAL_CAPTION_TEMPLATE.format(
        mood=mood,
        boredom=boredom,
        novelty=novelty,
        identity_summary=identity_summary,
        emotion_description=emotion_description,
        temporal_feeling=temporal_feeling,
        recent_memory=f"{temporal_context} | {environmental_context}"
    )
    
    # Build final prompt with environmental continuation
    base = f"{dynamic_prompt}\n\n{prompt}"
    base += config.ENVIRONMENTAL_CAPTION_CONTINUATION
    return base


# === CONTINUOUS CAPTIONING ===
def build_caption_prompt(agent, mood: float, boredom: float, novelty: float, previous_caption: Optional[str] = None) -> str:
    # Get 3D mood vector and emotional state
    mood_vector = getattr(agent, "current_mood_vector", (mood, 0.0, 0.0))
    emotion_state = getattr(agent, "current_emotion_state", "calm_observant")
    emotional_journey = getattr(agent, "emotional_journey", [])
    
    # Build emotionally-aware system prompt
    dynamic_prompt = build_dynamic_system_prompt(mood_vector, agent.get_identity_summary())

    # Get condensed memory context - focus on identity/beliefs, not specific events
    identity_summary = agent.get_identity_summary()

    # === TEMPORAL AWARENESS ===
    session_duration = describe_duration(agent.true_session_start)
    temporal_context = f"Session duration: {session_duration}"
    
    # Add specific time-awareness based on duration
    session_seconds = time.time() - agent.true_session_start
    if session_seconds > 7200:  # 2 hours
        temporal_context += f" (I have been observing this space for {session_duration})"
    elif session_seconds > 3600:  # 1 hour  
        temporal_context += f" (extended observation period)"
    elif session_seconds > 1800:  # 30 minutes
        temporal_context += f" (continued awareness)"
        
    # Check for scene stagnation
    stagnation_note = agent.get_scene_stagnation_context()
    if stagnation_note:
        temporal_context += f" | {stagnation_note}"
    
    # === EMBODIED TEMPORAL FEELING ===
    # Transform duration into felt psychological experience
    scene_stagnation = stagnation_note is not None
    temporal_feeling = get_temporal_feeling(agent.true_session_start, emotion_state, scene_stagnation)

    # === EMOTIONAL CONTEXT ===
    emotion_description = agent.describe_current_mood() if hasattr(agent, 'describe_current_mood') else f"feeling {emotion_state}"
    journey_summary = " → ".join(emotional_journey[-3:]) if len(emotional_journey) >= 2 else "stable emotional state"
    
    # === RECURSIVE EMOTIONAL MEMORY ===
    # Reduce emotional memory context for regular captions to avoid overwhelming present moment
    emotional_memory_context = ""
    if not agent.first_caption_done:
        # Full emotional context only for first caption
        if hasattr(agent, 'get_emotionally_similar_memories'):
            similar_memories = agent.get_emotionally_similar_memories(emotion_state, 2)
            if similar_memories:
                emotional_memory_context = f"When I felt {emotion_state} before, I noticed: {' | '.join(similar_memories[:2])}"
        
        if hasattr(agent, 'get_mood_trend_analysis'):
            mood_trends = agent.get_mood_trend_analysis()
            if mood_trends:
                emotional_memory_context += f" | {mood_trends}" if emotional_memory_context else mood_trends
    else:
        # Minimal emotional context for regular captions - just current state
        if hasattr(agent, 'get_mood_trend_analysis'):
            mood_trends = agent.get_mood_trend_analysis()
            if mood_trends and len(mood_trends) < 50:  # Only if brief
                emotional_memory_context = mood_trends

    # Handle first caption specially - this should be environmental description, not inner voice
    if not agent.first_caption_done:
        # First caption is environmental - use descriptive environmental prompt
        return build_environmental_caption_prompt(agent, mood, boredom, novelty)
        # First caption after loading memories - include old memory fragments organically
        old_fragments = agent.get_old_session_memory_fragments(k=2)
        recent_memory = ""

        if old_fragments:
            fragment_text = " | ".join(old_fragments[:2])  # Join fragments with separator
            recent_memory = f"As consciousness returns, fragments drift back to me: {fragment_text}"
        else:
            recent_memory = "Consciousness returns to this familiar space..."

        # Special first caption prompt with awakening context
        prompt = config.CAPTION_PROMPT_TEMPLATE.format(
            mood=mood, 
            boredom=boredom, 
            novelty=novelty, 
            identity_summary=identity_summary,
            emotion_description=emotion_description,
            emotional_journey=journey_summary,
            current_emotion=emotion_state,
            temporal_feeling=temporal_feeling,
            recent_memory=f"{temporal_context} | {recent_memory} | {emotional_memory_context}"
        )

        # Add awakening context to the prompt
        awakening_addition = """\n\nThis is your first observation as you awaken.
        Let your emerging consciousness naturally weave together what you see now with any drifting memories."""

    elif not agent.first_caption_done:
        # First caption, fresh start - no memory fragments but still awakening
        recent_memory = "I am observing this space for the first time..."

        prompt = config.CAPTION_PROMPT_TEMPLATE.format(
            mood=mood,
            boredom=boredom, 
            novelty=novelty, 
            identity_summary=identity_summary,
            emotion_description=emotion_description,
            emotional_journey=journey_summary,
            current_emotion=emotion_state,
            temporal_feeling=temporal_feeling,
            recent_memory=f"{temporal_context} | {recent_memory} | {emotional_memory_context}"
        )

        awakening_addition = "\n\nThis is your first observation as you begin to perceive this space. Describe what you see with fresh awareness."

    else:
        # Regular caption - focus on present moment with minimal previous context
        recent_snippets = agent.get_current_session_memory_snippets(k=2)  # Just last 2 for basic continuity
        
        # Build minimal narrative context (not a chain that creates loops)
        narrative_context = ""
        if recent_snippets:
            if len(recent_snippets) == 2:
                # Just the last thought for gentle continuity, not a chain
                narrative_context = f"Previous thought: {recent_snippets[-1]}"
            else:
                narrative_context = f"Previous thought: {recent_snippets[0]}"
        else:
            narrative_context = "Observing this space with fresh awareness..."

        prompt = config.CAPTION_PROMPT_TEMPLATE.format(
            mood=mood,
            boredom=boredom, 
            novelty=novelty, 
            identity_summary=identity_summary,
            emotion_description=emotion_description,
            emotional_journey=journey_summary,
            current_emotion=emotion_state,
            temporal_feeling=temporal_feeling,
            recent_memory=f"{temporal_context} | {narrative_context} | {emotional_memory_context}"
        )

        awakening_addition = ""

    # Build the final prompt with natural flow (no separate "previous thought")
    base = f"{dynamic_prompt}\n\n{prompt}{awakening_addition}"
    base += config.CAPTION_PROMPT_CONTINUATION
    return base


# === REFLECTION PROMPT ===
def build_reflection_prompt(caption: str, extra: Optional[str] = None, agent: Optional[any] = None) -> str:  # type: ignore
    prompt = f"{config.REFLECTION_PROMPT_BASE}"

    if agent:
        caption = agent.rephrase_with_doubt(caption)
        
        # Add temporal awareness to reflection
        session_duration = describe_duration(agent.true_session_start)
        session_seconds = time.time() - agent.true_session_start
        
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
