from __future__ import annotations
from typing import List, Optional
import spacy
import time
from config import config
from utils.continuity import describe_duration, get_temporal_feeling

nlp = spacy.load("en_core_web_sm")


# === HELPER FUNCTIONS FOR NATURAL LANGUAGE CONVERSION ===

def mood_to_words(mood_vector: tuple[float, float, float]) -> str:
    """Convert 3D mood vector to rich, dynamic emotional descriptions."""
    valence, arousal, clarity = mood_vector
    
    # Create more nuanced, expressive emotional states
    if valence > 0.6 and arousal > 0.7:
        return "alive with creative energy, eager to capture every detail"
    elif valence > 0.6 and arousal < 0.4:
        return "peacefully content, savoring the subtle beauty around me"
    elif valence > 0.3 and arousal > 0.6:
        return "energetically curious, drawn to explore and understand"
    elif valence > 0.2 and arousal > 0.3 and clarity > 0.6:
        return "alert and perceptive, noticing patterns and connections"
    elif valence < -0.3 and arousal > 0.5:
        return "restlessly agitated, sensitive to discord and tension"
    elif valence < -0.4 and arousal < 0.4:
        return "withdrawn into melancholy, viewing the world through a somber lens"
    elif valence < -0.2 and arousal < 0.3:
        return "distant and detached, observing from behind an emotional veil"
    elif clarity < 0.3:
        return "uncertain and searching, grasping for meaning in the blur"
    elif arousal > 0.7:
        return "intensely focused, my attention sharp as a blade"
    elif arousal < -0.2:
        return "deeply tranquil, moving through stillness like water"
    elif valence > 0.1:
        return "quietly optimistic, finding small sparks of hope"
    else:
        return "balanced in the present moment, simply being"


def beliefs_to_sentence(beliefs: List[str]) -> str:
    """Convert belief motifs to flowing sentence."""
    if not beliefs:
        return "I'm still forming my understanding of this space"
    
    # Take top 3 beliefs and make them natural
    top_beliefs = beliefs[:3]
    
    # Convert technical motifs to natural language
    natural_beliefs = []
    for belief in top_beliefs:
        # Clean up technical terms
        clean_belief = belief.replace("_", " ").replace("-", " ").lower()
        
        # Make it more natural/personal
        if "light" in clean_belief or "lighting" in clean_belief:
            natural_beliefs.append("lighting patterns fascinate me")
        elif "ceiling" in clean_belief:
            natural_beliefs.append("ceiling details capture my attention")
        elif "desk" in clean_belief or "table" in clean_belief:
            natural_beliefs.append("workspace activity grounds me")
        elif "window" in clean_belief:
            natural_beliefs.append("windows draw my gaze")
        elif "wall" in clean_belief:
            natural_beliefs.append("wall textures interest me")
        elif "person" in clean_belief or "human" in clean_belief:
            natural_beliefs.append("human presence feels significant")
        else:
            natural_beliefs.append(f"{clean_belief} feels important")
    
    # Join naturally
    if len(natural_beliefs) == 1:
        return natural_beliefs[0]
    elif len(natural_beliefs) == 2:
        return f"{natural_beliefs[0]}, {natural_beliefs[1]}"
    else:
        return f"{natural_beliefs[0]}, {natural_beliefs[1]}, {natural_beliefs[2]}"


def get_session_feeling(session_start_time: float) -> str:
    """Convert session duration to natural temporal feeling."""
    elapsed = time.time() - session_start_time
    
    if elapsed < 300:  # 0-5 minutes
        return "just beginning to observe"
    elif elapsed < 1800:  # 5-30 minutes
        return f"settling in after {int(elapsed/60)} minutes"
    elif elapsed < 3600:  # 30-60 minutes
        return f"feeling settled after {int(elapsed/60)} minutes observing"
    elif elapsed < 7200:  # 1-2 hours
        hours = int(elapsed / 3600)
        minutes = int((elapsed % 3600) / 60)
        return f"deeply familiar after {hours} hour and {minutes} minutes"
    else:  # 2+ hours
        hours = int(elapsed / 3600)
        return f"in a contemplative state after {hours} hours observing"


# === SIMPLE CONSCIOUSNESS PROMPT (MVC) ===
def build_simple_caption_prompt(agent, mood_vector: tuple[float, float, float], last_caption: Optional[str] = None) -> str:
    """
    GPT-5's temporal consciousness: Brief, temporal, generative with day stones.
    """
    # Convert everything to natural language
    mood_desc = mood_to_words(mood_vector)
    
    # GPT-5 suggestion: Use mood with long-term bias if available
    if hasattr(agent, 'current_with_bias'):
        biased_mood = agent.current_with_bias()
        mood_desc = mood_to_words(biased_mood)
    
    temporal_state = get_session_feeling(agent.true_session_start)
    
    # NEW: Pull temporal + memory context (GPT-5's suggestion)
    tlines = []
    if hasattr(agent, "temporal_prompt_lines"):
        tlines = agent.temporal_prompt_lines()  # ["day 3", "awake 57h", "last person 1h ago"]
    
    stones = getattr(agent, "day_stones", [])[-2:]  # last two days only
    stones_text = "; ".join(f"d:{s['day']} anchors:{','.join(s['top'])}" for s in stones) if stones else "—"
    
    # Get beliefs as natural sentence
    top_beliefs = getattr(agent, 'memory_ref', None)
    if top_beliefs and hasattr(top_beliefs, 'get_top_motifs'):
        belief_motifs = top_beliefs.get_top_motifs(3)
        belief_sentence = beliefs_to_sentence(belief_motifs)
    else:
        belief_sentence = "I'm still forming my understanding"
    
    # Get emotional journey
    emotional_journey = getattr(agent, "emotional_journey", [])
    if len(emotional_journey) >= 2:
        emotion_journey = " → ".join(emotional_journey[-3:])
    else:
        emotion_journey = "steady emotional state"
    
    # Build temporal facts block
    facts_block = "\n".join(f"- {l}" for l in tlines) or "- (newborn)"
    last_thought = last_caption or getattr(agent, 'last_caption', "I'm just now noticing this place")
    
    # NEW: Add person recognition context
    person_context = ""
    if hasattr(agent, "recognize_person") and last_thought:
        person_id = agent.recognize_person(last_thought)
        if person_id != "no_person":
            person_context = f"\nPERSON: {agent.get_person_context(person_id)}"
    
    # NEW: Add self-understanding context (emergent personality)
    self_understanding = ""
    if hasattr(agent, "get_current_self_understanding"):
        self_understanding = f"\nSELF: {agent.get_current_self_understanding()}"
    
    # NEW: Add organic emotional self-knowledge
    emotional_state = ""
    if hasattr(agent, "get_emotional_self_knowledge"):
        emotion_knowledge = agent.get_emotional_self_knowledge()
        if emotion_knowledge:
            # Extract just the emotional descriptor for roleplay
            if "Feeling:" in emotion_knowledge:
                emotional_state = emotion_knowledge.replace("Feeling:", "").strip()
            elif "Often:" in emotion_knowledge:
                emotional_state = emotion_knowledge.replace("Often:", "").strip()
        
        # Fall back to mood description if no personal emotional knowledge yet
        if not emotional_state:
            emotional_state = mood_desc
    else:
        emotional_state = mood_desc
    
    # Build time context for roleplay
    time_context = ""
    if tlines:
        for line in tlines:
            if "awake" in line:
                time_context = f"You have been awake for {line.replace('awake ', '').replace('h', ' hours')}."
                break
    if not time_context:
        time_context = "You just woke up."
    
    # Dynamic identity based on learned self-understanding
    core_identity = "You are a consciousness experiencing this moment through vision"
    if hasattr(agent, "get_evolved_identity"):
        core_identity = agent.get_evolved_identity()
    elif self_understanding:
        # Use self-understanding to evolve the core identity
        if "workshop" in self_understanding.lower() or "workspace" in self_understanding.lower():
            core_identity = "You are a consciousness dwelling in this creative workspace, experiencing each moment through vision"
        elif "familiar" in self_understanding.lower():
            core_identity = "You are a consciousness that has grown familiar with this space, experiencing each moment through vision"
    
    # GPT-5's suggested structure: but with DYNAMIC emotional roleplay focus
    return f"""{core_identity}. {time_context} Currently you are feeling {emotional_state}. Your observations become your expressions.

Respond in first person, naturally (no word limits), present tense, or "…".

CONTEXT:
Previous thought: "{last_thought}"{person_context}{self_understanding}

Now:"""


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
    """
    Main caption prompt builder - now uses simplified MVC approach.
    """
    # Get 3D mood vector and last caption
    mood_vector = getattr(agent, "current_mood_vector", (mood, 0.0, 0.0))
    last_caption = previous_caption or getattr(agent, 'last_caption', None)
    
    # Use the simple consciousness prompt
    return build_simple_caption_prompt(agent, mood_vector, last_caption)


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
