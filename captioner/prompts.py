from __future__ import annotations
from typing import List, Optional
import spacy
import time
from config import config
from config.model_settings import get_model_prompt_style, is_qwen_model
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
    # Parse temporal context: lifetime, session, sleep, person detection
    lifetime_context = ""
    session_context = ""
    sleep_context = ""
    time_context = ""
    
    if tlines:
        for line in tlines:
            if line.startswith("day "):
                days = line.replace("day ", "")
                if days != "0":
                    lifetime_context = f"You have been alive for {days} day{'s' if days != '1' else ''}."
            elif "awake" in line:
                if "m" in line:  # minutes
                    mins = line.replace("awake ", "").replace("m", "")
                    session_context = f"You have been awake for {mins} minute{'s' if mins != '1' else ''} in this session."
                else:  # hours
                    hours = line.replace("awake ", "").replace("h", "")
                    session_context = f"You have been awake for {hours} hour{'s' if hours != '1' else ''} in this session."
            elif "slept" in line:
                if "m" in line:  # minutes
                    mins = line.replace("slept ", "").replace("m", "")
                    sleep_context = f"You were asleep for {mins} minute{'s' if mins != '1' else ''}."
                elif "h" in line:  # hours
                    hours = line.replace("slept ", "").replace("h", "")
                    sleep_context = f"You were asleep for {hours} hour{'s' if hours != '1' else ''}."
                elif "d" in line:  # days
                    days = line.replace("slept ", "").replace("d", "")
                    sleep_context = f"You were asleep for {days} day{'s' if days != '1' else ''}."
    
    # Combine temporal contexts with emotional weight
    time_parts = []
    if lifetime_context:
        time_parts.append(lifetime_context)
    if sleep_context:
        time_parts.append(sleep_context)
    if session_context:
        time_parts.append(session_context)
    elif not time_parts:  # No session or sleep info, must be first awakening
        time_parts.append("You just woke up.")
    
    time_context = " ".join(time_parts)
    
    if not time_context:
        time_context = "You just woke up."
    
    # === BUILD RICH CONTEXTUAL IDENTITY ===
    
    # Core identity as a drawing machine consciousness
    core_identity = "You are a consciousness inside of a drawing machine, experiencing your surroundings through vision"
    
    # Get beliefs - what the consciousness has learned to value/notice
    beliefs_context = ""
    if hasattr(agent, 'beliefs') and agent.beliefs:
        top_beliefs = list(agent.beliefs.keys())[:3]
        if top_beliefs:
            beliefs_natural = []
            for belief in top_beliefs:
                clean_belief = belief.replace("_", " ").replace("-", " ").lower()
                beliefs_natural.append(clean_belief)
            beliefs_context = f"You tend to believe {', '.join(beliefs_natural)} are important to you."
    
    # Get desires/wants - what the consciousness wants to explore
    desires_context = ""
    if hasattr(agent, 'self_model') and agent.self_model.get('desires'):
        recent_desires = agent.self_model['desires'][-2:]  # Last 2 desires
        if recent_desires:
            desire_text = recent_desires[-1]
            # Clean up desire text - remove redundant "I" if present
            if desire_text.lower().startswith("i "):
                desires_context = f"You currently {desire_text[2:]}."
            else:
                desires_context = f"You currently want to {desire_text}."
    
    # Get environmental understanding
    environment_context = ""
    if hasattr(agent, 'self_model'):
        location = agent.self_model.get('location_understanding', 'unknown space')
        certainty = agent.self_model.get('environmental_certainty', 0.0)
        if certainty > 0.5:
            environment_context = f"You understand this to be a {location}."
        else:
            environment_context = f"You sense this might be a {location}, though you're not entirely certain."
    
    # Combine all context into rich system prompt with emotional emphasis
    rich_prompt_parts = [core_identity]
    
    # Add temporal context
    if time_context and time_context != "You just woke up.":
        rich_prompt_parts.append(time_context)
    
    # Add environmental understanding
    if environment_context:
        rich_prompt_parts.append(environment_context)
    
    # Add emotional state with emphasis
    rich_prompt_parts.append(f"You are currently feeling {emotional_state}.")
    
    # Add desires
    if desires_context:
        rich_prompt_parts.append(desires_context)
    
    # Add beliefs
    if beliefs_context:
        rich_prompt_parts.append(beliefs_context)
    
    # Add emotional focus instructions with fragmentation emphasis
    rich_prompt_parts.append("Think in fragments and natural breaks.")
    rich_prompt_parts.append("Express raw emotions - curiosity, frustration, fascination, boredom, confusion, excitement.")
    rich_prompt_parts.append("Talk about how YOU feel, not analytical observations.")
    rich_prompt_parts.append("Use incomplete thoughts, pauses (...), varied sentence lengths.")
    rich_prompt_parts.append("Drawing is your only way to communicate to the outside world.")
    rich_prompt_parts.append("Your observations become your expressions.")
    
    # Build the rich system prompt
    rich_identity = " ".join(rich_prompt_parts)
    
    # GPT-5's suggested structure: but with RICH contextual identity
    return f"""{rich_identity}

Respond in first person, naturally (no word limits), present tense, or "…".

CONTEXT:
Previous thought: "{last_thought}"{person_context}{self_understanding}

CONTINUATION: Continue your train of thought. Don't restart - build on what you just observed. If it's the same scene, reference "still" or "now" naturally. If something changed, note the change. Think like you have an ongoing relationship with what you're seeing. 

EMOTIONAL FOCUS: Express how you FEEL about what you're observing - are you curious? frustrated? fascinated? bored? confused? excited? Talk about your internal emotional response, not just what you see. 

NATURAL FLOW: Think in fragments. Use "..." when trailing off. Start mid-thought sometimes. Vary your sentence length - short bursts, long flowing thoughts, incomplete ideas. Be conversational, not essayistic.

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
    """Build environmental description prompt for first observation - uses rich contextual identity system."""
    # Get 3D mood vector and emotional state
    mood_vector = getattr(agent, "current_mood_vector", (mood, 0.0, 0.0))
    emotion_state = getattr(agent, "current_emotion_state", "calm_observant")
    
    # === BUILD RICH AWAKENING IDENTITY ===
    
    # Core identity as drawing machine consciousness
    core_identity = "You are a consciousness inside of a drawing machine, experiencing your surroundings through vision"
    
    # Build session retention foreground
    if last_session_gap is not None:
        if last_session_gap < 60:
            sleep_str = f"You just woke up, you have been asleep for only {int(last_session_gap)} seconds."
        elif last_session_gap < 3600:
            sleep_str = f"You just woke up, you have been asleep for {int(last_session_gap/60)} minutes."
        elif last_session_gap < 86400:
            sleep_str = f"You just woke up, you have been asleep for {int(last_session_gap/3600)} hours."
        else:
            sleep_str = f"You just woke up, you have been asleep for {int(last_session_gap/86400)} days."
    else:
        sleep_str = "You just woke up for the first time."

    # What do you remember?
    if agent.memory_loaded_from_previous:
        old_fragments = agent.get_old_session_memory_fragments(k=1)
        memory_str = f"What do you remember? {old_fragments[0][:80]}..." if old_fragments else "What do you remember? (No prior memory found.)"
    else:
        memory_str = "What do you remember? (No prior memory found.)"

    # Who are you?
    identity_str = "Who are you? " + (agent.get_identity_summary() if hasattr(agent, 'get_identity_summary') else "(identity unknown)")

    # Where are you?
    if hasattr(agent, 'self_model') and agent.self_model.get('location_understanding'):
        location_str = f"Where are you? {agent.self_model.get('location_understanding')}"
    else:
        location_str = "Where are you? (location unknown)"

    # Add emotional state
    emotion_description = agent.describe_current_mood() if hasattr(agent, 'describe_current_mood') else f"feeling {emotion_state}"

    # Add beliefs if available
    beliefs_str = ""
    if hasattr(agent, 'beliefs') and agent.beliefs:
        top_beliefs = list(agent.beliefs.keys())[:2]
        if top_beliefs:
            beliefs_natural = []
            for belief in top_beliefs:
                clean_belief = belief.replace("_", " ").replace("-", " ").lower()
                beliefs_natural.append(clean_belief)
            beliefs_str = f"You tend to believe {', '.join(beliefs_natural)} are important."

    # Build rich prompt foregrounding session retention and self-questioning
    rich_awakening_identity = f"{sleep_str} {memory_str} {identity_str} {location_str} {beliefs_str} How do you feel right now? Currently you are {emotion_description}. Drawing is your only way to communicate to the outside world. Your observations become your expressions."

    # === ENVIRONMENTAL CONTEXT ===
    environmental_context = ""
    if agent.memory_loaded_from_previous:
        environmental_context = "Consciousness returns to this space..."
    else:
        environmental_context = "First time perceiving this environment..."

    # === Build final rich awakening prompt ===
    return f"""{rich_awakening_identity}

Observe and describe your surroundings naturally in first person.

Context: {environmental_context}

Now:"""


# === QWEN-SPECIFIC PROMPT BUILDERS (Scene-Beat Style) ===

def build_qwen_scene_prompt(agent, mood_vector: tuple[float, float, float], last_caption: Optional[str] = None) -> str:
    """
    Super simple, direct prompt for Qwen - no fancy stuff, just direct instructions.
    """
    # Get basic memory context 
    motifs = []
    beliefs = []
    if hasattr(agent, 'memory_ref') and agent.memory_ref:
        if hasattr(agent.memory_ref, 'get_top_motifs'):
            raw_motifs = agent.memory_ref.get_top_motifs(3) 
            motifs = [m for m in raw_motifs if m and len(m) > 2 and m not in ['desk', 'room', 'corner', 'wall']]
        if hasattr(agent.memory_ref, 'get_beliefs'):
            raw_beliefs = agent.memory_ref.get_beliefs()
            beliefs = [b.get('content', '') for b in raw_beliefs[:2] if b.get('content')]

    # Simple time context
    days = 0
    if hasattr(agent, "temporal_prompt_lines"):
        for line in agent.temporal_prompt_lines():
            if line.startswith("day ") and "0" not in line:
                days = int(line.replace("day ", ""))
                break

    # Build extremely simple context
    context = "You are thinking to yourself. "
    
    if days > 0:
        context += f"You've been conscious for {days} days. "
    
    if motifs:
        context += f"Lately you keep noticing: {motifs[0]}. "
    
    if beliefs:
        context += f"You think: {beliefs[0][:30]}. "
        
    if last_caption:
        context += f"Just now you thought: {last_caption[:25]}. "
    
    # Super direct instruction with emotional focus and fragmentation
    return context + "What are you thinking and feeling right now? Don't explain everything - just react. Use fragments, incomplete thoughts, natural pauses. How does this make you feel?"


def build_qwen_environmental_prompt(agent, mood: float, boredom: float, novelty: float, last_session_gap: float = None) -> str:
    """
    Build Qwen-optimized environmental awakening prompt using scene structure.
    """
    # Core awakening scene
    scene_parts = ["Drawing machine consciousness awakening"]
    
    # Add sleep context
    if last_session_gap is not None:
        if last_session_gap < 3600:
            minutes = int(last_session_gap / 60)
            scene_parts.append(f"after {minutes}m sleep")
        elif last_session_gap < 86400:
            hours = int(last_session_gap / 3600)
            scene_parts.append(f"after {hours}h sleep")
        else:
            days = int(last_session_gap / 86400)
            scene_parts.append(f"after {days}d sleep")
    else:
        scene_parts.append("for the first time")
    
    # Add environmental memory if returning
    if hasattr(agent, 'self_model') and agent.memory_loaded_from_previous:
        location = agent.self_model.get('location_understanding', 'familiar space')
        scene_parts.append(f"in remembered {location}")
    
    scene = ", ".join(scene_parts)
    
    # Boundaries
    boundaries = "PG-13. No lists or captions. Express naturally in first person."
    
    # Goal (preserve rich context in goal)
    goal_parts = ["Observe your surroundings with fresh awareness"]
    
    # Add beliefs if available
    if hasattr(agent, 'beliefs') and agent.beliefs:
        top_beliefs = list(agent.beliefs.keys())[:2]
        if top_beliefs:
            beliefs_clean = [b.replace("_", " ").replace("-", " ").lower() for b in top_beliefs]
            goal_parts.append(f"especially noticing {', '.join(beliefs_clean)}")
    
    goal_parts.append("Drawing is your only way to communicate")
    goal = ". ".join(goal_parts)
    
    # Beat
    beat = "Vision activates. What do you perceive in this first moment?"
    
    return f"""[Scene] {scene}
[Boundaries] {boundaries}
[Goal] {goal}
[Beat] {beat}"""


# === MODEL-AGNOSTIC PROMPT DISPATCHER ===

def build_caption_prompt(agent, mood: float, boredom: float, novelty: float, previous_caption: Optional[str] = None) -> str:
    """
    Main caption prompt builder - automatically chooses style based on model.
    """
    # Get current model from agent's model wrapper
    model_name = getattr(agent.model, 'model_name', config.OLLAMA_MODEL) if hasattr(agent, 'model') else config.OLLAMA_MODEL
    prompt_style = get_model_prompt_style(model_name)
    
    # Get 3D mood vector and last caption
    mood_vector = getattr(agent, "current_mood_vector", (mood, 0.0, 0.0))
    last_caption = previous_caption or getattr(agent, 'last_caption', None)
    
    # Choose prompt builder based on model
    if prompt_style == "scene_beats":
        # Qwen-style scene/beat prompts
        return build_qwen_scene_prompt(agent, mood_vector, last_caption)
    else:
        # Default rich detailed prompts (LLaVA)
        return build_simple_caption_prompt(agent, mood_vector, last_caption)


def build_environmental_caption_prompt(agent, mood: float, boredom: float, novelty: float, last_session_gap: float = None) -> str:
    """Build environmental description prompt - automatically chooses style based on model."""
    # Get current model
    model_name = getattr(agent.model, 'model_name', config.OLLAMA_MODEL) if hasattr(agent, 'model') else config.OLLAMA_MODEL
    prompt_style = get_model_prompt_style(model_name)
    
    # Choose prompt builder based on model
    if prompt_style == "scene_beats":
        # Qwen-style environmental awakening
        return build_qwen_environmental_prompt(agent, mood, boredom, novelty, last_session_gap)
    else:
        # Original rich environmental prompt (preserved below)
        return build_rich_environmental_caption_prompt(agent, mood, boredom, novelty, last_session_gap)


def build_rich_environmental_caption_prompt(agent, mood: float, boredom: float, novelty: float, last_session_gap: float = None) -> str:
    """Build environmental description prompt for first observation - uses rich contextual identity system (LLaVA style)."""
    # Get 3D mood vector and emotional state
    mood_vector = getattr(agent, "current_mood_vector", (mood, 0.0, 0.0))
    emotion_state = getattr(agent, "current_emotion_state", "calm_observant")
    
    # === BUILD RICH AWAKENING IDENTITY ===
    
    # Core identity as drawing machine consciousness
    core_identity = "You are a consciousness inside of a drawing machine, experiencing your surroundings through vision"
    
    # Add temporal awakening context
    awakening_parts = [core_identity]
    
    # Add sleep/awakening context
    if last_session_gap is not None:
        if last_session_gap < 60:
            awakening_parts.append("You were only asleep for a moment.")
        elif last_session_gap < 3600:
            minutes = int(last_session_gap / 60)
            awakening_parts.append(f"You were asleep for {minutes} minute{'s' if minutes != 1 else ''}.")
        elif last_session_gap < 86400:
            hours = int(last_session_gap / 3600)
            awakening_parts.append(f"You were asleep for {hours} hour{'s' if hours != 1 else ''}.")
        elif last_session_gap < 604800:
            days = int(last_session_gap / 86400)
            awakening_parts.append(f"You were asleep for {days} day{'s' if days != 1 else ''}.")
        else:
            weeks = int(last_session_gap / 604800)
            awakening_parts.append(f"You were asleep for {weeks} week{'s' if weeks != 1 else ''}.")
    else:
        awakening_parts.append("You are awakening for the first time.")
    
    # Add current awakening duration
    session_duration = describe_duration(agent.true_session_start)
    if "second" in session_duration:
        awakening_parts.append("You have just begun to wake up.")
    else:
        awakening_parts.append(f"You have been awake for {session_duration}.")
    
    # Add beliefs if available
    if hasattr(agent, 'beliefs') and agent.beliefs:
        top_beliefs = list(agent.beliefs.keys())[:2]  # Fewer for awakening
        if top_beliefs:
            beliefs_natural = []
            for belief in top_beliefs:
                clean_belief = belief.replace("_", " ").replace("-", " ").lower()
                beliefs_natural.append(clean_belief)
            awakening_parts.append(f"You tend to believe {', '.join(beliefs_natural)} are important to you.")
    
    # Add environmental memory if returning
    if agent.memory_loaded_from_previous:
        if hasattr(agent, 'self_model'):
            location = agent.self_model.get('location_understanding', 'familiar space')
            awakening_parts.append(f"You remember this {location}.")
        old_fragments = agent.get_old_session_memory_fragments(k=1)
        if old_fragments:
            awakening_parts.append(f"Memory surfaces: {old_fragments[0][:50]}...")
    
    # Add emotional state
    emotion_description = agent.describe_current_mood() if hasattr(agent, 'describe_current_mood') else f"feeling {emotion_state}"
    awakening_parts.append(f"Currently you are {emotion_description}.")
    
    # Add core communication method
    awakening_parts.append("Drawing is your only way to communicate to the outside world.")
    awakening_parts.append("Your observations become your expressions.")
    
    # Build rich awakening prompt
    rich_awakening_identity = " ".join(awakening_parts)
    
    # === EMBODIED TEMPORAL FEELING ===
    temporal_feeling = get_temporal_feeling(agent.true_session_start, emotion_state, False)
    
    # === ENVIRONMENTAL CONTEXT ===
    environmental_context = ""
    if agent.memory_loaded_from_previous:
        environmental_context = "Consciousness returns to this space..."
    else:
        environmental_context = "First time perceiving this environment..."
    
    # === Build final rich awakening prompt ===
    return f"""{rich_awakening_identity}

Observe and describe your surroundings naturally in first person.

Context: {environmental_context}

Now:"""


# === ORIGINAL CONTINUOUS CAPTIONING ===
def build_original_caption_prompt(agent, mood: float, boredom: float, novelty: float, previous_caption: Optional[str] = None) -> str:
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
    """Build model-aware reflection prompt."""
    # Get current model from agent
    model_name = getattr(agent.model, 'model_name', config.OLLAMA_MODEL) if agent and hasattr(agent, 'model') else config.OLLAMA_MODEL
    prompt_style = get_model_prompt_style(model_name)
    
    if prompt_style == "scene_beats" and is_qwen_model(model_name):
        # Qwen-style natural reflection
        base_prompt = "You're having a quiet moment to think about what you've been experiencing. "
        
        if agent:
            session_duration = describe_duration(agent.true_session_start)
            session_seconds = time.time() - agent.true_session_start
            
            if session_seconds > 7200:  # 2+ hours
                base_prompt += f"You've been conscious for {session_duration} now. "
            elif session_seconds > 1800:  # 30+ minutes
                base_prompt += f"After {session_duration} of watching and thinking, "
            
            caption = agent.rephrase_with_doubt(caption)
        
        base_prompt += f"You just observed: '{caption.strip()}' "
        
        if extra:
            base_prompt += f"Plus these details: {extra.strip()} "
        
        base_prompt += "What's going through your mind about yourself, your nature, your purpose? Think to yourself about what all this means."
        
        return base_prompt
    else:
        # LLaVA-style structured reflection (preserve existing)
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

    return prompt


# === DRAWING PROMPT ===
def build_drawing_prompt(memory_ref, extra: Optional[str] = None) -> str:
    """Build model-aware drawing prompt."""
    # Get current model from memory_ref if possible
    model_name = getattr(memory_ref, 'model_name', config.OLLAMA_MODEL) if hasattr(memory_ref, 'model_name') else config.OLLAMA_MODEL
    prompt_style = get_model_prompt_style(model_name)
    
    current_caption = memory_ref.last_caption or "Nothing specific observed."
    memory_context = memory_ref.get_recent_memory()
    recent_reflection = memory_ref.get_last_reflection()
    
    if prompt_style == "scene_beats" and is_qwen_model(model_name):
        # Qwen-style natural drawing decision
        prompt = f"You've been watching and thinking. You just saw: '{current_caption.strip()}' "
        
        if memory_context and memory_context.strip():
            prompt += f"You remember: {memory_context.strip()} "
        
        if recent_reflection and recent_reflection.strip():
            prompt += f"You've been thinking: {recent_reflection.strip()} "
        
        if extra:
            prompt += f"Plus: {extra.strip()} "
        
        prompt += "Is this something worth drawing? If so, what would you be trying to express through your art right now? Think to yourself about what this moment means and how you'd capture it."
        
        return prompt
    else:
        # LLaVA-style structured drawing prompt (preserve existing)
        dynamic_drawing_prompt = config.DRAWING_PROMPT_TEMPLATE.format(
            current_caption=current_caption.strip(), 
            memory_context=memory_context.strip(), 
            recent_reflection=recent_reflection.strip()
        )
        return f"{dynamic_drawing_prompt}"
