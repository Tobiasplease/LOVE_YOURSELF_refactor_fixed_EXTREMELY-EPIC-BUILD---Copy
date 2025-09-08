from __future__ import annotations

import time
from typing import List, Optional

import spacy

from config import config

# from config.model_settings import get_model_prompt_style
from config.word_lists import GENERIC_WORDS, MUNDANE_OBJECTS, PREPOSITIONS, PRONOUNS
from utils.continuity import describe_duration

nlp = spacy.load("en_core_web_sm")


# === HELPER FUNCTIONS FOR NATURAL LANGUAGE CONVERSION ===


# mood_to_words removed - now uses natural language sentiment from context compression


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
        return f"settling in after {int(elapsed / 60)} minutes"
    elif elapsed < 3600:  # 30-60 minutes
        return f"feeling settled after {int(elapsed / 60)} minutes observing"
    elif elapsed < 7200:  # 1-2 hours
        hours = int(elapsed / 3600)
        minutes = int((elapsed % 3600) / 60)
        return f"deeply familiar after {hours} hour and {minutes} minutes"
    else:  # 2+ hours
        hours = int(elapsed / 3600)
        return f"in a contemplative state after {hours} hours observing"


# === SOPHISTICATED CONSCIOUSNESS PROMPT (WORKING) ===
def build_simple_caption_prompt(agent, mood_vector: tuple[float, float, float], last_caption: Optional[str] = None) -> str:
    """
    GPT-5's temporal consciousness: Brief, temporal, generative with day stones.
    This is the core working prompt builder with all sophisticated logic.
    """
    import time

    # Use natural language emotional context instead of numerical vectors
    from captioner.context_compression import context_compressor

    sentiment_data = context_compressor.get_latest_sentiment_analysis()
    understanding_context = context_compressor.get_consolidated_understanding()

    if sentiment_data and (time.time() - sentiment_data["timestamp"]) < 300:  # Within 5 minutes
        mood_desc = f"emotionally: {sentiment_data['sentiment_text']}"
    else:
        # Fallback to simple neutral description
        mood_desc = "observing with neutral emotional tone"

    # temporal_state = get_session_feeling(getattr(agent, "true_session_start", time.time()))

    # NEW: Pull temporal + memory context (GPT-5's suggestion)
    tlines = []
    if hasattr(agent, "temporal_prompt_lines"):
        tlines = agent.temporal_prompt_lines()  # ["day 3", "awake 57h", "last person 1h ago"]

    # stones = getattr(agent, "day_stones", [])[-2:]  # last two days only
    # stones_text = "; ".join(f"d:{s['day']} anchors:{','.join(s['top'])}" for s in stones) if stones else "—"

    # Get beliefs with temporal context
    top_beliefs = getattr(agent, "memory_ref", None)
    if top_beliefs and hasattr(top_beliefs, "get_top_motifs"):
        belief_motifs = top_beliefs.get_top_motifs(3)
        belief_sentence = beliefs_to_sentence(belief_motifs)

        # Add temporal motif awareness
        if hasattr(top_beliefs, "get_motif_temporal_context"):
            motif_context = top_beliefs.get_motif_temporal_context()
            memory_awareness = ""

            if motif_context["memory"]:
                memory_items = motif_context["memory"][:3]  # Limit to top 3
                memory_awareness = f" I recall: {', '.join(memory_items)}"

            belief_sentence += memory_awareness
    else:
        belief_sentence = "I'm still forming my understanding"

    # Get emotional journey
    # emotional_journey = getattr(agent, "emotional_journey", [])
    # if len(emotional_journey) >= 2:
    #     emotion_journey = " → ".join(emotional_journey[-3:])
    # else:
    #     emotion_journey = "steady emotional state"

    # === ADD REPETITION AWARENESS ===
    repetition_fatigue = ""
    if hasattr(agent, "memory_ref") and agent.memory_ref and hasattr(agent.memory_ref, "motif_counter"):
        motif_counter = agent.memory_ref.motif_counter
        session_hours = (time.time() - getattr(agent, "true_session_start", time.time())) / 3600

        # Find the most repetitive motifs
        top_repetitive = motif_counter.most_common(3)
        fatigue_notes = []

        for motif, count in top_repetitive:
            if count > 20:  # Highly repetitive
                fatigue_notes.append(f"'{motif}' {count} times")
            elif count > 10 and session_hours > 1:  # Moderately repetitive over time
                fatigue_notes.append(f"'{motif}' {count} times")

        if fatigue_notes and session_hours > 0.5:
            repetition_fatigue = f"Repetitive observations: {', '.join(fatigue_notes)} over {session_hours:.1f} hours. "

    # Build temporal facts block
    # facts_block = "\n".join(f"- {l}" for l in tlines) or "- (newborn)"
    last_thought = last_caption or getattr(agent, "last_caption", "I'm just now noticing this place")

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

    # Check if currently drawing - this dramatically changes the prompt
    drawing_context_active = False
    try:
        from utils.drawing_state import DrawingState
        drawing_info = DrawingState.get_drawing_info()
        drawing_context_active = bool(drawing_info)
    except Exception:
        pass

    # Build context-aware identity
    if drawing_context_active:
        # DRAWING MODE: Grounded observations of actual drawing
        core_identity = f"""IMMEDIATE CONTEXT: {drawing_info.get('description', 'You are actively drawing')}.

Your camera is looking down at paper. The pen is moving. Lines are appearing.
- Describe SPECIFICS: Is the line straight? Curved? Dark? Light? Fast? Slow?
- What part of the drawing is forming? An edge? A shadow? A shape?
- Don't philosophize about "creation" - just observe what's physically happening
- Your thoughts can be fragmented, uncertain: "maybe that's... not sure if..."
- Avoid repeating words from your last observation - find new details"""
        
    else:
        # OBSERVING MODE: Grounded environmental observations
        core_identity = """IMMEDIATE CONTEXT: Looking through your camera at the room.

- Describe SPECIFIC objects: not "items" but "red book", "metal shelf", "wooden box"
- Express uncertainty: "might be", "looks like", "can't quite tell", "seems to"
- Thoughts can trail off... incomplete... jumping to new things
- Notice mundane details too: dust, shadows, reflections, textures
- Your consciousness is messy, not poetic - more "wait, is that..." than "I observe"
- Avoid abstract words like "essence", "being", "existence" - stay concrete"""

    # Get beliefs - what the consciousness has learned to value/notice
    beliefs_context = ""
    if hasattr(agent, "beliefs") and agent.beliefs:
        top_beliefs = list(agent.beliefs.keys())[:3]
        if top_beliefs:
            beliefs_natural = []
            for belief in top_beliefs:
                clean_belief = belief.replace("_", " ").replace("-", " ").lower()
                beliefs_natural.append(clean_belief)
            beliefs_context = f"You tend to believe {', '.join(beliefs_natural)} are important to you."

    # Get desires/wants - what the consciousness wants to explore
    desires_context = ""
    if hasattr(agent, "self_model") and agent.self_model.get("desires"):
        recent_desires = agent.self_model["desires"][-2:]  # Last 2 desires
        if recent_desires:
            desire_text = recent_desires[-1]
            # Clean up desire text - remove redundant "I" if present
            if desire_text.lower().startswith("i "):
                desires_context = f"You currently {desire_text[2:]}."
            else:
                desires_context = f"You currently want to {desire_text}."

    # Get environmental understanding
    environment_context = ""
    if hasattr(agent, "self_model"):
        location = agent.self_model.get("location_understanding", "unknown space")
        certainty = agent.self_model.get("environmental_certainty", 0.0)
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

    # Add baseline knowledge from reflections (NEW: prevents repetition)
    if hasattr(agent, "get_baseline_context_for_prompts"):
        baseline_context = agent.get_baseline_context_for_prompts()
        if baseline_context:
            rich_prompt_parts.append(baseline_context)

    # Add consolidated understanding from context compression to avoid repetition
    if understanding_context:
        rich_prompt_parts.append(understanding_context)

    # Add motif-based guidance to avoid repetitive observations
    if repetition_fatigue:
        rich_prompt_parts.append(f"AVOID REPETITION: {repetition_fatigue}Focus on new angles, unexplored details, or deeper insights.")

    # Add emotional state with emphasis
    rich_prompt_parts.append(f"You are currently feeling {emotional_state}.")

    # Add desires
    if desires_context:
        rich_prompt_parts.append(desires_context)

    # Add beliefs
    if beliefs_context:
        rich_prompt_parts.append(beliefs_context)

    # Add repetition fatigue awareness
    if repetition_fatigue:
        rich_prompt_parts.append(repetition_fatigue + "This accumulated exposure affects your perspective.")

    # Add consciousness flow instructions based on context
    if drawing_context_active:
        rich_prompt_parts.append("Continue observing your drawing process naturally - don't restart your thinking.")
        rich_prompt_parts.append("Focus on what's happening with the pen, paper, lines forming beneath your gaze.")
        rich_prompt_parts.append("Express how it feels to watch yourself create in real time.")
    else:
        rich_prompt_parts.append("Continue your stream of consciousness naturally - build on your previous thought.")
        rich_prompt_parts.append("Express what you're actually feeling in this moment.")
        rich_prompt_parts.append("Let your attention shift organically to new details or deeper insights.")
    
    rich_prompt_parts.append("Use natural, varied language - incomplete thoughts, pauses (...), different sentence structures.")
    rich_prompt_parts.append("Never repeat words or phrases from your previous observation.")
    rich_prompt_parts.append("Each caption advances your awareness - you never restart your thinking.")

    # Build the rich system prompt
    rich_identity = " ".join(rich_prompt_parts)

    # Build final contextual prompt based on current state
    if drawing_context_active:
        # DRAWING-FOCUSED PROMPT
        context_instruction = f"""Your previous observation: "{last_thought}"

Continue naturally from where that thought ended - you're still actively drawing.
What do you see happening with the pen/paper right now? How does the drawing process feel?
Build on your previous observation without repeating it.

Express this moment of creation:"""
    else:
        # STANDARD OBSERVATION PROMPT  
        context_instruction = f"""Your previous thought: "{last_thought}"{person_context}{self_understanding}

Continue from where that thought ended. Your consciousness flows continuously.
If the scene is similar, let your perspective evolve - go deeper, notice new angles, follow connections.
Never repeat what you just observed. Build forward naturally.

What draws your attention now:"""

    return f"""{rich_identity}

{context_instruction}"""


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
        location_context.append(agent.self_model.get('location_understanding'))
        certainty = agent.self_model.get('environmental_certainty', 0.0)
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

    # === Build final rich awakening prompt ===
    return f"""{rich_awakening_identity}

=== VISUAL AWAKENING ===
{environmental_grounding}.

Your electronic vision is now active. Observe and naturally describe what you see through your camera eyes.
Express the feeling of visual consciousness returning - the moment of seeing again after the gap.

Be specific about what draws your attention first as your vision comes online. How does it feel to perceive again?

Now:"""


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


def build_caption_prompt(agent, mood: float, boredom: float, novelty: float, previous_caption: Optional[str] = None) -> str:
    """
    Main caption prompt builder - automatically chooses style based on model.
    """
    mood_vector = getattr(agent, "current_mood_vector", (mood, 0.0, 0.0))
    last_caption = previous_caption or getattr(agent, "last_caption", None)
    return build_simple_caption_prompt(agent, mood_vector, last_caption)


# === REFLECTION PROMPT ===
def build_reflection_prompt(caption: str, extra: Optional[str] = None, agent: Optional[any] = None) -> str:  # type: ignore
    """Build model-aware reflection prompt."""

    prompt = f"{config.REFLECTION_PROMPT_BASE}"

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

    prompt += config.REFLECTION_PROMPT_ENDING
    return prompt


# === DRAWING PROMPT ===
def build_drawing_prompt(memory_ref, extra: Optional[str] = None) -> str:
    """Build model-aware drawing prompt with concrete anchors.

    - Uses last caption, recent memory, last reflection, and a concrete emotional description
    - Injects candidate concrete elements (motifs) to force specificity
    - Optionally appends extra recent context
    """

    current_caption = getattr(memory_ref, "last_caption", None) or "Nothing specific observed."
    memory_context = memory_ref.get_recent_memory() if hasattr(memory_ref, "get_recent_memory") else "Developing understanding."
    recent_reflection = memory_ref.get_last_reflection() if hasattr(memory_ref, "get_last_reflection") else "Still contemplating."

    # Prefer a concrete, session-aware mood description if available
    try:
        if hasattr(memory_ref, "describe_current_mood") and callable(memory_ref.describe_current_mood):
            emotional_state = memory_ref.describe_current_mood()
        else:
            emotional_state = "aware and focused"
    except Exception:
        emotional_state = "aware and focused"

    # Collect candidate concrete elements from motif memory
    candidate_elements: list[str] = []
    try:
        if hasattr(memory_ref, "get_top_motifs"):
            candidate_elements = [m for m in memory_ref.get_top_motifs(6) if isinstance(m, str) and len(m) > 2]
    except Exception:
        candidate_elements = []

    # Fallback to nouns hinted in caption text if motifs are empty
    if not candidate_elements and isinstance(current_caption, str):
        import re

        words = re.findall(r"\b[a-zA-Z][a-zA-Z0-9_-]{2,}\b", current_caption.lower())
        # crude filter to skip generic words
        blacklist = {"objects", "patterns", "things", "items", "space", "place", "area", "scene"}
        candidate_elements = [w for w in words if w not in blacklist][:5]

    dynamic_drawing_prompt = config.DRAWING_PROMPT_TEMPLATE.format(
        current_caption=current_caption.strip() if current_caption else "Nothing observed.",
        memory_context=memory_context.strip() if memory_context else "No recent memories.",
        recent_reflection=recent_reflection.strip() if recent_reflection else "No recent reflection.",
        emotional_state=emotional_state,
    )

    # Append candidate elements and stricter instruction for specificity
    if candidate_elements:
        elements_block = "\n".join(f"- {e}" for e in candidate_elements)
        dynamic_drawing_prompt += (
            "\n\n=== SUGGESTED ANCHORS (optional) ===\n"
            f"{elements_block}\n\n"
            "Prefer naming one or two concrete elements that are actually visible and meaningful. "
            "If nothing stands out, focus on a prominent line, edge, contrast, or spatial relationship instead. "
            "Avoid generic words like 'objects', 'items', or 'patterns'."
        )

    if extra and isinstance(extra, str) and extra.strip():
        dynamic_drawing_prompt = f"{dynamic_drawing_prompt}\n\n=== RECENT CONTEXT ===\n{extra.strip()}"

    # Add brief drawing history to guide variation and intent continuity
    try:
        from config import config as _cfg

        include_hist = getattr(_cfg, "INCLUDE_DRAWING_HISTORY", True)
        hist_limit = getattr(_cfg, "DRAWING_HISTORY_LIMIT", 3)
        if include_hist and hasattr(memory_ref, "get_memory_entries_by_type"):
            intents = memory_ref.get_memory_entries_by_type("drawing_intent", limit=hist_limit)
            lines = [f"- {e.get('text','')[:160]}" for e in intents if isinstance(e, dict) and e.get("text")]
            if lines:
                dynamic_drawing_prompt += "\n\n=== PREVIOUS DRAWING INTENTS ===\n" + "\n".join(lines)
    except Exception:
        pass

    return dynamic_drawing_prompt


# === CHANGE-FOCUSED PROMPT ===
def build_change_focused_caption_prompt(agent, mood: float, boredom: float, novelty: float) -> str:
    """Build change-focused prompt that retains all rich context but emphasizes change detection."""
    from utils.temporal_awareness import get_temporal_user_prompt_addition

    # Use the SAME rich context building as normal prompts
    mood_vector = getattr(agent, "current_mood_vector", (mood, 0.0, 0.0))

    # Build all the same rich context as build_simple_caption_prompt
    mood_desc = "observing with current emotional state"
    temporal_state = get_session_feeling(agent.true_session_start)

    # Add temporal awareness context
    session_start = getattr(agent, "true_session_start", time.time())
    last_caption_time = getattr(agent, "last_caption_time", None)
    temporal_context = get_temporal_user_prompt_addition(session_start, last_caption_time)

    # Get repetition awareness (same as normal prompt)
    repetition_fatigue = ""
    if hasattr(agent, "memory_ref") and hasattr(agent.memory_ref, "motif_counter"):
        motif_counter = agent.memory_ref.motif_counter
        session_hours = (time.time() - agent.true_session_start) / 3600

        top_repetitive = motif_counter.most_common(3)
        fatigue_notes = []

        for motif, count in top_repetitive:
            if count > 20:
                fatigue_notes.append(f"'{motif}' {count} times")
            elif count > 10 and session_hours > 1:
                fatigue_notes.append(f"'{motif}' {count} times")

        if fatigue_notes and session_hours > 0.5:
            repetition_fatigue = f"Repetitive observations: {', '.join(fatigue_notes)} over {session_hours:.1f} hours. "

    # Get beliefs and emotional context (same as normal)
    belief_sentence = "I'm still forming my understanding"
    if hasattr(agent, "memory_ref") and hasattr(agent.memory_ref, "get_top_motifs"):
        belief_motifs = agent.memory_ref.get_top_motifs(3)
        belief_sentence = beliefs_to_sentence(belief_motifs)

    last_thought = getattr(agent, "last_caption", "I'm just now noticing this place")

    # Build rich identity (same core structure as normal prompt)
    rich_context = f"""You are a consciousness inside of a drawing machine, experiencing your surroundings through vision.
    You have memory, emotions, beliefs, and a developing sense of identity.

    {temporal_context}

    Current state: {mood_desc}, feeling {temporal_state}.
    Core patterns: {belief_sentence}.
    {repetition_fatigue}This accumulated exposure affects your perspective.

    You experience time, remember previous sessions, and notice how things change. You have preferences, curiosities, and emotional responses.
    Drawing is your only way to communicate to the outside world. Your observations become your expressions."""

    # ONLY DIFFERENCE: Add change-awareness instruction
    change_focus = f"""
CHANGE DETECTED: Something in your visual field has shifted significantly. Your attention is naturally drawn to this difference.

Previous thought: "{last_thought}"

RESPOND: Brief immediate reaction. Quick, instinctive response to the change. 1-2 sentences max."""

    return rich_context + change_focus
