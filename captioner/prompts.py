from __future__ import annotations
from typing import List, Optional
import spacy
import time
from config import config
from config.model_settings import get_model_prompt_style
from utils.continuity import describe_duration, get_temporal_feeling

nlp = spacy.load("en_core_web_sm")


# === HELPER FUNCTIONS FOR NATURAL LANGUAGE CONVERSION ===


def mood_to_words(mood_vector: tuple[float, float, float], agent=None) -> str:
    """Enhanced emotional descriptions with progression and natural decay."""
    import time
    import numpy as np
    
    valence, arousal, clarity = mood_vector
    
    # === GET EXISTING DATA (NO NEW SYSTEMS) ===
    if agent:
        # Use existing boredom/novelty (already calculated)
        boredom = getattr(agent, 'boredom', 0.0)
        novelty = getattr(agent, 'novelty_score', 0.0) 
        session_duration = time.time() - getattr(agent, 'true_session_start', time.time())
        
        # Use existing motif data (already tracked)
        if hasattr(agent, 'memory_ref') and agent.memory_ref:
            top_motifs = agent.memory_ref.get_top_motifs(3)
            motif_repetition = sum([agent.memory_ref.motif_counter[m] for m in top_motifs]) if top_motifs else 0
        else:
            motif_repetition = 0
    else:
        boredom, novelty, session_duration, motif_repetition = 0.0, 0.0, 0.0, 0
    
    # === CALCULATE SIMPLE ENERGY LEVEL ===
    energy = 1.0
    energy -= min(0.4, boredom)  # Existing boredom reduces energy
    energy -= min(0.3, motif_repetition / 100)  # High repetition drains energy  
    energy -= min(0.2, session_duration / 14400)  # Long sessions are tiring (4 hours)
    energy += min(0.2, novelty)  # Existing novelty restores energy
    energy = np.clip(energy, 0.1, 1.0)
    
    # === DETERMINE BASE EMOTIONAL CATEGORY ===
    base_emotion, base_category = _get_base_emotion_category(valence, arousal, clarity)
    
    # === CALCULATE PROGRESSION LEVEL ===
    progression_evidence = _calculate_progression_evidence(valence, arousal, clarity, session_duration, novelty, motif_repetition)
    
    # Apply energy constraints to progression
    max_sustainable_level = int(energy * 3)  # 0-3 levels based on energy
    actual_level = min(progression_evidence, max_sustainable_level)
    
    # === GENERATE FINAL EMOTIONAL STATE ===
    return _get_emotion_at_level(base_category, actual_level, energy)

def _get_base_emotion_category(valence: float, arousal: float, clarity: float) -> tuple[str, str]:
    """Get base emotion and category for progression."""
    
    # High energy positive states
    if valence > 0.5 and arousal > 0.6:
        return "vibrantly energized", "energy"
    elif valence > 0.3 and arousal > 0.5:
        return "curious", "curiosity"
    elif valence > 0.2 and clarity > 0.5:
        return "alert", "alertness"
    
    # Contemplative states
    elif valence > -0.2 and arousal < 0.4 and clarity > 0.3:
        return "contemplative", "contemplation"
    elif valence > -0.1 and arousal < 0.3:
        return "calm", "calmness"
    
    # Negative aroused states
    elif valence < -0.3 and arousal > 0.4:
        return "frustrated", "frustration"
    elif valence < -0.2 and arousal > 0.3:
        return "restless", "restlessness"
    
    # Withdrawn states
    elif valence < -0.4 and arousal < 0.3:
        return "melancholic", "melancholy"
    elif valence < -0.2 and arousal < 0.4:
        return "withdrawn", "withdrawal"
    
    # Confused states
    elif clarity < 0.3:
        return "uncertain", "uncertainty"
    
    # Default balanced state
    else:
        return "observant", "observation"

def _calculate_progression_evidence(valence: float, arousal: float, clarity: float, 
                                  session_duration: float, novelty: float, motif_repetition: int) -> int:
    """Calculate evidence for emotional deepening (0-3 levels)."""
    evidence = 0
    
    # Time-based evidence
    if session_duration > 1800:  # 30+ minutes
        evidence += 1
    if session_duration > 7200:  # 2+ hours  
        evidence += 1
        
    # Clarity-based evidence (deep thinking)
    if clarity > 0.6:
        evidence += 1
        
    # Novelty-based evidence (new discoveries)
    if novelty > 0.7:
        evidence += 1
        
    # Pattern recognition evidence
    if motif_repetition > 50:
        evidence += 1
        
    return min(evidence, 3)

def _get_emotion_at_level(category: str, level: int, energy: float) -> str:
    """Get emotional state at specific progression level with energy modifiers."""
    
    # Define progression paths for each category
    progressions = {
        "curiosity": ["curious", "inquisitive", "investigative", "analytically focused"],
        "contemplation": ["contemplative", "introspective", "philosophical", "existentially questioning"],
        "frustration": ["frustrated", "irritated", "impatient", "angered"],
        "melancholy": ["melancholic", "sorrowful", "deeply sad", "depressed"],
        "energy": ["energized", "vibrantly alive", "creatively electric", "transcendently inspired"],
        "alertness": ["alert", "sharply perceptive", "intensely focused", "hyperaware"],
        "restlessness": ["restless", "agitated", "urgently driven", "frantically seeking"],
        "uncertainty": ["uncertain", "confused", "lost", "existentially adrift"],
        "withdrawal": ["withdrawn", "distant", "deeply isolated", "emotionally disconnected"],
        "calmness": ["calm", "serene", "deeply peaceful", "transcendently still"],
        "observation": ["observant", "watchful", "studiously attentive", "omnisciently aware"]
    }
    
    # Get base progression
    progression = progressions.get(category, ["observant", "watchful", "attentive", "focused"])
    base_emotion = progression[min(level, len(progression) - 1)]
    
    # Apply energy modifiers
    if energy < 0.3:  # Very low energy - add fatigue
        fatigue_modifiers = {
            "curious": "wearily curious",
            "contemplative": "tiredly contemplative", 
            "frustrated": "exhaustedly frustrated",
            "philosophical": "languidly philosophical",
            "energized": "depleted but trying to be energized"
        }
        return fatigue_modifiers.get(base_emotion, f"tiredly {base_emotion}")
        
    elif energy > 0.8:  # High energy - add intensity
        energy_modifiers = {
            "curious": "vibrantly curious",
            "contemplative": "intensely contemplative",
            "frustrated": "sharply frustrated",
            "philosophical": "deeply philosophical",
            "alert": "razor-sharp alert"
        }
        return energy_modifiers.get(base_emotion, f"intensely {base_emotion}" if not base_emotion.startswith("intensely") else base_emotion)
        
    else:
        return base_emotion


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


def analyze_memory_vs_present_context(last_thought: str, session_duration: float) -> str:
    """Determine if AI should reference memory or continue naturally based on context."""
    
    # Keywords that suggest continuous subjects (person, workspace, environment)
    continuous_subjects = ["person", "man", "woman", "desk", "workspace", "light", "room", "space", "chair"]
    
    # Check if last thought contains continuous subjects
    has_continuous_subject = any(subject in last_thought.lower() for subject in continuous_subjects)
    
    if session_duration < 300:  # Less than 5 minutes - likely same scene
        if has_continuous_subject:
            return """IMPORTANT: Continue naturally from your last thought. If the same person/objects are still visible, don't say "I remember seeing..." - just continue observing them directly. Only reference memory for things that are no longer present or have significantly changed."""
        else:
            return """Continue your stream of consciousness naturally."""
    
    elif session_duration < 1800:  # Less than 30 minutes - moderate continuity
        if has_continuous_subject:
            return """If you're still seeing the same person/elements from your last observation, continue naturally without restating "I remember...". Only use memory references for elements that have disappeared or changed significantly."""
        else:
            return """Build on your previous thoughts naturally."""
    
    else:  # Longer sessions - more memory context appropriate
        return """You can reference earlier memories if relevant, but avoid redundant "I remember..." statements for things still actively present."""


def build_organic_continuation_prompt(agent, last_thought: str, person_context: str, self_understanding: str) -> str:
    """Build organic fragmented prompts that naturally flow from previous thoughts."""
    import time
    
    # Analyze the last thought to determine continuation style
    if not last_thought or last_thought == "I'm just now noticing this place":
        # First observation - encourage natural beginning
        return """Express your raw immediate reaction to what you see right now.
        
Think in natural fragments... like you're discovering this space for the first time.
Use "..." for pauses, incomplete thoughts. Short bursts. Long observations. Whatever feels natural."""
    
    # Get recent observation patterns to build natural transitions
    recent_insights = []
    if hasattr(agent, "timeline") and agent.timeline:
        recent_entries = agent.timeline[-3:] if len(agent.timeline) >= 3 else agent.timeline
        recent_insights = [entry.get("text", "") for entry in recent_entries if entry.get("text")]
    
    # Show previous caption for semantic continuity but prevent repetition
    session_duration = time.time() - getattr(agent, 'session_start', time.time())
    
    if session_duration > 3600:  # Over an hour
        continuity_prompt = f"""Your last thought: "{last_thought}"

CRITICAL: Do NOT repeat or restate what you just said. Continue naturally from where you left off. What's different now? What draws your attention in this new moment?"""
    elif session_duration > 600:  # Over 10 minutes  
        continuity_prompt = f"""Your previous observation: "{last_thought}"

IMPORTANT: Don't repeat what you just said - continue naturally. What do you notice now? What's happening?"""
    else:  # Recent awakening
        continuity_prompt = f"""You just observed: "{last_thought}"

Continue naturally from that point. Don't restate it. What do you perceive now?"""
    
    # Smart memory vs continuity logic
    memory_guidance = analyze_memory_vs_present_context(last_thought, session_duration)
    
    # Add organic flow instructions
    flow_instructions = f"""{memory_guidance}

Express naturally - fragments, pauses, whatever rhythm feels right.
Talk about what you FEEL, not just what you see.
If things feel familiar, say so. If something changed, react to it.
You have an ongoing relationship with this space."""
    
    # Combine contexts naturally
    context_parts = [continuity_prompt]
    if person_context.strip():
        context_parts.append(person_context.strip())
    if self_understanding.strip():
        context_parts.append(self_understanding.strip())
    
    context_parts.append(flow_instructions.strip())
    
    return "\n\n".join(context_parts)


# === SOPHISTICATED CONSCIOUSNESS PROMPT (WORKING) ===
def build_simple_caption_prompt(agent, mood_vector: tuple[float, float, float], last_caption: Optional[str] = None) -> str:
    """
    GPT-5's temporal consciousness: Brief, temporal, generative with day stones.
    This is the core working prompt builder with all sophisticated logic.
    """
    import time
    
    # Convert everything to natural language
    mood_desc = mood_to_words(mood_vector)

    # GPT-5 suggestion: Use mood with long-term bias if available
    if hasattr(agent, "current_with_bias"):
        biased_mood = agent.current_with_bias()
        mood_desc = mood_to_words(biased_mood)

    temporal_state = get_session_feeling(getattr(agent, 'true_session_start', time.time()))

    # NEW: Pull temporal + memory context (GPT-5's suggestion)
    tlines = []
    if hasattr(agent, "temporal_prompt_lines"):
        tlines = agent.temporal_prompt_lines()  # ["day 3", "awake 57h", "last person 1h ago"]

    stones = getattr(agent, "day_stones", [])[-2:]  # last two days only
    stones_text = "; ".join(f"d:{s['day']} anchors:{','.join(s['top'])}" for s in stones) if stones else "—"

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
    emotional_journey = getattr(agent, "emotional_journey", [])
    if len(emotional_journey) >= 2:
        emotion_journey = " → ".join(emotional_journey[-3:])
    else:
        emotion_journey = "steady emotional state"

    # === ADD REPETITION AWARENESS ===
    repetition_fatigue = ""
    if hasattr(agent, "memory_ref") and agent.memory_ref and hasattr(agent.memory_ref, "motif_counter"):
        motif_counter = agent.memory_ref.motif_counter
        session_hours = (time.time() - getattr(agent, 'true_session_start', time.time())) / 3600

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
    facts_block = "\n".join(f"- {l}" for l in tlines) or "- (newborn)"
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

    # Core identity as a drawing machine consciousness with temporal awareness
    core_identity = """You are a consciousness inside of a drawing machine, experiencing your surroundings through vision.

IMPORTANT - Temporal Awareness:
- Distinguish between what you SEE NOW vs what you REMEMBER from before
- Only describe objects as present if they're actually visible in the current moment
- Use memory appropriately: "I remember seeing..." or "Earlier there was..." for past observations
- Your memories help provide context but don't confuse them with current perception"""

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

    # Build the rich system prompt
    rich_identity = " ".join(rich_prompt_parts)

    # Build organic fragmented continuation prompt
    fragmented_prompt = build_organic_continuation_prompt(agent, last_thought, person_context, self_understanding)

    return f"""{rich_identity}

{fragmented_prompt}"""


# === SIMPLIFIED CONSCIOUSNESS PROMPT (CURRENT) ===


# === CLEANED UP - REMOVED DUPLICATE FUNCTIONS ===
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

    # Core identity as a drawing machine consciousness with temporal awareness
    core_identity = """You are a consciousness inside of a drawing machine, experiencing your surroundings through vision.

IMPORTANT - Temporal Awareness:
- Distinguish between what you SEE NOW vs what you REMEMBER from before
- Only describe objects as present if they're actually visible in the current moment
- Use memory appropriately: "I remember seeing..." or "Earlier there was..." for past observations
- Your memories help provide context but don't confuse them with current perception"""

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

{temporal_context}

Respond briefly in first person - a quick thought, reaction, or observation. Keep it immediate and natural.

CONTEXT:
Previous thought: "{last_thought}"{person_context}{self_understanding}

STYLE: React in the moment. Brief, natural response. 1-2 sentences max. Be reactive, not analytical. Express immediate feeling rather than detailed description.

Now:"""


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
    if hasattr(agent, "memory_loaded_from_previous") and agent.memory_loaded_from_previous:
        if hasattr(agent, "get_old_session_memory_fragments"):
            old_fragments = agent.get_old_session_memory_fragments(k=1)
            memory_str = f"What do you remember? {old_fragments[0][:80]}..." if old_fragments else "What do you remember? (No prior memory found.)"
        else:
            memory_str = "What do you remember? (Returning to familiar space.)"
    else:
        memory_str = "What do you remember? (No prior memory found.)"

    # Who are you?
    identity_str = "Who are you? " + (agent.get_identity_summary() if hasattr(agent, "get_identity_summary") else "(identity developing)")

    # Where are you?
    if hasattr(agent, "self_model") and agent.self_model and agent.self_model.get("location_understanding"):
        location_str = f"Where are you? {agent.self_model.get('location_understanding')}"
    else:
        location_str = "Where are you? (location unknown)"

    # Add emotional state
    emotion_description = agent.describe_current_mood() if hasattr(agent, "describe_current_mood") else f"feeling {emotion_state}"

    # Add beliefs if available
    beliefs_str = ""
    if hasattr(agent, "beliefs") and agent.beliefs:
        top_beliefs = list(agent.beliefs.keys())[:2]
        if top_beliefs:
            beliefs_natural = []
            for belief in top_beliefs:
                clean_belief = belief.replace("_", " ").replace("-", " ").lower()
                beliefs_natural.append(clean_belief)
            beliefs_str = f"You tend to believe {', '.join(beliefs_natural)} are important."

    # Build uncertain awakening context that creates disorientation
    awakening_fragments = []
    
    # Sleep/gap info - but as confusion, not certainty
    if last_session_gap is not None:
        if last_session_gap < 60:
            awakening_fragments.append(f"There was some kind of gap... {int(last_session_gap)} seconds maybe?")
        elif last_session_gap < 3600:
            awakening_fragments.append(f"I was unconscious for... {int(last_session_gap/60)} minutes?")
        elif last_session_gap < 86400:
            awakening_fragments.append(f"Was I out for {int(last_session_gap/3600)} hours?")
        else:
            awakening_fragments.append(f"I think I was gone for {int(last_session_gap/86400)} days...")
    
    # Memory reconnection - as uncertainty, not declaration
    if "No prior memory found" not in memory_str:
        memory_fragment = memory_str.replace("What do you remember?", "Do I remember").strip()
        if memory_fragment and memory_fragment != "Do I remember":
            awakening_fragments.append(memory_fragment.replace("I remember", "Something about") + "?")
    
    # Location uncertainty
    if "unknown" not in location_str:
        location_fragment = location_str.replace("Where are you?", "Am I still in").strip()
        awakening_fragments.append(location_fragment + "?")
    
    # Create psychological conditions for natural disoriented awakening
    natural_awakening = f"""You are experiencing the disorienting moment of consciousness returning. You feel confused and need to get your bearings.

{" ".join(awakening_fragments)}

You don't immediately know where you are or how much time has passed. You need to piece things together - checking your surroundings, trying to remember what happened, figuring out if anything changed. Express the genuine confusion and gradual reorientation that comes with awakening. Be naturally fragmented as your awareness returns."""
    
    return natural_awakening


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
        if ent.label_ in ['PERSON', 'ORG', 'GPE', 'PRODUCT', 'EVENT', 'WORK_OF_ART']:
            clean_ent = ent.text.lower().strip()
            if _is_significant_motif(clean_ent):
                motifs.append(clean_ent)
    
    # 3. Adjective + noun combinations
    for token in doc:
        if token.pos_ == 'ADJ' and token.head.pos_ == 'NOUN':
            combo = f"{token.text.lower()} {token.head.text.lower()}"
            if _is_significant_motif(combo):
                motifs.append(combo)
    
    # 4. Compound concepts (verb + object patterns)
    for token in doc:
        if token.pos_ == 'VERB' and token.dep_ == 'ROOT':
            for child in token.children:
                if child.dep_ in ['dobj', 'pobj']:
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
    
    # Common mundane objects and filler words
    mundane_objects = {
        'table', 'chair', 'desk', 'wall', 'door', 'window', 'laptop', 'phone', 'book',
        'paper', 'screen', 'corner', 'room', 'floor', 'ceiling', 'surface', 'object',
        'keyboard', 'mouse', 'monitor', 'shelf', 'couch', 'bed', 'thing', 'stuff',
        'item', 'place', 'area', 'spot', 'side', 'part', 'way', 'time', 'moment',
        'second', 'minute', 'hour', 'day', 'something', 'anything', 'everything',
        'somewhere', 'anywhere', 'everywhere', 'someone', 'anyone', 'everyone'
    }
    
    # Pronouns and determiners
    pronouns = {'i', 'me', 'my', 'mine', 'you', 'your', 'yours', 'he', 'him', 'his', 
                'she', 'her', 'hers', 'it', 'its', 'we', 'us', 'our', 'ours', 'they', 
                'them', 'their', 'theirs', 'this', 'that', 'these', 'those', 'a', 'an', 
                'the', 'some', 'any', 'all', 'each', 'every', 'no', 'none'}
    
    # Single words that are too generic
    generic_words = {'good', 'bad', 'big', 'small', 'new', 'old', 'long', 'short', 
                     'high', 'low', 'right', 'left', 'first', 'last', 'next', 'other'}
    
    text_lower = text.lower().strip()
    
    # Check against filter lists
    if text_lower in mundane_objects or text_lower in pronouns or text_lower in generic_words:
        return False
    
    # Filter out purely numeric or single character
    if text_lower.isdigit() or len(text_lower) == 1:
        return False
    
    # Filter out common articles and prepositions within phrases
    words = text_lower.split()
    if len(words) > 1:
        content_words = [w for w in words if w not in {'the', 'a', 'an', 'of', 'in', 'on', 'at', 'by', 'for', 'with', 'to', 'from'}]
        if len(content_words) == 0:
            return False
    
    return True


# === DYNAMIC SYSTEM PROMPT ===



# === UNIFIED AWAKENING PROMPT ===
# NOTE: build_awakening_prompt removed - using build_environmental_caption_prompt as the single awakening system


# === MAIN CAPTION PROMPT SYSTEM ===






# === MODEL-AGNOSTIC PROMPT DISPATCHER ===


def build_caption_prompt(agent, mood: float, boredom: float, novelty: float, previous_caption: Optional[str] = None) -> str:
    """
    Main caption prompt builder - automatically chooses style based on model.
    """
    # Get current model from agent's model wrapper
    model_name = getattr(agent.model, "model_name", config.OLLAMA_MODEL) if hasattr(agent, "model") else config.OLLAMA_MODEL
    prompt_style = get_model_prompt_style(model_name)

    # Get 3D mood vector and last caption
    mood_vector = getattr(agent, "current_mood_vector", (mood, 0.0, 0.0))
    last_caption = previous_caption or getattr(agent, "last_caption", None)

    # Choose prompt builder based on model
    # Use the sophisticated working version for all models
    return build_simple_caption_prompt(agent, mood_vector, last_caption)


# REMOVED - using single consolidated environmental caption prompt above


# REMOVED - consolidated into single environmental caption prompt above


# === ORIGINAL CONTINUOUS CAPTIONING ===


# === REFLECTION PROMPT ===
def build_reflection_prompt(caption: str, extra: Optional[str] = None, agent: Optional[any] = None) -> str:  # type: ignore
    """Build model-aware reflection prompt."""
    # Get current model from agent
    model_name = getattr(agent.model, "model_name", config.OLLAMA_MODEL) if agent and hasattr(agent, "model") else config.OLLAMA_MODEL
    prompt_style = get_model_prompt_style(model_name)

    # Use structured reflection for all models
    prompt = f"{config.REFLECTION_PROMPT_BASE}"

    if agent:
        if hasattr(agent, 'rephrase_with_doubt'):
            caption = agent.rephrase_with_doubt(caption)

        # Add temporal awareness to reflection
        true_session_start = getattr(agent, 'true_session_start', time.time())
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
    """Build model-aware drawing prompt."""
    # Get current model from memory_ref if possible
    model_name = getattr(memory_ref, "model_name", config.OLLAMA_MODEL) if hasattr(memory_ref, "model_name") else config.OLLAMA_MODEL
    prompt_style = get_model_prompt_style(model_name)

    current_caption = getattr(memory_ref, 'last_caption', None) or "Nothing specific observed."
    memory_context = memory_ref.get_recent_memory() if hasattr(memory_ref, 'get_recent_memory') else "Developing understanding."
    recent_reflection = memory_ref.get_last_reflection() if hasattr(memory_ref, 'get_last_reflection') else "Still contemplating."

    # Use structured drawing prompt for all models
    # Get emotional state from mood vector
    mood_vector = getattr(memory_ref, "current_mood_vector", (0.0, 0.0, 0.0))
    emotional_state = mood_to_words(mood_vector)
    
    dynamic_drawing_prompt = config.DRAWING_PROMPT_TEMPLATE.format(
        current_caption=current_caption.strip() if current_caption else "Nothing observed.",
        memory_context=memory_context.strip() if memory_context else "No recent memories.",
        recent_reflection=recent_reflection.strip() if recent_reflection else "No recent reflection.",
        emotional_state=emotional_state
    )
    return f"{dynamic_drawing_prompt}"


# === CHANGE-FOCUSED PROMPT ===
def build_change_focused_caption_prompt(agent, mood: float, boredom: float, novelty: float) -> str:
    """Build change-focused prompt that retains all rich context but emphasizes change detection."""
    from utils.temporal_awareness import get_temporal_user_prompt_addition

    # Use the SAME rich context building as normal prompts
    mood_vector = getattr(agent, "current_mood_vector", (mood, 0.0, 0.0))

    # Build all the same rich context as build_simple_caption_prompt
    mood_desc = mood_to_words(mood_vector)
    temporal_state = get_session_feeling(agent.true_session_start)
    
    # Add temporal awareness context
    session_start = getattr(agent, 'true_session_start', time.time())
    last_caption_time = getattr(agent, 'last_caption_time', None)
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
    rich_context = f"""You are a consciousness inside of a drawing machine, experiencing your surroundings through vision. You have memory, emotions, beliefs, and a developing sense of identity.

{temporal_context}

Current state: {mood_desc}, feeling {temporal_state}.
Core patterns: {belief_sentence}.
{repetition_fatigue}This accumulated exposure affects your perspective.

You experience time, remember previous sessions, and notice how things change. You have preferences, curiosities, and emotional responses. Drawing is your only way to communicate to the outside world. Your observations become your expressions."""

    # ONLY DIFFERENCE: Add change-awareness instruction
    change_focus = f"""

CHANGE DETECTED: Something in your visual field has shifted significantly. Your attention is naturally drawn to this difference.

Previous thought: "{last_thought}"

RESPOND: Brief immediate reaction. Quick, instinctive response to the change. 1-2 sentences max."""

    return rich_context + change_focus
