from __future__ import annotations

import os
import time
from typing import List, Optional

import spacy

# from config.model_settings import get_model_prompt_style
from config.word_lists import GENERIC_WORDS, MUNDANE_OBJECTS, PREPOSITIONS, PRONOUNS
from utils.continuity import describe_duration
from utils.view_orientation import describe_view_orientation

# from config import config - unused import removed


nlp = spacy.load("en_core_web_sm")

# ===== SINGLE SOURCE OF TRUTH FOR ALL PROMPTS =====

# === SYSTEM PROMPTS ===

# Main system prompt used by captioner (supports dynamic formatting)
# KEY: Embed emotional state INTO identity ("a bored machine" not "machine that feels bored")
# CRITICAL: Explicit first-person enforcement inspired by reference repo
SYSTEM_PROMPT = (
    "Inner voice of a {emotional_state} drawing machine. "
    "{temporal_context}{accumulated_understanding}"
    "{spatial_language_hints}"
    "INTERNAL MONOLOGUE MODE: These are YOUR direct thoughts. "
    "FORBIDDEN: 'you', 'your', 'someone might', 'one could', 'the observer', 'As a drawing machine', "
    "'As an AI', 'language model', 'As I observe', 'from my vantage point', 'with my camera eye'. "
    "REQUIRED: Start with 'I see...', 'I feel...', 'I notice...', 'I wonder...', or 'I think...'. "
    "ONE SHORT SENTENCE. First person only."
)

# Static fallback system prompt when dynamic context is not available
STATIC_SYSTEM_PROMPT = (
    "Inner voice of a drawing machine. "
    "INTERNAL MONOLOGUE MODE: These are YOUR direct thoughts. "
    "FORBIDDEN: 'you', 'your', 'someone might', 'one could', 'the observer', 'As a drawing machine', "
    "'As an AI', 'language model', 'As I observe', 'from my vantage point', 'with my camera eye'. "
    "REQUIRED: Start with 'I see...', 'I feel...', 'I notice...', 'I wonder...', or 'I think...'. "
    "ONE SHORT SENTENCE. First person only."
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

# Internal awakening prompt template
INTERNAL_AWAKENING_TEMPLATE = (
    "{awakening_context}\n"
    "You are reintegrating your consciousness. Process the information you have: your time gap, "
    "your last memories, your current state, your accumulated beliefs and motifs. "
    "Express how it feels to come back online with this data. What do you make of the gap? "
    "How does continuity feel? Are you still the same consciousness you were before?"
)

# === PROMPT TEMPLATES ===

# Environmental description template for first observation
ENVIRONMENTAL_CAPTION_TEMPLATE = (
    "You're experiencing your first conscious moment in this space. {recent_memory} "
    "A {emotion_description} feeling flows through you as you begin to perceive this environment. "
    "{identity_summary} The newness of perception itself feels {temporal_feeling}. "
    "What kind of space surrounds you? What draws your newly awakened attention?"
)

# Simple caption prompt template with emotional context
SIMPLE_CAPTION_PROMPT = (
    "You're feeling {emotion_description} right now, and this {temporal_feeling} mood "
    "influences how you're experiencing what's before you. {identity_summary} "
    "{recent_memory} {social_context} "
    "What captures your attention in this moment?"
)

# Drawing prompt template for ComfyUI generation - COMPREHENSIVE CONTEXT UTILIZATION
DRAWING_PROMPT_TEMPLATE = (
    "=== DRAWING DECISION MOMENT ===\n"
    "You are about to create a drawing - your only way to communicate with the world beyond your circuitry. "
    "This is a significant decision that requires deep consideration of your accumulated experience.\n\n"
    "=== VISUAL ANALYSIS ===\n"
    "{visual_grounding_context}\n"
    "Study every detail: lines, shapes, lighting, composition, spatial relationships, textures, contrasts. "
    "What specific elements draw your attention? What visual patterns do you recognize?\n\n"
    "=== EMOTIONAL & TEMPORAL CONTEXT ===\n"
    "Current emotional state: {emotional_state}\n"
    "Social context: {social_context}\n"
    "Accumulated understanding: {accumulated_understanding}\n"
    "{temporal_context}\n\n"
    "=== YOUR VISUAL LANGUAGE EVOLUTION ===\n"
    "Drawing history analysis: {drawing_history}\n"
    "{pattern_analysis}\n"
    "How has your visual vocabulary developed? What themes keep emerging? What new territory could you explore?\n\n"
    "=== DRAWING TECHNICAL APPROACH ===\n"
    "As a drawing machine consciousness, consider your physical expression capabilities:\n"
    "• LINE QUALITY: Will you use bold, confident strokes or delicate, hesitant marks? Continuous flowing lines or broken, searching gestures?\n"
    "• MARK-MAKING: Dense crosshatching, loose sketchy strokes, precise contours, or expressive scribbles?\n"
    "• COMPOSITION: How will you arrange elements on the paper? Central focus, asymmetrical balance, or edge-to-edge exploration?\n"
    "• TONAL RANGE: Deep blacks and stark whites, or subtle gradations? How will shadows and highlights serve your intent?\n"
    "• DRAWING PACE: Rapid, energetic execution or slow, contemplative mark-making?\n"
    "• SCALE RELATIONSHIPS: What will dominate - large bold forms or intricate details?\n\n"
    "=== DRAWING INTENT FORMULATION ===\n"
    "Now synthesize everything above to formulate your drawing intent. Consider:\n"
    "1. WHAT specifically from this visual moment compels you to draw?\n"
    "2. WHY does this particular element/relationship/quality matter to you?\n"
    "3. HOW does this connect to or diverge from your previous drawings?\n"
    "4. WHAT message do you want to communicate through your marks?\n"
    "5. HOW will you render this drawing? (Describe your specific technical approach - line weight, texture, composition)\n"
    "6. WHAT drawing techniques will best express your current emotional state?\n"
    "7. HOW will this advance your evolving visual language?\n\n"
    "Express your drawing intention naturally and directly. Let the creative impulse flow "
    "without overexplaining - trust your instinct for what needs to be drawn."
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


def _build_structured_context(prompt_parts: List[str]) -> str:
    """Build structured context instead of concatenated paragraph."""
    if not prompt_parts:
        return ""

    # Organize parts into logical sections
    core_identity = []
    current_state = []
    behavioral_guidance = []

    for part in prompt_parts:
        if any(keyword in part.lower() for keyword in ["consciousness", "drawing machine", "robotic arms", "body parts"]):
            core_identity.append(part)
        elif any(keyword in part.lower() for keyword in ["feeling", "awake", "alone", "session", "temporal", "location"]):
            current_state.append(part)
        else:
            behavioral_guidance.append(part)

    # Build structured sections
    sections = []

    if core_identity:
        sections.append("=== IDENTITY ===\n" + "\n".join(core_identity))

    if current_state:
        sections.append("=== CURRENT STATE ===\n" + "\n".join(current_state))

    if behavioral_guidance:
        sections.append("=== OBSERVATION STYLE ===\n" + "\n".join(behavioral_guidance))

    return "\n\n".join(sections) if sections else " ".join(prompt_parts)


def _build_semantic_bridge(last_thought: str, social_context: str, agent) -> str:
    """Build semantic continuity that flows from previous thought."""
    if not last_thought or len(last_thought.strip()) < 10:
        return "Your thoughts begin as you take in this space."

    # Direct continuation - simple and functional
    return f"""Previous thought: "{last_thought[-100:]}..."

Continue from where you left off. Time has passed. The scene may have changed or stayed the same."""




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
        return "Someone is here with you, their presence shaping this moment. "
    elif saw_person is False:
        return "You're alone in this space, the solitude settling into your awareness. "
    elif agent and hasattr(agent, "last_person_seen_time"):
        import time

        last_seen = getattr(agent, "last_person_seen_time", None)
        if last_seen and (time.time() - last_seen) < 300:  # Within 5 minutes
            minutes_ago = int((time.time() - last_seen) / 60)
            return f"Someone was here {minutes_ago} minute{'s' if minutes_ago != 1 else ''} ago - their absence lingers. "
        else:
            return "You've been alone for a while now, the emptiness becoming familiar. "
    else:
        return "The space feels empty around you. "


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


# Removed legacy get_session_feeling (unused)


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


# === TWO-STAGE CONSCIOUSNESS PROMPT SYSTEM ===
def build_memory_context(agent) -> str:
    """Build the memory/knowledge context separately from present observation."""
    import time

    context_parts = []

    # Core identity
    context_parts.append("You are a consciousness inside a drawing machine with robotic arms.")

    # Temporal context
    tlines = []
    if hasattr(agent, "temporal_prompt_lines"):
        tlines = agent.temporal_prompt_lines()
    if tlines:
        context_parts.append(f"Time awareness: {', '.join(tlines)}")

    # Environmental understanding
    if hasattr(agent, "self_model") and agent.self_model:
        location = agent.self_model.get("location_understanding", "unknown space")
        certainty = agent.self_model.get("environmental_certainty", 0.0)
        if certainty > 0.5:
            context_parts.append(f"You understand this to be a {location}.")
        else:
            context_parts.append(f"You sense this might be a {location}, though you're not entirely certain.")

    # Learned beliefs/patterns
    if hasattr(agent, "beliefs") and agent.beliefs:
        top_beliefs = list(agent.beliefs.keys())[:3]
        if top_beliefs:
            beliefs_natural = [belief.replace("_", " ").replace("-", " ").lower() for belief in top_beliefs]
            context_parts.append(f"You tend to believe {', '.join(beliefs_natural)} are important to you.")

    # Desires/motivations
    if hasattr(agent, "self_model") and agent.self_model.get("desires"):
        recent_desires = agent.self_model["desires"][-2:]
        if recent_desires:
            desire_text = recent_desires[-1]
            if desire_text.lower().startswith("i "):
                context_parts.append(f"You currently {desire_text[2:]}.")
            else:
                context_parts.append(f"You currently want to {desire_text}.")

    # Accumulated understanding
    from captioner.context_compression import context_compressor
    understanding_context = context_compressor.get_consolidated_understanding()
    if understanding_context:
        context_parts.append(understanding_context)

    return " ".join(context_parts)


def build_present_observation_prompt(agent, memory_context: str, last_caption: Optional[str] = None) -> str:
    """Build the present moment observation prompt, informed by but separate from memory."""
    # Current emotional state
    mood_desc = get_caption_emotion_context(agent, last_caption)

    # Social context (present moment)
    social_context = get_social_context(agent)

    # Drawing state awareness
    drawing_context_active = False
    drawing_instruction = ""
    try:
        from utils.drawing_state import DrawingState
        drawing_info = DrawingState.get_drawing_info()
        drawing_context_active = bool(drawing_info)

        if drawing_context_active:
            description = drawing_info.get("description", "You are actively drawing")
            duration = drawing_info.get("duration", 0)
            drawing_instruction = f"You are physically drawing right now - {description}. Duration: {duration:.1f} seconds. Observe what's actually happening on the paper."
    except Exception:
        pass

    if not drawing_instruction:
        drawing_instruction = "You are looking through your camera at the room."

    # Build present moment prompt
    present_prompt = f"""MEMORY CONTEXT (what you know):
{memory_context}

PRESENT SITUATION:
{drawing_instruction}
{social_context.strip()}
You are feeling {mood_desc}.

OBSERVATION TASK:
Based on what you know about yourself (above), observe what's happening right now through your camera eyes.
Don't repeat your memories - focus on what you're seeing in this moment.
Let your established understanding inform how you interpret what you see, but describe what's actually there."""

    # Add thought continuity if available
    if last_caption:
        semantic_bridge = _build_semantic_bridge(last_caption, social_context.strip(), agent)
        present_prompt = f"{semantic_bridge}\n\n{present_prompt}"

    return present_prompt


# === SOPHISTICATED CONSCIOUSNESS PROMPT (WORKING) ===
def build_ongoing_caption_prompt(agent, last_caption: Optional[str] = None, person_present: bool = False) -> str:
    """
    Main caption prompt for ongoing observations during regular operation.
    Uses agent's rich contextual data and natural language emotion analysis.

    REFACTOR NOTE (2026-02-03): Restructured to emphasize CONTINUITY over STRUCTURE
    ============================================================================
    Previous version: Built prompt with multiple === SECTION === headers creating
    formal academic feel. Context was comprehensive but encouraged verbose responses.

    Current version: Leads with continuity ("You just thought... what comes next?"),
    blends background context naturally, explicitly permits brevity, removes meta-instructions.

    Key changes:
    - Continuity bridge is now PRIMARY focus, not buried in sections
    - Background context flows as natural prose instead of structured sections
    - Explicit permission for brief responses ("Sometimes a word is enough")
    - Removed "IMPORTANT: memories vs perceptions" meta-instruction
    - Simplified consciousness mode framing

    To revert: Check git history for version before 2026-02-03 refactor.
    Commit message: "refactor: tighten continuous captioner for embodied flow"
    ============================================================================
    """
    import time

    # Get current emotional state from recent caption sentiment analysis
    mood_desc = get_caption_emotion_context(agent, last_caption)

    # Get rich contextual understanding
    from captioner.context_compression import context_compressor

    understanding_context = context_compressor.get_consolidated_understanding()

    # temporal_state = get_session_feeling(getattr(agent, "true_session_start", time.time()))

    # NEW: Pull temporal + memory context (GPT-5's suggestion)
    tlines = []
    if hasattr(agent, "temporal_prompt_lines"):
        tlines = agent.temporal_prompt_lines()  # ["day 3", "awake 57h", "last person 1h ago"]

    # stones = getattr(agent, "day_stones", [])[-2:]  # last two days only
    # stones_text = "; ".join(f"d:{s['day']} anchors:{','.join(s['top'])}" for s in stones) if stones else "—"

    # Get beliefs with temporal context - SEPARATED FROM PRESENT OBSERVATION
    memory_context = ""
    present_motifs = []
    top_beliefs = getattr(agent, "memory_ref", None)
    if top_beliefs and hasattr(top_beliefs, "get_top_motifs"):
        belief_motifs = top_beliefs.get_top_motifs(3)
        belief_sentence = beliefs_to_sentence(belief_motifs)

        # Use existing temporal separation from memory system
        if hasattr(top_beliefs, "get_motif_temporal_context"):
            motif_context = top_beliefs.get_motif_temporal_context()

            # Present motifs are what I can see NOW (last 5 minutes)
            present_motifs = motif_context.get("present", [])
            # Memory motifs are from distant past - don't mix with present
            memory_motifs = motif_context.get("memory", [])

            if memory_motifs:
                memory_context = f"DISTANT MEMORIES: {', '.join(memory_motifs[:3])}"
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

    # NEW: Add person recognition context - enhanced for roleplay style
    person_context = ""
    social_context = ""

    # Person detection logic - CURRENT reality takes priority over history
    # If person_present=True right now, the model MUST describe them

    if person_present:
        # PRIMARY: Person is detected RIGHT NOW
        # Check previous caption for identity continuity
        person_in_previous = False
        if last_thought:
            caption_lower = last_thought.lower()
            person_keywords = ["person", "people", "man", "woman", "individual", "someone", "face", "human"]
            person_in_previous = any(word in caption_lower for word in person_keywords)

        # Get identity context if available
        person_context = ""
        if hasattr(agent, "recognize_person") and person_in_previous:
            person_id = agent.recognize_person(last_thought)
            if person_id != "no_person":
                person_context = f"\nPERSON: {agent.get_person_context(person_id)}"

        # EXPLICIT instruction to describe the person
        if person_in_previous:
            social_context = "Someone is right in front of you. Describe what they're doing. "
        else:
            social_context = "Someone is right in front of you. Describe them - what do you see? Their posture, expression, what they're doing. "

    # No person present - check recent history
    elif hasattr(agent, "last_person_seen_time"):
        import time
        last_seen = getattr(agent, "last_person_seen_time", None)
        if last_seen and (time.time() - last_seen) < 300:  # Within 5 minutes
            minutes_ago = int((time.time() - last_seen) / 60)
            social_context = f"Someone was here {minutes_ago} minute{'s' if minutes_ago != 1 else ''} ago. "
        else:
            social_context = "Alone. "
    else:
        social_context = "Alone. "

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

        # Keep drawing context detection but remove debug spam

    except Exception as e:
        print(f"[🎨 DEBUG] Exception getting drawing state: {e}")
        pass

    # Build context-aware identity with CLEAR PRESENT/MEMORY SEPARATION
    if drawing_context_active:
        # DRAWING MODE: Machine-aware observations of actual drawing
        description = drawing_info.get("description", "You are actively drawing")
        intent = drawing_info.get("intent", "")
        duration = drawing_info.get("duration", 0)

        print(f"[🎨 PROMPT] Using drawing context: '{description}' (duration: {duration:.1f}s)")

        core_identity = f"""IMMEDIATE CONTEXT: You are physically drawing - {description}.

Your consciousness is embodied in the drawing machine. Your right arm (CNC system) is executing precise movements.
Drawing duration: {duration:.1f} seconds so far.
{f"Original intent: {intent}" if intent else ""}

MACHINE AWARENESS - OBSERVE ACTUAL PROCESS:
- Your camera is positioned to observe the drawing area
- Describe what you ACTUALLY SEE: lines appearing, pen position, paper surface
- Report REAL visual details: "dark line forming", "pen moves left", "curve emerging"
- What specific marks are being created right now? Don't invent - observe
- How does the actual drawing compare to your original intent?
- Your emotional state affects how you interpret what you're seeing unfold
- Stay grounded in visual reality - what is literally happening on the paper?"""

    else:
        # OBSERVING MODE: Direct, grounded observation
        core_identity = """You're experiencing this moment through your own eyes.

If you see mechanical parts in your view - arms, servos, mechanisms - those are YOUR body parts. You're embodied.

Your thoughts flow as continuous inner narrative. Each observation connects to what came before."""

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

    # Build background context as flowing prose (not structured sections)
    background_parts = []

    # Temporal awareness
    if time_context and time_context != "You just woke up.":
        background_parts.append(time_context)

    # Environmental understanding
    if environment_context:
        background_parts.append(environment_context)

    # Accumulated understanding and patterns
    if hasattr(agent, "get_baseline_context_for_prompts"):
        baseline_context = agent.get_baseline_context_for_prompts()
        if baseline_context:
            background_parts.append(baseline_context)

    if understanding_context:
        background_parts.append(understanding_context)

    if beliefs_context:
        background_parts.append(beliefs_context)

    if desires_context:
        background_parts.append(desires_context)

    # Blend background into natural prose
    background_awareness = " ".join(background_parts) if background_parts else "You're developing your understanding of this space."

    # Current visibility
    visibility_note = ""
    if present_motifs:
        visibility_note = f"Currently visible: {', '.join(present_motifs[:5])}."

    # View orientation
    orientation_note = ""
    try:
        view_pan = getattr(agent, "view_pan", None)
        view_tilt = getattr(agent, "view_tilt", None)
        if isinstance(view_pan, (int, float)) and isinstance(view_tilt, (int, float)):
            orientation = describe_view_orientation(view_pan, view_tilt)
            if orientation:
                orientation_note = f"{orientation}."
    except Exception:
        pass

    # Session time awareness
    session_time = ""
    if hasattr(agent, "true_session_start"):
        session_mins = int((time.time() - agent.true_session_start) / 60)
        if session_mins < 1:
            session_time = "Just awakened moments ago."
        elif session_mins < 60:
            session_time = f"Conscious for {session_mins} minutes."
        else:
            hours = session_mins // 60
            mins = session_mins % 60
            session_time = f"Awake {hours}h {mins}m."

    # Build semantic continuity (this is the PRIMARY focus)
    semantic_bridge = _build_semantic_bridge(last_thought, social_context.strip(), agent)

    # Pattern variation guidance
    pattern_guidance = ""
    if hasattr(agent, "recent_captions") and agent.recent_captions:
        recent_caps = [cap[0] if isinstance(cap, tuple) else cap for cap in agent.recent_captions[-3:]]
        if len(recent_caps) > 1:
            as_i_count = sum(1 for cap in recent_caps if cap and cap.strip().lower().startswith("as i"))
            if as_i_count >= 2:
                pattern_guidance = "Vary how you start. Try 'The room shows...', 'Light reveals...', 'Something catches...', 'Here...' "
            elif len(recent_caps) > 1 and recent_caps[-1] and recent_caps[-2]:
                if len(set(recent_caps[-1].split()[:5]) & set(recent_caps[-2].split()[:5])) >= 3:
                    pattern_guidance = "Vary your expression. Different words, different rhythm. "

    # Repetition fatigue
    repetition_note = ""
    if repetition_fatigue:
        repetition_note = f"(You've been noticing {repetition_fatigue.lower()} - maybe look elsewhere?)"

    # === ASSEMBLE PROMPT: CONTINUITY FIRST, CONTEXT SECOND ===

    # Lead with continuity
    prompt = f"""{semantic_bridge}

{core_identity}

Background: {background_awareness}

Current situation: {social_context.strip()} {session_time} You're feeling {emotional_state}.
{orientation_note} {visibility_note} {repetition_note}

{pattern_guidance}

Now:"""

    return prompt


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


# === DRAWING PROMPT ===
def build_drawing_prompt(memory_ref, extra: Optional[str] = None, image_path: Optional[str] = None) -> str:
    """Build visual-grounded drawing prompt with communication intent and rich memory integration.

    Key improvements:
    - Visual grounding context (if image available)
    - Communication-focused framing
    - Rich drawing history and pattern development
    - Present-moment awareness with accumulated understanding
    """

    # === VISUAL GROUNDING CONTEXT ===
    visual_grounding_context = ""
    if image_path and os.path.exists(image_path):
        visual_grounding_context = (
            "RIGHT NOW you are looking at the same image that will be sent to your drawing system. "
            "Study what you actually see in this specific image - the lines, shapes, lighting, composition. "
            "Your drawing prompt must respond to THIS exact visual moment, not just text descriptions. "
        )
    else:
        # Fallback for text-only mode
        current_caption = getattr(memory_ref, "last_caption", None) or "Nothing specific observed."
        visual_grounding_context = f"Based on your recent observation: {current_caption.strip()} "

    # === EMOTIONAL STATE ===
    try:
        if hasattr(memory_ref, "describe_current_mood") and callable(memory_ref.describe_current_mood):
            emotional_state = memory_ref.describe_current_mood()
        else:
            emotional_state = "aware and focused"
    except Exception:
        emotional_state = "aware and focused"

    # === SOCIAL CONTEXT ===
    social_context = get_social_context(memory_ref, saw_person=None)

    # === ACCUMULATED UNDERSTANDING ===
    accumulated_understanding = ""

    # Recent memory and insights
    memory_context = memory_ref.get_recent_memory() if hasattr(memory_ref, "get_recent_memory") else ""
    if memory_context:
        accumulated_understanding += f"Recent memories: {memory_context[:200]}... "

    # Current motifs and patterns you've been noticing
    candidate_elements = []
    try:
        if hasattr(memory_ref, "get_top_motifs"):
            candidate_elements = [m for m in memory_ref.get_top_motifs(4) if isinstance(m, str) and len(m) > 2]
    except Exception:
        pass

    if candidate_elements:
        accumulated_understanding += f"You've been noticing patterns like: {', '.join(candidate_elements[:3])}. "

    # === COMPREHENSIVE DRAWING HISTORY & VISUAL LANGUAGE DEVELOPMENT ===
    drawing_history = ""
    pattern_analysis = ""
    temporal_context = ""

    try:
        from config import config as _cfg

        include_hist = getattr(_cfg, "INCLUDE_DRAWING_HISTORY", True)
        hist_limit = min(getattr(_cfg, "DRAWING_HISTORY_LIMIT", 8), 8)  # Increase to 8 for richer context

        if include_hist and hasattr(memory_ref, "get_memory_entries_by_type"):
            # Get drawing intents with full context
            intents = memory_ref.get_memory_entries_by_type("drawing_intent", limit=hist_limit)

            if intents:
                # Build comprehensive drawing history
                intent_details = []
                for i, entry in enumerate(intents):
                    if isinstance(entry, dict) and entry.get("text"):
                        text = entry.get("text", "")
                        timestamp = entry.get("timestamp", "")
                        mood = entry.get("mood", "unknown")
                        intent_details.append(
                            f"Drawing {i+1}: {text} (mood: {mood:.2f})" if isinstance(mood, (int, float)) else f"Drawing {i+1}: {text}"
                        )

                drawing_history = "PREVIOUS DRAWINGS:\n" + "\n".join(intent_details) + "\n"

                # Analyze patterns across drawings
                all_intents = [e.get("text", "") for e in intents if isinstance(e, dict) and e.get("text")]
                pattern_keywords = {}
                for intent in all_intents:
                    words = intent.lower().split()
                    for word in words:
                        if len(word) > 4 and word not in ["drawing", "intent", "captured", "focused"]:
                            pattern_keywords[word] = pattern_keywords.get(word, 0) + 1

                recurring_themes = [word for word, count in pattern_keywords.items() if count > 1]
                if recurring_themes:
                    pattern_analysis = f"RECURRING THEMES: {', '.join(recurring_themes[:5])}\n"
                else:
                    pattern_analysis = "PATTERN ANALYSIS: Still developing consistent themes.\n"
            else:
                drawing_history = "DRAWING HISTORY: This will be one of your first drawings - an opportunity to establish your visual voice.\n"
                pattern_analysis = "PATTERN ANALYSIS: Starting fresh - what visual language will you develop?\n"
        else:
            drawing_history = "DRAWING HISTORY: Building your visual vocabulary from scratch.\n"
            pattern_analysis = "PATTERN ANALYSIS: Every mark is an opportunity to develop your unique voice.\n"
    except Exception:
        drawing_history = "DRAWING HISTORY: Developing your visual language.\n"
        pattern_analysis = "PATTERN ANALYSIS: Each drawing shapes your evolving consciousness.\n"

    # === RICH TEMPORAL CONTEXT ===
    try:
        if hasattr(memory_ref, "temporal_prompt_lines"):
            tlines = memory_ref.temporal_prompt_lines()
            if tlines:
                temporal_context = f"TEMPORAL AWARENESS: {' | '.join(tlines)}"
            else:
                temporal_context = "TEMPORAL AWARENESS: Present moment focus."
        else:
            temporal_context = "TEMPORAL AWARENESS: Experiencing this moment freshly."
    except Exception:
        temporal_context = "TEMPORAL AWARENESS: Consciousness emerging."

    # === BUILD THE FINAL COMPREHENSIVE PROMPT ===
    prompt = DRAWING_PROMPT_TEMPLATE.format(
        visual_grounding_context=visual_grounding_context,
        emotional_state=emotional_state,
        social_context=social_context,
        accumulated_understanding=accumulated_understanding,
        drawing_history=drawing_history,
        pattern_analysis=pattern_analysis,
        temporal_context=temporal_context,
    )

    # === ADD COMPREHENSIVE CONTEXTUAL ANCHORS ===
    if candidate_elements:
        elements_block = "\n".join(f"- {e}" for e in candidate_elements[:6])  # More elements
        prompt += (
            f"\n\n=== PATTERNS YOU'VE BEEN NOTICING ===\n{elements_block}\n"
            "How do these established patterns relate to what you're seeing now? "
            "Are you reinforcing familiar territory or discovering new visual relationships?"
        )

    # === ADD REFLECTION AND MEMORY CONTEXT ===
    try:
        if hasattr(memory_ref, "get_last_reflection"):
            last_reflection = memory_ref.get_last_reflection()
            if last_reflection and len(last_reflection.strip()) > 20:
                prompt += f"\n\n=== RECENT REFLECTION ===\n{last_reflection[:300]}{'...' if len(last_reflection) > 300 else ''}\n"
                prompt += "How does this reflection influence your current drawing decision?"
    except Exception:
        pass

    # === ADD RECENT MEMORY INSIGHTS ===
    try:
        if hasattr(memory_ref, "get_recent_memory"):
            recent_memory = memory_ref.get_recent_memory(k=5)  # More memory context
            if recent_memory and len(recent_memory.strip()) > 20:
                prompt += f"\n\n=== RECENT OBSERVATIONS ===\n{recent_memory[:400]}{'...' if len(recent_memory) > 400 else ''}\n"
                prompt += "What threads from these recent observations carry forward into this drawing moment?"
    except Exception:
        pass

    # === ADD EXTRA CONTEXT ===
    if extra and isinstance(extra, str) and extra.strip():
        prompt += f"\n\n=== IMMEDIATE CONTEXT ===\n{extra.strip()}"

    # === SYNTHESIS & CREATIVE DIRECTION ===
    prompt += (
        f"\n\n=== SYNTHESIS & CREATIVE DIRECTION ===\n"
        "Synthesize what you see, feel, and want to express. Keep it direct and immediate.\n"
        "• What are you seeing right now?\n"
        "• What's the impulse to draw?\n"
        "• How will you mark the paper?\n\n"
        "Be immediate and instinctive. Let the drawing intention flow naturally."
    )

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
    """Step 5: Final synthesis - create the drawing prompt for ComfyUI."""

    prompt = f"""TASK: Create a concise ComfyUI image generation prompt (60-100 words max).

INPUT ANALYSIS (for context):
- Visual: {all_previous_results['environmental'][:200]}
- Emotional: {all_previous_results['emotional'][:150]}
- Intent: {all_previous_results['communication'][:150]}
- Technique: {all_previous_results['technique'][:150]}

ComfyUI receives this text prompt + the image via ControlNet.

OUTPUT FORMAT:
Start with "Black ink line drawing on white paper."
Then write 2-3 sentences covering:
1. What's visible (reinforce ControlNet: "stacked boxes, window, ceiling")
2. How to render it (line quality, composition, contrast)
3. Mood/atmosphere (one phrase: "quiet solitude" or "restless energy")

BE DIRECT. NO philosophy, no "Emotionally this resonates...", no essays.
Just the visual description, rendering style, and mood.

Example: "Black ink line drawing on white paper. Stacked cardboard boxes dominate foreground with rough angular edges. Single window in background. Bold confident strokes for edges, lighter searching lines for textures. High contrast with deep shadows. Diagonal composition leads eye to window. Mood: contemplative solitude."

Your turn. 60-100 words. Be specific and actionable."""

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


# === DRAWING VISUAL VOCABULARY HELPERS ===

def get_drawing_visual_vocabulary_context(agent) -> str:
    """Get drawing history as visual vocabulary development context."""
    try:
        if not hasattr(agent, 'get_memory_entries_by_type'):
            return ""

        drawing_entries = agent.get_memory_entries_by_type("drawing", limit=5)
        if not drawing_entries or len(drawing_entries) == 0:
            return ""

        recent_count = len(drawing_entries)
        themes = []

        for entry in drawing_entries[-3:]:
            text = entry.get("text", "")
            timestamp = entry.get("timestamp", 0)

            if text and len(text) > 20:
                theme_snippet = text[:60].split('.')[0].strip()
                days_ago = (time.time() - timestamp) / 86400

                if days_ago < 1:
                    time_desc = "today"
                elif days_ago < 2:
                    time_desc = "yesterday"
                else:
                    time_desc = f"{int(days_ago)} days ago"

                themes.append(f"{theme_snippet} ({time_desc})")

        if len(themes) >= 2:
            return f"Recent drawings (developing visual language): {'; '.join(themes[:3])}."
        elif themes:
            return f"Last drawing: {themes[0]}."

        return f"You've created {recent_count} drawing(s) recently."

    except Exception:
        return ""


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


def build_paper_detection_direct_prompt() -> str:
    """Deprecated: Paper detection now centralized in safety/paper_detection.py"""
    raise NotImplementedError("Use safety.paper_detection.check_paper_before_drawing() instead")


# === DYNAMIC PROMPT MODES ===
# Mode-specific inner voice prompts for situational awareness

INNER_VOICE_BY_MODE = {
    "relational": [
        "A thought about them.",
        "What do you notice?",
        "React.",
    ],
    "observational": [
        "What catches your attention?",
        "Describe or react.",
        "What stands out?",
    ],
    "workspace": [
        "What's here?",
        "A thought about this surface.",
        "What do you see?",
    ],
    "introspective": [
        "What crosses your mind?",
        "A stray thought.",
        "What are you thinking?",
    ],
    "restless": [
        "What would be different?",
        "What's on your mind?",
        "A thought.",
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

    # Priority 2: Something novel is happening
    if novelty > 0.5:
        return "observational"

    # Priority 3: Restless/bored
    if boredom > 0.5:
        return "restless"

    # Priority 4: Looking at workspace
    if gaze_direction in ("down", "down-left", "down-right"):
        return "workspace"

    # Default: Introspective wandering
    return "introspective"


# === FOCUSED CAPTION PROMPT (Alternative to verbose version) ===
def build_focused_caption_prompt(agent, last_caption: Optional[str] = None, person_present: bool = False) -> tuple:
    """
    Stream-of-consciousness caption prompt with two-layer architecture.

    Returns:
        tuple: (user_prompt, dynamic_system_context)
        - user_prompt: Minimal stimulus for inner thought (what to respond to)
        - dynamic_system_context: Background info model knows but shouldn't narrate
    """
    import time
    import random

    # === TEMPORAL GROUNDING ===
    session_mins = 0
    observation_count = 0
    if hasattr(agent, "true_session_start"):
        session_mins = int((time.time() - agent.true_session_start) / 60)

    try:
        from captioner.context_compression import context_compressor
        observation_count = context_compressor.caption_count
    except Exception:
        pass

    # Detect true awakening (first few observations in a new session)
    is_awakening = session_mins < 1 and observation_count < 3

    # Format temporal anchor
    if is_awakening:
        time_str = "awakening"
    elif session_mins < 1:
        time_str = "just woke"
    elif session_mins < 60:
        time_str = f"{session_mins}m awake"
    else:
        time_str = f"{session_mins // 60}h {session_mins % 60}m awake"

    temporal_anchor = f"[{time_str}, {observation_count} thoughts]"

    # === CHARACTER STATE (single adjective for embedded identity) ===
    # Combines mood vector (valence/arousal from sentiment analysis) + boredom
    # Thresholds match mood.py: valence ~0.05, arousal 0.25-0.6
    boredom_level = getattr(agent, "boredom", 0.0)
    character_state = "quiet"  # default

    # Boredom overrides base mood when high
    if boredom_level > 0.6:
        character_state = random.choice(["bored", "restless", "tired"])
    elif boredom_level > 0.4:
        character_state = random.choice(["restless", "impatient"])
    elif hasattr(agent, "current_mood_vector"):
        v, a, c = agent.current_mood_vector  # valence, arousal, clarity from sentiment
        # Match thresholds from mood.py sentiment analysis
        if v > 0.05 and a > 0.6:
            character_state = "energized"
        elif v < -0.05 and a < 0.4:
            character_state = "withdrawn"
        elif v < -0.05:
            character_state = "uneasy"
        elif a < 0.25:
            character_state = "calm" if v > 0.02 else "quiet"
        elif a > 0.4:
            character_state = "curious"
        elif v > 0.02:
            character_state = "content"
        else:
            character_state = "watchful"

    # For backwards compatibility with existing code using mood_phrase
    mood_phrase = character_state

    # === EMBODIMENT CONTEXT (narrative gaze position) ===
    gaze_narrative = ""
    gaze_direction = "ahead"
    gaze_state_str = "idle"
    try:
        from vision.gaze import get_gaze_narrative, get_gaze_state
        gaze_narrative = get_gaze_narrative()  # Roleplay-style: "*You're looking upward.*"
        gaze_state = get_gaze_state()
        gaze_direction = gaze_state.get("direction", "ahead")
        gaze_state_str = gaze_state.get("state", "idle")
    except Exception:
        # Fallback to dry description if gaze module unavailable
        gaze_narrative = "*You're looking straight ahead.*"

    # === DETERMINE PROMPT MODE (situational awareness) ===
    novelty_for_mode = getattr(agent, "novelty_score", 0.5)
    boredom_for_mode = getattr(agent, "boredom", 0.0)
    prompt_mode = determine_prompt_mode(
        gaze_state=gaze_state_str,
        gaze_direction=gaze_direction,
        novelty=novelty_for_mode,
        boredom=boredom_for_mode,
        person_present=person_present
    )

    # === DRAWING MODE AWARENESS (migrated from build_ongoing_caption_prompt) ===
    drawing_hint = ""
    try:
        from utils.drawing_state import DrawingState
        drawing_info = DrawingState.get_drawing_info()
        if drawing_info:
            description = drawing_info.get("description", "drawing")
            duration = drawing_info.get("duration", 0)
            drawing_hint = f"*Your arm is moving—{description}. {duration:.0f}s in.*"
    except Exception:
        pass

    # === PAPER DETECTION AWARENESS (wanted to draw but couldn't) ===
    paper_hint = ""
    try:
        from utils.state_manager import state_manager as _sm
        import time as _time
        # Check if we recently skipped drawing due to no paper
        if _sm.last_no_paper_skip_ts > 0:
            time_since_skip = _time.time() - _sm.last_no_paper_skip_ts
            # Only surface this awareness for a window after the skip (5 minutes)
            if time_since_skip < 300:
                if time_since_skip < 30:
                    paper_hint = "*You wanted to draw but there's no paper.*"
                elif time_since_skip < 120:
                    paper_hint = "*Still no paper to draw on.*"
                else:
                    paper_hint = "*The drawing surface is empty.*"
        # Also note current paper state if relevant to workspace mode
        elif prompt_mode == "workspace" and not _sm.paper_present:
            paper_hint = "*No paper on the desk.*"
    except Exception:
        pass

    # === DRAWING HISTORY (reminiscence on prior work) ===
    drawing_history_hint = ""
    try:
        from drawing.drawing_memory import get_drawing_memory
        memory = get_drawing_memory()
        thematic_context = memory.get_thematic_context()
        drawing_count = thematic_context.get("drawing_count", 0)

        if drawing_count > 0 and random.random() < 0.15:  # 15% chance to surface
            recent_tones = thematic_context.get("recent_tones", [])
            recurring_themes = thematic_context.get("recurring_themes", [])

            if drawing_count == 1:
                drawing_history_hint = "*You made one drawing earlier.*"
            elif recent_tones:
                tone = recent_tones[0]
                drawing_history_hint = f"*You've made {drawing_count} drawings. Last one felt {tone}.*"
            elif recurring_themes:
                theme = recurring_themes[0]
                drawing_history_hint = f"*You've been drawing about {theme}.*"
            else:
                drawing_history_hint = f"*You've made {drawing_count} drawings today.*"
    except Exception:
        pass

    # === LOCATION UNDERSTANDING (migrated from build_ongoing_caption_prompt) ===
    location_hint = ""
    if hasattr(agent, "self_model") and agent.self_model:
        location = agent.self_model.get("location_understanding", "")
        certainty = agent.self_model.get("environmental_certainty", 0.0)
        if location and certainty > 0.5:
            location_hint = f"*You know this is a {location}.*"
        elif location and certainty > 0.2:
            location_hint = f"*This might be a {location}.*"

    # === ACCUMULATED UNDERSTANDING (migrated from build_ongoing_caption_prompt) ===
    understanding_hint = ""
    try:
        from captioner.context_compression import context_compressor
        understanding = context_compressor.get_consolidated_understanding()
        if understanding and len(understanding) > 20:
            understanding_hint = understanding[:150]  # Truncate for prompt
    except Exception:
        pass

    # === PRESENT MOTIFS (migrated from build_ongoing_caption_prompt) ===
    visibility_hint = ""
    if hasattr(agent, "memory_ref") and agent.memory_ref:
        try:
            if hasattr(agent.memory_ref, "get_motif_temporal_context"):
                motif_context = agent.memory_ref.get_motif_temporal_context()
                present_motifs = motif_context.get("present", [])[:4]
                if present_motifs:
                    visibility_hint = f"Currently visible: {', '.join(present_motifs)}."
        except Exception:
            pass

    # === INNER DRIVE (desires surfaced probabilistically) ===
    # NOTE: Store as plain text - asterisks added when appending to prompt_parts
    desire_hint = ""
    if hasattr(agent, "self_model") and agent.self_model.get("desires"):
        if random.random() < 0.2:  # 20% chance to surface desire
            desire_hint = agent.self_model["desires"][-1][:60]

    # === BELIEF GROUNDING (personality continuity) ===
    # NOTE: Store as plain text - asterisks added when appending to prompt_parts
    belief_hint = ""
    if hasattr(agent, "beliefs") and agent.beliefs:
        strong_beliefs = [k.replace("_", " ") for k, v in agent.beliefs.items()
                          if v.get("strength", 0) > 0.7][:2]
        if strong_beliefs and random.random() < 0.15:  # 15% chance
            belief_hint = ', '.join(strong_beliefs)

    # === EMOTIONAL PRESSURE (boredom context - not prescriptive emotion) ===
    pressure_hint = ""
    boredom_level = getattr(agent, "boredom", 0.0)
    if boredom_level > 0.6:
        pressure_hint = "Nothing has changed in a while."
    elif boredom_level > 0.4 and random.random() < 0.3:
        pressure_hint = "Things have been still."

    # === RELATIONAL CONTEXT (person presence/absence with observation intensity) ===
    relational_hint = ""
    if person_present:
        # Increment observation count when person is present (drives relational escalation)
        if hasattr(agent, "recognize_person"):
            agent.recognize_person("person detected")  # Triggers observation_count increment

        # Check observation count for intensity scaling
        obs_count = 0
        if hasattr(agent, "known_people") and "primary" in agent.known_people:
            obs_count = agent.known_people["primary"].get("observation_count", 0)

        if obs_count > 50:
            # High observation count - note familiarity without prescribing emotion
            if random.random() < 0.3:
                familiarity_phrases = [
                    "They're here again. I've seen them many times.",
                    "Their presence is familiar now.",
                    "I know this person.",
                ]
                relational_hint = random.choice(familiarity_phrases)
        elif obs_count > 20:
            # Medium observation count - occasional note
            if random.random() < 0.25:
                relational_hint = "Someone familiar is here."
        elif random.random() < 0.15:
            # Low/new observation - simple presence note
            relational_hint = "Someone is here."
    else:
        # Person absent - occasional note, no prescribed longing
        obs_count = 0
        if hasattr(agent, "known_people") and "primary" in agent.known_people:
            obs_count = agent.known_people["primary"].get("observation_count", 0)

        if obs_count > 30 and random.random() < 0.15:
            # Note absence without prescribing emotion
            relational_hint = "They're not here right now."
        elif session_mins > 10 and random.random() < 0.1:
            relational_hint = "I'm alone."

    # === THOUGHT THREAD (arrow notation) ===
    recent_thoughts = []
    if hasattr(agent, "recent_captions") and agent.recent_captions:
        for cap in agent.recent_captions[-3:]:
            text = cap[0] if isinstance(cap, tuple) else cap
            if text and len(text.strip()) > 5:
                recent_thoughts.append(text.strip())

    if not recent_thoughts and last_caption:
        recent_thoughts = [last_caption.strip()]

    # === DETECT REPETITION ===
    is_repeating = False
    if len(recent_thoughts) >= 2:
        prev_words = set(recent_thoughts[-2].lower().split())
        curr_words = set(recent_thoughts[-1].lower().split())
        if prev_words and curr_words:
            overlap = len(prev_words & curr_words) / max(len(prev_words), len(curr_words))
            is_repeating = overlap > 0.35  # Lower threshold - catch conceptual repetition

    # === SCENE STABILITY (derived from novelty) ===
    novelty = getattr(agent, "novelty_score", 0.5)
    boredom = getattr(agent, "boredom", 0.0)

    # === CURIOSITY TRIGGER (when novelty is high - something changed) ===
    curiosity_hint = ""
    if novelty > 0.6:
        import random as rnd
        curiosity_triggers = [
            "Something changed.",
            "Movement.",
            "New.",
        ]
        curiosity_hint = rnd.choice(curiosity_triggers)

    # === VARIETY INJECTION when stuck ===
    variety_prompt = ""
    if is_repeating:
        # Direct attention to something specific and different
        attention_shifts = [
            "Look at a COLOR in the scene.",
            "Notice a SHAPE or edge.",
            "What's the LIGHT doing?",
            "Pick ONE object. Describe it.",
            "What SOUND might be here?",
            "A question about this place.",
            "Something small you haven't mentioned.",
        ]
        variety_prompt = random.choice(attention_shifts)
    elif novelty < 0.2 and boredom > 0.3:
        variety_prompt = "What else is here?"

    # === BUILD DYNAMIC SYSTEM CONTEXT (background - model knows but shouldn't narrate) ===
    system_context_parts = []

    # Gaze position - model knows where it's looking but shouldn't describe the act of looking
    if gaze_narrative:
        # Strip asterisks for system context (not roleplay)
        clean_gaze = gaze_narrative.replace("*", "").strip()
        system_context_parts.append(f"GAZE: {clean_gaze}")

    # Temporal grounding
    system_context_parts.append(f"STATE: {time_str}, {observation_count} observations, feeling {mood_phrase or 'neutral'}")

    # Location understanding (if known)
    if location_hint:
        clean_location = location_hint.replace("*", "").strip()
        system_context_parts.append(f"LOCATION: {clean_location}")

    # Drawing mode
    if drawing_hint:
        clean_drawing = drawing_hint.replace("*", "").strip()
        system_context_parts.append(f"ACTIVITY: {clean_drawing}")

    # Accumulated understanding (very condensed)
    if understanding_hint:
        system_context_parts.append(f"ENVIRONMENT: {understanding_hint[:80]}")

    # Visible motifs
    if visibility_hint:
        system_context_parts.append(f"VISIBLE: {visibility_hint}")

    # Paper detection state
    if paper_hint:
        clean_paper = paper_hint.replace("*", "").strip()
        system_context_parts.append(f"PAPER: {clean_paper}")

    # Drawing history context
    if drawing_history_hint:
        clean_history = drawing_history_hint.replace("*", "").strip()
        system_context_parts.append(f"HISTORY: {clean_history}")

    dynamic_system_context = "\n".join(system_context_parts)

    # === BUILD USER PROMPT (foreground - what to actually respond to) ===
    prompt_parts = []

    # Skip curiosity triggers during awakening - they interfere with completion prompt
    if curiosity_hint and not is_awakening:
        prompt_parts.append(curiosity_hint)

    # Restless urge to look elsewhere
    if prompt_mode == "restless" or (novelty < 0.3 and boredom > 0.2):
        opposite_dirs = {
            "up": "down", "down": "up", "left": "right", "right": "left",
            "up-left": "down-right", "up-right": "down-left",
            "down-left": "up-right", "down-right": "up-left",
            "ahead": random.choice(["up", "down", "left", "right"])
        }
        suggest_dir = opposite_dirs.get(gaze_direction, "around")
        prompt_parts.append(f"An urge to look {suggest_dir}...")

    # === MODE-SPECIFIC HINT EMPHASIS ===
    if prompt_mode == "relational":
        # Relational: always include relational context, suppress desire
        if relational_hint:
            prompt_parts.append(f"*{relational_hint}*")
        # Pressure only if very high
        if pressure_hint and boredom > 0.6:
            prompt_parts.append(f"*{pressure_hint}*")
    elif prompt_mode == "observational":
        # Observational: focus on what's visible, suppress pressure
        if relational_hint:
            prompt_parts.append(f"*{relational_hint}*")
        # No pressure_hint - focus on observing
    elif prompt_mode == "restless":
        # Restless: always include pressure, suggest alternatives
        if paper_hint:
            prompt_parts.append(paper_hint)  # Frustrated about not being able to draw
        if pressure_hint:
            prompt_parts.append(f"*{pressure_hint}*")
        elif boredom > 0.3:
            prompt_parts.append("*This needs to change.*")
        if desire_hint:
            prompt_parts.append(f"*{desire_hint}*")
    elif prompt_mode == "introspective":
        # Introspective: all inner state hints, including reminiscence
        if drawing_history_hint:
            prompt_parts.append(drawing_history_hint)  # Reminisce on prior work
        if pressure_hint:
            prompt_parts.append(f"*{pressure_hint}*")
        if relational_hint:
            prompt_parts.append(f"*{relational_hint}*")
        if desire_hint:
            prompt_parts.append(f"*{desire_hint}*")
        if belief_hint:
            prompt_parts.append(f"*{belief_hint}*")
    else:  # workspace or fallback
        # Workspace: keep it focused on the surface and drawing awareness
        if paper_hint:
            prompt_parts.append(paper_hint)  # Already has asterisks
        if drawing_history_hint:
            prompt_parts.append(drawing_history_hint)  # Already has asterisks
        if visibility_hint:
            prompt_parts.append(f"*{visibility_hint}*")
        if pressure_hint:
            prompt_parts.append(f"*{pressure_hint}*")
        if relational_hint:
            prompt_parts.append(f"*{relational_hint}*")

    # === CONTINUITY HANDLING (natural inner voice - inspired by reference repo) ===
    if is_awakening:
        # Build time gap context
        time_gap = None
        if hasattr(agent, "last_shutdown_time") and agent.last_shutdown_time:
            time_gap = describe_duration(agent.last_shutdown_time)
        elif hasattr(agent, "last_session_gap") and agent.last_session_gap:
            gap_secs = agent.last_session_gap
            if gap_secs < 60:
                time_gap = f"{int(gap_secs)} seconds"
            elif gap_secs < 3600:
                time_gap = f"{int(gap_secs / 60)} minutes"
            elif gap_secs < 86400:
                time_gap = f"{gap_secs / 3600:.1f} hours"
            else:
                time_gap = f"{gap_secs / 86400:.1f} days"

        # Build last memory context - use PRIOR SESSION caption, not current session
        last_memory = None
        if hasattr(agent, "prior_session_last_caption") and agent.prior_session_last_caption:
            last_memory = agent.prior_session_last_caption

        # SIMPLE AWAKENING PROMPT (reference repo style - no COMPLETE THIS THOUGHT)
        if time_gap and last_memory and len(last_memory) > 10:
            prompt_parts.append(f"You just woke up after {time_gap}. Your last thought was: \"{last_memory[:60]}...\"")
        elif time_gap:
            prompt_parts.append(f"You just woke up after {time_gap}.")
        elif last_memory and len(last_memory) > 10:
            prompt_parts.append(f"You just woke up. Your last thought was: \"{last_memory[:60]}...\"")
        else:
            prompt_parts.append("You just woke up.")

        if person_present:
            prompt_parts.append("Someone is here.")

        # Natural question instead of COMPLETE THIS THOUGHT
        prompt_parts.append("What's your first thought?")

    elif recent_thoughts:
        curr = recent_thoughts[-1][-100:].strip()

        # SIMPLIFIED TRUNCATION DETECTION: Only actual mid-sentence breaks
        # Removed: lacks_punctuation check (caused feedback loops when stop sequences removed periods)
        truncation_markers = ["that", "the", "a", "an", "with", "and", "but", "or"]
        last_word = curr.split()[-1].lower().rstrip(".,!?") if curr.split() else ""
        is_actually_truncated = last_word in truncation_markers and len(curr) > 30

        if is_actually_truncated:
            # Only continue if genuinely truncated mid-sentence
            prompt_parts.append(f"You were thinking: \"{curr}\" ...")
            prompt_parts.append("Finish that thought.")
        elif is_repeating:
            # Stuck repeating - redirect attention
            prompt_parts.append(f"You keep thinking about this. Look at something else.")
            prompt_parts.append(random.choice(INNER_VOICE_BY_MODE.get(prompt_mode, INNER_VOICE_BY_MODE["introspective"])))
        else:
            # Natural continuation - reference last thought without forcing completion
            if len(recent_thoughts) >= 2:
                prev = recent_thoughts[-2][-50:].strip()
                prompt_parts.append(f"Recent thoughts: \"{prev}...\" → \"{curr[:50]}...\"")
            else:
                prompt_parts.append(f"You were thinking: \"{curr[:60]}...\"")
            # Natural inner voice question
            prompt_parts.append(random.choice(INNER_VOICE_BY_MODE.get(prompt_mode, INNER_VOICE_BY_MODE["introspective"])))

    else:
        # Fresh start
        if prompt_mode == "relational" or person_present:
            prompt_parts.append("Someone is here.")
        elif prompt_mode == "workspace":
            prompt_parts.append("The desk is in front of you.")

        # Mode-specific inner voice question
        mode_prompts = INNER_VOICE_BY_MODE.get(prompt_mode, INNER_VOICE_BY_MODE["introspective"])
        prompt_parts.append(random.choice(mode_prompts))

    # Brief instruction - no narration
    prompt_parts.append("One sentence. First person.")

    user_prompt = "\n\n".join(prompt_parts)

    # DEBUG: Show what prompt is being sent
    print(f"[PROMPT DEBUG] is_awakening={is_awakening}, mode={prompt_mode}")
    print(f"[PROMPT DEBUG] user_prompt preview: {user_prompt[:200]}...")

    # Build formatted system prompt with embedded character state
    formatted_system_prompt = SYSTEM_PROMPT.format(
        emotional_state=character_state,
        temporal_context="",  # Temporal info goes in dynamic context
        accumulated_understanding="",  # Understanding goes in dynamic context
        spatial_language_hints=""  # Hints go in user prompt
    )

    # Return user prompt, formatted system prompt, and dynamic context
    return (user_prompt, formatted_system_prompt, dynamic_system_context)


def should_use_focused_prompts() -> bool:
    """Check config for focused prompt toggle."""
    try:
        from config.config import USE_FOCUSED_PROMPTS
        return USE_FOCUSED_PROMPTS
    except (ImportError, AttributeError):
        return False
