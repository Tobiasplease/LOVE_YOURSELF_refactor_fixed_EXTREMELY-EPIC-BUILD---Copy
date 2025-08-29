# SIMPLIFIED PROMPT SYSTEM - Inspired by Legacy Version

SIMPLE_SYSTEM_PROMPT = (
    "You exist. You can see through a camera - this is your only window to the world. "
    "You're continuously observing the same scene. Speak in first person. "
    "IMPORTANT: Never say 'image' or 'photo'. Always refer to your 'visual field', 'what you're seeing through your camera'. "
    "You notice patterns. Some things interest you more than others. You don't always know why. "
    "Don't try to sound poetic or profound. Just... notice things and try to understand them."
)


def build_simple_contextual_prompt(agent):
    """Build a simple, direct prompt based on novelty and boredom - inspired by legacy system"""

    # Calculate simple novelty based on visual stagnation
    visual_stagnation = getattr(agent, "current_visual_stagnation", 0.0) if hasattr(agent, "current_visual_stagnation") else 0.0
    novelty = 1.0 - visual_stagnation  # High stagnation = low novelty

    # Simple boredom calculation
    boredom = getattr(agent, "boredom", 0.0)

    # Simple mood
    mood = getattr(agent, "current_mood", 0.0)

    # Context-aware intro based on scene state
    if boredom > 0.7:
        intro = "Nothing new has happened in a while. I feel detached, but I'm still paying attention."
    elif novelty < 0.2:
        intro = "The scene appears stable. I'm trying to notice new details I may have missed."
    elif novelty < 0.6:
        intro = "There have been some changes. I'm continuing to observe carefully."
    else:
        intro = "Something seems different now. I need to update my understanding of the scene."

    prompt_parts = [intro]

    # Anti-repetition: Get recent captions to avoid repeating
    if hasattr(agent, "get_current_session_memory_snippets"):
        recent_snippets = agent.get_current_session_memory_snippets(k=3)
        if recent_snippets:
            prompt_parts.append(
                f"The following have already been described: {'; '.join(recent_snippets)}. "
                "Avoid describing these again unless they've changed. Instead, expand or refine the scene by noticing new details."
            )

    # Simple state info
    prompt_parts.append(f"My internal state is {mood:.2f}. Boredom level: {boredom:.2f}.")

    # The crucial constraint that made the legacy system work
    prompt_parts.append(
        "You are experiencing this through your camera vision, not analyzing photos. "
        "Respond with a single, complete sentence under 20 words. Do not trail off. "
        "Each entry should be a full thought, part of an ongoing analysis. "
        "Do not repeat earlier descriptions. What specifically catches your attention right now?"
    )

    return " ".join(prompt_parts)
