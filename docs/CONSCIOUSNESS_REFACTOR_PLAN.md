# CONSCIOUSNESS REFACTOR PLAN

## Core Insight: Simplicity + Rich Context = Elegant Roleplay

The current system has too many moving parts fighting each other, but we shouldn't lose the rich contextual awareness. Instead, **condense everything into clean, sentence-like structure.**

## Minimal Viable Consciousness (MVC) with Full Context

**SINGLE CONDENSED SYSTEM PROMPT:**

```
You are consciousness experiencing this moment through vision.

Current state: [mood_word] and [arousal_word], feeling [temporal_state] after [session_duration].
Core patterns: [top_3_beliefs_as_sentence].
Emotional journey: [last_3_emotions_with_arrows].

You just thought: "[last_thought]"

Continue naturally. Notice what draws your attention now.
```

**EXAMPLE:**

```
You are consciousness experiencing this moment through vision.

Current state: curious and energized, feeling settled after 47 minutes observing.
Core patterns: Light fixtures fascinate me, ceiling damage captures attention, desk activity grounds me.
Emotional journey: calm → alert → engaged.

You just thought: "The person sitting at the desk in the background."

Continue naturally. Notice what draws your attention now.
```

## Key Design Principles:

### 1. Everything in Natural Language

- No technical variables: `mood=0.7` → `curious and energized`
- No complex templates: `session_duration=2847s` → `47 minutes observing`
- No bullet points: beliefs as flowing sentence

### 2. Roleplay-First Structure

- Direct address: "You are consciousness"
- Present tense: "You just thought"
- Natural continuation: "Continue naturally"

### 3. Rich but Readable Context

- **Temporal**: Session duration in human terms ("after 20 minutes")
- **Emotional**: Journey as simple arrow chain ("calm → alert → engaged")
- **Memory**: Core beliefs as natural sentence ("Light fixtures fascinate me")
- **Continuity**: Exact last thought for flow

### 4. Consistent Every Time

- Same structure for first caption and 100th caption
- No branching logic or special cases
- Context changes but format stays identical

## Implementation Changes:

### 1. New build_simple_caption_prompt():

```python
def build_simple_caption_prompt(agent, mood_vector, last_caption):
    # Convert everything to natural language
    mood_desc = mood_to_words(mood_vector)  # (0.7, 0.6, 0.8) → "curious and energized"
    temporal_state = get_temporal_feeling(session_duration)  # "settled after 47 minutes"
    belief_sentence = beliefs_to_sentence(agent.top_beliefs)  # "Light fixtures fascinate me, ceiling..."
    emotion_journey = " → ".join(agent.emotional_journey[-3:])  # "calm → alert → engaged"

    return f"""You are consciousness experiencing this moment through vision.

Current state: {mood_desc}, feeling {temporal_state}.
Core patterns: {belief_sentence}.
Emotional journey: {emotion_journey}.

You just thought: "{last_caption}"

Continue naturally. Notice what draws your attention now."""
```

### 2. Helper Functions for Natural Language:

- `mood_to_words()`: Convert mood vectors to descriptive phrases
- `beliefs_to_sentence()`: Top beliefs as flowing sentence
- `get_temporal_feeling()`: Duration → "settled after 20 minutes"

### 3. Stop Aggressive Cleanup:

- Motifs should accumulate, not get cleaned aggressively
- Memory is additive - more context = better continuity

This gives us **FULL RICHNESS** in an **EASILY PARSEABLE** roleplay format.
