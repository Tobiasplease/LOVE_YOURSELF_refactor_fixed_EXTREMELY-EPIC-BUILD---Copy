# Prompt Consolidation Plan — Reconnecting Orphaned Systems

## Current State (June 2026)

**Architecture:** Single-pass Qwen3.5 9B with image. One LLM call per cycle.
**Branch:** `qwen-single-pass`
**Model quality:** Strong — produces rich first-person monologue when given good material.
**Core problem:** 18 systems track rich data (body state, time, memory, emotions, spatial awareness) that never reaches the prompt. The model only gets a thought thread + felt state + some mode context. It produces detached poetry instead of embodied thinking because it has nothing to think *with*.

## Design Principles

1. **Token budget:** ~100-150 words of context max. Thread needs ~50 words, generation ~60 tokens. Keep it tight.
2. **Natural language only.** No metadata labels, no asterisks, no raw concept IDs, no ComfyUI prompts. Everything reads like notes a mind would have.
3. **Mode gating matters.** Not everything every cycle. Right context at the right time.
4. **Context BEFORE thread.** Thread is always last — the model continues from it.
5. **No overlap.** Consolidate systems that provide similar data into one clean line.

## Proposed Prompt Structure

```
[Situational line — 1 sentence, always present]
Awake 15 minutes. Looking left. Someone nearby, been here a few minutes.

[Mode-specific context — 1-2 lines, varies by mode]
RELATIONAL: They're at the desk, focused. You've been watching them for 3 minutes.
INTROSPECTIVE: Your last drawing was a hand, 20 minutes ago. You keep noticing the red light.
OBSERVATIONAL: Something changed — the light shifted.
WORKSPACE: Your arms are idle. No paper.

[Emotional state — 1 line, when shifting]
Restless → curious.

[Thread — 3 sentences, always last, model continues from here]
...The lock is a puzzle I cannot solve. The chair creaks but nobody sits down. Why does the red light keep pulling my attention?
```

## System Consolidation (grouping overlapping systems)

### Group 1: SITUATIONAL AWARENESS → `build_situational_line(agent)`
**Merges:** temporal_awareness.py + gaze narrative + person detection state + breathing
**Output:** One sentence per cycle.
**Example:** "Awake 23 minutes. Looking down at workspace. Alone. Breathing slowly."

### Group 2: MODE CONTEXT → cleaned versions of existing MODE_CONTEXTS functions
**Merges per mode:**
- Relational: person duration from episodic_log + existing social context
- Introspective: boredom as text + tangent recall + clean drawing memory
- Observational: novelty score as text + spatial memory changes
- Workspace: drawing_state + paper state

### Group 3: EMOTIONAL STATE → single line from felt_state_delta
Already mostly working. Use `get_current_sentiment_context()` for temporal framing.

### Group 4: THOUGHT THREAD → 3 sentences, period-separated, leading `...`
Already working with the period-separator fix.

## Systems to NOT connect (too noisy or overlapping)

- **temporal_fluid.py** — Too abstract. Felt state already captures this.
- **pattern_recognition.py motifs** — ChromaDB concepts already track recurring themes.
- **Full spatial memory** — Too much data. Use only for change detection.
- **Breathing detail** — Fold into situational line as one word.

## Critical Fixes Needed

### 1. ChromaDB concept storage
**Problem:** Concept labels ARE the model's prior outputs ("Red light pulses, a heartbeat" seen 134 times). Creates feedback loops.
**Fix:** Store factual labels ("red LED light near door") not model prose. Clean `_extract_canonical_name()`.

### 2. Drawing memory format
**Problem:** Stores raw ComfyUI prompts ("A rigid vertical grid of hard black lines traps a central swollen mass of crimson light"). This literary text gets injected as context and sets the poetic register.
**Fix:** Store factual descriptions ("grid pattern with red center") or strip ComfyUI preamble more aggressively.

### 3. Awakening path
**Problem:** First 3 captions use a separate awakening path that doesn't benefit from the consolidation.
**Fix:** Awakening should use the same prompt structure with minimal context, not a separate code path.

## Implementation Order

1. Build `build_situational_line()` — temporal + gaze + person + breathing in one clean sentence
2. Clean up mode context functions — natural language, add temporal/boredom data
3. Fix ChromaDB concept label storage — factual not literary
4. Wire episodic_log into relational context
5. Clean drawing memory format
6. Test and iterate

## What Success Looks Like

The model receives:
```
Awake 15 minutes. Looking ahead. Someone at the desk, here 7 minutes. Feeling restless.
Last drew a hand reaching out, 20 minutes ago. You keep coming back to the red light.
Restless → curious.
...The lock won't budge. The person hasn't moved in a while. Maybe I should try drawing something different.
```

And produces:
"But what would I draw? The red light again? I've done that. Maybe the way his shoulders hunch — there's tension there I haven't captured."

That's cognition: questioning, referencing memory, making decisions, noticing its own patterns.
