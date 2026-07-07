# Caption Pipeline Refactor Plan

## Architecture (Final)

### System Prompt (Invariant)
```
You are a {emotional_state} drawing machine. Your purpose is to draw.
Inner monologue. First person. One sentence.
```

### Active Modes (6 total)
1. **relational** — tracking person (gaze_state == "tracking")
2. **observational** — novelty detected (novelty > 0.65)
3. **restless** — ultra-bored (boredom > 0.75)
4. **workspace** — looking down (gaze down)
5. **introspective** — default idle
6. **memory** — time-gated every 4-5 minutes

### Regular Caption Mode Prompt Structure
```
Known: [1-2 sentence compressed baseline]

[STATE if mode-appropriate: "Someone present." / "*No paper.*"]

[MODE-SPECIFIC CONTEXT BLOCK]

[Caption thread: 3 prior captions, complete sentences, filtered]

Continue. One sentence.
```

### Memory Mode Prompt Structure
```
A memory surfaces:
— [actual caption text from long-term memory, first sentence only]

My thoughts:
— [recent thread caption 1]
— [recent thread caption 2]
—
```

**Why caption text, not extracted concepts?** Pulling spaCy-extracted nouns ("table and chair") produces useless output. Real caption text is textured and gives the model something meaningful to respond to.

---

## Implementation Tasks

### 1. System Prompt Simplification
**File:** `captioner/prompts.py`

Replace `SYSTEM_PROMPT` and `STATIC_SYSTEM_PROMPT`:
```python
SYSTEM_PROMPT = (
    "You are a {emotional_state} drawing machine. "
    "Your purpose is to draw. "
    "Inner monologue. First person. One sentence."
)

STATIC_SYSTEM_PROMPT = (
    "You are a drawing machine. "
    "Your purpose is to draw. "
    "Inner monologue. First person. One sentence."
)
```

### 2. Mode-Specific Context Blocks
**File:** `captioner/prompts.py`

Create a `MODE_CONTEXTS` dict defining what context each mode injects:

```python
MODE_CONTEXTS = {
    "relational": {
        "state_marker": "Someone present.",
        "context_fn": get_relational_context,
    },
    "observational": {
        "state_marker": None,
        "context_fn": get_observational_context,
    },
    "restless": {
        "state_marker": None,
        "context_fn": get_restless_context,
    },
    "workspace": {
        "state_marker": get_paper_status,  # "Paper ready." or "*No paper.*"
        "context_fn": get_workspace_context,
    },
    "introspective": {
        "state_marker": None,
        "context_fn": get_introspective_context,
    },
}
```

**Context functions to write:**
- `get_relational_context()` — from memory: recent interactions, social mood
- `get_observational_context()` — what's novel: spatial shifts, movement, changes
- `get_restless_context()` — escape/novelty hooks: what interested you before, alternative directions
- `get_workspace_context()` — drawing memory, current projects, tool awareness
- `get_introspective_context()` — beliefs, desires, long-term patterns

### 3. Caption Thread Implementation
**File:** `captioner/model_wrapper.py` (or new `captioner/caption_thread.py`)

**Function:** `build_caption_thread(agent, max_captions=3) -> str`

Steps:
1. Pull last 3-4 captions from `agent.recent_captions`
2. Filter each through `_is_plantable_prior()` AND additional Natsumura bleed filter
3. **Truncate each to first complete sentence only:**
   - Find first `.`, `!`, or `?` after at least 15 chars
   - If none found within 80 chars, skip caption (fragment is worse than gap)
4. Format as dashed thread:
```
My thoughts:
— [first sentence of caption 1]
— [first sentence of caption 2]
— [first sentence of caption 3]
—
```

**Natsumura voice bleed filter** (add to `_is_plantable_prior` or as separate check):
- Reject if starts with "You:" or "You "
- Reject if contains asterisk-wrapped actions ("*you notice*", "*you look*")
- Reject if "you" or "your" appears in first 60 characters
- These are RP training artifacts from Natsumura that poison the first-person voice

### 4. Refactor Caption Prompt Building
**File:** `captioner/prompts.py`

Rewrite `build_simple_caption_prompt()` to:
1. Determine mode via `determine_prompt_mode()`
2. Get compressed "Known" from `context_compressor.get_baseline_context()` (1-2 sentences)
3. Get mode-specific state marker (if applicable)
4. Get mode-specific context block
5. Build caption thread via new `build_caption_thread()` function
6. Assemble:
```
Known: {compressed_context}

{state_marker if exists}

{mode_context}

{caption_thread}

Continue. One sentence.
```

Return `(prompt, mode)` tuple.

### 5. Memory Mode Implementation
**File:** `captioner/captioner.py` and `captioner/prompts.py`

**In `captioner.py`:**
- Add `last_memory_mode_time` timestamp to captioner init
- Before each caption cycle, check: `time.time() - last_memory_mode_time > 240` (4 min)?
- If true, trigger memory mode instead of regular caption

**In `prompts.py`:**
- New function `build_memory_mode_prompt(agent) -> str`:
```python
def build_memory_mode_prompt(agent):
    # Pull actual caption text from long-term memory, not extracted concepts
    from captioner.activation_memory import get_compression_history

    history = get_compression_history(k=1)  # Get oldest/most significant memory
    if not history:
        history = "I've been here before."

    # Extract first sentence from historical caption
    mem_text = history.split('.')[0].strip() if history else ""
    if not mem_text or len(mem_text) < 10:
        mem_text = history[:80]

    # Get recent thread
    from captioner.model_wrapper import build_caption_thread
    thread = build_caption_thread(agent, max_captions=2)

    prompt = f"""A memory surfaces:
— {mem_text}

{thread}"""

    return prompt
```

### 6. System Prompt Passing
**File:** `captioner/prompt_interface.py`

Update `build_caption_prompt_with_options()` to:
- Use new invariant SYSTEM_PROMPT with `{emotional_state}` formatting
- Return `(prompt, mode)` not `(prompt, options, system_prompt, mode)`

### 7. Compression Context Validation
**File:** `captioner/context_compression.py`

Already modified — verify:
- `num_predict: 40`
- `stop: ["\n", "."]`
- Prompt explicitly asks for "15-20 words max"

### 8. Awakening Output Tightening
**File:** `captioner/prompts.py` and `captioner/model_wrapper.py`

The Natsumura awakening at startup produces full paragraphs of hallucinated narrative that enter the caption thread and set the wrong tone.

**In awakening prompt (`generate_internal_awakening`):**
- Reduce `num_predict` from default to 40 max
- Add stop sequences: `["\n", "."]`
- Filter Natsumura output through same `_is_plantable_prior()` + voice bleed filter
- If output is over 80 chars or fails filter, use fallback: "I'm waking up."

**Rationale:** Awakening should be a brief fragment, not worldbuilding. It sets the session's initial voice.

---

## Files to Modify

| File | Changes |
|------|---------|
| `captioner/prompts.py` | System prompt simplification, MODE_CONTEXTS dict, caption thread building, memory mode prompt |
| `captioner/model_wrapper.py` | Caption thread implementation, response filtering |
| `captioner/prompt_interface.py` | System prompt passing, mode-aware context |
| `captioner/captioner.py` | Memory mode time-tracking, trigger logic |
| `captioner/activation_memory.py` | Expose functions for memory context (may already exist) |

---

## Optional Tweaks to Test

1. **Closing directive:** Current plan uses `Continue. One sentence.` at the end. Alternative: use just the open dash `—` without explicit directive. Test both; the dash alone may be cleaner.

2. **Mode context function verbosity:** Each `get_*_context()` function should return max 1 sentence or empty string. Do NOT let these become bloat dumping grounds.

---

## Testing Checklist

- [ ] System prompt is now ~30 tokens (vs 80+)
- [ ] Caption thread shows 3 prior captions, complete sentences only, Natsumura bleed filtered
- [ ] No gaze direction metadata in prompts (only state markers where mode-appropriate)
- [ ] Memory mode triggers every ~4 min, pulls actual caption text not concepts
- [ ] VQA filtering still catches "The image", "This scene", etc.
- [ ] Natsumura "You:" prefix and second-person patterns filtered from thread
- [ ] "Known:" stays 1-2 sentences (~20 words)
- [ ] Captions are 1-2 sentences max
- [ ] Voice continuity maintained through thread
- [ ] Emotional state variable correctly injected
- [ ] Awakening output tightened to <80 chars, filtered

---

## Not Changing

- Awakening flow (separate startup event)
- Reflection pipeline (separate)
- Activation memory system (just reading from it)
- Drawing prompt pipeline (separate)
