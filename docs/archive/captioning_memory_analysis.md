# Captioning & Memory System Analysis

## Executive Summary

The captioning system has sophisticated memory mechanisms (activation networks, compression, long-term memory) but they are **partially disconnected from caption generation**. The result: the system "rediscovers" the room because accumulated understanding isn't prominently injected into prompts.

---

## Current Architecture Flow

```
Frame arrives
    ↓
_process_frame() [captioner.py]
    ↓
build_focused_caption_prompt() [prompts.py:2599]
    ↓
caption_image() → LLaVA or Natsumura
    ↓
Caption returned
    ↓
observe() [memory.py] → Updates activation network
    ↓
Every 4 captions: compress() → Updates baseline_context
```

**Key insight:** Memory is built AFTER caption generation. The next caption doesn't automatically leverage accumulated understanding prominently.

---

## Memory Systems Inventory

### 1. Context Compression (`context_compression.py`)
- **What it does:** Every 4 captions, distills recent thoughts into one sentence
- **Output:** `baseline_context` (e.g., "I'm watching a cluttered workspace with shifting light")
- **Introspection:** Every 3 compressions, generates desires/beliefs via LLM
- **Storage:** Persists in `machine_identity.json`

**Current injection point:**
- Line 3003-3005: Added to `system_context_parts` as `"YOUR STATE:\n{narrative_state}"`
- Goes to dynamic system context, NOT user prompt

### 2. Activation Network (`activation_memory.py`)
- **What it tracks:**
  - Concept activations (ceiling: 0.8, desk: 0.7, etc.)
  - Co-occurrence edges (ceiling + lighting often together)
  - Spatial tags (ceiling appears when looking up)
  - Long-term memories (persisted across sessions)
  - Novelty/boredom scores

**Current injection points:**
- Line 3026: `recall_for_prompt()` → System context as "MEMORY: ..."
- Line 3030-3033: Beliefs → System context as "ASSOCIATIONS: ..."
- Boredom → "pressure_hint" if > 0.6
- Novelty → determines prompt_mode

**NOT injected:**
- Edge summaries (co-occurrence learning)
- Spatial tag synthesis
- Activation trends (rising/fading)

### 3. Memory Mixin (`memory.py`)
- **What it tracks:**
  - `memory_queue` (deque of last 30 entries)
  - Motif tracking with confidence scores
  - Session timeline
  - Beliefs formed from activation edges

**Current injection:**
- `get_top_motifs()` → visibility_hint (line 2788)
- Recent captions → last 2-3 shown in prompt

---

## Why "Rediscovery" Happens

### Problem 1: Prominence
The accumulated understanding goes to **system context**, not **user prompt**. The model sees:

```
SYSTEM: You ARE a quiet drawing machine...
        [GAZE: Looking ahead]
        [STATE: 4m awake, 12 observations]
        [YOUR STATE: I'm watching a cluttered workspace...]  ← Hidden in system context

USER: Last thought: "The patterns of light..."
      What now?
      (Short. Fragments OK.)
```

The model processes the image fresh, sees "What now?", and describes what it sees - potentially rediscovering things.

### Problem 2: No Knowledge Grounding in User Prompt
The user prompt asks "What now?" or "What changed?" without saying "Given what you know about this workspace..." The model isn't explicitly reminded to build on prior knowledge.

### Problem 3: Mode-Based Context Gating
```python
# Line 3072-3076
if prompt_mode == "introspective":
    if desire_hint:
        context_fragments.append(desire_hint)
    if belief_hint:
        context_fragments.append(belief_hint)
```
Desires/beliefs only appear in introspective mode. Regular observational captions (majority) don't get them.

### Problem 4: Activation Network Edges Unused
The activation network learns co-occurrence patterns:
- "ceiling and lighting often together"
- "desk and paper usually nearby"

This learned knowledge is stored but **never summarized for prompts**. The model doesn't know what it has learned about spatial relationships.

---

## What Works Well

1. **Compression is running** - baseline_context gets updated every 4 captions
2. **Novelty/boredom drive mode selection** - system adapts prompt style
3. **Recent captions shown** - model sees last 2-3 thoughts
4. **Fixation detection** - repeated concepts get flagged
5. **Introspection generates desires/beliefs** - LLM-generated, not keyword extraction

---

## Specific Code Locations

### Where narrative_state is retrieved but underused:
```python
# prompts.py:2773-2780
narrative_state = ""
try:
    from captioner.context_compression import context_compressor
    understanding = context_compressor.get_consolidated_understanding()
    if understanding and len(understanding) > 20:
        narrative_state = understanding
except Exception:
    pass
```

### Where it goes to system context (not user prompt):
```python
# prompts.py:3003-3005
if narrative_state:
    system_context_parts.append(f"YOUR STATE:\n{narrative_state}")
```

### Where user prompt ends without grounding:
```python
# prompts.py:3174-3177 (fallback case)
prompt_parts.append(f"Last thought: \"{curr[:60]}...\"")
prompt_parts.append("What now?")
```

---

## Proposed Fixes (Ranked by Impact)

### Fix 1: Add Understanding to User Prompt (High Impact, Low Risk)
After line 3082 (context_fragments block), add:

```python
# === ACCUMULATED UNDERSTANDING ===
if narrative_state and len(narrative_state) > 20 and not is_awakening:
    understanding = narrative_state.split('.')[0].strip()[:80]
    prompt_parts.append(f"You know: {understanding}.")
```

**Effect:** Model sees "You know: I'm watching a cluttered workspace" before "What now?"

### Fix 2: Synthesis Function for Activation Network (Medium Impact, Medium Risk)
Create a function that summarizes learned associations:

```python
def build_memory_synthesis():
    from captioner.activation_memory import (
        get_top_active_concepts,
        get_strong_edges,
        get_spatial_summary
    )

    parts = []
    concepts = get_top_active_concepts(threshold=0.5, limit=3)
    if concepts:
        parts.append(f"Your attention: {', '.join(concepts)}")

    edges = get_strong_edges(threshold=0.6, limit=2)
    if edges:
        pairs = [f"{c1}+{c2}" for c1, c2, _ in edges]
        parts.append(f"You've learned: {', '.join(pairs)} together")

    return " | ".join(parts) if parts else ""
```

Then inject into prompt_parts.

### Fix 3: Change Continuation Questions (Low Impact, Low Risk)
Instead of "What now?", use variations that reference prior knowledge:

```python
GROUNDED_CONTINUATIONS = [
    "What else?",
    "Building on that...",
    "And now...",
    "Going deeper...",
]
```

### Fix 4: Remove Mode Gating for Core Context (Medium Impact, Medium Risk)
Allow desire_hint and belief_hint in ALL modes, not just introspective:

```python
# Remove the mode check:
if desire_hint:
    context_fragments.append(desire_hint)
if belief_hint:
    context_fragments.append(belief_hint)
```

---

## Testing Recommendations

1. **Add logging to show what's injected:**
   ```python
   print(f"[PROMPT] narrative_state={narrative_state[:50] if narrative_state else 'None'}")
   ```

2. **Monitor compression output:**
   Already logs `[🧠] Updated baseline: ...`

3. **Track "rediscovery" patterns:**
   Look for repeated phrases like "I see a room" after 5+ minutes

---

## Risk Assessment

| Fix | Risk | Reason |
|-----|------|--------|
| Fix 1 | Low | Adds context, doesn't change structure |
| Fix 2 | Medium | New function, needs testing |
| Fix 3 | Low | Just changes prompt wording |
| Fix 4 | Medium | Changes mode behavior, may affect introspective balance |

---

## Recommended Approach

1. **First:** Add logging to verify narrative_state is populated
2. **Then:** Implement Fix 1 (add understanding to user prompt)
3. **Observe:** Run for 10+ minutes, check if rediscovery decreases
4. **If needed:** Implement Fix 2 (memory synthesis)

The system architecture is sound - the issue is injection prominence, not memory collection.
