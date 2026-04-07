# Memory Injection Fix Plan

## Executive Summary

The captioning/memory system is **collecting data correctly** but **injecting it incorrectly**. The accumulated understanding (`narrative_state`) goes to system context where it's easily ignored, instead of being prominently placed in the user prompt.

**Result:** The system rediscovers the room because the model sees "What now?" without being reminded what it already knows.

---

## Current State (What Works)

✅ **Memory Collection**
- Activation network tracks concepts, builds edges, calculates novelty/boredom
- Context compression distills every 4 captions into `baseline_context`
- Introspection generates desires/beliefs every 3 compressions
- Spatial tags, temporal awareness, pattern recognition all functional

✅ **Data Quality**
- `narrative_state` contains good 3-line compressed understanding
- Activation edges capture learned associations
- Novelty/boredom metrics drive mode selection appropriately

---

## Current State (What's Broken)

❌ **Injection Points**

### Problem 1: narrative_state Hidden in System Context
**Location:** [prompts.py:3006-3007](captioner/prompts.py#L3006-L3007)
```python
if narrative_state:
    system_context_parts.append(f"YOUR STATE:\n{narrative_state}")
```

The model receives:
```
SYSTEM: [GAZE: ahead] [STATE: 4m awake, 12 observations]
        YOUR STATE:
        I'm watching a cluttered workspace...  ← Buried here

USER: Last thought: "The patterns of light..."
      What now?  ← No grounding!
```

### Problem 2: Continuation Questions Lack Grounding
**Location:** [prompts.py:3161](captioner/prompts.py#L3161), [prompts.py:3179](captioner/prompts.py#L3179)
```python
prompt_parts.append("What now?")
```

Asks for continuation without referencing accumulated knowledge.

### Problem 3: Mode Gating Limits Context
**Location:** [prompts.py:3072-3076](captioner/prompts.py#L3072-L3076)
```python
if prompt_mode == "introspective":
    if desire_hint:
        context_fragments.append(desire_hint)
    if belief_hint:
        context_fragments.append(belief_hint)
```

Desires/beliefs only appear in introspective mode. Regular observations (majority) don't get them.

### Problem 4: Activation Edges Never Summarized
The activation network learns:
- "ceiling and lighting often appear together"
- "desk and paper are usually nearby"

But this learned spatial/conceptual knowledge is stored and never injected into prompts.

---

## Proposed Fixes (Prioritized)

### Fix 1: Add Understanding to User Prompt ⭐ **START HERE**
**Impact:** HIGH | **Risk:** LOW | **Lines:** ~5

**Location:** After [prompts.py:3082](captioner/prompts.py#L3082) (after context_fragments block)

```python
# === ACCUMULATED UNDERSTANDING ===
if narrative_state and len(narrative_state) > 20 and not is_awakening:
    understanding = narrative_state.split('.')[0].strip()[:80]
    prompt_parts.append(f"You know: {understanding}.")
```

**Result:** Model sees "You know: I'm watching a cluttered workspace" before "What now?"

**Test:** Run 10+ minutes, check if "rediscovery" patterns decrease.

---

### Fix 2: Add Memory Synthesis Helper
**Impact:** MEDIUM | **Risk:** MEDIUM | **Lines:** ~30

**Location:** Create new function in [prompts.py](captioner/prompts.py)

```python
def build_memory_synthesis(activation_network) -> str:
    """Summarize learned associations from activation network."""
    parts = []

    # Top active concepts
    concepts = activation_network.get_top_active_concepts(threshold=0.5, limit=3)
    if concepts:
        parts.append(f"Your attention: {', '.join(concepts)}")

    # Strong learned edges
    edges = activation_network.get_strong_edges(threshold=0.6, limit=2)
    if edges:
        pairs = [f"{c1}+{c2}" for c1, c2, _ in edges]
        parts.append(f"You've learned: {' and '.join(pairs)} appear together")

    return " | ".join(parts) if parts else ""
```

Then inject into prompt_parts in `build_focused_caption_prompt()`.

**Test:** Check that learned associations appear in prompts and influence captions.

---

### Fix 3: Improve Continuation Questions
**Impact:** LOW | **Risk:** LOW | **Lines:** ~10

**Location:** Replace "What now?" at [prompts.py:3161](captioner/prompts.py#L3161), [prompts.py:3179](captioner/prompts.py#L3179)

```python
import random

GROUNDED_CONTINUATIONS = [
    "What else?",
    "Building on that...",
    "And now...",
    "Going deeper...",
]

# Replace: prompt_parts.append("What now?")
# With:
prompt_parts.append(random.choice(GROUNDED_CONTINUATIONS))
```

**Test:** Observe if language feels more connected/flowing.

---

### Fix 4: Remove Mode Gating for Core Context
**Impact:** MEDIUM | **Risk:** MEDIUM | **Lines:** ~3

**Location:** [prompts.py:3072-3076](captioner/prompts.py#L3072-L3076)

**Before:**
```python
if prompt_mode == "introspective":
    if desire_hint:
        context_fragments.append(desire_hint)
    if belief_hint:
        context_fragments.append(belief_hint)
```

**After:**
```python
# Always include desires/beliefs if available
if desire_hint:
    context_fragments.append(desire_hint)
if belief_hint:
    context_fragments.append(belief_hint)
```

**Test:** Run 20+ minutes, ensure introspective balance isn't disrupted.

---

## Implementation Order

1. **Phase 1:** Implement Fix 1 only
   - Add narrative_state to user prompt
   - Run 10+ minutes
   - Check logs for reduced rediscovery

2. **Phase 2:** If Fix 1 helps but isn't enough
   - Implement Fix 3 (continuation questions)
   - Observe language flow

3. **Phase 3:** If deeper context needed
   - Implement Fix 2 (memory synthesis)
   - Wire activation network edges to prompts

4. **Phase 4:** If appropriate
   - Implement Fix 4 (remove mode gating)
   - Monitor introspective/observational balance

---

## Testing Strategy

### Before Changes
```bash
# Run system, capture baseline
python machine.py --config_override config/debug_config.json > baseline.log 2>&1 &
# Let run 15 minutes
# Count "rediscovery" patterns (grep for "I see a" after 10+ minutes)
```

### After Fix 1
```bash
# Run with changes
python machine.py --config_override config/debug_config.json > fix1.log 2>&1 &
# Compare:
# - How often does model describe room from scratch?
# - Does it build on prior observations?
# - Does accumulated knowledge show in captions?
```

### Metrics to Watch
- Frequency of phrases like "I see a room" after 10+ minutes (should decrease)
- References to prior knowledge (should increase)
- Caption continuity (should improve)
- Compression quality (should remain stable)

---

## Code Locations Reference

| Component | File | Lines | Purpose |
|-----------|------|-------|---------|
| narrative_state retrieval | [prompts.py](captioner/prompts.py) | 2775-2780 | Gets compressed understanding |
| narrative_state injection (BROKEN) | [prompts.py](captioner/prompts.py) | 3006-3007 | Goes to system context |
| **FIX 1 LOCATION** | [prompts.py](captioner/prompts.py) | ~3083 | Add to user prompt here |
| Continuation questions | [prompts.py](captioner/prompts.py) | 3161, 3179 | "What now?" |
| Mode gating | [prompts.py](captioner/prompts.py) | 3072-3076 | Limits desires/beliefs |
| Activation network | [activation_memory.py](captioner/activation_memory.py) | - | Full implementation |
| Context compression | [context_compression.py](captioner/context_compression.py) | - | Working correctly |
| Memory observe | [memory.py](captioner/memory.py) | - | Working correctly |

---

## Success Criteria

After Fix 1 implementation, success looks like:

- [ ] Model references accumulated knowledge in captions
- [ ] Fewer "room discovery" captions after 10+ minutes
- [ ] Captions build on previous observations naturally
- [ ] narrative_state appears in user prompt (verify in logs)
- [ ] No degradation in caption quality
- [ ] Compression continues working normally

---

## Rollback Plan

If Fix 1 causes issues:
1. Remove the added lines (3 lines total)
2. System returns to previous behavior
3. No data loss, no broken state

The fix is purely additive - it doesn't remove or change existing behavior, just adds context to prompts.

---

## Notes

- The memory system architecture is **sound** - data collection works
- The issue is **presentation** - where/how data is shown to the model
- Fix 1 is minimal, targeted, and low-risk
- Can build incrementally based on results
- No need to refactor the entire system
