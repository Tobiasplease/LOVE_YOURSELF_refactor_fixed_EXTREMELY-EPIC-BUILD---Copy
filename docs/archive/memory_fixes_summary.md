# Memory Injection Fixes - Implementation Summary

## Changes Implemented

### Fix 1: Narrative State to User Prompt ✅
**Status:** IMPLEMENTED
**Location:** [prompts.py:3090-3094](../captioner/prompts.py#L3090-L3094)

Adds compressed understanding from `narrative_state` prominently to user prompt:
```python
if narrative_state and len(narrative_state) > 20 and not is_awakening:
    understanding = narrative_state.split('.')[0].strip()[:80]
    prompt_parts.append(f"You know: {understanding}.")
```

---

## Proposed Changes

### Fix 2: Reweight Mode Selection
**Status:** PROPOSED
**Location:** [prompts.py:2308-2314](../captioner/prompts.py#L2308-L2314)

**Current thresholds:**
```python
if novelty > 0.5:      # observational mode
    return "observational"

if boredom > 0.5:      # restless mode
    return "restless"
```

**Proposed (more selective):**
```python
if novelty > 0.65:     # only truly novel things trigger observational
    return "observational"

if boredom > 0.75:     # only extreme boredom triggers restless
    return "restless"
```

**Effect:** Lets introspective mode trigger more often (when novelty=0.5-0.65 or boredom=0.5-0.75)

---

### Fix 3: Inject Desires/Beliefs in User Prompt for Internal Modes
**Status:** PROPOSED
**Location:** [prompts.py:3078-3082](../captioner/prompts.py#L3078-L3082)

**Current (hidden in context_fragments):**
```python
elif prompt_mode == "introspective":
    if desire_hint:
        context_fragments.append(desire_hint)  # Goes to bracketed context
    if belief_hint:
        context_fragments.append(belief_hint)
```

**Proposed (prominent in user prompt):**
```python
# After narrative_state injection (around line 3095):

# Internal states get desires/beliefs prominently
if prompt_mode in ("introspective", "restless") and not is_awakening:
    if desire_hint:
        if prompt_mode == "restless":
            prompt_parts.append(f"You want: {desire_hint}")
        else:  # introspective
            prompt_parts.append(f"You feel: {desire_hint}")

    if belief_hint:
        prompt_parts.append(f"You sense: {belief_hint}")
```

**Effect:**
- Introspective: "You feel: I want to explore each object in detail..."
- Restless: "You want: I want to explore..." (more active framing)
- Both get semantic beliefs: "You sense: This space has both natural and artificial light"

---

## What This Unlocks

**Before (current state):**
```
SYSTEM: [hidden context]
        MEMORY: I remember 15 minutes ago...
        ASSOCIATIONS: desk and papers often together
        YOUR STATE: I'm watching a cluttered workspace

USER:   Looking down.
        Last thought: "Papers scattered on the surface"
        What now?
        (Short. Fragments OK.)
```

**After (with all fixes):**
```
SYSTEM: [cleaner, less buried]

USER:   Looking down.
        You know: I'm watching a cluttered workspace.
        You feel: I want to explore each object in detail and understand their purpose.
        You sense: This space has both natural and artificial light.
        Last thought: "Papers scattered on the surface"
        What now?
        (Short. Fragments OK.)
```

---

## Rich Context Available (From machine_identity.json)

**Current desire (LLM-generated):**
> "My interest has shifted from simply observing the light to exploring each object in detail; I want a deeper understanding of how they work, their purpose within this space. I no longer wish solely for visual exploration but now crave knowledge about these devices' functionality."

**Current belief (LLM-generated):**
> "This space has both natural and artificial light"

**This semantic richness EXISTS** - it's just hidden!

---

## Testing Plan

1. **Implement Fix 2** (reweight thresholds)
   - Run 15+ minutes
   - Count mode distribution: should see more `introspective`

2. **Implement Fix 3** (inject desires/beliefs to user prompt)
   - Watch for desires/beliefs in captions
   - Should see continuity building on identity

3. **Combined effect**
   - Captions should show accumulated understanding
   - Less "rediscovery" of the room
   - More coherent narrative development

---

## Rollback Plan

All changes are minimal and reversible:
- Fix 2: Change 2 numbers back
- Fix 3: Remove 4-8 lines of injection code
