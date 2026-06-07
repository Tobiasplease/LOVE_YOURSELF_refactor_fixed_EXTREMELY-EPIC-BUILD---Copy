# Memory System Redesign Plan

## Status: June 2026

## The Core Problem

The memory system has ~4000 lines of code across 6 modules. Most of the infrastructure
works correctly in isolation, but **the outputs are disconnected from the prompt pipeline**.
The model generates desires, beliefs, discoveries, reflections, baseline understanding,
and attention threads — but almost NONE of these reach `build_simple_caption_prompt()`.

The result: sophisticated memory computation happens in the background, but the model
sees almost none of it during regular captioning. The only memory that actually reaches
the prompt is the thought thread (last caption) and the felt-state delta.

## Architecture Map (Current)

```
Camera Frame (30fps)
    |
    v
machine.py main loop
    |
    +---> frame_buffer.push() [for video mode]
    +---> captioner.update(frame)
              |
              v
         _process_frame() [every 10s]
              |
              +---> build_simple_caption_prompt()  <-- THE SINGLE INJECTION POINT
              |         |
              |         +-- 1. Situational line (time + gaze + person) [WORKING]
              |         +-- 2. Mode-gated context (relational/observational/workspace/introspective) [WORKING]
              |         +-- 3. Introspective context (drawing memory + semantic memory) [BROKEN - session_greeting disabled]
              |         +-- 4. Drawing/paper state [WORKING]
              |         +-- 5. Felt-state delta [WORKING]
              |         +-- 6. Thought thread (last caption sentence) [WORKING]
              |         +-- MISSING: baseline_context (environmental understanding)
              |         +-- MISSING: desires/beliefs (from compression introspection)
              |         +-- MISSING: discoveries (from compression introspection)
              |         +-- MISSING: reflections (from semantic memory)
              |         +-- MISSING: drawing history in non-introspective modes
              |
              +---> query_model() --> caption text
              |
              +---> Post-caption processing:
                    +-- semantic_memory.match_or_create_concepts(caption) [BROKEN - stores monologue fragments]
                    +-- activation_network.observe(concept_ids) [WORKS but input quality is garbage]
                    +-- context_compressor.add_caption(caption) [WORKS - generates baseline + desires + beliefs]
                    +-- episodic_log.record() [WORKS - person events only]
```

## What Exists But Is Disconnected

| System | Generates | Stored Where | Reaches Prompt? |
|--------|-----------|--------------|-----------------|
| context_compression | baseline_context | In-memory + machine_identity.json | NO (only via dead build_monologue_prompt) |
| context_compression | current_desire | machine_identity.json | NO (only via dead _build_concept_context) |
| context_compression | current_belief | machine_identity.json | NO (only in introspective system prompt) |
| context_compression | discoveries | machine_identity.json | NO (only fed back to introspection) |
| context_compression | felt_state | In-memory | YES (system prompt + user prompt) |
| semantic_memory | reflections | ChromaDB observations collection | NO (get_current_thread never called) |
| semantic_memory | concept labels | ChromaDB concepts collection | NO (get_session_greeting disabled) |
| activation_memory | attention state | activation_snapshot.json | NO (generate_state_summary never called) |
| activation_memory | boredom/novelty | In-memory | YES (drives mode selection only) |
| drawing_memory | artistic arc | drawing_memory.json | Only in introspective mode |
| episodic_log | events | episodic_events.json | Only person_arrived/left via situational line |

## Dead Functions (Safe to Remove)

~400 lines of dead code in the runtime path:

- `captioner/prompts.py`: `build_monologue_prompt()`, `select_perception_prompt()`, `_build_concept_context()`, `PERCEPTION_PROMPTS` dict
- `captioner/activation_memory.py`: `observe_and_store()`, `recall_for_prompt()`, `get_current_thread()`, `generate_state_summary()`
- `captioner/semantic_memory.py`: `after_perception()`, `get_established_labels()`, `get_session_greeting()` (disabled)
- `captioner/context_compression.py`: `get_current_sentiment_context()` (never called)

## Implementation Plan

### Phase 1: Reconnect Existing Wires (HIGH IMPACT, LOW EFFORT)

Wire the existing data that's being generated but never reaches the prompt.
Add to `build_simple_caption_prompt()` between felt-state and thought thread:

```python
# 5b. DESIRE/BELIEF (from compression introspection)
try:
    from captioner.context_compression import context_compressor
    desire = context_compressor.get_current_desire()
    belief = context_compressor.get_current_belief()
    if desire and len(desire) > 5:
        prompt_parts.append(f"Preoccupied with: {desire}")
    if belief and len(belief) > 5 and mode == "introspective":
        prompt_parts.append(f"What I know: {belief}")
except Exception:
    pass

# 5c. BASELINE CONTEXT (rolling environmental understanding)
if mode in ("observational", "workspace"):
    try:
        from captioner.context_compression import context_compressor
        baseline = context_compressor.get_baseline_context()
        if baseline and len(baseline) > 10:
            # Only first sentence — full baseline would dominate
            first_sent = baseline.split(".")[0] + "."
            prompt_parts.append(first_sent)
    except Exception:
        pass
```

### Phase 2: Core Memory Block

Extend existing `machine_identity.json` to include stable facts.
The compression introspection already generates beliefs/discoveries — add a
`core_facts` section that the consolidation cycle updates:

```json
{
  "current_desire": "...",
  "current_belief": "...",
  "discoveries": ["..."],
  "core_facts": {
    "place": "Workshop. Pink shelves left. Monitor on desk. Cables everywhere.",
    "people": "One regular visitor. Sits at desk, types.",
    "drawings": "47 total. Cables, shadows, hands recurring.",
    "self": "I fixate on cables. I draw better with company."
  }
}
```

The `core_facts.place` field replaces `get_session_greeting()` with curated content.
Updated by the reflection worker when concepts pass a threshold (>10 observations).

### Phase 3: Fix Concept Storage Quality

Replace `_extract_canonical_name()` regex with LLM-based extraction.
Run ONCE per compression cycle (every 8 captions), not per caption.
Extract from the compression output, which is already a clean summary.

Prompt for COMPRESSION_MODEL:
```
From this summary, list physical objects or spatial facts as noun phrases (2-4 words each).
Only concrete things that would be there next time. One per line. Max 3.
Summary: "{compression_output}"
```

This is batched and async — no latency impact on the caption cycle.

### Phase 4: Delete Dead Code

Remove orphaned functions listed above. Reduces cognitive overhead for future work.

### Phase 5: Episodic Consolidation

Every 30 minutes, scan episodic log + recent compressions:
- Person visited >5 times this session → update core_facts.people
- Same object in >10 concepts → promote to core_facts.place
- Drawing theme repeats → update core_facts.drawings

Runs on the compression background thread. No new threads needed.

## Token Budget for Prompt Injection

The prompt has a 150-word budget (enforced at end of build_simple_caption_prompt).
Current usage per mode:

| Section | Tokens (approx) |
|---------|-----------------|
| Situational line | 10-20 |
| Mode context | 10-30 |
| Introspective context | 15-40 |
| Drawing/paper state | 5-10 |
| Felt-state delta | 5-15 |
| Thought thread | 15-25 |
| **TOTAL current** | **60-140** |
| **Budget remaining** | **10-90** |

New injections (desires, baseline) should fit within the remaining budget.
If over, the existing line-level trimmer handles it (drops middle lines first).

## Risk Mitigations

1. **VRAM**: No new per-caption LLM calls. Fact extraction piggybacks on compression cycle.
2. **Latency**: All new injections are reads from in-memory or JSON. Zero network calls in prompt path.
3. **Quality**: Core memory block is LLM-curated (by compression/reflection), not regex-extracted.
4. **Corruption**: machine_identity.json already has load/save. Add a `.bak` write-ahead.
5. **Bloat**: Strict character limits on all core_facts fields (200 chars each).

## Files to Modify

| File | Change |
|------|--------|
| `captioner/prompts.py` | Add desire/belief/baseline injection to build_simple_caption_prompt |
| `captioner/context_compression.py` | Add core_facts persistence + promotion logic |
| `captioner/semantic_memory.py` | Replace _extract_canonical_name with LLM gate (in reflection cycle) |
| `captioner/prompts.py` | Delete dead functions (~400 lines) |
| `captioner/activation_memory.py` | Delete dead functions (~100 lines) |
