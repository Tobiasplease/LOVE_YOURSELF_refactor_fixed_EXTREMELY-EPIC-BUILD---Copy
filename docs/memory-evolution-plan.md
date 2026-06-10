# Memory Evolution Plan — Natural Learning + Personality Development

## Status: June 2026, follows memory-redesign-plan.md phases 1-4 (completed)

## Implementation Status (June 10, 2026)

| Phase | Status | Where |
|-------|--------|-------|
| A1 Familiarity line | ✅ DONE | `prompts.py: get_familiarity_line()`, section 3c; `captioner.py` stashes `_last_matched_concepts`; `semantic_memory.py` returns `session_count` in matches |
| A2 Awakening recognition | ✅ DONE | `captioner.py: generate_internal_awakening()` — top-2 multi-session concepts |
| B1 Real memories | ✅ DONE | `semantic_memory.py: get_random_old_memory()`; `prompts.py: build_memory_mode_prompt()` quotes verbatim with age |
| B2 Tangent recall | ⏸ deferred | watch prompt budget first |
| C1 Self-synthesis | ✅ DONE | `context_compression.py: _synthesize_self_model()` — every 3rd introspection, from discoveries + desire/belief history |
| C2 Persona block | ✅ DONE | `prompts.py: get_monologue_system_prompt()` appends core_facts.self; excluded from user-prompt facts string |
| C3 Desire gating | ✅ DONE | `prompts.py` section 5b — max 3 injections per desire via `desire_injection_count` |
| D1 Journal write | ✅ DONE | `context_compression.py: _maybe_write_journal()` every 30 min on compression thread + `write_journal_now()` on shutdown |
| D2 Journal read | ✅ DONE | awakening injects last entry ("From my diary, last time: ...") + entry count |
| E1 Visit consolidation | ⬜ TODO | episodic log → core_facts.people |
| E2 Person identity | ⬜ TODO | vision branch (OSNet) |
| F Belief injection + hygiene | ⬜ TODO | |

Verified: imports clean, familiarity gating works (1 injection per ~3 captions,
same-concept repeat suppressed).

## The Goal

The machine should **naturally evolve over time**: learn the space around it,
recognize what it has seen before, remember actual past thoughts, and develop
a stable personality from accumulated experience. The scaffolding exists —
this plan is about *utilization*, not new infrastructure.

## Current State (what phases 1-4 achieved)

| Data | Generated | Reaches prompt? |
|------|-----------|-----------------|
| baseline_context | every 8 captions | YES (observational/workspace, 1st sentence) |
| felt_state | every compression | YES (system + user) |
| current_desire | every ~3 compressions | YES — but unconditionally (echo-loop risk) |
| current_belief | every ~3 compressions | **NO** |
| discoveries | introspection | **NO** (only fed back to introspection) |
| core_facts (place/people/drawings/self) | introspection | YES (caption 3b + awakening + memory mode) |
| ChromaDB concepts (clean LLM labels) | per compression | **NO** — matched per caption, familiarity never surfaces |
| ChromaDB observations (past thoughts) | per caption | **NO** — write-only store |
| episodic person events | debounced arrive/leave | YES (situational line) |
| ByteTrack person IDs | per detection | **NO** — not connected to memory |
| drawing arc | per completed drawing | introspective mode only |

## The Five Gaps

1. **Familiarity is invisible.** The machine sees the pink shelf for the 50th
   time and the prompt says nothing. Recognition is the core experience of
   "learning a space" — without it, every caption is a first encounter.
2. **No genuine remembering.** Memory mode confabulates from a static summary.
   ChromaDB holds hundreds of actual past thoughts that are never read back.
3. **Personality doesn't consolidate.** Discoveries accumulate (capped at 10)
   but never become self-knowledge. core_facts.self is distilled only from
   spatial compression history, not from desires/discoveries.
4. **No session narrative.** Nothing summarizes a session. Awakening gets the
   last caption + desire, but there's no diary, so no long-term arc.
5. **People are statistics, not memories.** "Come and gone 11 times" — no sense
   of a *regular*, no per-person memory. ByteTrack IDs now exist but unused.

## Design References

- **Generative Agents (Stanford)**: memory stream + retrieval (recency ×
  importance × relevance) + periodic reflection that synthesizes higher-level
  insights. We have the stream (ChromaDB) and the reflection slot
  (introspection cycle); we're missing retrieval and synthesis.
- **MemGPT/Letta**: core memory blocks — a *persona block* the agent edits
  about itself, injected into the system prompt to color voice. core_facts is
  exactly this pattern; core_facts.self belongs in the SYSTEM prompt.
- **LENS finding**: don't over-structure. Inject natural sentences, never
  data dumps.
- **May 2026 lesson (echo loop)**: desire injected every caption → monologue
  yearning → compressed back into desire → runaway loop. Any always-on
  injection of model-generated affect must be gated.

## Implementation Phases (ordered by impact)

### Phase A: Recognition — make familiarity visible (1-2h)

**A1. Familiarity line in caption prompt.**
`match_or_create_concepts()` already returns `times_seen`, `is_new`,
`first_seen`, `session_count` per matched concept — store the result on the
agent (`agent._last_matched_concepts`) and let `build_simple_caption_prompt()`
inject ONE line for the most notable concept:

- `is_new` → "Something you haven't noticed before: {label}."
- `times_seen` ≥ 10, `session_count` ≥ 2 → "That {label} again — it's always there."
- 3 ≤ times_seen < 10 → "{label} — you've noticed it a few times now."

Guards (lessons from the old triple-echo bug):
- Max 1 concept per caption; skip person concepts (situational line covers people).
- Never inject the same concept twice in a row (track last injected id).
- Only inject every 2nd-3rd caption — recognition is occasional, not constant.

**A2. Cross-session recognition at awakening.**
Concepts with `session_count > 1` exist → top-2 by times_seen into the
awakening prompt: "I know this place — the pink shelf, the desk." This is
get_session_greeting reborn, now safe because labels are LLM-clean.

### Phase B: Genuine remembering (1-2h)

**B1. Memory mode pulls real memories.**
Replace the static core_facts text in `build_memory_mode_prompt()` with an
actual stored observation from ChromaDB:
- Filter: timestamp > 1 hour old; prefer a different session.
- Pick randomly among the candidates (variety over relevance).
- Inject verbatim with age: `Earlier (2 days ago) you thought: "{text}"`.
- Fall back to core_facts while the store is thin (< 20 observations).

This converts memory mode from confabulation into actual remembering — the
machine quotes its own past thoughts.

**B2 (later). Tangent recall on boredom.**
`recall_tangent()` exists, never called. When boredom > 0.7 and nothing new is
matched, inject one associative old thought. Deferred — watch prompt budget.

### Phase C: Personality consolidation (2h)

**C1. Self-model synthesis.**
Every ~3rd introspection (~10 min), run a small LLM call over
`discoveries` + `desire_history` + `belief_history`:
"From these, state one or two stable facts about yourself — preferences,
fixations, habits. Under 20 words." → updates `core_facts.self`.
This is the Generative-Agents reflection step applied to identity.

**C2. Persona block in the SYSTEM prompt.**
`core_facts.self` moves from user-prompt data (section 3b) into
`_MACHINE_IDENTITY_BASE`: "You are a drawing machine... {self_knowledge}".
e.g. "You fixate on cables. You're calmer when someone is in the room."
Personality should *color the voice*, not be recited as fact. (MemGPT persona
block pattern.) Keep place/people/drawings facts in the user prompt.

**C3. Desire injection gating (regression guard).**
Use the existing `desire_injection_count`: inject "Preoccupied with:" only for
the first 3 captions after a desire changes, then stop until introspection
produces a new one. Prevents the documented May 2026 yearning echo loop.

### Phase D: Session journal — the long-term arc (2h)

**D1. Journal writing.**
On shutdown + every 30 min (compression thread): one LLM call summarizing the
session so far — compression history, discoveries, drawing events, visitor
presence — into 2-3 first-person past-tense sentences. Append to
`machine_identity.json["journal"]`: `{date, summary}`, keep last 30.

**D2. Journal reading at awakening.**
Awakening prompt gets the most recent entry: "Last time: {summary}". For gaps
> 1 day, also the entry count: "I have 12 days of memories of this place."
This is the genuine cross-session arc — the machine wakes up *with a past*.

### Phase E: People memory (builds on ByteTrack) (1-2h)

**E1. Visit-pattern consolidation** (old Phase 5).
Every 30 min on the compression thread, scan episodic log: visits today,
typical duration → `core_facts.people`: "One regular. Comes most days, stays
about an hour." Debounced events (this session) make these stats meaningful.

**E2 (vision branch). Appearance-based identity.**
OSNet embeddings → persistent person IDs across sessions → per-person memory
("the one with glasses is back"). Tier 2 of the vision plan; memory side just
needs a `people` dict in machine_identity.json keyed by person id.

### Phase F: Hygiene (30 min)

- Inject `current_belief` in introspective mode only ("What I know: ...").
- Single persistence path for desires (machine_identity.json only).
- Cap and dedup `discoveries` against core_facts.self (consumed discoveries
  can be dropped once consolidated).

## Token Budget

Caption prompt budget is 150 words; current usage 60-140.
- A1 adds ~10 words, every 2nd-3rd caption only.
- C2 moves words OUT of the user prompt into the system prompt.
- B1 replaces existing memory-mode content (no growth).
- D2 only fires at awakening.
Net effect on the regular caption prompt: roughly neutral.

## Risk Notes

- **Echo loops**: anything the model generates that gets re-injected must be
  gated (C3) or grounded in verbatim quotes (B1) rather than paraphrase.
- **Latency**: C1 and D1 piggyback on the existing compression background
  thread — zero new per-caption calls.
- **Corruption**: journal + core_facts live in machine_identity.json which
  already has load/save; add .bak write-ahead when implementing D1.

## Suggested Order

1. A1 — familiarity line (biggest perceived change, smallest diff)
2. C2 + C1 — persona block + self-synthesis (the personality goal)
3. B1 — real memories in memory mode
4. D1 + D2 — journal
5. C3 — desire gating
6. A2, E1, F — follow-ups
