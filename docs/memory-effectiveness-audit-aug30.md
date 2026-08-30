# Memory Effectiveness Audit — Aug 30, 2026

The audit `memory-runtime-review-2026-07-30.md` ranked #5 and never ran:
"Activation-memory audit — retire or repurpose; suspected legacy." Traced
symbol-by-symbol on the current head (post trim-pass). The question answered
for every layer: what writes it, what reads it, and which reads actually land
text in a model prompt or change a call's parameters.

Note on model-specific docs: there is NO dedicated memory doc for Qwen3.8.
The memory docs are era-stamped 9B/3.6 (`memory-redesign-plan`,
`memory-evolution-plan`, `mood-novelty-audit`, `continuity-plan`); the only
3.8-era memory writing is caption-system-audit-aug28 Part 2, plus the
`run_38.sh` comments (STREAM_WINDOW/STREAM_CONSOLIDATE_CHARS "until 3.8's
verbosity is measured" — still unmeasured). This doc is now the 3.8-era
memory reference.

## Verdict in one paragraph

The memory that works is the *text* memory: the stream window, hour_log,
the concepts ledger, the reflection loop + its ChromaDB store, the identity
slots (self/desire/belief/felt), the durable ledger, and the journal. The
**activation network is vestigial** — its two outputs are a novelty scalar
with literally zero behavioral effect and a boredom scalar whose entire
effect is a 0.05 temperature nudge + 30 tokens — while it costs a
synchronous full-state JSON dump every caption, an extra ChromaDB
embed+query every 8 captions, and (worse) that feedback query **silently
inflates `times_seen` on the concepts ledger**, corrupting the counters the
familiarity line and memory mode actually read. `mood-novelty-audit.md:55`
predicted exactly this outcome on Aug 12; it has fully come true.

## 1. ActivationNetwork + ContextualMemory — VESTIGIAL

- **Novelty → nothing.** `_last_novelty` is passed to `determine_prompt_mode`
  (`prompts.py`), whose body no longer references it (the observational
  branch was the last consumer), and to `should_draw` (`drawing/drawing.py:180`)
  where it appears only in `_log_trigger_decision` — the verdict comes from
  the drive/desire ladder. A printed number and a visualizer field.
- **Boredom → one knob.** Computed in `calculate_boredom`, consumed only at
  `captioner.py:1590`: `boredom > 0.7` selects num_predict 110 vs 80 and
  temperature 0.85 vs 0.9 (and the p=0.2 short beat pre-empts it 20% of the
  time). The `prompts.py:1402-1404` docstring claiming boredom reaches the
  model "via the identity line" is FALSE (audit-aug28 flagged it; still
  unfixed). ~80% of the boredom signal derives from `times_seen`/`is_new` —
  preservable in ~15 lines without the network.
- **Edges never persist.** Module `save_state()` has zero callers, so
  `activation_edges.json` is never written by the live process; every
  session's learned edges die at exit, and `_load_edges` reads a fossil.
  Cross-session edge learning has silently not existed.
- **Costs:** `save_comprehensive_snapshot` does a synchronous `json.dump`
  of the full edge matrix (O(concepts²), 219 concepts) EVERY caption — the
  dominant per-caption memory cost. `boost_from_compression` fires a second
  `match_or_create_concepts` every 8 captions, which **bumps
  `times_seen`/`last_seen`/`session_count`** on matched concepts — the
  activation feedback loop contaminates the semantic ledger that memory
  mode ("countless times") and the familiarity gate (`times_seen >= 3`)
  read. Net-negative, not just wasted.
- **Retiring it loses:** the boredom nudge (replaceable), the visualizer
  snapshot (`debug/activation_visualizer.py`, manual-run only). Nothing else.

## 2. What is doing real work (keep, this is the memory that matters)

stream window + consolidation (the continuity spine) · `hour_log` (80
verbatim captions to reflection, 30 to drawing intent — the richest
read-back) · SemanticMemory `concepts` (familiarity line, memory mode,
place inventory, awakening recognition, gaze nudge) · `reflections` store +
the reflection loop (the only becoming pipe) · felt/desire/belief/
`core_facts["self"]` (dosed into prompts) · durable ledger (the only
cross-day surface) · journal (awakening + time organ) · spatial registry
(gaze targeting + the absence line) · `presence_identity.singular_regime`
(the "He's come in" vs "Someone's come in" decision — and the code source
of "forever the man").

## 3. Write-only archives (decide: give them a reader, or stop writing)

1. **`observations` collection** — the single largest write volume (one
   embedded, deduped, pruned row per caption) with ZERO readers
   (`get_concept_observations` has no callers). Either a per-concept
   "what you've said about this before" surface, or stop writing.
2. **Episodic person spans** — arrivals/departures DO reach the visitor
   reflection organ (so "became nothing" was half right), but no
   visit-pattern distillation slot exists = audit-aug28 B2.
   **`drew` events have no reader at all.**
3. **`baseline_context`** — an LLM call every 8 captions whose output
   reaches only the drawing system prompt and reflection, never a caption.
4. **`activation_snapshot.json` + ContextualMemory's 200-entry ring** —
   feed only the manual visualizer.

## 4. Dead compute and broken bits found on the way

- `get_activation_summary_for_compression` + `activation_context` +
  `historical_context` + `duration_description`
  (`context_compression.py:302-334`): built every compression, never
  interpolated — the template has three placeholders and none of these is
  one of them.
- **MemoryMixin dead state**: `timeline` is NEVER appended (only
  save/restored) so `temporal_prompt_lines`' last-person clause emits
  "no person yet" into the drawing system prompt **permanently and
  falsely**; `self_model`, `day_stones`, `known_people`, `primary_person`
  written-never-read or never written; all still serialized every save.
- `semantic_memory.py:849-861`: an orphaned `delete_concept` body sits
  unreachable after a `return` inside `get_concept_observations` — live
  delete calls inside a getter that lost its `def` line.
- Stale doc claims: `runtime-map.md:339` says mood takes a "novelty nudge"
  (it doesn't — `mood/mood.py` nudges on `saw_person` only); the boredom
  docstring above; `memory-redesign-plan.md:106,129` list
  `activation_memory get_beliefs/get_desires` as live readers (both gone).

## 5. How the field does it (survey, Aug 30 2026) — and what to steal

Checked against the current state of the art in agent memory (Letta/MemGPT,
Stanford generative agents, Zep/Graphiti, Mem0, and the 2025-26 consolidation
literature). The honest headline: **this system independently converged on
most of the right architecture.** The reflection loop IS what Letta ships as
a "sleep-time agent" (a background consolidator that edits the main agent's
memory, deliberately separated from the talking agent); the identity slots
ARE Letta's "memory blocks" (labeled, size-limited, re-injected context
sections); the ledger-not-transcript reframe and the storage gates are what
the literature now calls write-gating, invented here from spiral scars. The
gaps are specific and small:

1. **Importance scoring at store time (generative agents).** Park et al.'s
   retrieval is recency × relevance × IMPORTANCE — each memory gets a
   one-shot salience rating when stored. Our stream and reflection echo have
   recency and relevance; nothing anywhere scores importance, so a caption
   about the wall and a caption about a visitor's arrival weigh the same in
   every downstream fold. Cheapest structural upgrade on the list.
2. **Validity windows instead of TTLs (Zep/Graphiti).** Temporal-KG memory
   stores facts as bi-temporal — valid_at/invalid_at set by contradiction,
   never deletion. That is the general form of the paper-state redesign
   (event + relevance, kill the standing 30-min TTL) and of the north-star
   would-it-lie test: a fact that can go stale carries when-it-was-true, and
   a new observation CLOSES the old fact rather than coexisting with it.
   Worth adopting as the pattern for every stateful fact (paper, presence,
   pen), not just paper.
3. **Provenance on confirmations (the faulty-update literature).** 2025-26
   work on continuously self-updated memories documents exactly the durable
   ledger's failure: an agent that re-reads its own memory and re-stores
   agreement inflates confidence without evidence. The mitigation is the
   already-planned B1 (discount in-prompt confirmations) plus a provenance
   field per confirmation (distill vs echo vs fresh observation). The field
   considers this table stakes; we planned it and haven't built it.
4. **Reflections that cite their evidence (generative agents).** Park's
   reflections store pointers to the observations they derive from, so a
   conclusion can be traced and re-weighed. Our distillates land in three
   anonymous slots. B2/B3's outcome slots are a step toward typed,
   evidenced self-knowledge.
5. **What NOT to import:** the retrieval-heavy frameworks (Mem0-style
   per-turn fact extraction into a graph) solve chat personalization, not
   a continuous single-viewpoint perception stream — our stream window is
   already the right short-term memory for that. And LongMemEval-style
   benchmarks don't transfer; the north-star "how we'll know it's working"
   week-test remains the right eval.

Sources: [Letta sleep-time agents](https://docs.letta.com/guides/agents/architectures/sleeptime/),
[Letta memory blocks](https://www.letta.com/blog/memory-blocks/),
[Zep temporal knowledge graph](https://www.getzep.com/ai-agents/temporal-knowledge-graph/),
[Zep architecture paper](https://arxiv.org/abs/2501.13956),
[Mem0 vs Zep comparison](https://vectorize.io/articles/mem0-vs-zep),
plus the 2025-26 arXiv consolidation/faulty-update literature
(e.g. "Useful Memories Become Faulty When Continuously Updated by LLMs").

## 6. Recommendation (order)

1. **Retire ActivationNetwork + ContextualMemory.** Keep the boredom
   scalar as ~15 lines off `matched_concepts` (`times_seen`/`is_new`/
   person). Delete the per-caption snapshot dump, the every-8
   ledger-contaminating boost, novelty end-to-end, `get_beliefs`, the
   unreachable persistence. This ends the `times_seen` inflation — a
   correctness fix for the memory that IS read, not just a cleanup.
2. Delete the §4 dead-compute pocket and MemoryMixin dead state; fix the
   "no person yet" lie (drop the clause or feed it from episodic_log).
3. Decide the `observations` collection (reader or retirement) — it's the
   biggest ongoing write cost in the system.
4. Then the becoming work (audit-aug28 B1-B4) — visit-pattern distillation
   is what episodic spans are waiting for; that, not more stores, is the
   effectiveness gap. The system remembers plenty; it distills almost none
   of it into anything the voice can use.
