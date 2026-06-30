# Memory Redesign — the ledger model (June 2026)

Companion to `north-star.md` (spec), `voice-analysis.md` (diagnosis), and
`runtime-map.md` (wiring). The holistic plan for rebuilding the stores so they
enable development WITHOUT poisoning the base voice. Written after a full
three-part audit of every store, generator, and injection point on branch
`rebuild/north-star`. Checkpoint commit before this work: `582555b`.

> SUPERSEDES the earlier "reconnect the disconnected wires" plan (see History at
> end). That work is done — the wires are connected, and the problem has
> inverted: memory now over-reaches the prompt and contaminates the voice.

## The core reframe

We have been storing the model's **prose** and replaying it as **voice**. Every
contaminated channel takes the model's output sentences — purple, interpretive,
present-tense — and pastes them back as "who you are / what you saw / how you
sound." Memory became a *transcript of the voice*, so the voice imitates itself
and ratchets ("grid/dread" re-grew in 3 minutes even after a reset).

The fix is to make memory a **ledger, not a transcript**:

> Separate the **ledger** (what is true — neutral, structured, dated) from the
> **voice** (how it's said — regenerated fresh each time). The model never
> re-reads its own prose; it reads neutral facts and re-voices them now.

Development then lives in the ledger *changing over time* (new objects, new
drawings, evolving patterns, desires that resolve or curdle), not in stylistic
drift — a more authentic version of north-star's "develops, over time, its own".

## Two contamination mechanisms (we had only been addressing one)

1. **Injection** — stored prose pasted into a prompt. `BASE_VOICE_DETOX` gates
   this, but only for the caption prompt (20 of 52 sites). See blind spots below.
2. **Generation** — the compressors/synthesizers that WRITE the stores run on
   the model's prose and emit more prose. Even with perfect injection-gating the
   stores refill purple (this is why the persona re-grew). The ledger fixes this
   by making every generator emit **structured/neutral** output at low temp
   behind a hard validator.

`BASE_VOICE_DETOX` is a *test scaffold*, not the destination. The destination is
memory clean enough to leave in with the flag OFF.

## Detox blind spots (close these first — they make clean-room valid system-wide)

Three pipelines never check the flag and keep injecting stored prose:

| Pipeline | Sites | Where | Note |
|---|---|---|---|
| Awakening | 8 | `captioner.generate_internal_awakening` | seeds the whole session |
| Reflection loop | 9 | `reflection.py` + `build_reflection_loop_prompt` | separate thread |
| Drawing pipeline | 23 | `prompts.py` step1–5 builders | pulls raw `recent_captions[-20:]`; produced the contaminated art |

## The full ledger map — per channel

Format today / target / readers to migrate (orphan list). `cc` =
context_compression.py, `sm` = semantic_memory.py, `dm` = drawing_memory.py,
`p` = prompts.py, `c` = captioner.py. Line numbers are from the June audit —
re-grep before editing.

### 1. Concepts / objects  (LOWEST RISK — already mostly a ledger)
- **Now:** concept records ARE structured (label, times_seen, first/last_seen,
  session_count, spatial). The prose leak is the **observations** collection
  (stores raw monologue per concept) and surfacing.
- **Target:** keep concept records; stop storing raw monologue as observations
  (store extracted noun-facts or nothing); surface neutral familiarity
  ("the foam finger again — many times now"); add **decay** + **diversity**
  (don't resurface the same concept twice running) so no theme monopolizes.
- **Readers:** `p get_familiarity_line`, `p get_random_old_memory`,
  `sm _get_relevant_observation`, `sm _get_related_thoughts`,
  `c awakening concepts`, `activation_memory boost`. Generator: `cc` extract
  (temp 0.1, keep).

### 2. Drawings  (MEDIUM)
- **Now:** `dm` has `theme_tags` (structured) BUT also `compressed_summary`
  (prose) and `comfy_prompt` (raw purple); readers inject the prose.
- **Target:** surface from `{theme_tags, outcome, date}` — "you've drawn the
  desk 3 times; last one failed (no paper)". Keep `comfy_prompt` STORED (needed
  to reference the art) but NEVER inject it.
- **Readers:** `p get_last_drawing_description`, `dm.get_recent_drawings_summary`
  (reflection, journal, drawing step4/5), introspective context.

### 3. Place / baseline  (MEDIUM — sentence-split reader must change)
- **Now:** `baseline_context` prose (sentence-split in build_simple_caption_prompt),
  `core_facts[place]` prose.
- **Target:** structured object inventory derived from concepts; surface as a
  neutral list. Retire the baseline prose generator or reformat it to emit a list.
- **Readers:** the baseline sentence-split (must change), `cc` stagnation check,
  journal writer.

### 4. Felt-state  (LOW–MEDIUM — candidate for retirement)
- **Now:** prose adjective phrase + punctuation/word-count validators.
- **Target:** a valence/arousal **tag** from a fixed vocabulary, or retire the
  injection entirely (low value, high register-risk). Both injection points are
  detox-gated already.
- **Readers:** system prompt `Right now:`, caption felt-delta, `cc get_felt_state`
  (validators).

### 5. Desire / belief  (MEDIUM–HIGH — many readers)
- **Now:** prose, raw-injected; `activation_memory` expects strings; drawing
  pipeline injects.
- **Target:** desire as `{goal_object, status: open/resolved/abandoned, since}`;
  surface re-voiced as a current preoccupation. The status field is what makes a
  desire ARC across days (north-star principle 4) instead of regenerating.
- **Readers:** drawing step3, caption `Preoccupied with:`, awakening,
  `activation_memory get_beliefs/get_desires`, reflection context.

### 6. Reflections  (HIGH — core to development)
- **Now:** long prose, injected as echo + drawing pipeline + reflection context.
- **Target:** store the reflection's **conclusion as a neutral proposition** +
  subject; surface the *subject* to re-think, not the prose to re-read. Keep full
  text only for the reflection thread's own continuity, gated from captions.
- **Readers:** `p get_reflection_echo_line`, `reflection get_recent_reflections`.
- **Generator:** `reflection.py` (temp 0.75) — must emit a structured conclusion
  alongside the prose.

### 7. Persona (core_facts[self])  (HIGHEST RISK — do LAST)
- **Now:** a quoted purple sentence, raw-injected into the system prompt.
- **Target:** a small set of structured **trait records** (recurring fixations,
  what shifts its mood, preferences) built from now-plain captions under a strict
  low-temp plainness gate; expressed fresh, quoted only if it passes the gate.
- **Readers:** system prompt persona block. Generator: `cc _synthesize_self_model`
  (temp 0.4) fed by desire/belief histories + discoveries — so it can only be
  clean once 5 and 6 are.

## Orphan-risk readers (assume prose; must be migrated WITH their channel)
- baseline sentence-split in build_simple_caption_prompt
- felt-state word/punctuation validators (system prompt + `cc get_felt_state`)
- `activation_memory get_beliefs/get_desires` (treat desire/belief as strings)
- `cc _synthesize_self_model` / journal writer (discoveries list-iteration)
- raw-inject sites: `core_facts[self]` (system prompt), desire (drawing step3,
  caption, awakening)

Already structured & robust (safe): `journal`, `desire_history`,
`belief_history`, `compression_history`, episodic_log, concept metadata.

## Three mechanisms that make it develop AND stay clean
1. **Neutral storage gates** — every generator emits structured/neutral data at
   low temp behind a hard validator; non-neutral output is rejected, not stored.
2. **Decay + diversity** — unreferenced records fade; don't resurface the same
   theme repeatedly. Turns "fixate forever" into "learn over time".
3. **Re-express, don't replay** — stored facts are inputs to a fresh thought in
   the current voice; the past surfaces as a prompt to think, not a sentence to
   imitate.

## Migration order (risk-ascending) + validation protocol
0. **[DONE — commit 1101f74]** Closed the detox blind spots. NOTE: there were
   FOUR, not three — memory mode (a separate caption branch every 4 min that
   quoted a raw old caption) was the fourth, closed in step 1.
1. **[DONE — commit 6ae0d2d]** Concepts/objects ledger: memory mode reframed to
   a neutral concept fact (re-express not replay) + gated; familiarity gained
   decay/diversity; per-caption garbage-concept creation removed; noun-phrase
   gate on creation + surfacing; 8 dead methods removed. PENDING: the existing
   concept store has inflated counts/garbage from prior runs — a purge is a
   separate data decision (like the persona reset).
2. **[DONE — June 28]** Drawings ledger: `get_recent_drawings_summary` and
   `get_last_drawing_description` now surface the structured **theme tags** (+
   recency + outcome) — "chair, cables (about 10 minutes ago)" — never the
   stored `compressed_summary` / raw `comfy_prompt` prose (kept for reproducing
   the art, never injected). Readers (workspace/introspective context, journal,
   core-facts, reflection, drawing pipeline) take the bare phrase + their own
   framing. PENDING: `get_artistic_arc` still LLM-narrates from `comfy_prompt`
   — but it's drawing-pipeline-only + detox-gated; folds into the separate
   drawing-pipeline cleanup (its 5 step system-prompts also push metaphor).
3. **[DONE — June 28]** Place/baseline ledger: `place` is now DERIVED from the
   concepts store (`get_place_inventory` → "desk, mannequin head, humming fan,
   red foam finger") instead of LLM prose; `get_core_facts_string` surfaces
   place (concepts) + people (pattern, awakening only) and drops drawings
   (drawing_memory is the sole channel — one channel per fact). The
   `baseline_context` sensory-prose caption injection (5c, the sentence-split
   reader) is RETIRED — redundant with place + familiarity. core_facts['place']/
   ['drawings'] LLM prose is now stored-unused (stop generating = follow-up).
4. **[DONE — June 28]** Felt-state: kept (its purpose — translating the abstract
   mood vector into something the LLM grasps — is valid), but now DERIVED
   deterministically from the valence/arousal vector via `mood.mood_to_feeling`
   instead of LLM prose. Plain + DEGREED: "a little happy", "very anxious",
   "really excited", "calm" — emotion word + intensity adverb (per the user:
   "describe an emotion to an autistic person", but with gradation). Set via
   `set_felt_state` from the captioner where the mood vector lives; compression
   no longer generates or parses felt (single source of truth). The old LLM
   felt-storage block + parser felt-extraction removed.
5. **[DONE — June 28]** Desire/belief ledger: the `WANT:/NOTICED:/DISCOVERED:`
   fill-in template (temp 0.7, formulaic, label-leaked "NOTICED:" into beliefs)
   replaced with OPEN questions (WANT/THINK) that allow "nothing", at temp 0.4.
   Parser rewritten to strip any leaked label robustly. DISCOVERED retired (it
   was the most purple input to the persona). ARC (principle 4): a desire now
   persists — `desire_since` is kept while the wish stays roughly the same
   (`_roughly_same`), only resetting on a genuinely different desire; persisted
   across restarts. Readers unchanged (current_desire stays a string). FOLLOW-UP:
   surfacing the "since" ("you've wanted X for days") and closing the arc
   (desire → drawing → resolved/abandoned) need the drawing pipeline (detox-gated).
6. **[DONE — June 28]** Reflections: `get_reflection_echo_line` now surfaces the
   reflection's SUBJECT to re-think ("Something that was on your mind 6 days ago:
   the room.") instead of quoting its prose first-sentence ("…the residue of what
   almost happened"). So even the 87 old contaminated reflections surface safely
   (subject only). The full long-form text stays ONLY for the reflection thread's
   own continuity (reflection.py get_recent_reflections → next reflection,
   principle 3) — never quoted into a caption. PENDING: the reflection→drawing
   injection (captioner.get_last_reflection → drawing extra_context) is
   drawing-pipeline-scoped + detox-gated; and the reflection generator's own
   long-form register (temp 0.75) is a separate quality axis, now gated from
   captions. Both fold into the drawing-pipeline cleanup.
7. **[DONE — June 28]** Persona: `_synthesize_self_model` now GROUNDS the
   synthesis in the concepts ledger ("Things I keep noticing: desk, mannequin
   head, foam finger…") + the now-plain desire/belief histories (step 5), at
   temp 0.4→0.3. That grounding is the primary defense — it anchors "I fixate
   on X" to real objects so the register can't drift to "silhouettes breaking my
   grid". `_valid_self_fact` strengthened as a BACKSTOP (rejects similes
   "like a"/"as if", >24 words, in addition to surveillance/reality/third-person)
   — but it honestly can't catch metaphor-without-markers; that relies on the
   grounding + clean inputs not GENERATING it (north-star: fix what's stored,
   not the mouth). Persona is empty (reset step 1) and will re-form from the
   clean grounded synthesis — what it re-grows is the test.

## Migration complete (steps 0–7). Remaining before detox OFF
- **Drawing-pipeline cleanup** (its own bucket): the 5 step system-prompts push
  metaphor; get_artistic_arc LLM-narrates comfy_prompt; reflection→drawing
  injection; core_facts['place']/['drawings'] LLM prose still generated unused.
- **Turn detox OFF** channel-by-channel and re-judge each surfaces clean, per the
  validation protocol — the stores are now ledgers, but this has not yet been
  run end-to-end with memory back on.

**Per step:** reformat store → reformat generator (structured, low temp, gate) →
migrate EVERY reader in its orphan list → re-enable that channel with detox OFF →
run `debug/test_base_voice.py` and a short `machine.py` clean-room → confirm the
naked voice holds plain AND the channel's facts surface correctly → commit. Never
two channels at once; the harness is the regression test.

## How development is preserved (the whole point)
Personality emerges from WHICH facts accumulate (real drawings, objects, visitor
patterns) + HOW it consistently reacts (traits extracted over time), grounded in
events — not from replaying a self-reinforcing mood. Desire arcs because it's a
tracked object with a status. It learns because the ledger evolves and decays
with real salience. What it stops doing is mistaking its own diary for itself.

## History (superseded plan, kept for the runtime-map cross-reference)
The prior version of this file was the **"reconnect the disconnected wires"**
plan: at the time, compression/desire/belief/baseline were generated but never
reached `build_simple_caption_prompt`. Phases 1–5 of that plan wired them in
(desire → "Preoccupied with:", baseline first sentence, core_facts block,
LLM concept extraction at compression time, episodic consolidation) and removed
the dead `build_monologue_prompt`/`get_session_greeting`/`_build_concept_context`
paths. Completing it created today's inverse problem — too much connected,
contaminating the register — which this ledger plan addresses by changing WHAT is
stored/surfaced (facts, re-voiced) rather than WHETHER it connects.
