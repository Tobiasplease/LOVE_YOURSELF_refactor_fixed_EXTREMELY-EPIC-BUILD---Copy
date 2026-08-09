# Prompt Inventory — every LLM pass in the system (August 2, 2026)

**Purpose.** The complete, verified list of every place this codebase talks to a
model. This is the foundation for the prompt panel (`docs/prompt-panel-plan.md`)
and the answer to "are we sure we've got all of them?" — a question that has
burned this project before (silently-dead features, half-retired paths, two
awakening paths coexisting for a month).

**Method (so it can be re-derived, not trusted).** Every `prompt_type="..."`
call site was located, its enclosing function resolved, and that function's
callers counted outside its own definition. Liveness below means *reachable
from machine.py*, not *fires often*. Re-run:

```bash
grep -rn 'prompt_type="' --include=*.py captioner/ utils/ drawing/ mood/ machine.py
```

Anything added later without a `prompt_type` is invisible to this audit — that
is the one hole in the method, and the panel's manifest (Phase 0) closes it by
making the assembly itself the record.

**The panel found one on its first run (Aug 2).** A pass labelled `general` was
talking to the model: the video path's degrade-to-a-single-still fallback called
`query_llama_server` without a `prompt_type`, so genuine captions were logged
under the default label and were invisible to every per-type measurement made
this week. Fixed. This is the panel's real job — an undocumented pass shows up
as ⚠ in the sidebar rather than waiting to be grepped for.

---

## A. Live passes (18)

| # | prompt_type | Trigger / cadence | Builder | Backend call |
|---|---|---|---|---|
| 1 | `caption` | every caption cycle (4/7/12s breathing) | `prompts.build_simple_caption_prompt` + `get_monologue_system_prompt` | `query_model` / `query_model_video` (captioner.py `_generate`) |
| 2 | `memory` | memory mode, ~every 4 min | inline in `_process_frame` (memory_system) | `query_model` + stream — **on the live path since Aug 2** |
| 3 | `awakening` | first caption of a session | `generate_internal_awakening` | `query_model` |
| 4 | ~~`awakening` (legacy)~~ | **DELETED Aug 2** — one awakening path now | — | — |
| 5 | `drawing_watch` | every 20s while GRBL executes | `_watch_drawing` | `query_model` |
| 6 | ~~`drawing_thematic_consolidation`~~ | **REMOVED Aug 5** — a closed loop. Its output (theme_tags / emotional_tone / narrative_thread) was read only by `get_thematic_context`, whose consumers are the two dormant 5-step prompts, and by the next consolidation reading its own prior output. The live intent call referenced it **zero** times | — |
| 7 | `stream_consolidation` | when joined stream > `STREAM_CONSOLIDATE_CHARS` | `_consolidate_stream_if_needed` | `query_model` |
| 8 | `compression` | every 8 captions (the memory diff) | `context_compression._perform_compression` | `query_model` |
| 9 | `concept_extraction` | per compression | `_extract_concepts_from_compression` | `query_model` |
| 10 | `journal` | every 30 min + shutdown | `_write_journal_entry` | `query_model` |
| 11 | `reflection` | every ~20 quiet min | `reflection.py _reflect` | `query_model` |
| 12 | `reflection_distill` | after each reflection | `context_compression.distill_reflection` | `query_model` |
| 13 | ~~`drawing_critique`~~ | **REMOVED Aug 5** — artist: "not useful and underutilised... I'd like to redesign that system anyway". It was also the most tangled pass in the stack (two conflicting critiques, one invisible to this audit, and the one whose timeouts kept being stored as the machine's reflection). When it returns it should critique **the paper**, not the ComfyUI image — judge what the pen made, not what was intended | — |
| 14 | `drawing_intent` | drawing triggered | `prompts.stream_drawing_analysis` (call 1) | `query_model` |
| 15 | `drawing_render` | drawing triggered | `prompts.stream_drawing_analysis` (call 2) | `query_model` |
| 16 | `drawing_summary` | during drawing flow | `drawing.handle_drawing_flow` | `query_model` |
| 17 | `artistic_arc` | when drawing context is built | `drawing_memory.get_artistic_arc` | `query_model` |
| 18 | `caption_blind` | **exception path only** — frame not on disk yet | `build_simple_caption_prompt` (same builder as #1) | `query_model`, no image — **reworked Aug 2** |

⚠️ = needs a decision, see section C.

## B. Not live

| prompt_type / function | Status | Note |
|---|---|---|
| `monologue` — `generate_monologue()` | **DEAD** — 0 callers | model_wrapper.py |
| `perception` — `perceive()` | **DEAD** — 0 callers (only a docstring mention) | model_wrapper.py |
| ~~second critique in `grbl_utils.execute_gcode_file`~~ | **REMOVED Aug 5** — critiqued the same drawing a second time, text-only, unlabelled (logged as `general`, invisible to this audit). The GRBL path now reuses `drawing.last_critique` and records the completion; the judgement happens once, the memory is written when the pen actually finishes | artist: "it should definitely only critique it once" |
| `drawing_step1_environmental` … `drawing_step5_synthesis` | **DORMANT** — reachable only when `DRAWING_ANALYSIS_MODE="multi_step"`; live value is `"stream"` | the retired 5-step committee, kept for A/B |

## C. Findings — ALL RESOLVED August 2

Every item below was found by the audit above and fixed the same day; kept as
the record of what was wrong and why the fix took the shape it did.

1. **RESOLVED — the shadow caption path (#18).** When the primary caption call raises
   "No image found"/"does not exist", `_process_frame` retries through
   `model_wrapper.caption_image` — a *completely different prompt builder* that
   predates the June teardown. It does not use the reflexive frame, the stream,
   the seam, or any current gating, and whatever it returns enters the stream
   like any other caption. Options: delete the fallback (preferred — a missed
   cycle is cheaper than a caption in the old voice), or route it through the
   same builder. **Nothing should be able to speak in a voice the panel cannot
   show.**
2. **RESOLVED (renamed `drawing_critique`) — `reflection` named two different passes** (#11 the reflection loop, #13 the
   post-drawing critique). Every per-type metric we have ever computed on
   "reflection" silently mixed them. Rename #13 to `drawing_critique`.
3. **RESOLVED (deleted) — `_call_ollama` was a fossil name** — Ollama was retired July 9; the method
   now calls llama-server. It is the entry point for #2, #4 and #18. Rename, or
   fold those three onto the single `query_model` path.
4. **RESOLVED — two awakening paths coexisted** (#3 live seed, #4 logged-only) —
   flagged in `runtime-map.md` since June and still true.
5. **RESOLVED (mode now logged) — the caption event did not record its mode.** The console prints
   `(relational)`, `(workspace)` etc.; the event log stores `mode: None`. Any
   analysis of mode behaviour is currently impossible.
6. **OPEN — modes are half-retired.** `relational / observational / workspace /
   introspective / awakening` each still route a distinct context function, but
   observational/workspace/introspective have had their elicitation suppressed
   in document/world/hybrid — so three of five change what *arrives* but not
   what is *asked*. Decide whether they earn their keep; the panel makes the
   answer visible either way.

## D2. The drawing intent, and what feeds it (Aug 5)

`stream_drawing_analysis` builds its intent from: the current frame · the
stream tail · drawing-flavoured musings from this session · felt state ·
desire/identity lines · **the executed body of work** · looped words ·
matched reflections.

The body of work comes from `get_executed_sequence`, and it used to prefer
`comfy_prompt` — so the machine was shown its own history in image-generator
prose ("A high-angle view looking down at a pile of rough, splintered wood
scraps scattered...") and, asked what to draw next, continued in that
register. It now prefers `compressed_summary`, which is the intent in the
machine's own voice, stored for exactly this purpose in July and then bypassed
here. The same list now reads: "I will draw that red foam finger, ripped from
its high perch and lying flat against my..." Error sentinels are skipped
defensively; `debug/scrub_drawing_memory.py` removes them from the ledger.

## D. The caption pass, slot by slot

The largest pass, and the one the panel must render first. Order is assembly
order; hybrid/world move the world's turn last.

**System prompt** (`get_monologue_system_prompt`)
| Slot | Source | Kind |
|---|---|---|
| situation | `_SITUATION` (reflexive) / `_SITUATION_WORLD` (log genre, world only) | static |
| drawing state | `state_manager` | live fact, gated |
| genre clause | `_monologue_clause()` — varies by STREAM_MODE | static |
| felt state | `context_compressor.get_felt_state()` | **model-generated, re-injected** |
| persona | `core_facts.self` | **model-generated, re-injected** |
| mode elicitation | `_MODE_ADDITIONS[mode]` | static, suppressed for 3 modes |

**User prompt** (`build_simple_caption_prompt`)
| Slot | Source | Kind |
|---|---|---|
| situational delta | `build_situational_line` | live |
| salience event | `_salience_event` | live |
| reorientation | `get_reorientation_line` | live, windowed |
| mode context | `MODE_CONTEXTS[mode].context_fn` | mixed |
| introspective ctx | drawing_memory | stored |
| core facts | concepts-derived place list | stored, occasional |
| familiarity / drawing echo / reflection echo | ChromaDB | stored, one per call max |
| drawing + paper state | `state_manager` | live |
| felt delta | compression | **model-generated** |
| desire | `current_desire` | **model-generated**, 3-injection cap |

**History**: `_stream_history()` — raw (document) or timestamped log (world/hybrid).
**Seam**: hybrid only — `_seam_of(newest entry)` as trailing assistant prefill.
**Images**: 1 still or up to 6 frames (`VIDEO_MODE=multi`), dropped on inward beats.

The "kind" column is the one that matters most: **model-generated, re-injected**
is the class that has caused every spiral. The panel should colour it distinctly.

## E. Gates

**Mouth** (`_caption_reject_reason`, retry-once-then-silence): template_echo ·
refrain_echo · outward_address · assistant_speak · cjk_drift · numeric_fragment ·
word_salad · number_chain · phantom_drawing · tail_echo · prompt_parrot

**Salvage strips** (`_strip_list_shape`): enum prefix · countdown · log stamp ·
log label / Status: · hashtag tail

**Stream admission** (`_stream_admissible`): meta markers · outward hooks ·
telemetry register · markdown · list shape

**Storage**: `_valid_self_fact` (persona) · `_felt_phrase_held_reason` (felt) ·
event provenance (`_had_perception_event_in_window`) · `_is_plantable_prior`
(session seed) · `_journal_entry_clean` · `_is_abstract_label` (concepts)

**Escapes**: erosion (drop oldest stream entry after 3 stuck cycles) ·
seam exclusion in prefill modes (`_comparable_stream`)

## F. What the panel needs from this

Phase 0 turns section D into a runtime artifact: each pass emits a **manifest**
(slot name, source ref, resolved value, provenance, age, gate verdict) beside
the string it sends. Sections A/B/C become generated rather than hand-written —
this document should eventually be produced by the code it describes, which is
the only way it stays true.

**Open verification debt:** passes #2, #6, #16, #17 were confirmed reachable but
their prompt bodies have not been read line-by-line in this pass. Do that before
Phase 1 renders them.
