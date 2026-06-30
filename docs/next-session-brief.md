# Next Session Brief — memory is a ledger now; validate + finish (June 28, 2026)

**Branch:** `rebuild/north-star`
**Read first, in order:**
1. `docs/north-star.md` — the spec (unchanged).
2. `docs/memory-redesign-plan.md` — the ledger migration, step by step, with the
   final-state notes and what's still pending.
3. `docs/runtime-map.md` — the live wiring (detox is now OFF; memory is live).
4. `docs/voice-analysis.md` — the original diagnosis; its Status block is the
   short version of where it landed.

## What happened this session

Two intertwined efforts, both done:

1. **Base-voice rework.** Per-mode elicitations; sticky-uncertain presence belief
   (killed the false-arrival jam); constant continuity via a **delta line** (the
   per-call prompt carries only what CHANGED, not a re-stated scene); **DRY
   sampling** (the real fix for verbatim looping — `repeat_last_n`=64 was shorter
   than the stream, so prior captions had no repetition penalty); brevity
   (`num_predict` 60); **inward beats** (`INTROSPECT_INTERVAL=4` — every 4th quiet
   caption drops the image so it thinks instead of describing); face-detection
   fix (mannequin heads were tripping eye-contact). Stream is ON (`STREAM_WINDOW=6`).

2. **Memory ledger migration (steps 0–7), then a consolidation.** Every store that
   fed the prompt was rebuilt as a clean ledger that stores FACTS and re-voices
   them, instead of storing the model's purple prose and replaying it. concepts
   (purged + creation-gated) · drawings (theme tags, never comfy_prompt) · place
   (derived from concepts) · felt-state (deterministic + degreed translation of
   the mood vector, e.g. "a little restless") · desire/belief (open questions,
   persistent arc) · reflections (echo the SUBJECT, not the prose) · persona
   (grounded in concepts). Then the big move: **the reflection loop became the
   identity engine** — `context_compression.distill_reflection` pulls TRAIT/
   BELIEF/WANT from each reflection into the persona/belief/desire ledgers — and
   the **inert compression-thread introspection/self-synthesis/core-facts layers
   were retired** (~345 lines). `BASE_VOICE_DETOX` is the regression harness.

## Validated vs not

- **VALIDATED:** the feedback loop is dead. Cold-started fresh, run 1.7h with
  memory on → no store re-grew purple; concepts are clean noun-phrases; place is
  a clean object list; reflection echoes are subjects. The "grid/dread" obsession
  did not return.
- **NOT YET:** the reflection-distillation was wired AFTER that run, so the
  persona forming **clean** from a real reflection is unproven — needs a run where
  ≥1 reflection fires (~20–40 min) and you check the `What you've come to know
  about yourself: "…"` line. (Unit-tested: a reflection distills to "I keep coming
  back to the pink shelf" etc.)

## What to do next (priority order)

1. **Run and validate the identity engine.** Restart, let it run 30–40+ min, watch
   the persona/desire/belief form from reflections. Green = grounded/plain
   ("I keep coming back to the tools"); red = "grid"-style abstraction.
2. **Stop injecting the place inventory every caption.** It's the main reason the
   *relational* captions feel isolated (every one re-describes the same object
   list — `times_seen` hit 321 on "mannequin heads"). Make it occasional/delta
   like the situational line. This is the highest-value voice fix left.
3. **Drawing-pipeline cleanup** (the last contamination hole, currently dormant
   because drawing rarely fires without ComfyUI): the 5 step system-prompts
   actively ask for metaphor; `get_artistic_arc` LLM-narrates raw `comfy_prompt`;
   the reflection→drawing injection (`get_last_reflection`). When drawing runs,
   the ARTWORK + its post-reflection will be purple — that's expected, but it's
   the next thing to clean.
4. **The base-voice purple floor** (dust motes, similes, in fresh captions) is the
   model prior — north-star Principle 7 says this likely needs the fine-tune. Do
   NOT chase it with fences. Separate problem from everything above.

## Standing cautions

- **Mind the grandfathered consolidation systems.** There have been many
  iterations (compression/introspection/reflection at various intervals). The
  live set is now small: compression (spatial+concepts, every 4 captions),
  journal (30 min), reflection+distillation (20 min). Still-wired-but-old:
  `generate_awakening_message` (machine.py calls it AND
  `generate_internal_awakening` — two awakening paths, worth reconciling),
  `_call_natsumura_introspective`. Don't add an Nth layer; consolidate.
- Features fail SILENTLY — verify via logs / live state files, not by reading code.
- Fix what's STORED, never the mouth (no fences). The ledger holds because the
  gates are at the storage side.
- Memory cold-start backup: `event_log/_pre_coldstart_bak_*` (restore to undo the
  wipe). Persona/identity history backups: `*.purple-bak`.
- Keep `runtime-map.md` updated as wiring changes — it's the artist's window.
