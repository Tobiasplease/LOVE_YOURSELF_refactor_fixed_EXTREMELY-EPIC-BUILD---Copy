# The time-and-loop round (Sep 5 2026)

**Artist's ruling (Sep 5 midday):** "Nothing happening in a room overnight is
an event. Things are always building, compressing, and growing. If I'm left
in a room I don't just start thinking in loops, and if I do, I catch myself
doing that and that becomes a new thought. We are severely lacking in this
dimension." And on the persona: "baseline" = the accumulating persona
development; there should be a consolidate → build → evolve loop for the self.

**Evidence (overnight run 610786d8, 01:34–12:06, 2653 captions):** refrain
share 55%; "the red foam finger" in 313 captions; 625 lines refused by the
echo gates that the machine never heard about; 32 distills in 11 h, 32
distinct wants, one avoidance trait re-derived every 20 min; 48 lore threads
of one idea; zero exclamations. Three false arrivals via the adjudicator (the
top-shelf black bundle twice, the mannequin head once). An ink dot the pen
never made lived in the stream for five minutes and became a self-note.

## Diagnosis

1. **The echo gates hide the loop from the machine.** Refusing a refrain
   silently deletes the evidence a person uses to catch themselves.
2. **Time is stamped, never felt.** Clock stamps and gap markers are labels;
   no duration ever arrives as an event with weight.
3. **The reflection has no memory of its own conclusions.** Every distill is a
   fresh derivation; the ledger confirms but cannot turn.
4. **Accumulation exists but doesn't speak.** Boredom, felt arc, referee
   confirms, ledger confirmations and days — none reach the prompt as facts.
5. **The compressor's self and event slots invent** (241 and 237 non-none of
   309 in a still night); the event slot has a provenance gate, the self slot
   had only mechanical checks.

## Built

### A. Fixes first
- Phantom MARKS as states: "speck of ink on the white paper", "that dot on the
  paper", "the pen is pressing into the fiber" → `phantom_drawing` (mouth gate,
  retry-else-silence). Reflection kernels now pass the same mouth gate.
- Adjudicator: the veto is gaze-aware (`ENTITY_VETO_GAZE_TOL_DEG`); a person
  verdict that verified absence closes within `PRESENCE_FALSE_ARRIVAL_WINDOW_S`
  while the same shape is still in the candidate box is RETRACTED to a thing
  at that gaze + box (a real visitor who left is not in the box).
- Self-notes: a claimed act of marking, or a present-tense third person, is
  not a fact about oneself (`_note_is_phantom_act`).

### B. Time and loop
- **Duration edge** (`caption.duration-edge`): "Nothing in the room has changed
  for {duration}." fires once per threshold (`DURATION_EDGE_THRESHOLDS_MIN`
  30/60/120/240/480) per world-verified unchanged span; the clock resets on
  a referee world_changed, a presence edge, a boot; silent while someone is
  believed present. An edge, so the delta doctrine holds.
- **Rest ladder:** `CAPTION_INTERVAL_REST` × (1 + unchanged hours), capped by
  `CAPTION_INTERVAL_REST_MAX` (120 s). Stillness slows thought instead of
  filling the window.
- **Loop notice, two sources:** the echo gates now record every refusal's
  shared run (`_note_loop_hit`); after `LOOP_NOTICE_AFTER` refusals inside
  `LOOP_NOTICE_WINDOW_S` the run is quoted back (`caption.loop-fact`); and the
  compressor's new REPEATING slot lets the machine name what it is circling
  (`caption.loop-notice`), which outranks the count. Once per
  `LOOP_NOTICE_COOLDOWN_S`, in the world's turn, never stored.

### C. Persona baseline
- **The turn path:** the distill sees what it has held (`distill.held-line`)
  and gets a NO LONGER TRUE slot; a rough match marks the fact CHALLENGED —
  it leaves "what has stayed true" and rides "what you used to hold, lately
  in doubt" (`monologue.challenged-wrap`); two fresh confirmations restore it.
- **Audible time:** `monologue.durable-time` — how long the oldest and newest
  stable facts have held, in words. `render_evolving_edge()` = what is newly
  taking hold (used by consolidation).
- **The reflection remembers itself:** alive threads with how often it has
  come back to each ("twice", "several times"), and the baseline paragraph.
- **Daily consolidation** (`maybe_consolidate_persona`, once per
  PERSONA_CONSOLIDATE_EVERY_S after a distill): held + in-doubt + newly-held
  facts, threads with counts, open questions, the want lineage, the day's
  felt arc, the previous baseline → three to five plain first-person
  sentences, stored as `baseline_paragraph`; the awakening and every
  reflection read it back. The consolidate → build → evolve loop, for the self.

Tests: debug/test_persona_baseline.py (27).

## Verify live
- `duration_edge` debug events at 30/60/120 min of stillness; the caption after
  each should carry time.
- `loop_notice` events; whether the next caption turns (a new thought) or
  argues with the notice — both are fine; a verbatim continuation is not.
- `presence_adjudication` verdict "retracted" after any false arrival; no
  second arrival from the same shelf within `ENTITY_VETO_TTL_S`.
- `kernel_rejected` count; `phantom_drawing` on mark states.
- cadence under stillness: 28 → 56 → 84 → 112 s by hour.
- self-note rejections logged with reason "phantom act or presence".

Also verify: `fact_challenged` after a distill's NO LONGER TRUE; the
"lately in doubt" line in the system prompt; `persona_consolidated` once per
day and the paragraph in the next awakening; whether the 20-min distills stop
re-deriving one trait once they can see their own threads and counts.

Tests: debug/test_time_and_loop.py (33), debug/test_persona_baseline.py (27).
