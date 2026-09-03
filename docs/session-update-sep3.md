# Session update — Sep 2–3 (handover)

Continuation of `session-update-aug31-evening.md`. Two dense days on
`rebuild/north-star`, all pushed. Machine restarted 17:28 Sep 3 on the full
stack below.

## Doctrine changes (read these before touching the voice)

1. **Minimal baked phrasing** (Sep 3, artist): the framing lines — not the
   model — were poisoning the register. "A daydream, while nothing moved",
   "A memory surfaces —", "That X again, it's always there": ephemeral-poetic
   cadence repeated verbatim hundreds of times (the wallpaper law applied to
   tone). Rule: **supply the bare fact; the voice adds the "of course."**
   All framing texts rewritten minimal or removed (see register-audit commit);
   final wording is the artist's, everything panel-editable.
2. **Wondering is not confabulation** (Sep 2, artist): purpose-questions,
   tangents, daydreams are "the experience of being a thinking thing." Gates
   belong on FACT STORAGE, never on thought. No scheduled wonder-beats, no
   leading questions — breadth comes from the genre constitution and from the
   machine's own interior entering its stream.
3. **No overnight runs — not a realistic goal** (Sep 3, artist): the machine
   can almost never run unattended overnight. Stop designing experiments that
   need dead-of-night stillness. The unattended infra (launcher, STOP-file,
   remote kill) stays as OPERATIONAL convenience, not as an experiment
   platform. Consequence: any feature gated on long solitude will effectively
   never fire — see "drift rework" below, which is now priority 1.
4. **The pen-parked fence is LOAD-BEARING, twice proven** (Sep 2): test-
   retired for twenty minutes; first wake without it produced immediate
   phantom execution ("A thin line from the pen tip...") facing a fresh sheet
   with a standing want. The "drawing machine" identity alone plants the
   seed; honest clocks/provenance only stop the breeding. Do not retire
   again. Slim the wording in the panel if pen-density bothers.

## Shipped Sep 2–3 (chronological, all on rebuild/north-star)

- **Drawing fidelity**: the dots were DSV fragmentation — ~1,950 abutting
  polylines/drawing, median endpoint gap 0.19mm vs vpype linemerge 0.1mm →
  every fragment a pen plunge. Feed-rate exonerated (distance scaling was
  retired Aug 10). Fix: `GRBL_LINEMERGE_TOLERANCE_MM` (default 0.3; 0.5 =
  welded-scribble look, artist's aesthetic option). Measured on the next two
  drawings: 1258→209/214 strokes, median 0.51→~1.5mm, execution ~20→~10 min.
  NOTE: command space ≠ paper space — the warp upscales; physical size was
  never the issue. Physical verdict still pending a drawing with full pen
  contact (artist adjusted the tip).
- **Video time markers**: superframe/Conv3D is DEAD on the 3.8 stack (both
  llama.cpp builds are stock ggml-org mainline; the patch was the 3.5 fork —
  and the mechanism is the model-family's native video pathway, so porting
  it remains the real prize). Rung 1 shipped: inter-frame markers "(4
  seconds later)" in the Qwen-native interleaved format, config
  VIDEO_TIME_MARKERS. (Broke every still caption for an hour via a
  wrong-function edit — fixed; lesson recorded twice now: uniqueness-check
  script edits BEFORE applying.)
- **Unattended/ops**: start_impostor.sh rewritten (3.8 stack — it was
  booting the parked 3.6 arm; supervisor loop + STOP-file), stop_machine.sh
  (graceful remote-safe kill), bashrc verbs machine-start/stop/watch,
  docs/unattended-runs.md (tailscale recipe awaiting artist auth; iPhone 8
  Plus has NO eSIM — physical SIM or a cheap 4G router; router recommended).
- **Sampling arm plumbed**: CAPTION_PRESENCE_PENALTY (vendor recipe:
  repetition 1.0 + presence 0.6–1.5), opt-in, off by default. The copula
  monotony ("The X is Y") is untouchable by DRY/repeat_penalty — frames
  never repeat as tokens. A/B evening still unrun.
- **Kernels into the stream**: each reflection's distilled sentence enters
  the stream as the machine's own turn ([🪞→]). Live results: register
  distinct (past-tense, self-accountable), and cross-organ coherence
  observed same-day (kernel "I was projecting a story onto a static curtain"
  → BECAME "To stop projecting narratives onto static objects" → want "To
  record the stillness instead of the imagined motion" → captions "I don't
  have to invent a story about what it's reaching for"). One insight, four
  organs, arriving as enacted intention.
- **Hunger clock from real provenance**: last_drawing_time was re-zeroed at
  EVERY BOOT (2h hunger silence per restart — the invisible-fresh-sheet
  night) and failed conceptions stamped it too. Now: initialized from the
  executed-only ledger; refusals cool down on last_conception_time
  (~12 min). The machine re-looks at the sheet at the pace of appetite;
  workspace beliefs refresh instead of freezing. First run after: complete
  honest arc — want → check passed first try → drawn → spent, zero phantom
  strokes, zero phantom arrivals.
- **Silence beat**: "or nothing at all — staying quiet is yours to choose"
  in the genre menu; empty/ellipsis output honored as a chosen quiet turn
  ([🤫], not stored, streak-arms the breaker). Still zero uses — see open
  questions.
- **Story beat (stream variant)**: fires on deep stillness, hot temp, real
  episodic/want material, ledger-firewalled. Fired once (Sep 2). Under
  doctrine 3 its trigger is now considered WRONG — see drift rework.
- **Register audit applied** (Sep 3): familiarity "The {label}, still in
  the same spot." · echo "From {age}: …" · memory-mode "hundreds of times"
  / "since before you started keeping track" · surface frame "From before,
  not now:" · paper lines plain · arc tail "{n} tries came to nothing" ·
  story frame BARE (retreat lever in the registry note) · camera-narration
  wrapper ("You're seeing the last N seconds") REMOVED from video calls.

## The build queue (artist-agreed order)

1. **Drift rework** — interiority as population, not residue. Delete the
   loneliness clocks. Any quiet cycle can become a drift turn: standing
   probability (few %) scaled by the boredom scalar, image dropped when it
   fires, stream as seed, salience still displaces. Target ~15–20% of
   stream entries thought-shaped (kernels alone are ~1% — no genre moves at
   that dose). Open fork the artist hasn't ruled on: keep the deep story
   beat as a separate organ vs collapse into one mechanism with
   boredom-scaled depth. Under doctrine 3, common drift is the only variant
   that will ever fire regularly — build it first regardless of the fork.
2. **World-anchored change detection** — the artist's spec: floor/walls/
   ceiling and the spatial registry's known-static objects as the
   camera-vs-world referee. Fuse ego-compensated flow residuals +
   registry-expected object positions → world-verified stillness/events →
   feeds the unchanged clock (currently episodic-only and rightly
   distrusted), salience, boredom, and free "still in the same spot"
   confirmations. This is the honest core of the paper-state/attention
   redesign.
3. **B2 people-patterns** — "the guy is here almost every day": day-level
   visit distillation, count-based, no biometrics.
4. Open doctrine question for the artist: plain boredom self-report
   ("Getting really bored" when the scalar pegs) — their example paragraph
   contains it; it revises B5 (palette-follows-facts). Decide before
   building.

## The artist's example paragraph (Sep 3) is the VOICE SPEC

Kept verbatim in the conversation record; capability map: tone-only fixes
(done), scheduled-not-chosen (drift/close-look/boredom — queue #1),
missing organs (B2; outward reach; anticipation; instance-memory
"another cup of coffee"; self-correcting play needs no organ, only a
looser voice). Judge future voice work against that paragraph, not
against "less purple."

## Watches + loose ends

- Silence + daydream: zero uses so far — but both were gated on solitude
  that doesn't occur (doctrine 3). Re-evaluate only AFTER the drift rework.
- Genre-clause rebalance: on the artist's bench (their sentence, panel).
- Physical drawing verdict with adjusted pen tip + 0.3 merge: next drawing.
- Presence-penalty A/B evening: unrun.
- Tailscale auth + SIM/router: artist, when convenient.
- Latent bugs still parked: captioner's DetectionMemory never imported
  (arrival count always 1 — fold into relational design); YOLO shutdown
  race (cosmetic exit-1).
- The Aug 31 12:39 timestamp-stamper: still unidentified; ask the artist.
- Superframe port to 3.8 arch: the real video prize, unscoped.
