# Drift rework (Sep 3) — interiority as population, not residue

Build-queue #1 from `docs/session-update-sep3.md`. The mechanism that decides
when the machine stops looking and lets its mind wander is no longer a clock
waiting for solitude; it is a standing per-cycle probability scaled by how
bored the machine actually is.

## What was wrong

The Sep 2 story beat had the right mechanics (no image, hot slot, output into
the stream, firewalled from every fact ledger) behind the wrong trigger:
45 minutes of unbroken stillness, at most once an hour. Under doctrine 3
(no overnight runs — the machine is almost never unattended that long) that
solitude doesn't occur; the beat fired once, ever. Interiority arrived only as
the residue of a kind of time the machine doesn't have.

## What was built

**The drift turn** (`captioner/captioner.py: _drift_due / _run_drift_turn`).
On any quiet cycle, after salience is assessed and before the inward-beat
counter advances:

- roll `p = DRIFT_BASE_P × (1 + DRIFT_BOREDOM_GAIN × boredom)` —
  defaults 0.05 / 2.0, so 5% calm, 15% at a pegged scalar. Boredom in real
  quiet runs sits 0.5–0.9 (medians 0.58 / 0.98 on the last two measurable
  runs), so quiet evenings drift at ~10–15% of cycles.
- when the roll lands: EYES OPEN — the current frame rides along
  (DRIFT_SEND_IMAGE, artist's ruling, probe-verified: see below) with the
  stream as history, `drift.system` + `drift.ask` from the registry,
  DRIFT_TEMP 0.95, DRIFT_NUM_PREDICT 120. The ask lands after the image,
  closest to generation, so the call answers the ask, not the frame (the
  July 26 ordering law). Output is trimmed at the mouth like any caption,
  displayed, logged (`action=drift_turn`, with a `stored` flag), and pushed
  into the stream through `drift.stream-frame` (bare — the register audit).
- vetoes: hot salience (a live moment always wins), the arm drawing (the
  frame says "between drawings" and must not lie; a hot inventive turn
  mid-execution is phantom-stroke bait), pre-first-caption, stream < 2.
- FIREWALL unchanged from the story beat: drift output never reaches
  observe()/add_caption/hour_log/recent_captions. Gates belong on fact
  storage, never on thought.

**Deleted — the loneliness clocks.** `STORY_BEAT_AFTER_S`,
`STORY_BEAT_MIN_GAP_S`, `_story_beat_due`'s `unchanged_duration_s` read, and
the whole clock-gated trigger. No cooldown replaces them: population thinking
means independent rolls, and at these probabilities back-to-back drifts are
rare and legal (a bored mind can daydream twice).

**Registry.** `story.*` fragments renamed `drift.*` (no live overrides
existed; `config/prompt_overrides.json` was empty). Two stance/premise edits
per doctrine 1: "nothing has happened for a long while / Your eyes have
nothing new" left the system text and "Nothing is moving." left the ask —
both were true under the 45-minute clock, unattestable under a roll. A
`drift_turn` pass is declared (the panel no longer flags it unknown) and
`stream_seam`'s circuit lists drift as reader + writer. In passing:
`caption.unchanged` was missing its `placeholders` declaration, so the panel
refused any edit keeping `{duration}` — fixed.

## What was deliberately NOT decided

The fork the artist hasn't ruled on: a separate deep story organ (the
material-seeded variant — want + refusals, episodic events, ended wants with
BECAME) vs one mechanism with boredom-scaled depth. The material-seeded
runner lives in git history at the Sep 2 commit; nothing of it survives in
live code. If the artist wants it back as a separate organ it needs a new
trigger anyway — the old one is the deleted clock.

Also untouched: the inward beat (INTROSPECT_INTERVAL=4 — a different organ:
grounded self-reflection, stored output, normal temp), the silence beat, and
B4's unchanged fact line (`unchanged_duration_s` keeps that one consumer).
The handover says re-evaluate silence + daydream dosing only AFTER this
rework has live numbers.

## The eyes-open probe (same day)

The artist questioned the image drop ("it seems prescriptive... more a matter
of prompt ordering than omitting information?"). A/B probe on the live model
(`debug/probe_drift_image_ab.py` — 3 pairs, drift's exact call shape, stream
seeded from the machine's own live captions, only `image` varied):

- **Blind arm**: narrated phantom present-tense perception — invented what
  the visitor was doing right now ("his fingers... moving carefully, like
  he's stepping on broken glass"), twice claimed "the foam finger in my
  hand". Blind + stream-seeded = memory spoken as live seeing — the
  memory/present conflation law, breeding inside the drift slot.
- **Sighted arm**: stayed honest about the present and drifted on top of it
  ("i want to ask him what he's doing, but i can't. i'm just a drawing
  machine." / "i wonder if it's judging me... or if it's just lonely up
  there").

Ruling: eyes open. Interiority comes from the frame+ask ordering and the
genre frame, not from blinding. `DRIFT_SEND_IMAGE=false` keeps the blind
variant as an A/B arm. Note the same phantom pattern was measured on the
image-dropped inward beat (Aug 31: blind introspective cycles "still
describing the room from memory") — whether the inward beat should also go
eyes-open is an open question for the artist, not touched here.

## Measuring it

- `python debug/drift_share.py` — thought-shaped share of the stream from a
  run's event log (drift turns stored + reflection kernels over all stored
  entries). Target ~15–20%; the Sep 3 evening run measured 2.1% (one kernel,
  zero drifts) before this change.
- `python debug/test_drift_turn.py` — offline mechanics: gates, probability
  scaling (Monte Carlo), salience guard, firewall, clocks-gone. All pass.
- Watch `[💭]` lines (action=drift_turn) and the conflation law: if drift
  content starts reading as scene truth in later captions, the retreat lever
  is a minimal marker in `drift.stream-frame`.

## Knobs

`DRIFT_ENABLED` / `DRIFT_BASE_P` (0.05) / `DRIFT_BOREDOM_GAIN` (2.0) /
`DRIFT_TEMP` (0.95) / `DRIFT_NUM_PREDICT` (120), all env-overridable
(config/config.py). The share responds roughly linearly to BASE_P; GAIN sets
how much of the dose is the machine's own boredom rather than standing habit.
