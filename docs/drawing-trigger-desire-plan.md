# Drawing trigger → desire arc (north-star step 5) — scope and plan

Aug 12 2026. Full scope agreed before build. Related arcs: mood revamp (motor-panel
session Aug 9-10), sentiment/text-agitation proposals, kinetic bus emotion mapping.

## Findings (verified in code Aug 12)

**The "state-motivated" trigger is a timer in practice.** `drawing/drawing.py`
`_should_draw_state_motivated`:

- Mood is double-normalized: `(mood+1)/2` on a value already 0..1
  (`MoodEngine.current_mood`), so mood contributes ~0.225 baseline regardless of state.
- The +0.3 "first drawing" startup bonus fires on **every** evaluation — its
  `time_since_last > 300` guard is always true because the function already returned
  below `DRAWING_MIN_INTERVAL` (900).
- Net: even with novelty=boredom=0 the score meets the 0.45 threshold at the 900s
  floor. Novelty and boredom are decorative. The machine draws at the first quiet,
  non-executing caption cycle past 15 minutes.
- Supporting rot: `reflection` arg accepted, never read; `novelty_score` has two
  racing writers (activation network via `captioner/memory.py` vs mood engine via
  `machine.py` every 10s); `DRAWING_PERSON_WEIGHT/BONUS` have no runtime consumer;
  cooldown resets at prompt generation, not execution (deliberate — prompts must not
  stack when paper/ComfyUI absent).

**Dead mood wiring** (matters to the wider mood revamp, not fixed in this arc):
`captioner.update()` is never passed `mood_vector`/`emotion_state`
(`machine.py` call site), so the captioner vector is frozen at (0,0,0.5) —
awakening always "calm", workspace hints always "slow", every drawing's
`emotional_tone` is `calm_observant`. The mood engine's if/elif ladder pins to
`alert_curious` in practice; a 30-min sine oscillator from the flatline era is
still layered on top.

**The content side is already self-directed; the timing is not.** The desire slot
(`current_desire` / `desire_since` / `desire_history` / `spend_desire`, in
`captioner/context_compression.py`) forms wants in distillation with no drawing
framing, persists them across distills, and discharges them post-GRBL. The
stocktake beat (`drawing_direction`) carries the considered arc. Nothing the model
says can currently cause a drawing — the WANT slot only decorates the intent prompt.

## Design principles (agreed Aug 12)

1. **Never ask "do you want to draw?"** — compliance bias makes yes ~free; we'd get
   manufactured justifications (same failure class as the 5-step's manufactured
   drama). The model *can* decline when the frame isn't a question: the salience
   gate defers regularly; the WANT prompt's "(if any)" produces honest absence.
2. **The no is structural**: absence of a formed, persisted drawing-directed want.
   Desire forms elsewhere, for other reasons; the trigger only reads it.
3. **Persistence is the sincerity test** — a want surviving distills means it, same
   doctrine as durable-ledger confirmations. One-shot answers confabulate.
4. **Timer becomes guardrails only**: floor/cooldown/safety gates stay (immune
   system); ceiling stays as hunger (a machine that never wants still eventually
   draws — or we decide it doesn't; open question for the artist).
5. **Storage over mouth**: the desire slot is storage; any deliberate decision beat
   is a mouth. Trust storage. If a choice beat is ever added, it is a choice between
   equals ("stay with the room" / "go to the paper"), never yes/no, and "name the
   one image" is the cost of yes.
6. **No substring matching** — `_drawing_intentions` died as "a coin flip wearing a
   label" ("ink" in "th**ink**"). Any lexicon runs on word boundaries.

## Build order

**Phase A — shadow mode (this commit).** `Drawing.desire_shadow_verdict()` reads the
desire slot on every real trigger evaluation and logs verdict + formula verdict side
by side (`LogType.DECISION`, `decision: desire_shadow`). No behavior change — the
formula still rules. Run at least a day; read the log:
- formula fired / shadow silent → today's over-eager timer, expected often
- shadow would-fire rate ~100% → framing is leaky, fix before it ever controls the arm
- want formed → persisted → would-fire → spent → the arc working end to end

**Phase B — cutover.** After shadow validates: `should_draw` = guardrails +
desire verdict; retire novelty/boredom weights, double-normalization, startup
bonus, the `reflection` param, `DRAWING_PERSON_*`. Config flag for A/B
(`DRAWING_TRIGGER_MODE = "desire" | "formula"`). Decide max-interval semantics
with the artist (forced hunger vs allowed silence). Update runtime-map.

**Phase C — mood revamp step 3 (separate arc, motor-panel plan).** Continuous
(v,a,c) coordinates, proximity-weighted temperament pooling, retire the 5-label
ladder. Fix the `captioner.update()` dead wiring and novelty writer race here —
they belong to the mood plumbing, not the trigger.

**Phase D — text-agitation signal (separate, small).** Per-caption repetition /
exclamation / sentence-collapse metric driving movement temperature + tremor
(kinetic bus) within seconds; candidate secondary input to the trigger
(agitation as "not now" / "now").

Parked, tracked in motor-panel plan: per-recording modifiers (step 4),
person-tracking temperament, prompt-mode modifier profiles, arousal-coupled LLM
sampling temperature (currently inverted: 0.9 bored / 1.0 engaged),
felt-phrase figurative-vocabulary gate.
