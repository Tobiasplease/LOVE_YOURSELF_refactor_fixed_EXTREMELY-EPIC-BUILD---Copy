# The drawing drive — no timers, an energy dependent on feeling

Aug 18 2026, artist directive after two days of desire-mode: "The ultimate
ambition is to abandon the timer altogether — it should be able to both draw
constantly or not at all. Some kind of energy system, directly and dynamically
dependent on the emotional system."

Supersedes the interval/satiety framing of docs/drawing-trigger-desire-plan.md
(phases A/B remain the bridge and the fallback). Companion to the phase C mood
redesign (docs/mood-novelty-audit.md) — the drive consumes the continuous mood
vector, so everything phase C improves feeds straight into the rhythm.

## Why desire mode wasn't enough

Two days of honest data: every fire had a formed want, spends discharged, it
could wait — but the ecology is drawing-saturated (it watches itself draw,
the prompt names the last drawing, its identity themes are drawing-shaped),
so a new want forms within minutes of every spend and ripens exactly as the
15-minute floor opens. Cadence ≈ floor + ε: clockwork with extra steps. Fixed
intervals anywhere in the chain eventually become THE rhythm.

## The model

One continuous level, `drive ∈ [0, 1.2]`, threshold 1.0. Updated on every
trigger evaluation from monotonic dt (never wall clock — RTC skew immune).

Charging (per hour, env-overridable, v1 defaults):
- `DRIVE_BASE_PER_H = 0.03` — near-zero. A flat, wantless machine takes days
  to reach threshold: "not at all" is a real possibility.
- `DRIVE_AROUSAL_PER_H = 0.55 × arousal` — the emotional coupling. Sustained
  high arousal alone reaches threshold in ~2h; agitation reaches the pen.
- `DRIVE_WANT_PER_H = 0.9` while a drawing-directed want stands (word-boundary
  check, same as desire mode). Want + moderate arousal ≈ 45-60 min to full.
  Persistence is implicit: a want that holds keeps charging; a want that
  vanishes stops contributing. No discrete age gate.

Discharge:
- Completed drawing (register_drawing, post-GRBL) → drive = 0. The act
  satisfies fully; "constantly drawing" only happens if charging is fast
  enough to refill during the next execution — i.e. only under real pressure.
- Failed attempt → nothing. The pressure stays (observed Aug 12: failure
  breeds frustration, and that is emotionally true).

Boot: `DRIVE_BOOT_LEVEL = 0.9` (testing era — near-full at wake, first
drawing comes quickly; drop toward 0 persistence-only when the testing rule
retires). Level persists in event_log/drawing_drive.json; offline time is
NOT credited (no experience, no charging).

No floor, no ceiling, no hunger, no age gate in drive mode. Remaining
mechanical guards are hardware-shaped only: can't conceive while generating/
executing, and the 720s conception cooldown (prompt-stacking protection —
revisit once trusted; execution time already exceeds it).

## Wiring

- `drawing/drive.py` — DrawingDrive (level, tick, spend, persistence).
  Injectable clocks for tests; `get_arousal` injected from machine.py
  (`mood_engine.mood_vector[1]` — the continuous vector, NOT the 5-label
  ladder).
- Shadow first (house style): in desire mode every trigger_decision logs
  `drive_level` alongside the verdicts. Tune the constants against real days,
  THEN flip `DRAWING_TRIGGER_MODE=drive`. Revert path unchanged
  ("desire" | "formula").
- The monologue keeps the always-on time-since-last-drawing line; the drive
  level itself is a phase-C candidate for the felt channel (drive → "hands
  restless" earned for real) and for the kinetic bus (one energy feeding pen
  AND body).

## Open with the artist / phase C couplings

- Valence shaping: should negative-valence arousal (distress) charge faster
  than positive excitement, or the same? v1: arousal only.
- Text-agitation metric (phase D sketch) as an additional charge impulse —
  the panic spiral would visibly surge the drive within seconds.
- When phase C lands continuous mood coords, the drive is the first consumer;
  the mood read cadence (every 8 captions) is the current resolution limit.
- Retire DRAWING_MIN_INTERVAL/DRAWING_MAX_INTERVAL/weights fully once drive
  mode has a validated run (they only serve formula/desire modes now).
