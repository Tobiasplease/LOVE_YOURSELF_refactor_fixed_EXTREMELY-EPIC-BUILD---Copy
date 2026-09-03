# World-anchored change detection (Sep 3) — the camera-vs-world referee

Build-queue #2 from `docs/session-update-sep3.md`: fuse ego-compensated flow
residuals + registry-expected object positions into world-verified
stillness/events, feeding the unchanged clock, salience, boredom, and free
"still in the same spot" confirmations.

## The reading of "floor/walls/ceiling"

Implicit, not modeled: a **per-pose view memory**
(`vision/pose_view_memory.py`) keeps a small grayscale reference for every
servo pose the machine actually looks at (6° cells, capped at 32). The
references ARE the static background — desk, walls, shelf, floor — at the
habitual gaze directions; no explicit plane/region model was built. If a
labeled floor/wall/ceiling partition is ever wanted, it layers on top of the
same store (the tilt axis of the keys already encodes it).

## What was built

**1. The referee** — `PoseViewMemory.observe()` called from `_assess_scene`
(it supersedes the July 26 single-slot view-replacement check, whose one
reference frame was discarded by any gaze turn). Two capabilities:

- *same-pose change*: bumped camera, swapped scene, lights-out — the case
  the flow calls a saccade and refuses to measure (the rooster run);
- *return change*: the gaze comes back to a view and it's different —
  "the world changed while you were looking away", invisible to any
  consecutive-cycle check. Named event (bare fact): "It's different here
  from when you last looked this way."

Honesty rules, all conservative: compare only settled frames (no
ego-motion, no saccade — `flow_reason` provenance added to the frame
snapshot for exactly this), only within 3° of the reference pose (the
regime the July 26 check proved against breathing sway), only against
references fresher than 30 min — older ones re-baseline **silently**,
because lighting drifts and a change the code can't attest must not mint an
event. Confirmed-unchanged looks roll the reference forward, so slow drift
never accumulates into a false change. Not persisted: a restart starts from
fresh baselines.

**2. The consumers** (the spec's four):

- *salience*: a `changed` verdict sets `view_changed` → `_salience_hot`,
  exactly the old wiring, plus the new away-case event line.
- *the unchanged clock*: `world_changed` is recorded in the episodic log
  (with pan/tilt/away_s/score metadata) and added to
  `unchanged_duration_s`'s anchor tuple — the clock the Sep 3 handover
  called "episodic-only and rightly distrusted" now has perceptual backing
  in both directions (verified changes reset it; the referee is actively
  watching while it runs).
- *boredom*: world-verified stillness blends into the `boredom` property —
  needs ≥3 confirmed-unchanged looks since the last world change / salience
  spike (absence of evidence isn't stillness), saturates over an hour, caps
  at 0.6, deliberately below the 0.7 bored threshold: the world being still
  raises drift propensity on its own but never flips the sampling regime
  alone. The scalar is no longer purely linguistic.
- *"still in the same spot" confirmations*: the spatial registry stamps
  `last_verified_ts` when a detector re-sighting lands within 10° of the
  stored anchor (position stability, not mere existence; a far sighting
  moves the EMA but verifies nothing). The familiarity line's "The {label},
  still in the same spot." now **requires** a verification within 900s and
  falls back to "You've noticed the {label} a few times now." otherwise —
  the minimal-phrasing doctrine's other half: prompts must not claim what
  the code can't vouch for. Per the artist's spec line, the fact (same
  spot) is the prompt's; the "of course" stays the machine's.

**3. Flow provenance** — `scene_motion.update()` now names WHY it couldn't
measure (`first_frame` / `few_features` / `no_transform` / `saccade` /
`low_coverage`), and the frame snapshot carries `camera_shift_px` +
`flow_reason`. "Saccade" is load-bearing: it means the CAMERA jumped, not
the world, and the referee skips those frames.

## Deliberately NOT built

- Passive in-view **misses** never count against registry entries — CPU
  detector recall is noisy (occlusion, light, conf floors); deliberate
  revisit glances keep sole miss authority via the existing absence ladder.
- Relocation events ("the X moved over there") — a far sighting could be a
  second instance of the same term; deferred until wanted.
- Bi-temporal validity windows for registry entries (the Zep/Graphiti
  pattern from memory-effectiveness-audit-aug30) — `last_verified_ts` is a
  small step in that direction; the full pattern belongs to the paper-state
  redesign.
- No new prompt lines beyond the one named-event string — the referee's
  output flows through existing channels (salience, episodic, boredom,
  familiarity), per one-channel-per-fact.

## Knobs (config/config.py, world-anchor block)

`WORLD_POSE_MEMORY_ENABLED` (false = no view-change detection at all) ·
`WORLD_POSE_CELL_DEG` 6 · `WORLD_POSE_COMPARE_DEG` 3 ·
`WORLD_POSE_REF_MAX_AGE_S` 1800 · `WORLD_POSE_MAX_REFS` 32 ·
`WORLD_VIEW_DIFF_THRESHOLD` 0.30 (kept) · `WORLD_ANCHOR_CONFIRM_DEG` 10 ·
`WORLD_SAME_SPOT_WINDOW_S` 900 · `WORLD_STILL_MIN_CONFIRMS` 3 ·
`WORLD_STILLNESS_SATURATION_S` 3600 · `WORLD_STILLNESS_BOREDOM_MAX` 0.6.
Retired: `WORLD_VIEW_SERVO_STILL_DEG` (meaning lives on as COMPARE_DEG).

## Verifying it

- `python debug/test_world_anchor.py` — 33 offline checks: the referee's
  honesty rules on synthetic views, anchor verification + EMA asymmetry,
  the familiarity gate both ways, the clock anchor, the boredom blend and
  its cap, flow reason codes. All pass.
- Live watches: `world_changed` entries in the event log (pan/tilt/away_s);
  `[interior]` familiarity lines — the same-spot phrasing should now
  correlate with recent looks at the object; boredom values in
  drawing_check logs during quiet stretches should rise even when caption
  concepts vary (the world input). If false `world_changed` events appear
  (e.g. strong sunlight shifts), raise WORLD_VIEW_DIFF_THRESHOLD or lower
  WORLD_POSE_REF_MAX_AGE_S — both narrow what the referee will attest.
