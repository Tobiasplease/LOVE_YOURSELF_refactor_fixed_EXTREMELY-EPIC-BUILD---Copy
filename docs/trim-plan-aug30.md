# Codebase Trim Plan — Aug 30, 2026

A full-repo dead-code inventory for `rebuild/north-star` (head `ed1b3f3`), verified
against the code symbol by symbol. It extends `docs/caption-system-audit-aug28.md`
(the caption-layer audit) to the repository level, and re-checks every §2 verdict
in that audit. Judged against `docs/north-star.md`.

Baseline: ~62,000 lines of project Python. A third of it — ~21,000 lines across
159 files — is `debug/`. The runtime proper reaches 99 files.

## 0. Corrections to the Aug 28 audit (code moved since it was written)

- **Reflection-echo pacing — ALREADY FIXED.** The per-source counter is back
  (`captioner/prompts.py:891-900`, gated on `REFLECTION_ECHO_EVERY_N`, config = 3).
  The audit's "fix before adding anything new" item 1 is done; strike it.
- **Paper-state redesign — DOC AHEAD OF CODE.** Commit `ed1b3f3` ("paper state
  redesigned") touched only the audit doc. The standing injection is still live
  (`prompts.py:1953-1968`, `PAPER_STATE_TTL_S=1800`), and the only writer of
  `paper_state` is still the drawing attempt + image monitor — no natural-glance
  check exists. The design in the doc addendum is the spec; the build is still owed.
- **Spoken-not-stored visibility — HALF DONE.** The terminal marker exists
  (`captioner.py:1858` `[🔂] … spoken, but kept out of the stream`); the caption
  display itself (`utils/caption_display.py`) still gets no gate provenance.
- Everything else in §2 BROKEN verified still broken: ledger in-prompt-confirmation
  discount unbuilt (no such concept in `durable_ledger.py`), memory-mode 240 s
  hardcoded (`captioner.py:1407`), mention-boost fatigue absent
  (`perception/spatial_registry.py:142` — flat 4.0 boost, no habituation).

## 1. Zero-behavior-change deletions (~1,750 lines) — do first

> **DONE Aug 30 2026** — executed as five independently revertable commits
> ("trim 1a" … "trim 1e") on this branch; see each commit message for exact
> contents. Two deviations: sdk_uarm was KEPT (it's a submodule pointer to
> the uArm SDK the live teach backend installs from, not an empty dir), and
> the beliefs store's read fn (`get_beliefs`) was kept pending the
> ship-or-delete decision (§1a note below).

### 1a. The audit's §2 DEAD list — confirmed, 359 lines

All verified dead with caller lists; the audit's ~350 estimate was accurate.

| Item | Location | Lines |
|---|---|---|
| `build_caption_prompt_with_options` | `captioner/prompt_interface.py:25-130` | 106 |
| `_build_drawing_introspection_prompt` + purple system prompt | `prompt_interface.py:213-282` | 70 |
| `build_focused_caption_prompt` | `captioner/prompts.py:2088-2096` | 10 |
| `_build_simple_system_context` (incl. beliefs/story injection) | `prompts.py:1627-1671` | 48 |
| `PERCEPTION_SYSTEM_PROMPTS` block | `prompts.py:70-118` | 49 |
| `STATIC_SYSTEM_PROMPT` trio + `_MISTRAL_MODEL` + getter | `prompts.py:17-47` | 31 |
| `should_include_context` + vestigial import | `activation_memory.py:506-523`, `prompts.py:1798` | 19 |
| observational mode branch + `get_observational_context` | `prompts.py:1617-1618`, `:1061-1080`, registry entry | 24 |
| dead imports | `prompt_interface.py:15-16`, `captioner.py:45` | 3 |

Notes that de-risk it:
- The debug scripts that were the only callers are **already broken** (they unpack
  a 3-tuple from a function returning 4, and one calls a `caption_image` method
  that no longer exists anywhere). Deleting the layer breaks nothing that works.
- Keep the `"observational"` *fragment*: `captioner.py:2825` still calls
  `get_monologue_system_prompt("observational")` for the drawing-watch beat.
  Only the router branch + context fn go.
- Beliefs-line decision (the audit's open question): the store's read path is
  reachable only via this dead layer. Deleting it makes `get_beliefs` fully dead
  too — decide "ship as a live slot" or delete store access with the layer.

### 1b. Whole dead files — 1,041 lines

| File | Lines | Why dead |
|---|---|---|
| `perception/spatial_memory.py` | 449 | zero importers, not even debug |
| `vision/spatial_awareness.py` | 290 | Ollama-era (`localhost:11434`, `/api/generate` — backend removed July 9); one debug importer |
| `vision/drawing_inspection.py` | 159 | zero importers; functions only call each other |
| `config/model_settings.py` | 143 | see §3 — inert end to end |

### 1c. Orphaned symbols in live files — ~350 lines

Zero references repo-wide, verified individually:
`prompts.build_identity_line` (:492, 57 ln — superseded by the inline dose at
:242) · `prompts.get_social_context` (:946, 31 ln) · `ContextualMemory.recall` +
`_format_memory` + `get_drawing_context` + `format_drawing_context` +
`get_desires` + `_truncate_at_sentence` (activation_memory, ~96 ln — the whole
activation-recall READ path; only `store()` is live) · `gaze.get_gaze_narrative`
(:694, 46 ln), `get_self_motion`, `is_paper_search_active` ·
`continuity.describe_sleep_duration`/`get_presence_phrase`/`format_timestamp`
(41 ln) · `Captioner.capture_mood_snapshot`, `truncate_caption` ·
`MultimodalModel._clean_response` · `ContextCompressionEngine.reset_context` ·
`MemoryMixin.get_identity_summary`, `get_recent_memory` (shadowed by an identical
override in Captioner), `save_activation_state` · `aruco_detector.is_paper_present`
· `semantic_memory.TIER_NEW/TIER_FAMILIAR` · unused imports in
`context_compression.py:19`, `memory.py:23`.

Do NOT touch: `_query_superframe` / `_query_multi_image` in `utils/llama_server.py`
— cold but config-gated (`VIDEO_MODE`), a live toggle, not dead code.

### 1d. Dead-by-construction runtime branches

- **`machine.py` uArm "controller" backend**: `machine.py:141-144` imports
  `uarm_control.uarm_controller` / `motion_manager` / `simple_api` — **those files
  do not exist**. The import always raises, so `UARM_BACKEND` always falls back to
  `"teach"` and the whole controller branch (`machine.py:276-347`) is unreachable.
- **`grbl/idle_movement_manager.py` start path** (`:65-110`): spawns
  `grbl/run_idle_movements.py`, which was deleted in the Aug 12 teardown. machine.py
  imports only `stop_idle_movements` ("kills stray wanderer processes"). Keep the
  stop, delete the permanently-failing start.

### 1e. Root and data clutter

Delete: `run_machine.py` (restart wrapper superseded by the tmux loops; spawns
bare `python` outside the venv) · `ohbaby.txt` (orphaned uArm teach recording —
or move into `movement_recordings/uarm/`) · `default_grid.svg` ·
`left_arm_calibration.json` (pre-kinetic-bus, zero refs) ·
`grbl/warp_calibration.json.pre_grow`, `.pre_grow2`, `.pre_manual` · `sdk_uarm/`
(empty directory; also drop its lint-skip globs) · `labs/` (3 warp experiments,
"not integrated" per runtime-map, untouched for a year).

## 2. One thing to ADD, not delete

`motor_panel/kinetic_bus.py:107` reads `motor_panel/arm_calibration.json` — the
9-point pose grid for `_bilinear_pose`. It is **not in the repo and not
gitignored**; it exists only on the machine, written by `motor_panel/panel.py`.
The remote is the only backup (offline exhibitions — your own standing caution).
Commit it.

## 3. The Qwen 3.8 question — where output quality actually lives

Checked the whole model-config path; the code is NOT sabotaging 3.8 sampling:

- `config/model_settings.py` looks alarming (no `qwen3.8:27b` key → every lookup
  falls through to the `llava:7b-v1.6-mistral-q5_1` entry) but it is **inert**:
  `utils/llama_server.py` forwards only temperature/top_p/max_tokens/
  repeat_penalty/seed + the `_SAMPLER_PASSTHROUGH` set. The llava stop-token
  chants, `top_k`, `num_ctx` never reach a payload, and the one surviving field
  (`num_predict: 200`) is swallowed by `max(…, 400)` in the drawing builder. Real
  sampling comes from `run_38.sh` env + config, as intended. Delete the file
  (§1b); the emitted payloads are byte-identical.
- So the "not on 3.8's level" verdict is what audit Part 2 already names: the
  **becoming bottleneck**. The mouth is fixed; the middle layer is one narrow
  pipe (TRAIT/BELIEF/WANT slots + a self-confirming ledger). The 27B amplifies
  the self-story it's fed, and it's fed one sentence four ways. The fixes are
  the already-planned B1–B4 — not sampling work, not new fences (P7: if it
  slips ornate, lower temp / cleaner gate / sharper genre frame, never a fence).
- Two standing sampling notes remain queued, unchanged: vendor-shaped
  presence_penalty experiment (repeat 1.0 + presence 0.6–1.0) behind a full
  evening of metrics on 1.05/DRY; `--image-min-tokens 1024` is already in
  `run_38.sh`, still absent from the 3.6 arm before any perception A/B.

## 4. Needs-artist-decision (don't let a cleanup pass decide these)

- **`start_impostor.sh` / `start_impostor_panes.sh` pin the parked 3.6 arm**
  (`run_27b.sh`) and hardcode `/home/impostor/…`. The exhibition launchers boot
  the wrong model today. Repoint to `run_38.sh` or retire to one launcher.
- **`arduino_src/`**: 15 `.ino` in near-duplicate families (`hand_controller`,
  `_clean`, `_debug`, `_fixed`; 8× `lightbulb*`). Only you know which variants
  are flashed; mark the flashed ones, archive the rest.
- **Per-machine config overrides** (`config/gpu-peon/` ×4, `config/jbe-osx/`,
  `config/impostor-bot-win/`, `config/debug_captions_only.json`): zero code refs,
  still carry `DEBUG_OLLAMA_PROMPTS`-era keys, but they encode other install
  sites. Archive or delete knowingly.
- **`movement_recordings/arms/projects/`**: the engine globs `arms/`, not the
  subfolder — artist archive or lost takes?
- **`bcnc/svg_centerliner.py`** (v1): `bcnc/__init__.py` says keep until v2 is
  confirmed on paper. Has that confirmation happened?
- **Config knobs nothing reads**: `USE_HAND_CONTROLLER`, `KINETIC_SAFE_NEIGHBOURS`,
  `GRBL_IDLE_CENTER/RADIUS_MIN/RADIUS_MAX/FEED_RATE/UPDATE_INTERVAL`,
  `DEBUG_EMOTION_CHANGES`, `DRAWING_HISTORY_LIMIT`.

## 5. Keep — looks legacy, is load-bearing (the traps)

- `debug/caption_monitor.py` — spawned by machine.py at boot. Runtime, not debug.
- `bcnc/` — NOT dormant vendor code: `image_monitor` → `bcnc.raster_to_centerline_svg`
  is the raster→centerline→G-code path. Its internal imports go through a
  `sys.path.insert` hack in `bcnc/__init__.py`, so naive import-scanners will
  wrongly call the converters dead.
- `grbl/svg_to_grbl.py` the FILE is a deletable CLI, but the `svg_to_grbl`
  FUNCTION lives in `grbl/__init__.py:19` and is live via image_monitor. Delete
  the file, never the function.
- `motor_panel/panel.py` / `arm_studio.py` / `arm_model.py` — offline studio, but
  panel.py is the author of `arm_calibration.json` (§2).
- `hand_control/` (3 files) — torn down Aug 12, deliberately kept for left-arm
  servo calibration via debug scripts.
- `prompt_panel/`, `tools/`, grbl setup/calibration UIs, `calibration/` — hand-run
  tooling. (Fix `calibration/README.md`: it points at `debug/test_paper_detection.py`,
  which doesn't exist.)
- `utils/` — all 17 files live. `captioner/` — all 12 live. `drawing/` — all 4 live.

## 6. `debug/` — the biggest surface, its own pass

159 files, ~21k lines. Not audited per-file here. Suggested split:
- **runtime-adjacent, keep at top level or move to `tools/`**: `caption_monitor.py`
  (runtime), `clear_sticky_slots.py` (the only safe slot clear),
  `sanitize_future_timestamps.py` + `test_clock_guard.py` (clock-step remedy),
  `log_viewer.py`, `archive_future_runs.py`, servo/aruco calibration tools.
- **evidence of finished experiments** (think-probe, 27B replay, phase0 reports,
  crisp A/B, …): they answered their questions; the answers live in docs/. Archive
  the lot to `debug/archive/` (or delete — git remembers) rather than curating
  one by one.
- ~40 debug scripts are the only consumers keeping `hand_control/`,
  `bcnc/svg_centerliner.py`, `grbl/segmented_executor.py` + `gcode_segmenter.py`,
  and `vision/spatial_awareness.py` alive — decide those pairs together.

## 7. Order of operations

1. **§1 wholesale** (one commit per subsection; each is zero-behavior-change and
   independently revertable). Commit `arm_calibration.json` first (§2).
2. **Audit Phase 2 remainder** (the self-fighting): ledger dedup +
   in-prompt-confirmation discount (B1, prerequisite for a second self-fact ever
   existing) · memory-mode cadence → config · mention-boost habituation.
   Reflection-echo pacing is already done.
3. **Build the paper-state redesign** the latest commit specced (event +
   relevance, natural-glance checks) — currently doc-only.
4. **Collapse the mode vocabulary** to relational | quiet + beats (audit Phase 3),
   then the **interior-budget scheduler** (Phase 4) — one policy replacing eight
   counters, scheduling only living lines.
5. **The becoming expansion** (B2 people-history, B3 unbound want + curdling,
   B4 unchanged-ness as fact; B5 deliberately nothing) — this, not sampling, is
   the 3.8-quality lever.
6. `debug/` archive pass (§6) whenever convenient.
