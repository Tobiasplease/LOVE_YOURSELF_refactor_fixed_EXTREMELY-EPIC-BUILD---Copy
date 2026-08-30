# Dead-Code Census — Aug 30, 2026 (the holistic view)

One document of record for everything found dead, vestigial, or silently
broken across the ENTIRE codebase, before any further deletion. Method:
vulture 2.16 over all runtime packages (231 candidates at ≥60% confidence),
then EVERY candidate hand-verified for dynamic dispatch — getattr/setattr,
the P("…") prompt registry, utils/hooks.py assignments, thread targets,
HTTP-handler and dataclass conventions, config-override setattr, state
save/load round-trips, debug/ and shell consumers. Companions:
`trim-plan-aug30.md` (repo-level, phase 1 executed), and
`memory-effectiveness-audit-aug30.md` (memory layers + field survey).

## The whole codebase in one table

~62,000 lines of project Python at branch start; ~21,000 of it debug/.

| Bucket | Lines | Status |
|---|---|---|
| Deleted in trim phase 1 (commits trim 1a–1e) | 4,128 | done, each commit revertable |
| Newly confirmed dead (this census, §1) | ~730 | verified, awaiting go |
| Debug-only code in runtime dirs (§4) | ~770 | decide with the debug/ pass |
| Vestigial-but-live memory machinery (§5) | ~500 | decision items (memory audit) |
| Hand-run tooling that LOOKS dead (§6) | — | keep; documented so nobody trims it |
| Live runtime | the rest | — |

Verification counts: 231 vulture candidates → 154 confirmed dead,
20 debug-only, 12 false positives (framework methods, dataclass fields,
signal-handler ABI, loop vars), the rest tool-kept or duplicates of
already-known findings. No candidate was saved by hooks, the prompt
registry, threads, or serialization round-trips — the two dynamic-access
saves in the whole run were `DEBUG_HAND_CONTROLLER` (getattr in
hand_control/quiet_print.py:22) and debug-script getattr reads.

## 1. Confirmed dead — safe deletions, by file

**Whole modules / features (one coherent abandoned feature each):**
- `utils/view_orientation.py` — entire 74-line module: `describe_view_orientation`
  is imported at `captioner/prompts.py:10` and never called; its intended
  inputs `Captioner.view_pan/view_tilt` (captioner.py:338-339) are write-only.
  The whole egocentric-orientation feature, ~76 lines.
- `utils/continuity.py` TimeKeeper — the class, its 4 methods, the module
  singleton, plus `describe_time_gap` (imported by memory.py, never called):
  ~41 lines.
- `image_monitor/image_monitor.py:123 _quick_paper_check` — 132 lines +5
  collateral. Unreachable nested def; comments at :256 say the check moved
  post-homing (`grbl_utils._paper_check_after_homing`), and its body only
  wrapped that same function anyway. No flag reaches it.
- `reactivity/camera_reactive.py` — `_calculate_motion_intensity`,
  `_calculate_chaos_multiplier`, `_calculate_speed_multiplier`,
  `get_status_info`, `set_smoothing`, `recent_peak`: ~70 lines. The first
  three read attributes `__init__` never creates — they would crash with
  AttributeError if called, independent proof they haven't run since the
  inline metric path (:149-151) replaced them.
- `grbl/gcode_optimizer.py:354 optimize_file` — 45 lines; live entry is
  `optimize_gcode` via grbl_utils.py:819.

**config/config.py — 27 dead constants** (zero code readers; the
event-logger metadata dump getattr's everything and doesn't count):
`USE_HAND_CONTROLLER`, `IDLE_AMPLITUDE_X/Y`, `IDLE_CENTER_X/Y`,
`IDLE_EASING`, `SWEEP_PROBABILITY`, `BODY_REGION_OVERLAP`,
`KINETIC_SAFE_NEIGHBOURS`, `GRBL_IDLE_CENTER/RADIUS_MIN/RADIUS_MAX/
FEED_RATE/UPDATE_INTERVAL`, `GRBL_ENABLE_SEGMENTED_EXECUTION`,
`GRBL_MAX_SEGMENT_SIZE`, `GRBL_ENABLE_PERSON_DETECTION_PAUSE`,
`UARM_CONNECT_ON_STARTUP`, `UARM_HOME_ON_CONNECT`, `UARM_DEFAULT_SPEED`,
`UARM_SWIFT_PORT`, `CAPTIONER_TEMPERATURE`, `ENVIRONMENTAL_TEMPERATURE`,
`DEBUG_EMOTION_CHANGES`, `NO_HANDS`, `DRAWING_HISTORY_LIMIT`,
`PAPER_USE_DRAWING_TILT` — plus their dead import/fallback lines in
gaze.py:7-26, body_schema.py:10, grbl_utils.py:24-51,:1766-1779 (~15 lines).
Debug-only constants to keep: `LIGHTBULB_SENSITIVITY`, `SERIAL_PORT`,
`UARM_PORT`, `UARM_MOVEMENT_NAMES`. Note `NO_HANDS` is SET by four JSON
overrides yet read by nothing — the loader happily sets values nobody
consults (harmless today; hand teardown removed the reader).

**vision/gaze.py — ~34 lines + 6 dead config imports:** write-only globals
(`target_x/y`, `idle_next_move_time` + its pause_duration block,
`pan/tilt_offset_time`, `pan/tilt_easing_variance`, `pan/tilt_target_time`,
`searching_target_pan/tilt`, `tracking_person_position`), dead constants
(`MAX_PAN/TILT_VELOCITY`), dead physics attrs (`orbital_phase`,
`param_blend_factor`), `get_current_glance` (runtime-map:514 still
advertises it — stale doc; live consumers use `get_glance_info`/`get_last_glance`).

**captioner/ — ~85 lines:**
- captioner.py: `caption_window`, write-only presence attrs
  (`_presence_since/_last_seen/_seen_now`), `sessions_since_boot`,
  `is_processing`, `view_pan/tilt`, `_current_reactivity_data`,
  `_last_sent_caption` (~23 ln).
- context_compression.py: `hashlib` import, `recent_image` loop (dropped
  visual-grounding feature, 6 ln), `historical_context` (12 ln),
  `duration_description` local (keep `_format_duration` — live via
  session_info), `activation_context` (17 ln). See §3 — these document
  disconnected features.
- semantic_memory.py: TWO orphaned method bodies whose `def` lines were
  lost in past edits — :454-462 (a truncation helper stranded inside
  `_find_any_person_concept`) and :849-861 (the body of `delete_concept`
  stranded after `get_concept_observations`'s return, containing live
  ChromaDB delete calls). Also `concept_name` (:570) — assigned and unused
  while the comment claims the relevance query uses it (latent behavior
  gap, see §3).
- memory.py: `Set` import, `describe_time_gap` import, `BOREDOM_THRESHOLD`,
  `CAPTION_SAVE_THRESHOLD`, `CaptionTuple` (~5 ln).
- prompts.py: `describe_view_orientation` import, `SELF_CRITIQUE_PROMPT` +
  `SELF_CRITIQUE_SYSTEM_PROMPT` (their only importer drawing.py:31 never
  uses them), `NUMBER_GENERATOR_SYSTEM_PROMPT` (~17 ln). Neither critique
  constant is in the prompt registry — not panel-editable, truly dead.
- frame_buffer.py: `_target_fps`.
- activation_memory.py: `Set` import.

**machine.py — ~6 lines:** `VERBOSE`, `previous_beliefs`,
`last_printed_caption_time`, `prev_gray`, `smoothed_pwm` (superseded by
reactivity metrics), dead `mood_frame` param (positional — change both
sites together). `signal_frame` is ABI, keep.

**utils/ — ~20 lines:** state_manager `cnc_start_time` (never read, not
serialized), `last_no_paper_skip_ts` (written from two live paths, read by
nobody — a skip-throttle never wired up), `awaiting_environmental_phase`
(set on the captioner object which never reads it);
caption_display `chunk_delay`; inference `reload_model` (see §3);
llama_server `import io`.

**perception/ + safety/ + mood/ + event_logging/ — ~35 lines:**
detection_memory `_timestamp` + `get_person_tracks`; open_vocab_detector
`get_detections` (live one is `get_detections_for_drawing`);
person_detection_state `departure_delay`; vocab_promotion
`get_promotion_history`; aruco_detector `is_marker_visible` +
`get_detection_confidence` (superseded by `get_status()` dict reads;
independent of the still-reachable PAPER_CHECK_METHOD="aruco" path);
paper_detection `detection_count` (write-only); mood.py
`last_person_detected` + `last_caption` (abandoned change-comparison);
event_logger `set_run_id`; LogType `SYSTEM` + `INTROSPECTION` (no reverse
lookup exists, old logs can't break — keep-or-delete is taste).

**drawing/ + motion stack — ~70 lines:** comfy `set_config`/`update_config`;
drawing.py `last_prompt`/`last_drawing_prompt`/`last_critique`/
`quota_manager` + the dead critique import; drawing_memory
`update_last_drawing`/`get_last_failure`/`get_artistic_arc_context`;
grbl_utils dead critique block (:1440-1467 — builds `critique_prompt`, then
hardcodes `self_critique = ""`, making the `if self_critique:` branch
dead-by-construction; the `if True:` at :1477 is INTENTIONAL, documented
at :1469, don't "fix" it); kinetic_bus `pose_of` (aspirational, never
wired); arms_markov `freeze` + `_pause_until` + its wait-loop (superseded
by `_hold_until`); session.py `GROUPS`; lightbulb legacy aliases
(`set_brightness`/`set_base_brightness`/`set_pwm`/`caption_boost`/
`LightbulbController` — live calls are `caption_flash` +
`set_frame_diff_brightness`); servo_control `set_lung`; teach_menu
`_smoothed_path`; warp_calibration `similarity_transform` (the ONE orphan
among its 9 flagged functions — the other 8 are the calibration tool's API);
gcode_segmenter `end_line`; idle_movement_manager `RETIRED` flag;
segmented_executor `fallback_to_original` attr.

## 2. Latent bugs surfaced by the census (not deletions — decisions)

1. **`grbl/grid_drawing_ui.py:19` broken import**: imports `inverse,
   scale_x, scale_y, translate` from warp_transform, which exports none of
   them → `WARP_AVAILABLE` is permanently False → the tool's warp toggle is
   a silent no-op and `corrected_warp_transform_line` (85 ln) is
   unreachable. Fix the import or knowingly retire the toggle.
2. **`semantic_memory.delete_concept` doesn't exist** (def line lost);
   `debug/test_semantic_memory.py:113` AttributeErrors. Restore the def or
   drop the caller when deleting the orphan body.
3. **Three dangling config switches**: `GRBL_ENABLE_SEGMENTED_EXECUTION`,
   `GRBL_MAX_SEGMENT_SIZE`, `GRBL_ENABLE_PERSON_DETECTION_PAUSE` are
   imported, re-defaulted, and never read — flipping them changes nothing.
   Same class: `NO_HANDS` in four override JSONs.
4. **`utils/inference.reload_model` has no caller** — half the VRAM dance
   is missing; recovery happens lazily via `ensure_server_up` guards.
   Works today; deleting it should note the docstring at inference.py:11.
5. **semantic_memory.py:570** — the concept-relevance query uses the raw
   monologue `text`, not the concept name the comment describes.
6. Already ticketed in the memory audit: the never-appended `timeline` →
   permanent false "no person yet" in the drawing system prompt; the false
   boredom docstring; activation edges never persisted.

## 3. Dead code that documents disconnected features (flag before deleting)

Each of these is correct to delete AND is evidence a designed behavior is
silently off — deleting removes the only trace:
- `activation_context` (compression prompt was meant to see the activation
  summary — ties into the activation-network retirement decision),
- `historical_context` (compression was meant to see prior compressions),
- `recent_image` (compression was meant to be visually grounded),
- `last_no_paper_skip_ts` (a no-paper skip throttle never completed),
- `reload_model` (explicit post-drawing model restore),
- drawing.py `last_critique` chain + grbl_utils critique block (the
  self-critique feature was removed Aug 5; these are its stumps).

## 4. Debug-only (decide together with the debug/ archive pass)

`grbl/segmented_executor.py` (388) + `grbl/gcode_segmenter.py` (246) — a
matched pair reachable only from `debug/test_person_responsive_cnc.py`;
their config flags are already dead (§2.3). `captioner.py
_extract_character_insights` (51). `gaze.set_llm_zone` (48) +
`tracking_person_movement` + `FACE_VELOCITY_SMOOTHING`.
`drawing_memory.get_thematic_context` (15, its producer prompt removed
Aug 5). `arm_model.separation` (9 — arm_studio computes its own; only
debug/test_arm_model calls it). `body_schema.add_reference` (11, manual
enrollment). frame_buffer `get_recent`/`seconds_buffered`
(debug/test_llama_server). semantic_memory `get_concept_observations` (32)
+ `merge_concepts` (33) (debug/test_semantic_memory). state_manager
`get_drawing_status` + `current_gcode_file` (debug readers only).

## 5. Vestigial-but-live (decision items, from the memory audit)

The activation network + ContextualMemory (~450 lines live compute whose
sole behavioral output is the 0.05-temp boredom nudge, while its
compression boost inflates `times_seen` on the real ledger); the
`observations` ChromaDB collection (biggest write volume, zero readers);
`baseline_context` (an LLM call every 8 captions reaching only the drawing
system prompt + reflection); MemoryMixin's never-written state
(`self_model`, `timeline`, `day_stones`, `known_people`, `primary_person`)
still serialized every save. Full analysis + retirement plan:
`memory-effectiveness-audit-aug30.md`.

## 6. Looks dead, is NOT — the keep list (so future passes don't cut it)

Framework/ABI: `prompt_panel/server.py do_POST`/`log_message`
(SimpleHTTPRequestHandler dispatch — serves the live prompt panel);
`machine.py signal_frame`; `PaperCheckResult.check_image_path`/
`llm_response` (dataclass fields, debug readers). Params live callers pass:
`compression kind`, `captioner preview`, `get_monologue_system_prompt
emotional_state`. Hand-run tools (runtime-map:1450 "do not mistake them
for dead code"): the 8 warp_calibration functions, setup_grbl*/
generate_grid_image/grid_drawing_ui, motor_panel panel/arm_studio/
arm_model, tools/, prompt_panel, hand_control (servo calibration),
bcnc/svg_centerliner.py (until v2 confirmed on paper). Config-gated cold
paths: `_query_superframe`/`_query_multi_image` (VIDEO_MODE), the ArUco
paper path (PAPER_CHECK_METHOD). `debug/caption_monitor.py` (spawned at
boot). `GRBL_IDLE_ZONE` (tools/arm_gui_tk). `DEBUG_HAND_CONTROLLER`
(getattr). sdk_uarm submodule pointer.

## 7. Proposed execution order (each its own revertable commit, pending go)

1. §1 mechanical deletions, clustered as: (a) whole modules/features,
   (b) config constants + their dead imports, (c) gaze globals,
   (d) captioner/utils/perception/mood/event_logging strays,
   (e) drawing/motion strays. Zero behavior change; compile + grep gate
   after each.
2. §2 bug decisions folded in where trivial (restore `delete_concept` def
   OR drop its debug caller; delete the three dangling GRBL flags with
   their imports).
3. §5 memory retirements per the memory audit (activation network with
   boredom preserved, observations decision, MemoryMixin state) — behavior
   change is intended there (ends the times_seen inflation), so it runs as
   its own series after §1 proves quiet.
4. §4 with the debug/ archive pass.
