# Mood / novelty / emotional-state audit — the dead-weight map

Aug 12 2026, ahead of the mood revamp (phase C of docs/drawing-trigger-desire-plan.md).
Two exhaustive traces, verified against the LIVE run 387e699a. Rule for the teardown:
work the tiers in order; each tier is safe once the one above is gone.

**STATUS: tiers 0–3 EXECUTED Aug 12** (same day, artist-approved), plus the
stale-push ordering fix. Tiers 4–5 (the frozen captioner island — VOICE
changes) and the redesign surface remain, pending artist decisions.

## The headline: two disconnected mood systems

**Pipeline A — LIVE and working.** LLM mood read (memory-diff call, every 8 captions,
context_compression._absorb_mood) → MoodEngine.mood_vector → 5-label ladder →
body (kinetic bus temperament pull, gaze PHYSICS_PATTERNS, breathing) + felt-state
phrase → system prompt. Empirically alive today: mood scalar moved 0.25→0.50,
felt phrases vary, ladder output varies (measured distribution over reachable
inputs: alert_curious 40%, calm_observant 25%, energized_engaged 22%,
withdrawn_distant 11%, quiet_detached 2% — an earlier claim that it "pins to
alert_curious" was wrong; only the captioner-side copy is frozen).

**Pipeline B — LIVE-BUT-FROZEN.** Captioner.current_mood_vector /
current_emotion_state are never written (machine.py:1611-1616 omits the kwargs;
no caller anywhere passes them). Everything downstream emits constants, proven
in today's log: "Everything feels slow." 28× this run (arousal frozen at 0.0);
"Right now I feel calm." 67/67 historical awakenings; emotional_tone
calm_observant 24/24 drawings; the DRAWING_SYSTEM_PROMPT "You are feeling
{emotional_state}" slot gets the same constant sentence on every drawing.

**Novelty is three unrelated numbers wearing one name.** (1) activation-network
novelty (the only meaningful one — gates prompt mode at >0.65), (2) motif
novelty from pattern_recognition (saturates at 1.0 on most captions:
min(1.0, x*1.2)), (3) whichever of the two last wrote Captioner._novelty_score
(the race: memory.py:115 vs machine.py:1066). Live evidence: activation said
0.359 while system_state.json carried 1.0; today's two shadow entries logged
0.0 then 0.81. Boredom has ONE writer and is coherent (~0.77 today).

**Genuinely load-bearing consumers (the KEEP list):**
- ActivationNetwork novelty → determine_prompt_mode (prompts.py:2634-2639)
- boredom → caption temperature + num_predict (captioner.py:1341-1371, CAPTION_TEMP_BORED)
- boredom > 0.6 "mood" context gate (activation_memory.py:555)
- mood_vector → get_emotion_for_hand_controller → kinetic bus / gaze / breathing
- get_arousal (mood_vector[1]) → kinetic move sampler
- felt-state read channel (set_felt_state source="read" → prompts.py:294, 1868)
- current_mood scalar → drawing trigger + stored intent "(mood: X)" read-back (prompts.py:1690)
- activation_snapshot.json novelty/boredom fields → debug/activation_visualizer.py
- activation_memory.py:25-32 decay/boost constants (the real tuning surface;
  there are NO NOVELTY_*/BOREDOM_* knobs in config.py — that belief was stale)

## Bugs found while auditing (fix-not-delete)

- drawing/drawing.py:229 double-normalization `(mood+1)/2` on a 0..1 scalar
  (known, phase B retires the formula anyway)
- machine.py:1012 samples the emotion for the push BEFORE analyze_mood at :1027 —
  push is always one 10s tick stale (harmless only because the bus pull wins)
- mood momentum is neutered: the 10s tick re-analyzes an unchanged caption ~3×,
  so momentum 0.2 is a ~5-tick transient, not smoothing
- last_mood_read is not persisted — every restart runs neutral for several
  minutes until the first compression lands
- captioner/memory.py:126-129 get_temporal_mood_modifier silently overwrites
  current_mood after 30 min of scene stagnation (hidden third writer)
- utils/error_tracking.py:23 expects a "mood_engine" heartbeat that is never sent
- pattern_recognition motif_history: unbounded list, +1 entry per caption forever,
  only consumer is an uncalled summary — memory leak
- log_viewer matches event_type "mood" against entries typed "mood_update" and
  reads a nested "data" dict that log_json_entry flattens — doubly broken, so
  LogType.MOOD is a write-only sink (2 writes/20s)
- machine.py:1006-1008 writes mood_{ts}.jpg EVERY 10s tick purely so log_mood can
  record a path nothing reads — dead disk output (distinct from the caption
  images mood_{reason}_{ts}.jpg, which are load-bearing)

## Removal plan (dependency-ordered, merged from both traces)

### Tier 0 — pure orphans, zero runtime edges
1. hand_control/controller.py + hand_controller_bridge.py:70-82 (never instantiated)
2. config/word_lists.py — whole file, zero importers (byte-identical duplicates
   of pattern_recognition's constants)
3. prompts.py:1247-1418 build_environmental_caption_prompt — zero callers, ~170 lines
4. prompts.py:2400-2412 _get_persistent_motifs — zero callers
5. activation_memory.py:651-669 get_activation_summary_for_introspection — zero callers
6. prompts.py:2486-2490 agent.get_mood_phrase() elif — method does not exist,
   silent no-op; plus activation_memory.py:554-555 "mood" branch feeding it
7. mood.py dead attrs: previous_mood_vector (:77), self.memory (:81), long_bias
   (:85), unused imports (:6, :13); dead params memory_context/temporal_feeling
   (:96-97) + machine.py:1030 arg + utils/continuity.py:140-176
   get_temporal_feeling (zero callers)
8. machine.py:1068 current_motifs_from_mood write (attr documented as removed)
9. vision/gaze.py:119 self.current_emotion (never read)
10. utils/error_tracking.py:23 mood_engine expectation
11. event_logging/log_type.py MOTIF_SCORE + commented siblings; utils/llm_log.py:35
    "motif_scoring" emoji; log_viewer motif_score/tinyllama parsing branches
12. debug/comprehensive_state_reset.py:58 motif_counter read (stale key)
13. movement_recordings/*.json top-level (13 files) — bus scans only arms/session_*
14. perception/person_detection_state.py:313-345 get_breathing_modifiers
    (only a debug script calls it)

### Tier 1 — legacy subtrees (must precede ANY MoodEngine API change)
15. hand_control/ ENTIRE directory (~5000+ lines: hand_control_interface,
    organic_left_arm, direct_hand_control, current_emotion.json…) — zero machine.py
    imports; last remaining MoodEngine importers outside machine/debug
    + debug/test_machine_mood_to_hand.py
16. grbl/idle_movements.py emotion params + grbl/run_idle_movements.py + both
    update_emotion no-ops + drawing/drawing.py:542-545 call site
    (keep pause_for_drawing / resume_after_drawing / stop_idle_movements)

### Tier 2 — dead branches inside live functions
17. prompts.py:2430-2454 the `if mode is None:` block (second determine_prompt_mode
    call site + _last_novelty/_last_boredom reads there)
18. activation_memory.py:523-552 every should_include_context branch except
    beliefs/story/mood-successor; removes "pressure"/"curiosity"/"restless" literals
19. prompts.py:1056-1066 unreachable bare-word activation probe
    ("Something has shifted in the space." can never fire — keys are concept IDs)
20. activation_memory.py:627-628, 667-668 dead summary keys (+ trends :642-646)
21. prompts.py:2371 unused boredom param of determine_prompt_mode (after 17)

### Tier 3 — the novelty race + pattern engine
22. machine.py:1065-1068 get_pattern_data block (kills racing writer #2)
23. captioner.py:2415-2419 set_novelty_score; mood.py:203-209 get_pattern_data
24. debug/test_drawing_trigger_values.py (exists only to inspect the race);
    debug/test_caption_to_emotion_flow.py _last_novelty read
25. DECIDE: mood.py:132 arousal nudge `+0.15*novelty` — motif novelty saturates,
    so this is a near-constant bias; recommend drop (re-sourcing from the
    activation network would create a mood→captioner import edge)
26. mood.py:14,88 PatternRecognitionEngine import/instantiation;
    utils/pattern_recognition.py whole file (kills the leak + LogType.MOTIF's
    only producer → then retire LogType.MOTIF + log_viewer motif branch)
27. ~~spaCy can go~~ EXECUTION CORRECTION (Aug 12): spaCy STAYS —
    perception/vocab_promotion.py needs the singleton (the audit missed this
    importer); it moved to utils/nlp.py. Also kept: hand_control/
    hand_expression.py + quiet_print.py (debug/test_left_arm_servos.py
    calibration tool imports them).

### Tier 4 — the frozen captioner island (delete as ONE unit; prompt text changes)
⚠ items marked VOICE change what the model reads — artist should sign off.
28. captioner.py:293-321 update() mood_vector/emotion_state params + unreachable
    felt fallback + journey append; then context_compression.py:1374-1378 vector
    priority guard + source= param
29. captioner.py:162-164 attrs; :2024-2072 describe_current_mood;
    _get_emotional_description; emotional_journey
30. Readers, in order: prompts.py:1105-1109 (VOICE: deletes constant "Everything
    feels slow."), prompts.py:1014-1015 (dead branches — v==0 hits neither),
    prompts.py:1147-1151, captioner.py:2199-2204+2307 awakening line (VOICE:
    "Right now I feel calm." — replace with felt-state read?),
    prompt_interface.py:194-206 + DRAWING_SYSTEM_PROMPT {emotional_state} slot
    (VOICE: every drawing's system prompt), captioner.py:1990 emotional_tone +
    drawing_memory tone fields (arc prompt reads a constant today),
    captioner.py:1797-1798 + memory.py:90-101 stored mood_vector/emotion_state
    (write-only payload)
31. prompts.py:1476-1548 build_step2_emotional_prompt — dead-by-config; goes if
    the multi_step A/B branch is retired (DECIDE: retire
    context_rich_multi_step_drawing_analysis entirely?)
32. state_manager.py:108-110, 215-217 emotional_expressions /
    personal_emotional_vocabulary / emotional_patterns (+ memory.py:71-73) —
    saved/restored, never written with content, never read

### Tier 5 — logging/persistence + docs
33. mood.py log_mood + LogType.MOOD + log_viewer mood branches (write-only sink)
34. machine.py:1000-1008 mood-tick snapshot jpg (dead output; note
    debug/test_spatial_compression.py:25 globs them)
35. state_manager mood_engine save/restore block (:115-118, :271-285) — MUST
    update :311 required_keys in the same edit or restore breaks;
    also :94-95 novelty_score/boredom write-only fields (:189 comment goes too)
36. Doc corrections: CLAUDE.md:102 (boredom does not drive prompt mode),
    :104 (no "restless" mode), spaCy dep line; README.md:21;
    .github/copilot-instructions.md:82; runtime-map:542-544 & :558-560 stale
    "keyword sentiment" watch items; runtime-map :934-935 push-vs-pull
    contradiction (:901-904 is correct)

### The redesign surface (phase C proper — after the teardown)
- mood.py:163-170 oscillator → drop
- mood.py:174-201 ladder → continuous (v,a,c) coordinates. HARD CONSTRAINT:
  kinetic_bus.bundle_for needs a discrete dataset key (recordings are named by
  label). Options: keep labels as a quantizer over continuous coords, or
  proximity-weighted pooling over datasets placed in (v,a,c) space (the
  motor-panel step-3 plan). gaze PHYSICS_PATTERNS + breathing patterns + panel
  UI need adapters or the same mapping.
- clarity (word-count proxy) free to redefine — one boolean of influence today
- fix momentum by analyzing only on caption change; persist last_mood_read
- machine.py:1611-1616: either pass mood_vector/emotion_state for real or
  (preferred) delete Pipeline B per Tier 4

### Debug scripts that break on signature changes (sweep when the tier lands)
test_drawing_motivation, test_state_motivated_drawing, test_debug_drawing_system,
test_with_debug_enabled, test_drawing_intervals, test_person_presence_drawing
(should_draw signature); test_runtime_wiring (bus pull — KEEP, update);
test_gaze_physics, test_idle_manager, test_kinetic_bus, test_multi_step_drawing,
test_caption_continuation, test_refactored_prompts, test_memory_injection,
activation_visualizer (keep; degrade gracefully), test_felt_gates (KEEP).

### Orphan tripwire after each tier
grep -rn "mood_vector|current_emotion_state|describe_current_mood|mood_to_feeling|
emotional_journey|LogType.MOOD|novelty_score|_last_boredom|restless|motif" \
  --include='*.py' . — expected survivors shrink per tier; anything unexpected
is a missed edge.
