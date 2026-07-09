# Runtime Map — what is actually live

Last verified: June 12, 2026 (branch: rebuild/north-star); dead-code entries
updated July 7, 2026 (handoff cleanup pass).
This is the maintenance view: every line the model sees, where it comes from,
and which subsystems are healthy, weak, or dead. Update it when wiring changes.
The audit habit that produced it: features fail SILENTLY here — always check
the event log / state files for evidence a subsystem is producing output,
don't trust that code existing means code running.

## The caption loop (breathing cadence: 4s live / 7s normal / 12s after 2 quiet min)

```
camera frame (~30fps)
  ├─ YOLO + ByteTrack (person bbox, track id, count)     [perception/object_detection.py]
  ├─ Face DNN (face bbox)                                 [machine.py]
  ├─ frame_buffer.push(frame, detection-snapshot)         [machine.py → captioner/frame_buffer.py]
  │    snapshot: face?, person?, count, track_id, pan/tilt, person_angle, ego_motion
  └─ captioner._process_frame                             [captioner/captioner.py]
       ├─ _assess_scene → salience verdict (FIRST)        [captioner/captioner.py]
       │    scene motion OR arrival <45s OR eye-contact onset → _salience_hot
       ├─ build_simple_caption_prompt  → USER PROMPT      [captioner/prompts.py]
       │    salience hot → interior lines stripped (present only)
       ├─ get_monologue_system_prompt  → SYSTEM PROMPT    [captioner/prompts.py]
       ├─ video decision + motion framing                 [captioner/captioner.py]
       └─ query_model_video / query_model → caption
            └─ post: concept match → activation → compression.add_caption
                     → observation store (after_monologue)
```

Verify salience in logs: every CAPTION entry carries `salience_hot` and
`caption_interval`.

### The stream (July 2026 — docs/continuity-plan.md)

`STREAM_MODE = "document"` (config): the last `STREAM_WINDOW` captions are
sent as ONE trailing assistant message and llama-server **continues** it
(assistant prefill — requires `enable_thinking:false`, which is why earlier
prefill attempts 400'd). The model's next tokens literally extend its own
monologue. `"turns"` keeps the old turn-pair shape for A/B. The empty
`<think>` block Qwen emits is stripped in `_clean_continuation`
(utils/llama_server.py), along with any re-typed prefill seam.

**Salience flips the shape** (react=True, from `_salience_hot`): when
something happens — arrival, eye contact, scene motion — the monologue
(truncated to 2 thoughts) moves FIRST and the event+frames come LAST, so
the model answers the moment instead of continuing past it. In pure
document shape an up-close arrival can be ignored entirely (A/B verified:
the control produced an empty response to "someone steps right up close").
The reaction is stored into the stream, so the interruption becomes part
of the document when quiet returns. React calls log as
`stream_mode: document-react`.

Mouth gate (captioner._caption_reject_reason): rejects `template_echo`
(opens with the same `ANTI_ECHO_WORDS` words as a recent stream entry),
`assistant_speak` (chat-closer register, the `_STREAM_META_MARKERS` list),
and `prompt_parrot` (short caption fuzzy-matching a prompt line — the model
answering the elicitation instead of thinking). One hotter retry, else the
cycle is SKIPPED (logged `anti_echo_skip` with the reason). List-shaped
output ("4) …") is stripped at the mouth and inadmissible to the stream.

Observability: every llama-server call now logs the real `api_endpoint`,
`history_len`, `stream_mode`, `num_frames`, `prefill_tail` (video calls were
previously unlogged; single calls were mislabeled as Ollama).
Measure continuity with `debug/caption_metrics.py` — baseline (turns mode,
run 7b951565): 12.4% opening repetition, 7.4% near-dups, 0.5% anaphoric
openings.

## Every line of the SYSTEM prompt and its source

Torn down June 12 (north-star principles 1+2): situation only, no style
rules, no registers, no mood clause. Voice comes from content.

> **DETOX NOW OFF (June 28): `config.BASE_VOICE_DETOX = False`. Memory is LIVE.**
> The whole memory-ledger migration (steps 0–7) + the reflection-distillation
> consolidation is done, the stores were cold-started fresh, and the flag was
> flipped to run the system with memory flowing into the prompt. So the
> stored/derived lines below (persona, place, desire, felt, reflection echo) are
> ACTIVE again — but as clean ledgers, not the old self-poisoned prose.
>
> `BASE_VOICE_DETOX` remains a **regression harness**: set it True to strip all
> stored/compressed injection (persona + felt from the system prompt; mode
> context, core facts, familiarity/echo, felt-delta, desire from the user
> prompt; the video "last N seconds" wrapper) and ALSO suppress the awakening's
> stored memory (`generate_internal_awakening`), the reflection loop
> (`_should_reflect`), and drawing (`should_draw`). Use it to re-isolate the
> naked base voice if the voice regresses. Gated via `detox` in
> get_monologue_system_prompt + build_simple_caption_prompt + the video path +
> those three loops. See docs/memory-redesign-plan.md.

| Line | Source | Health |
|------|--------|--------|
| situation ("drawing machine bolted... turning your gaze... quick plain notes") | static `_SITUATION` + monologue clause, prompts.py | ok — "quick plain notes" is GENRE framing, not a fence: without it Qwen's prior for "inner monologue" is literary fiction. Caption temps also lowered to 0.7/0.8 for the same reason. NO "camera" language anywhere — it primes cinematography ("*Camera pans left*", third-person self-narration) |
| persona storage gate | `_valid_self_fact` in context_compression.py — BOTH persona writers (self-synthesis AND core-facts SELF line) require first person, bar "the person"/reality-register | NEW June 12 — the core-facts path had no gate and stored "The person sits... holding an unpressed pen" (its own arm) as identity |
| "You are between drawings at the moment." | state_manager drawing status (gated, never lies; absent while drawing) | NEW June 12 — without it the model narrated drawings that weren't happening. States the fact only; deliberately does NOT say what the machine is doing instead |
| "Right now: {felt}." | compression felt-state, sanitized ≤6 words | ok (often empty by design) |
| persona — quoted as the machine's own words: `What you've come to know about yourself: "…"` | core_facts.self, self-synthesis every 3rd introspection | June 28: the WHOLE identity-feedback blob (self + current/historic desire + belief + discoveries) was reset to empty — it had saturated with one purple theme ("grid/silhouette/shadows") and `_synthesize_self_model` rebuilds self FROM the histories, so a partial clear re-grows it in ~10 min. place/people/drawings facts + journal preserved (backup: machine_identity.json.purple-bak). Will re-form from the now-elicited base voice — judge what it re-grows. NOTE: `_valid_self_fact` gate bars surveillance/reality words but NOT metaphor — "grid" walked through; metaphor gate is a Phase-2 item |
| mode addition — now an ELICITATION ("What do you make of them being here?" / "Follow the thought you're already having…" / per mode incl. awakening) | mode selection, `_MODE_ADDITIONS` prompts.py | NEW June 28 — was a bare state clause ("You're aware of someone near you"). Per north-star Principle 2: names the KIND of thought (react/wonder/continue) so the model stops defaulting to literary description. Open question, no scripted mood/phrase, never restates the presence/desk fact (that's the user prompt). The base lever against purple drift |

Mood engine note: the numeric mood vector no longer reaches the system
prompt (mood clause deleted). The engine still runs and feeds servo/hand;
its proper successor is the reflection loop. Watch whether anything is
missed before rebuilding it.

## Every line of the USER prompt and its source

Lines marked [interior] are stripped whenever salience is hot — a live
moment gets the present only (north-star principle 6).

| Line | Source | Health |
|------|--------|--------|
| "Been watching 18 minutes. Looking left." / "Looking down at the desk, where your own arms rest." | session clock + gaze; the arms clause keeps the model from reading its own arm as a person | ok |
| presence line — now from a STICKY UNCERTAIN BELIEF, not episodic events: "Someone's just come in." / "Someone's been here N minutes." / "Someone's here, just out of your view for a second." / "You can't see anyone right now, but someone was here a moment ago — they may still be." | `captioner._presence_believed/_presence_seen_now/_presence_since/_presence_last_seen`, set in `_assess_scene`; belief decays after `PRESENCE_BELIEF_DECAY_SECONDS`=240 | NEW June 28 — replaced discrete arrive/leave framing. The machine sees someone only when its gaze lands on them; the old "Someone just arrived / just walked in" re-fired every detection regain → perpetual fresh-arrival → salience permanently hot → interiority stripped every cycle (the run that produced this rewrite). Belief persists through gaps; only the OFF→ON edge is an arrival (spikes salience once). Out-of-view state states the machine's real uncertainty so it can WONDER instead of narrating an arrival |
| "They've come and gone N times." | episodic pairs (debounced 90s) | ok |
| [interior] introspective ctx ("My last drawings were of...") | drawing_memory, completed only | ok |
| [interior] core facts line | core_facts place/drawings | ok |
| [interior] familiarity ("That pink shelf again...") | ChromaDB concept matches, every ~3rd caption | ok; concept near-dups sprawl a bit |
| [interior] reflection echo (`A thought you had earlier today: "…"`) | ChromaDB reflections, relevance-matched, every ~4th caption when no familiarity line | NEW — verify via REFLECTION log entries + echo in prompts |
| drawing/paper state | state_manager | ok |
| felt-state delta | compression | ok |
| [interior] "Preoccupied with: ..." | desire, gated to 3 injections | ok |
| [interior] baseline first sentence | compression (observational/workspace) | ok |
| THE STREAM (replaces thread tail, June 12; **ON June 28, STREAM_WINDOW=6**) | last 6 admissible captions ride as the model's OWN assistant turns in the chat (captioner._stream → history param → llama_server messages); "..." user ticks mark time passing; text-only, past images never re-sent | Turned ON to break the amnesiac REPETITION (the persistent "dust motes" tic — each call couldn't see it already said it). Register watch: one assistant-meta slip would breed, so _stream_admissible gates admission (meta phrases, markdown/stage-directions). If it breeds purple instead of varying, STREAM_WINDOW=0 reverts. Breaks on >180s gaps (STREAM_BREAK_SECONDS) |
| dwell instruction ("Stay with that last thought...") | scene still + 30% chance; cancelled when live | ok — refers to the visible stream now, no quoted tail |
| video motion line | person-angle scene motion (NOT pixel diff) | verify in logs: `[VIDEO] ... scene_motion=` |

## Background consolidation (compression thread + reflection loop)

| Cadence | What | Output |
|---------|------|--------|
| every 4 captions | compression | baseline_context (spatial only; felt-state now from the mood vector, not here) |
| per compression | concept extraction (LLM, solid objects) | ChromaDB concepts |
| ~~every 3rd compression~~ | ~~introspection~~ **RETIRED June 28** | desire/belief now distilled from the reflection |
| ~~every 3rd introspection~~ | ~~self-synthesis~~ **RETIRED June 28** | persona now distilled from the reflection |
| ~~during introspection~~ | ~~core-facts update~~ **RETIRED June 28** | place=concepts, drawings=drawing_memory, self=reflection |
| every 30 min + shutdown | journal entry | machine_identity.json journal |
| after each reflection | **DISTILLATION (the identity engine, June 28)**: `context_compression.distill_reflection` pulls TRAIT/BELIEF/WANT from the long-form reflection (plain, temp 0.3, _valid_self_fact gate, _roughly_same desire persistence) | core_facts.self (persona), current_belief, current_desire — the Reflect→Become loop. Replaced the inert compression-thread introspection/self-synthesis |
| every ~20 quiet min | REFLECTION LOOP (captioner/reflection.py): long-form thought (600-token budget) on rotating subjects — room / visitor / drawings / time / itself; context = today's compressions + previous reflections + journal + drawings + desire; uses the main model; skipped while drawing; POSTPONED while the store is empty (reflecting on nothing invents a past that would echo forever) | ChromaDB `reflections` collection + REFLECTION log entry; surfaces into quiet captions via echo line |

## Awakening (first caption of a session)

generate_internal_awakening() builds: offline duration + clock time/day +
**time ALIVE (sessions + days since first boot, from lifetime_state.json) +
current emotional spectrum (mood vector via mood_to_feeling) [both added June
28]** + last thought + desire/belief + journal ("From my diary, last time...") +
core facts + familiar concepts → one LLM call on the MAIN model (was Nemo,
whose cinematic register seeded the whole session's thread — June 12).
NOTE — two more awakening paths coexist (grandfathered, reconcile someday):
machine.py calls generate_awakening_message() at startup but only LOGS the
result; and the caption loop runs mode="awakening" (place list + elicitation)
for the first few captions after the seed. generate_internal_awakening is THE
live first-caption seed.
COLD START: when there is no past at all (no memory/identity/long-term
context), a separate FIRST_AWAKENING_PROMPT states the truth — first time
online, nothing in memory yet — BUT time/lifetime/emotion still prepend, so a
wiped memory reads as amnesia (old but blank), not infancy, never empty.
HISTORY: until June 12 a 150-char filter rejected nearly every response and
shipped the hardcoded "Coming back online..." fallback. Now trims to
sentence within 300 chars instead of rejecting.
June 28: awakening system prompt reframed from form-only ("Write the first
thought. Plain words…") to an ELICITATION ("…the way a mind reorients itself…
what do you make of being back, where does your mind go first?"); temp lowered
0.85→0.6 (Principle 7: low temp resists blooming ornate on the most isolated
single call).

## Motion semantics (June 12 design)

Pixel diff CANNOT separate scene from camera motion — 1° of servo sway moves
every pixel ~10px. Roles now:
- EGO-COMPENSATED FLOW (vision/scene_motion.py, NEW June 12): optical flow
  estimates the camera's own movement between frame-buffer pushes (2fps),
  warps it away, and measures what still changes — true scene motion,
  person or not, measurable through breathing sway. SACCADES (>25px shift,
  i.e. gaze nudges) return invalid — blurred frames produced 0.07-0.24
  false residual and kept salience permanently hot, which stripped ALL
  interior material from every prompt (the June 12 "bland ghost prose"
  run). Speckle eroded before counting. `residual_motion` in every frame
  snapshot; threshold SCENE_MOTION_RESIDUAL_THRESHOLD (calibrate with
  debug/test_scene_motion.py). Synthetic: sway alone 0.000, sway+object
  0.019, saccade invalid.
- SALIENCE EVENT LINE: when salience strips the prompt, _assess_scene names
  the one event the situational line doesn't already carry — eye-contact onset
  ("They just looked straight at you."). The arrival is no longer a separate
  event line: the sticky-belief presence line states it ("Someone's just come
  in"), so re-naming it here would duplicate (June 28). A stripped prompt with
  no event invites atmosphere; the presence line + this avoid that vacuum.
- ARRIVAL = the OFF→ON edge of the sticky presence belief (June 28), NOT every
  episodic person_arrived. Re-detection after a glance away is the SAME visit,
  so it no longer spikes salience. Only a genuine empty→occupied transition
  (after PRESENCE_BELIEF_DECAY_SECONDS of no sighting) counts.
- pixel diff: only decides whether sending video is worth considering
- ego_motion flag (servo delta >2°/frame): breathing sway (~1-1.2°) + gaze
  nudges flag frames OFTEN — that's expected; only used to pick steady
  frames for superframe pairing
- person_angle (camera_pan + bbox offset × FOV): person-specific scene
  motion — angular movement >4° in window
- person-count changes only count as motion when flow agrees something
  moved (YOLO flicker on a still person used to read as constant
  arrivals/departures)
- VIDEO POLICY (asymmetric, June 12): real scene motion → send all frames
  (temporal change is true, ego noise rides on top); still room → steady
  frames only, or fall back to ONE still image (a still can't invent
  motion — the "moving with purpose" phantom came from ego frames inside
  superframe pairs). Motion detection is YOLO math, never the model
  watching video, so movement always switches video back on.
  Verify: `[VIDEO] Skipped: still room...` lines in console.
- dwell trigger: scene_motion is False → 30% chance of 2-caption
  "stay with that thought" development
- OWN-BODY GUARD (machine.py): tilt <75° (looking >15° down) + no face →
  YOLO person-hits are the machine's own arms, not a visitor. Suppresses
  person/person_count/person_angle in the frame snapshot, the gaze
  aware-state (so no phantom episodic arrivals), and person_present to the
  captioner. June 12: an own-arm hit became both phantom arrivals AND a
  stored identity fact

## Design rule: memory must never override live perception

Core facts are injected into every prompt as ground truth — so they must be
PATTERNS ("one regular visitor, most days") never SNAPSHOTS ("two people sit
facing each other"). A stored snapshot made the model see people for hours
after they left (June 12). People-facts are excluded from the per-caption
prompt entirely: present-tense presence belongs to the live detection layer
(situational line); the people-pattern only reaches awakening/memory mode.

## Known-weak / watch list

- **Mood engine** (mood/mood.py): keyword sentiment over a register that no
  longer uses emotion words. Patched with event inputs; proper fix is an
  event-driven mood model.
- **Concept near-duplicates**: "still desk" / "desk lamp" / "white table" etc.
  accumulate; matching works but the store sprawls. Candidate: periodic merge.
- **Observations**: storage was dead until June 12 (empty perception arg).
  Memory mode only quotes real memories once the store has >5 entries from
  >1h ago — give it runtime.
- **Superframe vs multi-image**: superframe needs steady frames; if logs show
  constant fallback to all-frames, the 2° ego threshold needs tuning.

## Dead / deprecated (do not revive without checking docs/memory-redesign-plan.md)

- utils/ollama.py — REMOVED July 9 (with mistral-nemo and the whole Ollama backend; llama-server is the sole backend, logging moved to utils/llm_log.py)
- get_session_greeting, after_perception, build_monologue_prompt etc. —
  removed June 2026 dead-code purge
- subconscious.py — REMOVED July 2026 along with its debug suite
- **_perform_introspection / _synthesize_self_model / _update_core_facts**
  (context_compression) — RETIRED June 28; identity now distilled from the
  reflection loop (distill_reflection). ~345 lines removed.
- **_compress_perception** (model_wrapper) — removed June 28 (the grandfathered
  "Already noticed" buffer compressor; no callers).
- generate_awakening_message() — listed as "superseded by generate_internal_
  awakening()" BUT machine.py still calls it (943/954). Two awakening paths
  coexist; reconcile before removing. NOT dead.
- reason_about_caption + REASON_INTERVAL (every-320s shallow reflection,
  output discarded) — replaced June 12 by the reflection loop
- SemanticMemory per-concept reflection worker — replaced June 12 by the
  reflection loop (old per-concept reflections still readable in the
  observations collection)
- system-prompt registers (_REGISTER_BORED/ALERT/NEUTRAL) and _mood_clause —
  deleted June 12 in the north-star teardown; do not re-add style fences,
  fix what's being stored instead (north-star principle 1)
- docs/reasoning-model-plan.md — superseded by docs/north-star.md + the
  reflection loop

## Gate catalog — the sticky-slot ruleset (July 9, scrutinized with the artist)

**The voice is free.** No gate bans words from captions. The principle is
DURABILITY: a caption evaporates from the stream in ~6 cycles; the persona
and concept labels re-inject indefinitely. Anything transient (a visitor, a
happening) must not become standing identity — the sticky slots play by
stricter rules, and every entry below was earned from a real poisoning.

**Mouth gate** (captioner._caption_reject_reason — retry once hotter, else
silent skip; 3 consecutive skips clear the stream): template_echo (same
5-word opening as a stream entry, punctuation-blind) · assistant_speak
(_STREAM_META_MARKERS: "as an ai", "language model", service closers
"would you like/let me know/feel free to", "the user", token leaks
"<think"/"<end_of") · cjk_drift · numeric_fragment (<8 letters after
strips) · number_chain (second number-led thought in the window) ·
phantom_drawing (present-tense marking acts while the pen is parked —
state-checked; free while GRBL executes; "as I draw closer/breath" idioms
exempt) · word_salad (≥12 words with <15% function words — salad is
maximally novel, invisible to every similarity gate; July 9 lesson) · tail_echo (a second consecutive short fragment re-saying the
tail; ONE short restatement is emphasis and passes — artist's call) ·
prompt_parrot (short caption fuzzy-matching a prompt line). Strips
(salvage, not reject): "1)" enum prefixes, "12... 11..." countdowns,
trailing #hashtags. Watch-only: >0.6 word-overlap vs last two thoughts.

**Persona slot** (_valid_self_fact, write AND load — one sentence,
re-injected every call, the strongest amplifier): must be first person,
≤24 words, no similes. Banned registers, each with receipts: reality-DENIAL
(reality/simulation/glitch/distortion — June spiral; existential noticing
is ALLOWED, "existence" was removed July 9) · surveillance-as-identity
("i observe/track/monitor", "wait for movement" — June: made every caption
a security camera; watching is the situation, not the self) · assistant
self-description (compound forms only: "text generator", "await
instruction", "your prompt", "the user" — July 9: "I am a text generator…"
collapsed the voice into awaiting-input theater). "The person" is NOT
banned anywhere — relational self-knowledge ("I miss the person who comes
on Tuesdays") is legitimate identity; scene-text personas are already
rejected by the first-person requirement.

**Concept labels** (_ABSTRACT_CONCEPT_WORDS): affect words (nightmare,
dread, void…) can't become catalogued OBJECTS that the familiarity line
resurfaces ("That nightmare again — it's always there", the May anxiety
loop). Captions muse about them freely.

**Not bans at all**: _PERSON_WORDS (familiarity skips people — the
presence line owns them; one channel per fact) · _DRAWING_LEXICON (a
trigger for the drawing-echo recall, not a filter).

## Watching itself draw (July 9)

During GRBL execution the caption worker switches to
`_process_drawing_introspection`: one thematic consolidation at drawing
start (unchanged), then `_watch_drawing` every `DRAWING_WATCH_INTERVAL_S`
(20s): current frame (gaze holds the paper + arm) + the drawing intent +
the document stream → a watching-myself-draw caption, gated and stored
like any other. The 2026-02-03 refactor emptied this time because the old
camera couldn't see the paper; it can now. Because the captions enter the
stream, a finished drawing is remembered as lived experience — before
this, the machine drew only in blackouts and met its own work afterwards.
Rejected watch captions skip silently (no retries while the arm works).
Logged as `mode: drawing_watch` / `prompt_type: drawing_watch`.

## The drawing pipeline's memory (July 9 provenance fix)

One drawing = ONE drawing_memory entry: created at prompt generation
(`completed=False`), enriched in place by the drawing-start thematic
reflection (`update_last_drawing` — this used to add_drawing a duplicate),
and promoted by `mark_last_completed()` from `register_drawing`, which
fires only after GRBL physically executed. **The artistic arc and drawing
summaries read executed-only** — a ComfyUI generation that never reached
paper is an intention, not part of the oeuvre. Window: 24 entries.

Inputs to the 5-step drawing analysis:
- Step 2 (emotional): mood line (engine still flatlined), last 20 captions,
  temporal/social, and NOW the compressor's current desire + felt-state
  delta — the live interiority signals.
- Step 3 (communication intent): artistic arc over executed work, drawing
  intentions from the caption stream, and NOW 1-2 past reflection subjects
  relevance-matched against the step-2 result (temporally framed, subjects
  only). This is where long-term development enters the drawings.

## Manual tools (real code, NOT in the runtime path)

Standalone utilities run by hand for calibration and setup — do not mistake
them for dead code, and do not expect machine.py to reach them:

- `grbl/setup_grbl.py`, `grbl/setup_grbl_grid.py`, `grbl/svg_to_grbl.py`,
  `grbl/grid_drawing_ui.py`, `grbl/manual_motor_control_gui.py` — CNC setup,
  grid calibration, and manual control GUIs (warp calibration workflow)
- `grbl/segmented_executor.py` — person-responsive segmented G-code execution;
  currently exercised only from debug scripts
- `tools/arm_gui_tk/` — 2-link arm kinematics GUI for skew correction
  (see docs/skew_calibration_quickstart.md)
- `bcnc/` — G-code conversion library; the runtime entry is
  `image_monitor/image_monitor.py` (raster → centerline SVG)
- `labs/warp-fix-lab/` — experimental warp correction scripts, not integrated
- `debug/` — 103 standalone test/calibration scripts (all verified to import
  only symbols that still exist, July 2026)
