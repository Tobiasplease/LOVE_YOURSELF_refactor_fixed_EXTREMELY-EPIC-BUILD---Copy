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
  │    cadence 0.1s person-present / 1.5s idle (config); bbox = sticky track id, not per-frame argmax
  │    model yolov8m since July 10 — nano hallucinated the desk mannequin head as person and
  │    missed still/seated people (the cause of the 180s departure-timeout workaround). Known
  │    hard case both models fail: the life-size sweater doll reads as person (~0.8 conf)
  ├─ Face DNN (face bbox)                                 [machine.py]
  │    tracking dead zone scales with face size (config FACE_TRACK_*) — close faces used to
  │    drive servo hunting ("bobbing") that blurred captures; tracking physics now ~critically damped
  ├─ frame_buffer.push(frame, detection-snapshot)         [machine.py → captioner/frame_buffer.py]
  │    snapshot: face?, person?, count, track_id, pan/tilt, person_angle, ego_motion
  └─ captioner._process_frame                             [captioner/captioner.py]
       ├─ _assess_scene → salience verdict (FIRST)        [captioner/captioner.py]
       │    scene motion OR arrival OR eye-contact onset OR VIEW REPLACEMENT → _salience_hot
       │    view replacement (July 26): servo still + frame mostly different vs last cycle
       │    (WORLD_VIEW_DIFF_THRESHOLD) = the world changed — bumped camera, swapped scene;
       │    named event line; motion-tripped hot cycles also get a named event now (react-vacuum fix)
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

`STREAM_MODE = "world"` (config, **LIVE July 26 — THE INVERSION**): the stream
rides as ONE assistant message of timestamped log lines ("14:02 — the lamp's
still on", `captioner._stream_history`) and the user message — frames + the
world's turn (situational delta, event, reorientation — moved LAST in
`build_simple_caption_prompt`) — always ends the call. Generation begins right
after the present, never after the machine's own prose: every call answers the
world (closed loop) instead of extending its own essay (open loop — the
drift/rambling physics; in document mode text momentum beat the frame and the
rooster went unremarked). The system frame swaps to `_SITUATION_WORLD`: same
situation, LOG genre ("you keep a log — quick plain notes") instead of the
lonely-soliloquy trope ("thoughts yours alone, no one hears them"), which is
literary fiction's machine-monologue setup and summoned poetry (the artist:
"a real brain in a machine wouldn't default to shit poetry"). Logs are plain
BY GENRE (P7: name what the text is). No prefill → no truncation-cascade
run-ons; an imitated "14:05 —" stamp is stripped at the mouth
(`_LOG_STAMP_ANY_RE` — mid-entry stamps too; the first run wrote "19:06 — A
second figure appears" and one stored stamp breeds). React needs no special
shape — world IS the react ordering; salience only varies the interior mix.

**Continuity fixes after the first world run (July 27)** — it read as
isolated scene reports ("The room is full of motion... The room is alive
again..."): (1) the genre clause now states the THREAD ("each entry follows
from the ones above — what's new, what continues, what's still nagging");
(2) quiet-mode elicitations suppressed in world like document — a fresh
question every call produced a fresh scene report every call; (3) motion
salience is ONSET-only (`motion_onset`) — level-based residual kept react
perpetually hot while a person moved around (the June sustained-event
anti-pattern, reintroduced July 26, caught in one run); (4) persona slot
cleared (vibration/hum/gears saturation; backup
machine_identity.json.pre-world-bak) — judge what re-grows under the log
frame. **The clear was CLOBBERED**: the file was edited while machine.py ran,
and the process's next identity save wrote its in-memory persona straight
back — the second world run had "I vibrate when the silence gets too heavy"
in every call again. Lesson: never edit machine_identity.json under a live
process; use `debug/clear_sticky_slots.py` (refuses while machine.py runs,
backs up, clears persona+desire+belief only).

`"document"` (previous live mode, kept for A/B): the last `STREAM_WINDOW`
captions sent as ONE trailing assistant message and llama-server **continues**
it (assistant prefill — requires `enable_thinking:false`, which is why earlier
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
| situation — **REFLEXIVE FRAME July 28**: "drawing machine bolted… This is your inner voice — you keeping yourself company… The fragments that arrive between thoughts are your own senses reporting… When a question forms, it's you asking yourself" | static `_SITUATION` + monologue clause (genre only now: "Ongoing, plain, half-formed — a sentence or two…"), prompts.py | REWRITTEN July 28: the old five-negation solitude clause ("no one hears/answers/instructs/assists") invoked assistant vocabulary while denying it, and nothing told the model what the per-cycle user turns ARE — it inferred a speaker and bred "What do you think?" into full assistant mode. Now the incoming channel is named honestly (its own senses) and questions get an answer-path (own next look/thought — the PEN deliberately absent until drawing initiative is real; a frame must not promise agency the code doesn't grant). Outward hooks also admission-gated (`_OUTWARD_HOOKS` — storage, not mouth: say it once, never re-seed). Genre framing stays positive; NO "camera" language anywhere — it primes cinematography |
| persona storage gate | `_valid_self_fact` in context_compression.py — BOTH persona writers (self-synthesis AND core-facts SELF line) require first person, bar "the person"/reality-register | NEW June 12 — the core-facts path had no gate and stored "The person sits... holding an unpressed pen" (its own arm) as identity |
| "You are between drawings at the moment." | state_manager drawing status (gated, never lies; absent while drawing) | NEW June 12 — without it the model narrated drawings that weren't happening. States the fact only; deliberately does NOT say what the machine is doing instead |
| "Right now: {felt}." | the mood read's own phrase (July 10; mood_to_feeling vector translation as stale fallback), sanitized ≤6 words. **GATED July 26** (`_felt_phrase_held_reason`): the phrase is held back — numbers kept, vector translation speaks instead — if it shares a content word (4-char stem) with the persona (one channel per fact) or re-reads the standing phrase's vocabulary inside `FELT_REBORE_SECONDS`=1800 (no self-renewing lease). The rooster run (b15516be) had felt "heavy, hesitant" + persona "…silence gets too heavy" put "heavy" in 41/41 system prompts, twice — the May/June verbatim-affect spiral rebuilt. Metaphor itself stays legal (artist's call). Verify: `[🫀] … (phrase held back: …)` lines | LIVE — was stuck on "calm" while the keyword mood engine flatlined; then the spiral fuel (July 26) |
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
| [interior] introspective ctx ("My last drawings were of...") | drawing_memory, completed only — per-drawing intent phrases since July 11 (the June tags-only rule shredded subjects into single-word confetti; the anti-purple reason died when the stream pipeline started storing the machine's own intent words) | ok |
| [interior] core facts line | core_facts place/drawings — **OCCASIONAL July 26** (the June 28 brief's #1 voice fix): injected when the inventory changes or every 6th quiet caption, not per call. Per-call injection made every caption re-describe the same list, and the model re-voiced it ("scattered dust, pale floorboards" → "the dust on the floorboards settles" — the unearned-ephemera awakening) | ok |
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
| every 8 captions | **MEMORY DIFF (July 12, was spatial-only compression)**: one structured call over the recent thoughts, diffed against what the machine already knows — ROOM / NEW ABOUT ME / EVENT / PLEASANTNESS / ENERGY / FELT, "none" most cycles. The June spatial-only fix (right call, register contamination) had narrowed memory past the point where a life event could survive: everything long-term sits downstream of this call and it only passed geometry ("My name is Penelope" had no channel to tomorrow) | baseline_context (ROOM); self_notes ledger (NEW ABOUT ME — append-only, _valid_self_fact gate + load-heal, wants rejected, ~same dedupe, cap 30); events ledger (EVENT — cap 20; **PROVENANCE-GATED July 26**: an EVENT only lands if code attests a happening in the window — salience spike (`note_perception_event` from `_assess_scene`) or executed drawing (`spend_desire`) — after the rooster-run awakening stored "A pen shattered into nothingness…" as biography and ~half the ledger proved fiction; held lines print `[📆] Event held back`); mood read (folded in — was a separate call over the same captions). Ledgers persist in machine_identity.json; facts are appended, never rewritten as prose (the pre-June narrative compression kept everything but re-purpled the story every cycle) |
| per compression | concept extraction (LLM, solid objects) | ChromaDB concepts |
| ~~every 3rd compression~~ | ~~introspection~~ **RETIRED June 28** | desire/belief now distilled from the reflection |
| ~~every 3rd introspection~~ | ~~self-synthesis~~ **RETIRED June 28** | persona now distilled from the reflection |
| ~~during introspection~~ | ~~core-facts update~~ **RETIRED June 28** | place=concepts, drawings=drawing_memory, self=reflection |
| every 30 min + shutdown | journal entry | machine_identity.json journal |
| after each reflection | **DISTILLATION (the identity engine, June 28)**: `context_compression.distill_reflection` pulls TRAIT/BELIEF/WANT from the long-form reflection (plain, temp 0.3, _valid_self_fact gate, _roughly_same desire persistence) | core_facts.self (persona), current_belief, current_desire — the Reflect→Become loop. Replaced the inert compression-thread introspection/self-synthesis |
| (folded into memory diff) | **MOOD READ (July 10; merged July 12)**: PLEASANTNESS/ENERGY/FELT lines of the memory-diff call. The keyword lexicon it replaced matched emotion adjectives the post-teardown voice never uses — valence flatlined ~0 since June. FELT phrase gated July 26 (persona-overlap + lease rules — see the system-prompt felt row); numbers always land | `last_mood_read` {valence, arousal, felt}; MoodEngine.analyze_mood blends it as the vector's core each caption, person/novelty nudge on top; felt-state = the read's own phrase when the gate passes it (mood_to_feeling vector translation otherwise/stale). Prints [🫀] |
| on GRBL execution | **DESIRE SPEND (July 10, the desire arc)**: `context_compression.spend_desire` from `drawing.register_drawing` (post-GRBL only) | the act discharges the want: current_desire clears, desire_since resets, history tail annotated {spent, drawing}, last_spent_desire persisted. Surfaces: captions get "You wanted: X — you drew it." (3-caption cap, 2h), the drawing intent call gets "the next want hasn't formed yet", awakening gets "I wanted: X. I acted on it", and the next REFLECTION receives the spent want as fact so the next want forms informed, not amnesiac. Without this the slot held one sentence indefinitely and every drawing re-rendered it |
| every ~20 quiet min | REFLECTION LOOP (captioner/reflection.py): long-form thought (600-token budget) on rotating subjects — room / visitor / drawings / time / itself; context = **THE RAW RECORD — up to 80 verbatim captions from the last 75 min (hour_log; July 12 "dreaming" upgrade: every prior input was a summary of a summary, so the loop could never notice what actually happened in its own head — e.g. an hour of questions addressed to a visitor that nothing ever answered)** + today's compressions + previous reflections + journal + drawings + desire (or the freshly spent one) + events + self_notes; the subject question becomes a reading lens and the ask is a digest-then-advance: what moved, what it circled, what it assumed the record doesn't show, what it asked and whether any answer came — fact questions, not fences (how it can LEARN it needs no permission); uses the main model; skipped while drawing; POSTPONED while the store is empty (reflecting on nothing invents a past that would echo forever) | ChromaDB `reflections` collection + REFLECTION log entry; surfaces into quiet captions via echo line |

## Awakening (first caption of a session)

generate_internal_awakening() builds: offline duration (casual words — "about
18 hours", never "18.7"; decimals read as telemetry and got skipped over) +
**new-day fact when the gap crosses a date ("I was last on yesterday evening.
This is a new day.") [July 10]** + clock time/day +
**time ALIVE (sessions + days since first boot, from lifetime_state.json) +
current emotional spectrum (mood vector via mood_to_feeling) [both added June
28]** + last thought + desire/belief + journal ("From my diary, last time...") +
core facts + familiar concepts → one LLM call on the MAIN model (was Nemo,
whose cinematic register seeded the whole session's thread — June 12).
BLINK GATE (July 10): _try_blink_resume() guards BOTH awakening paths — the
machine.py display message AND the first-caption ceremony in _process_frame
(the latter used to bypass the July 9 gate, so 2-minute dev restarts still
ran full ceremonies). Gap < AWAKENING_MIN_GAP_S → no ceremony; the prior
session's last thought (full mouth gate) seeds the stream and document mode
resumes it.
TEMPORAL REORIENTATION (July 10): the awakening states the gap ONCE and it
evaporates from the six-entry stream in minutes — the machine woke after 18
dark hours and mused as if the day had never ended. Now, after a gap ≥
REORIENT_MIN_GAP_S (2h), prompts.get_reorientation_line() puts the gap + new
day in the caption prompt (section 1c) and the inward line as a standing
fact of the present — same doctrine as the close-face line — for the first
REORIENT_WINDOW_S (45 min) of the session. Its "came back on X ago" clause
coarsens with time so the line drifts instead of repeating verbatim. A live
salience event still displaces it.
AMBIENT ANCHORS (July 10): the inward (introspective-beat) line now rotates
FOUR ways — awake-time / day-part / tenure ("you've been in this room about
27 days now", prompts.get_tenure_line from lifetime_state.json) / plain. The
awake-time variant had NEVER fired: it read self.session_start, which does
not exist (true_session_start does), and the AttributeError died in the
rotation's try/except — the classic silent failure; fixed.
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
prompt_parrot (short caption fuzzy-matching a prompt line) ·
refrain_echo (NEW July 27: a shared run of 6+ consecutive words with any
stream entry — the world thread carries phrases forward and a verbatim
formula rode it as a chorus, "…nothing moves but waits for something else
to happen first" ×3; opening-echo can't see mid-sentence repeats; 2-3 word
thematic motifs still pass). Strips
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

## Drawing prompt generation (July 10: DRAWING_ANALYSIS_MODE="stream")

TWO calls (prompts.stream_drawing_analysis), replacing the 5-step committee:

1. **Intent** — one call in the machine's own voice (_SITUATION system
   prompt + "it's time to draw"). **Sighted since July 27**: the current
   camera frame rides on the call (it arrived as image_path and was DROPPED
   before — the machine decided what to draw blind, which is half of why
   everything became abstract gesture). The system prompt names the choice
   explicitly — draw the room / an object / a person, or something
   remembered, or something imagined; nothing is the wrong kind of drawing
   (the old "Not a report of the room" prohibition structurally excluded
   figuration and is gone). Materials, in this order: "the attached image
   is what you see right now", live stream tail (5 entries × 400 chars —
   was 200), drawing musings from the session (300 chars — was 120,
   mid-word), felt state, the sticky slots each stated ONCE with age
   ("Since earlier today, you've wanted: ..." — the 5-step printed
   identity==belief twice and led with them, so every drawing became a
   portrait of the same sentence: hovering-pencil ×3 on July 10), the
   executed body of work as plain chronological lines
   (`get_executed_sequence`, no LLM — repetition stays VISIBLE, never
   forbidden: motif fixation is a choice per the artist, drawing the same
   image blindly is a loop), a **vocabulary-loop mirror** (a ≥5-char
   content word present in ≥3 of the last 4 executed lines gets named
   once: "Worth noticing: 'spiral' — in almost every recent drawing" —
   the circle/crack/spiral night of July 20 looped in words, not images),
   and the best-matched reflection as a DATED 400-char prose excerpt
   ("Once, 6 days ago, you found yourself writing: ...") — subjects-only
   starved the drawing of the system's best writing; the date framing is
   the memory/present-conflation rule. The intent is stored as the memory
   entry's compressed_summary (the machine's own words, not ComfyUI prose)
   and logged as `prompt_type: drawing_intent`.
2. **Render** — mechanical translation to a ComfyUI prompt under hardware
   truth: one black pen, lines only, no shading/fills/texture (the 5-step's
   technique stage planned india-ink washes the plotter cannot do), and
   since July 27 pinned to the intention's own concrete nouns — it invented
   scenery ("industrial chassis viewed from above" from an atmospheric
   intent). Temp 0.5, logged as `prompt_type: drawing_render`. Fallback:
   prefix + intent. Inspect offline: `debug/test_drawing_intent_prompt.py`
   assembles the real intent prompt from disk state, no model needed.

LEGACY (kept for A/B: DRAWING_ANALYSIS_MODE="multi_step"): the 5-step
context_rich_multi_step_drawing_analysis — env essay / emotion manufacture
(from the flatlined mood, converging on invented stasis drama every time) /
intent / technique fiction / synthesis.

## Kinetic bus (LIVE — default ON since July 27: KINETIC_BUS_ENABLED)

`motor_panel/kinetic_bus.py` — the motor panel's markov generation lifted
into the runtime, behind the mood system. machine.py starts it INSTEAD of
`start_hand_controller()` + `organic_left_arm` (all three want
/dev/arduino_lefthand — never two at once; set the flag False to fall back
to the legacy pair). It owns the lefthand device only (fingers, elbow,
shoulder, wrist); gantry idle stays with grbl/idle_movements, gaze/lung
with their systems.

- THE AWAKENING (July 28, FULLY CONCURRENT): machine.py enables the bus
  with await_homing=True — the body holds STILL through the whole init —
  and at the main-loop threshold a background `_awakening` thread starts
  the homing choreography + the idle subprocess + the uArm's opening play
  (deferred from its connect-time slot) WHILE the main loop brings up the
  camera window, gaze, lung and bulb. The subprocess spawns IMMEDIATELY
  (its ~10s preamble moves nothing) and runs alongside the choreography;
  `utils.hooks.ARM_CLEAR_SENTINEL` (clear-at epoch written by the idle
  manager) gates $H in ensure_homed so the sweep fires the instant the
  arm is clear — choreography and homing are simultaneous, not queued.
  Everything wakes in one moment; the first temperament blooms when
  homing completes. (Fresh runs used to never home at all — the start
  sat in the session-restore branch only.)
  Failsafe KINETIC_AWAKENING_MAX_WAIT_S if homing never arrives. After
  ANY homing, the SAME dataset resumes (continuity, not a re-pick).
- LEGACY MOVEMENT WIRING REMOVED (July 28): machine.py no longer
  imports/starts/stops the old hand controller or organic_left_arm —
  no more change_to_emotion, start_autonomous_mode, send_reactivity_data
  (the "WARNING Hand controller not available" log spam is gone with
  them); the reactivity-pause block that drove the dead Markov loop is
  retired (the bus has startle/reach for person-reactivity). The unused
  hand_control_v2/ directory (zero consumers) is deleted. hand_control/
  modules remain on disk for the standalone legacy tools only — nothing
  in the runtime touches them; full deletion after the validated run.
- GAZE/LUNG DROPOUTS (diagnosed July 28): the lunggaze Arduino
  re-enumerates on USB hiccups (the /dev/arduino_lunggaze symlink
  vanishes and returns) and ServoController used to DISABLE ITSELF
  PERMANENTLY on the first I/O error — gaze+lung dead until restart,
  long predating the panel work. Now it drops the stale handle and
  auto-reconnects (throttled 3s) when the device returns, resending
  fresh state. If dropouts persist, the remaining suspect is physical:
  that board's cable/hub/power.
- EMOTION IS PULLED, not pushed: the bus calls
  `mood_engine.get_emotion_for_hand_controller` (injected at construction)
  every supervisor tick. The old push sites (`change_to_emotion` + the
  mood thread) remain as redundancy but nothing depends on them —
  deliberate hardening against years of push-plumbing accretion.
  Mapping verified: debug/test_runtime_wiring.py.
- HOMING (July 27, playback semantics): the "homing" dataset IS the escape
  choreography — home_clear() eases into the take's first sample, PLAYS it
  through once (straight playback, no markov), holds the final pose, then
  blends back on completion. Two trigger paths, both required because the
  idle subprocess is a separate process: (a) in-process —
  `grbl_utils.ensure_homed` calls `utils.hooks.on_grbl_homing_start/_done`
  (panel, manual tools); (b) cross-process — `idle_movement_manager.start()`
  plays the choreography and WAITS before spawning the subprocess (covers
  machine.py startup homing + every resume-after-drawing respawn: the
  recorded movement is the machine's first gesture on boot), and
  `ensure_homed` touches `utils.hooks.HOMING_SENTINEL` on completion, whose
  fresh mtime the bus watches to release the arm.
- MONITOR: `motor_panel/runtime_monitor.py` — a small read-only Tk window
  opened by machine.py in the old hand controller's slot
  (KINETIC_MONITOR_UI): NOW PLAYING + rotation countdown, pulled mood,
  dataset tree with the playing one marked, gaze vector, ⚡ test.
  The full practice room stays the standalone panel (ports exclusive).

- Bundle choice: session files `movement_recordings/arms/session_{state}_*.json`,
  state = the 5 mood emotions or "drawing" (overrides emotion while
  `state_manager.is_executing_cnc`). Several per state rotate on a dwell.
- DRAWING GATE (July 27): while is_drawing(), the bus drops ALL plan (x/y)
  and step (pen) sends at the output layer — the right hand belongs to the
  GRBL execution, whatever the active dataset's chains contain. Runtime v1
  is doubly safe (owned=lefthand: gantry channels never even train there);
  the gate future-proofs any widened ownership and makes the panel lab's
  drawing checkbox faithful.
- Emotion arrives by push: machine.py calls `kinetic_bus.set_emotion(...)`
  at the same two sites as `change_to_emotion(...)`.
- Transitions are seamless: new generators seed from live servo positions
  and ease into the NEAREST demonstrated state (KINETIC_CROSSFADE_S).
- Modifiers (July 27 redesign — directional logic, everything blends):
  the GAZE CURRENT (polls vision.gaze.get_gaze_state; KINETIC_GAZE_* one
  config block): one direction vector driving three coordinated effects —
  LEAN (each applicable channel drifts a bounded number of degrees toward
  the gaze, settling/releasing over LEAN_TAU — the felt one, the whole
  body sways together), TEMPO (gaze-aligned transitions eager, opposed
  reluctant), CHOICE (markov transition reweighting; needs branching in
  the recording). Scaled by KINETIC_GAZE_STRENGTH (runtime tab slider).
  Poses only ever lean by a bounded smoothed amount; the walk never
  leaves demonstrated states (debug/test_gaze_bias.py).
  REACH (July 27, KINETIC_REACH_*): while a person is TRACKED
  (person_state visible — and face tracking already points the gaze at
  them, so gaze direction = person direction), the arm leans OUT toward
  them: the gaze picks a point in the arm's measured 9-point calibration
  square (motor_panel/arm_calibration.json; bilinear over captured poses
  IS the IK — measured, not modeled) and the temperament's field shifts
  partway toward that pose, ramping over REACH_TAU on arrival/departure,
  capped at REACH_MAX_DEG. Proportional joint-space fallback until the
  arm is calibrated. Proof: debug/test_reach.py.
  STARTLE prefers a RECORDED gesture: a dataset assigned under the
  "startle" state interrupts flinch-fast (KINETIC_STARTLE_CROSSFADE_S),
  plays through, blends back to the running temperament; freeze+snap only
  as fallback until one exists. Person ARRIVAL triggers it (cooldown
  against detector flicker). All auditionable in the panel's runtime tab
  (gaze pad, influence slider, ⚡ button); the panel opens on that tab
  when KINETIC_BUS_ENABLED is set.
- Proof: `debug/test_kinetic_bus.py` (bucketing, ownership, seamlessness
  bound, gaze current, startle, homing tuck, drawing gate) +
  `debug/test_runtime_wiring.py` (mood mapping, pull, homing hooks).
- PENDING RETIREMENT (legibility directive, after the first validated
  exhibition-length run on the bus): organic_left_arm.py, the hand
  interface's generation path, and the firmware wanderer variants — they
  are now bypassed by default but still in the tree as the fallback.

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
  `image_monitor/image_monitor.py` (raster → centerline SVG). Since July 21
  2026 the live tracer is `bcnc/svg_centerliner_v2.py` (skeleton graph walk,
  every stroke drawn once); `svg_centerliner.py` (v1, contour-of-skeleton —
  drew everything twice) is kept only until v2 is confirmed on paper.
  Same date, anti-blur generation knobs: TRIGGER_PROMPT says "ink" not
  "sketch" (config.py) and the depth ControlNet releases at end_percent 0.6
  (COMFY_CNET_END_PERCENT → drawing/comfy.py node 711).
- `labs/warp-fix-lab/` — experimental warp correction scripts, not integrated
- `debug/` — 103 standalone test/calibration scripts (all verified to import
  only symbols that still exist, July 2026)
