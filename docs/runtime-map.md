# Runtime Map — what is actually live

Last verified: June 12, 2026 (branch: rebuild/north-star); dead-code entries
updated July 7, 2026 (handoff cleanup pass).
This is the maintenance view: every line the model sees, where it comes from,
and which subsystems are healthy, weak, or dead. Update it when wiring changes.
The audit habit that produced it: features fail SILENTLY here — always check
the event log / state files for evidence a subsystem is producing output,
don't trust that code existing means code running.

## Known failure, SOLVED Aug 9 — the 27B was running on the CPU

Runs went off a CLIFF, not a slope: healthy for ~20 minutes, then EVERY call
failed for the rest of the run, always starting just after a drawing cycle.
The two calls per hour that got through took 34.8s and 58.8s against a 60s
timeout — so nothing was hung, generation had simply collapsed to ~2 tok/s.

`ggml_cuda_init: failed to initialize CUDA: no CUDA-capable device is detected`
in `event_log/llama_server.log`, on every recent server start. llama-server was
loading the whole 27B into system RAM (23.5GB RSS, 1300% CPU, 0 MB VRAM) while
the GPU sat idle. Cause: **ultralytics implements `device="cpu"` by setting a
PROCESS-GLOBAL `os.environ["CUDA_VISIBLE_DEVICES"] = "-1"`** and never restoring
it (`utils/torch_utils.py:185`). The open-vocab detector (Phase 1, Aug 5) calls
that every 4s, so from its first pass onward every subprocess machine.py spawned
inherited "-1". The cliff lands after a drawing because a slow call trips the
wedge watchdog, the watchdog restarts llama-server, and the NEW server is the
one that inherits the poisoned env — which is also why it never recovers and
why "restart fixes it" only worked when the restart beat the first detection.

Fixed in two layers: the detector restores the variable itself
(`_cpu_without_hiding_the_gpu`), and `start_server` hands the child the
`CUDA_VISIBLE_DEVICES` captured at import rather than whatever an imported
library has since done to `os.environ`. `_warn_if_running_on_cpu` now reads the
server's own startup lines and says so loudly — this presented as "healthy but
slow" because a CPU-loaded server answers `/health` perfectly.

NOTE the related weakness this exposed: `ensure_server_up()` returns True as
soon as `/health` answers, so a crippled-but-responding server is declared fine
by the recovery path. `/health` is a liveness check, not a health check.

## Known failure, SOLVED Aug 5 — the stderr pipe

`llama-server` was started with `stderr=subprocess.PIPE` and nothing in the
codebase ever read it. The pipe buffer is 64KB; the server logs every request;
once full, the writing thread BLOCKS in `write()` and generation stops while
`/health` keeps answering from another thread. That is the "hangs mid-generation
but health is ok" failure the wedge watchdog was built for, and the artist's
description exactly: works for a while, then hangs forever, restart fixes it.
Multi-image looked guilty only because it is the chattiest path, so it filled
64KB fastest. Measured at 20-53 timeouts/hour on every 27B run since Aug 1.
Fixed: stderr goes to `event_log/llama_server.log` (rotated at 50MB), which also
gives the server's own account of any future failure. Reproduced and verified
in isolation — an unread PIPE blocks, a file does not.

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

**LIVE MODE: `"document"` again since July 28** — the rooster was a detection
failure, not a shape failure; document + the new detectors/gates had never
been tested, and the artist judged the continuous flow the more alive machine
(world's delta framing also bred fake-delta tropes: "the light feels
different now"). VIDEO_MODE=multi since the same day (plain stills, standard
mtmd, no patched fork). World remains below for A/B.

`STREAM_MODE = "world"` (the July 26 inversion, now the A/B arm): the stream
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
| every ~20 quiet min | REFLECTION LOOP (captioner/reflection.py): long-form thought (600-token budget) on rotating subjects — room / visitor / drawings / time / itself. **SUBJECTS ARE ORGANS (July 31, piece 1 of `docs/reflection-organs-handover.md`): each subject now gets its OWN slice of memory, not one shared bundle behind five different questions.** Shared spine = **THE RAW RECORD — up to 80 verbatim captions from the last 75 min (hour_log; July 12 "dreaming" upgrade: every prior input was a summary of a summary, so the loop could never notice what actually happened in its own head — e.g. an hour of questions addressed to a visitor that nothing ever answered)** + that subject's OWN prior-reflection thread. Then per organ (`_diet_*`): room = compressions + concept-ledger place inventory; visitor = episodic arrival/departure spans + people pattern + events; drawings = the framed drawing scrap + full executed sequence (8) + artistic arc (an LLM call, 2-sentence trim, purple-prone — watch it) + current desire + desire_history; time = journal chronology (6) + session duration + durable-ledger multi-day spans + events; yourself = identity slots (persona/belief/desire) + self_notes. The subject question becomes a reading lens and the ask is a digest-then-advance: what moved, what it circled, what it assumed the record doesn't show, what it asked and whether any answer came — fact questions, not fences (how it can LEARN it needs no permission); uses the main model; skipped while drawing; POSTPONED while the store is empty (reflecting on nothing invents a past that would echo forever) | ChromaDB `reflections` collection + REFLECTION log entry; surfaces into quiet captions via echo line |
| per reflection | REFLECTION SYSTEM PROMPT is **subject-gated** (July 31): the standing persona (`core_facts['self']`) rides `yourself` ONLY, the durable ledger rides `yourself` + `time passing`. Both used to ride all five, asserting one identity at the top of every lens while `distill_reflection` wrote that same persona back from every reflection — the frame re-homogenised whatever the data did. Identity material now enters `yourself` as DATA (the `identity` block) instead of as frame | `get_reflection_system_prompt(subject)`, captioner/prompts.py |

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

## Open-vocabulary object detection (Phase 1, Aug 5 — OPEN_VOCAB_ENABLED)

Zero-shot object naming so the machine can find the things the machine talks about.
`perception/open_vocab_detector.py` (`OpenVocabDetectorThread`) wraps YOLO-World-S
(`models/yolov8s-worldv2.pt`) — **CPU-only** (60ms/frame; the 3090 belongs to the
27B and Flux). Fed from the camera loop next to the YOLO/ArUco `set_frame` calls;
runs every `OPEN_VOCAB_INTERVAL` (4s) THROUGH normal movement like the YOLO
tracker (second live tuning, Aug 5 — a settle gate starved detection). Only a
genuine mid-saccade frame is skipped (`OPEN_VOCAB_SETTLE_VELOCITY` 20 deg/s,
near the physics velocity caps); below that, velocity just sets
`settled: True/False` on results — provenance, not suppression (efference-copy
principle). Results carry capture-time pan/tilt; each term is flagged `novel`
if absent from the previous pass. Overlay: capped at `OPEN_VOCAB_MAX_BOXES`
(4) — best box per term, new arrivals ranked before confident regulars; boxes
hide once the gaze drifts `OPEN_VOCAB_DRAW_MOVE_TOL` (6°) from capture — the
tolerance must exceed tremor jitter (~2°/axis position noise, invisible to
velocity) or boxes flash, the first live run's symptom. Magenta boxes + labels in the preview window, drawn only while the gaze
hasn't moved since capture.

- Vocabulary: `OPEN_VOCAB_VOCABULARY` in config — the machine's promote-ready terms from the
  Phase 0 scan (`debug/phase0_report/`). Hot-swappable via `set_vocabulary()`
  (recompiles CLIP embeddings, ~4s) — the Phase 2 promotion hook.
- Per-term confidence floors (`OPEN_VOCAB_TERM_FLOORS`): mannequin heads are real
  but faint; book/cardboard box need extra proof.
- Person suppression: detections mostly inside the person-tracker bbox are dropped
  ("sculpted human head" matched a live person in Phase 0).
- Location dedup: strongest box wins a patch; same-term nesting collapses, small
  object in front of a big one survives.
- NOT yet wired: no LLM/prompt consumer, no spatial registry, no gaze targets —
  detector output is structured data only, by design (session brief Phase 5:
  keep the channels separate). Kill switch: `OPEN_VOCAB_ENABLED = False` restores
  prior behaviour exactly.
- Standalone test: `debug/test_open_vocab_detector.py [frame.jpg]`.

**Phase 3 — spatial registry + registry glances (LIVE, Aug 5:
SPATIAL_REGISTRY_ENABLED, GAZE_REGISTRY_GLANCES_ENABLED).** The world map:
`perception/spatial_registry.py` (`spatial_registry` singleton) folds each
settled detector pass into per-term anchors in servo angles (box center →
absolute angle via the machine.py person_angle convention: HFOV 60°, VFOV 34°;
image-y down = tilt down). EMA-smoothed (`SPATIAL_REGISTRY_EMA`), 7-day decay,
persists in `event_log/spatial_registry.json`. Unsettled detections never move
anchors. Idle gaze consumes it: `_update_registry_glance` in `vision/gaze.py`
sits ABOVE the LLM-zone override in the idle target chain — every
`GAZE_GLANCE_INTERVAL` (45s jittered) the gaze commits to a target for
`GAZE_GLANCE_DWELL`: staleness-weighted revisit of a known object (recent
discoveries boosted 3x) or, `GAZE_GLANCE_EXPLORE_WEIGHT` (25%) of the time, an
under-visited pan bucket ("look around"). Arrival triggers the existing
stillness logic → stillness settles the gaze → detector fires on settle → the
anchor sharpens: look→arrive→see is one loop with no new choreography. Person
tracking outranks everything, untouched. Console: `[👁️] Glance (revisit):
pink shelf → ...`. Structured consumers: `get_current_glance()`,
`get_self_motion()` (efference signal, surfaced as `info["self_motion"]` in
`_assess_scene` — NO prompt consumer yet, deliberately; "I was turning"
framing is a separate prompt-tree step). Test:
`debug/test_spatial_registry.py` (angle math, EMA, policy split).

**Aware-churn fix (Aug 10 evening).** The gaze flickered idle↔aware every few
seconds on marginal YOLO hits ("person detected but no tracking target" →
5s timeout → look away → re-trigger), and each 90s-spaced flicker minted an
episodic "someone arrived" (vision/gaze.py ~1160 — a THIRD arrival system,
feeding identity events). Fixes: (1) `YOLO_PERSON_MIN_AREA_FRAC` in
object_detection.py — tiny "persons" are phantoms; size filter beats raising
YOLO_CONFIDENCE_THRESHOLD (which worsens the known seated-still misses);
(2) idle→aware entry debounced (`AWARE_ENTRY_CONFIRM_S`, 2s ≈ two idle-cadence
passes) AND requires an actual bbox to aim at — no more aware-with-no-target.
Face DNN sensitivity knob remains `CONFIDENCE_THRESHOLD` (0.72, raised June 28
for mannequin faces) — untouched, not implicated in this churn.

**Presence identity — the "106th man" fix (Aug 10).** The machine read the one
regular man as endless strangers. Causes: the presence line said "Someone's
come in." on every ON-edge (an anonymous stranger each return — the model
confabulated a count), and core_facts.people was empty. MEASURED dead end:
CLIP cross-outfit similarity (0.70) is BELOW cross-person (0.76) — appearance
embeddings cannot say "same man, new jacket"; do not retry. (Face embeddings
are the real cross-day tool — future, new dependency.) The fix is the
single-occupant PRIOR: the presence line is now the definite singular —
"He's come in." (person_count 1) / "People have come in." (count > 1) /
"He's back." (session re-ID resumed, only when PRESENCE_REID_ENABLED — OFF
since ~Aug 6 by the artist's hand; all re-ID calls no-op while off).
Relational "Someone is here" → "He's here." Gaze-aware belief decay
(re-applied — the Aug 5 version was lost to another session's
`git checkout -- captioner/captioner.py`, see transcripts): absence only
accumulates while the gaze points near last-seen
(PRESENCE_ABSENCE_LOOK_TOLERANCE); not-looking is not evidence of absence.
The singular register is a CONCLUSION, not a hardcoded fact (artist's call):
the arrival ledger (`presence_identity.record_arrival` →
`event_log/presence_arrivals.json`) tracks {ts, person_count} per genuine
arrival; `singular_regime()` (7-day window, ≥80% single-person) picks "He's
come in." vs "Someone's come in." — an exhibition's crowds flip the register
back within hours, no config change. Re-ID re-measured Aug 10: CLIP is dead
at every scope (same-outfit-same-session 0.70-0.74 vs cross-person
0.67-0.76); PRESENCE_REID_ENABLED stays False until an OSNet/face-embedding
backend replaces embed_crop.

**Body schema — visual self-recognition (LIVE, Aug 10: BODY_SCHEMA_ENABLED).**
The arms were a hole in the world model: YOLO called the drawing hand a person
(→ phantom presence while drawing), the object detector labelled an arm
"rooster figurine". `perception/body_schema.py` (`body_schema` singleton,
persistent `event_log/body_schema.json`): TWO-FACTOR self test, place +
appearance — measured (Aug 10) that appearance alone fails here (arm mid-draw
scores 0.75–0.83 vs its own last pose; the hanging wooden figure scores 0.87 —
CLIP tracks pose, not identity, and the studio is full of sculpted limbs).
Inside the harvested reach envelope (pan/tilt-conditioned image regions) the
bar is `BODY_SELF_THRESHOLD` 0.72; outside it `BODY_SELF_STRICT` 0.92.
References harvest by proprioception: CNC phase "executing" + gaze drawing
lock + faceless person-box at the workspace = own arm (`maybe_harvest`, called
each caption cycle; face in view blocks harvest so a visitor can't poison the
gallery). Consumers: captioner presence — a faceless "person" matching the
schema no longer counts as seen (face evidence always wins; veto never fires
against an actual face), sets `info["own_arm_visible"]`; open-vocab detector —
self-matching detections dropped pre-storage (never reach overlay/registry/
audit). Person tracker itself untouched. Seeded from real frames; verified
5/5 verdicts (arm re-seen = self; artist standing in the arm region = not;
wooden figure = not, saved by the place gate). Test/seeding:
`debug/test_body_schema.py [--seed]`.
Consumption-edge gating (same day, second pass — the blue person box on the
drawing hand persisted): machine.py now reads `cached_person_verdict()` (never
computes — an embed in the main loop would hitch the servo physics; the
detector thread keeps the cache warm each pass) and when the verdict is self,
`update_yolo_detection(False)` — so gaze awareness/tracking, bbox smoothing
and the person overlay all see no-person. The overlay draws a gray "SELF" box
instead. The old `own_body_likely` tilt heuristic (machine.py, June) stays for
person_angle only.

**Label audit — the self-correction loop (LIVE, Aug 10: LABEL_AUDIT_ENABLED).**
A wrong label looks healthy from the inside (hits accumulate either way; the
artist: labels "don't correct themselves" — cable bundle stayed "wire basket",
styrofoam head stayed "wooden mannequin torso"). `perception/label_audit.py`
(`LabelAuditThread`, started in machine.py beside the detector): every
`LABEL_AUDIT_INTERVAL` (10 min, skipped while a drawing generates) the
well-detected registry term least recently audited has its latest settled crop
(detector keeps per-term crops) shown to the 27B — "what is this? plain
appearance names" (`prompt_type: label_audit`). Same name → confirmed.
Different → CLIP head-to-head on the crop via a PRIVATE YOLO-World judge (live
model's compiled vocab untouched); challenger beating the incumbent by
`LABEL_AUDIT_MARGIN` is promoted (origin "audit", bypasses recurrence
threshold) — the rooster pattern, automated. Incumbent never removed: it keeps
its true referent, loses only the stolen patch in future contests. Verdicts
(confirmed/relabelled/held/no_candidates) land on the registry entry and in the
event log as `vocab_promotion {event: audit}`. Verified on the real cases:
styrofoam-head crop → incumbent scores 0.00 on its own crop (its wins were
context), "white styrofoam head" 0.25 → relabel fires; cable-tangle-on-face-
cast crop → all names ~0 → held (honest: that patch is YOLOE visual-prompt
territory). Test: `debug/test_label_audit.py`.

**Phase 2 — vocabulary promotion (LIVE, Aug 5: OPEN_VOCAB_PROMOTION_ENABLED).**
The recursive part: what the machine says shapes what it can see.
`perception/vocab_promotion.py` (`vocab_promoter` singleton) is fed each
accepted caption from `captioner._process_frame` (right after context
compression). spaCy noun chunks (reuses `utils/pattern_recognition.nlp`) →
rolling window count (`OPEN_VOCAB_PROMOTE_WINDOW` 300 captions) → threshold
(`OPEN_VOCAB_PROMOTE_THRESHOLD` 10) → `set_vocabulary()` hot-swap. Filters keep
mythology upstairs: no proper nouns (coinages), no person heads
(`OPEN_VOCAB_PERSON_NOUNS`), no undetectable heads (shadow/corner/tonight —
`OPEN_VOCAB_STOP_HEAD_NOUNS`), no self-narration (`OPEN_VOCAB_SELF_NOUNS`:
machine/gear/lens — the mirror doesn't chart itself), no abstract suffixes, no
near-duplicates of existing terms (subsumption: "monitor" vs "computer
monitor"). Eviction under cap (`OPEN_VOCAB_MAX_TERMS` 40): ghosts first, then
fewest detector hits; evicted terms get a re-promotion cooldown (churn fix).
Ghosts = promoted, never detected for `OPEN_VOCAB_GHOST_AFTER` — kept and
logged; looking for what isn't there is data. Every promote/evict/ghost event →
event log (`LogType.VOCAB_PROMOTION`) + readable history in
`event_log/vocab_promotion.json` (survives restarts; re-compiled into the
detector on attach). Replay test: `debug/test_vocab_promotion.py [n]` — on 800
real captions promotes pen/workbench/screen/paper/table/curtain/blank
page/glass and nothing else.

## Presence belief: gaze-aware decay + re-ID scaffold (Aug 5)

The "genuine new presence" misfire (the artist greeted as new dozens of times a
day) had two causes; one is fixed, one is scaffolded:

**Fixed — gaze-aware decay** (`captioner._assess_scene`): the presence belief's
240s decay clock now only ticks while the gaze is pointed within
`PRESENCE_ABSENCE_LOOK_TOLERANCE` (30°) of the last-seen spot
(`person_detection_state.is_looking_at_last_known_location`). Previously the
machine's own wandering decayed the belief — absence-of-evidence read as
evidence-of-absence — and every look-back fired a false OFF->ON arrival. Same
bug class as the blur/efference issue. Departure now requires having actually
looked and found nobody, accumulated (`_absence_watch_s`), not wall-clock.

**Scaffolded, DISABLED — session re-ID** (`perception/presence_identity.py`,
`PRESENCE_REID_ENABLED = False`): rolling gallery of person-crop embeddings;
on an OFF->ON edge, a match against recent sightings suppresses the arrival
(`presence_resumed` in scene info). Disabled because
`debug/test_presence_reid.py` proved CLIP image embeddings rank scene
similarity over identity (different people 0.87, same person 0.49 across
scale change). Needs a real person-reid embedding (OSNet-class, ~2MB, CPU)
dropped into `PresenceIdentity.embed_crop`, then re-run the test and enable.
`DetectionMemory.get_person_crop()` added for the crop path.

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
thematic motifs still pass) ·
outward_address (NEW July 28, register measured not enumerated: ≥2
second-person tokens = the text has acquired a reader — the machine's own
voice says "you" at most once, to a rooster; also planning openers "I'll
begin by…" and parenthetical meta "(Note: …)". The marker list alone leaked
18/58 captions the day the reflexive frame landed. Same density screen on
`_is_plantable_prior` so an assistant tail can't seed the next session's
awakening). Strips
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

## Drawing prompt generation (July 10: DRAWING_ANALYSIS_MODE="stream"; Aug 10: stocktake beat + register freedom)

Up to THREE calls (prompts.stream_drawing_analysis), replacing the 5-step committee:

0. **Stocktake** (Aug 10, `DRAWING_REVIEW_ENABLED`, logged as
   `prompt_type: drawing_review`) — before choosing, the machine reads the
   SAME materials the intent will get (whole executed ledger — now all 24
   entries, was 8 — plus reflections retrieved by TWO keys: the stream tail
   for the moment, the body-of-work text for the work itself) and writes a
   2-4 sentence first-person note on where the work has been going and what
   it's missing. The note joins the intent materials AND is stored
   (memory type `drawing_direction`); the PREVIOUS note is read back with
   its age — successive drawings answer a remembered direction instead of
   starting from amnesia. Not the 5-step returning: it reads only real
   material (ledger, own reflections) and speaks in first person.
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
2. **Render** — translation to a ComfyUI prompt. **REWRITTEN Aug 10**: the
   old negation wall ("no shading, no fills… detail is lost") was echoed
   verbatim into the Flux prompt, and the plotter-meta language correlated
   with the LoRA's photographed-defocus mode while every drawing collapsed
   to one sparse object. Constraints are now positive pen-and-ink craft
   language (contour lines of varying weight, hatching/cross-hatching for
   tone, tone = line density never washes), the render matches the
   intention's REGISTER (small thing → focused study with white space;
   scene → full composition with foreground/background depth), stays pinned
   to the intention's own concrete nouns (may render the implied setting,
   invents no new objects), and MUST NOT mention plotters/tracing/vectors/
   machines in the emitted prompt. Temp 0.5, logged as
   `prompt_type: drawing_render`. Fallback: prefix + intent. The intent
   system prompt likewise frees register (close study or built-up scene)
   and frames the materials as the sketchbook the decision comes from.
   Inspect offline: `debug/test_drawing_intent_prompt.py`
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

- THE AWAKENING (July 28, final form): machine.py enables the bus with
  await_homing=True — the body holds STILL through the whole init — and
  at the main-loop threshold a background `_awakening` thread HOMES THE
  GANTRY DIRECTLY (find_grbl_port + ensure_homed in-process, port closed
  after — no subprocess in the path) and starts the uArm's opening play,
  WHILE the main loop brings up the camera window, gaze, lung and bulb.
  KINETIC_HOMING_WAIT_CLEAR=False (artist's call): the homing
  choreography and the $H sweep run SIMULTANEOUSLY — the dance is
  recorded to stay clear of the gantry; set True to restore
  clear-first gating. The first temperament blooms when homing completes.
- IDLE WANDERER RETIRED (July 28): grbl/idle_movement_manager.start()
  refuses (RETIRED flag) — the Lissajous wanderer no longer runs. Its
  pause/resume module functions now forward to the gantry arbitration
  hooks below; stop_idle_movements kept in cleanup to kill strays.
- THE RIGHT ARM IN THE TEMPERAMENT (July 28, KINETIC_GANTRY): the bus
  owns a headless `motor_panel/gantry.py` GantryLink between drawings —
  the datasets' recorded x/y play through the same markov chains as the
  servos (reach-clamped, G1 at chain tempo, ≤3 segments pipelined, pen
  UP unless KINETIC_GANTRY_PEN). The awakening `gantry_acquire()`s:
  port open (resets GRBL) → ensure_homed (tuck choreography fires,
  simultaneous with $H) → the link KEEPS the port and generation flows.
  Drawing arbitration: the legacy pause/resume call sites (grbl_utils
  completion ritual, image_monitor) fire `utils.hooks.on_gantry_pause/
  on_gantry_resume` → bus releases (pen up, port closed) before a
  drawing and re-acquires (re-home + choreography) after. The
  is_drawing() hard gate still drops every plan/pen send as backstop.
  Proof: debug/test_gantry_runtime.py (pty GRBL discipline + bus flow/
  gate/release/re-acquire).
  Failsafe KINETIC_AWAKENING_MAX_WAIT_S if homing never arrives. After
  ANY homing, the SAME dataset resumes (continuity, not a re-pick).
- GANTRY LISTENING + RELEASE RACE (July 28 evening): the first live run
  parked the right arm despite a healthy pipeline (offline repro with
  the real datasets flows 60+ targets/12s — debug/repro pattern in
  test_gantry_runtime). Two fixes: (1) Player.stop() could race
  start() during the homing choreography assembly — a crash inside
  home_release that ensure_homed silently swallowed, which could also
  eat the HOMING_SENTINEL write (sentinel now writes BEFORE the hook;
  Player start/stop take a lifecycle lock and a stopped latch).
  (2) GantryLink was DEAF: it never read GRBL's replies, so an
  alarm-locked or error-rejecting GRBL looked identical to success.
  The link now probes '?' at attach (logs the status line, shouts if
  Alarm), logs its first three streamed G1s, and reports any non-ok
  reply (error:N / ALARM) deduplicated. Diagnosis on hardware:
  debug/test_gantry_live.py (machine.py stopped) homes and sends
  dataset coordinates echoing every GRBL reply.
- EMPIRICAL COLLISION SAFETY (Aug 1, KINETIC_SAFE_*): the camera/IK
  calibration idea was dropped — the machine never needs to know where its
  hands ARE, only which motor COMBINATIONS are safe, and the recordings
  are thousands of those. Both arms train as ONE chain group (x, y, elbow,
  shoulder, wrist, fingers), so a normal walk is demonstrated by
  construction; the danger is the glue. Measured: 8400 pooled
  combinations, neighbours ~1.2-2.0 units apart, but straight-line
  crossfade midpoints land 6.9 (worst 12.1) from anything performed and a
  full gaze lean 11.0 — continuously. motor_panel/safe_envelope.py is a
  cKDTree over the pooled recordings; kinetic_bus._guard runs every send
  (~0.02ms) against the MERGED both-arm pose, since sends arrive split.
  Three details paid for by measurement: the pull target is the average of
  8 neighbours (a single nearest one flips and snaps, 3.6 units), with the
  true nearest as fallback when that average sits off the curved cloud;
  only the channels IN this send may move (a correction written elsewhere
  is discarded, leaving the combination stray); and the correction is
  eased AND slew-capped (KINETIC_SAFE_SLEW), because a raw projection jump
  is exactly the snapping recorded movement exists to avoid. A/B on the
  live bus (crossfades + full lean + startle): median stray 7.5 -> 3.9,
  worst 14.2 -> 10.1; isolated crossfade/lean cases land exactly on the
  3.0 threshold. Inactive (everything passes) without scipy or
  recordings — the guard must never be why the body stops. Proof:
  debug/test_safe_envelope.py.
- THE CHAINS WEREN'T CHAINS (July 31, measured): keying identity at 1 degree
  on all seven servo channels at once meant two moments had to agree on
  every joint to merge — a 600-sample take trained ~568 states with
  BRANCHING FACTOR 1.00. The markov walk was the recording as a linked
  list, and the gaze CHOICE bias (which reweights candidates) had nothing
  to choose between; only the lean offsets ever did anything. Fixed in
  three parts: (1) IDENTITY IS COARSE, POSE IS FINE —
  KINETIC_STATE_BIN_SCALE (8x) merges states while train() stores each
  state's real averaged pose in `state_poses`, so the walk forks without
  quantizing the movement; (2) a transition's dt is now the DWELL time
  spent in the state, not one sample tick (without this, coarse states
  replayed ~14x too fast); (3) THE BODY'S SAMPLER in _pick — temperature,
  repetition penalty, min_p, same vocabulary as the model's, read live so
  mood moves them mid-walk (KINETIC_MOVE_*; arousal from
  mood_vector[1] via the bus's get_arousal). Second-order contexts are
  ~99% single-exit (momentum lock), so _candidates lets temperature above
  1.0 drop to first order — heat is what breaks the groove. Measured
  after: 10-14% of states offer a real choice, two walks share 57% of
  their steps instead of 100%, hot runs agree 34% vs cold 56%, tempo
  preserved. Proof: debug/test_move_sampler.py.
- PAPER-CHECK INTERRUPT + ORGANIC STARTLE (July 30): new bus state
  "paper" (session_paper_*.json) — when safety/paper_detection starts
  its ArUco search it fires hooks.on_paper_check_start → bus.paper_clear
  plays the recorded get-clear move (BOTH arms, gantry included via raw
  plan sends), waits the returned clearing time, searches, then
  on_paper_check_done → paper_release blends the SAME dataset back.
  Config: KINETIC_PAPER_TUCK_S / KINETIC_PAPER_MAX_HOLD_S. Panel: the
  runtime lab gained a "📄 paper check" preview button; the state list
  picks "paper" up automatically. STARTLE reworked: the startle take now
  plays RELATIVE (live pose + (sample − first) × NUDGE — zero-offset
  entry, no more frozen median pose), whole body including gantry;
  startle is suppressed while homing/paper own the body. Proof:
  debug/test_paper_startle.py.
- THE ACTUAL PARKED-ARM BUG (July 28, caught live in the terminal):
  KeyError 'x' — generators seed from the body's LIVE state, and the
  servo device reports no x/y (gantry position is commanded, not
  sensed), so the group chain owning the gantry DIED on its first step
  at every bloom (one dead thread per temperament switch; the surviving
  solo chain kept some servos moving, masking it). Fixed at both ends:
  engine._loop fills missing channels from the nearest demonstrated
  state instead of dying; kinetic_bus._live_state merges
  gantry.position ((0,0) after homing) into the seed. Regression:
  test_gantry_runtime's bus test now uses the production shape (a
  servo-only get_state).
- ONE MACHINE PER BODY (July 28): the "phantom left arm" (moving with
  machine.py 'off', glitching during runtime) traced to a forgotten
  login autostart — ~/.config/autostart/impostor.desktop →
  start_impostor.sh → hidden tmux session `impostor-system` running
  machine.py in a 5s restart loop since Sep 2025. Two machine.py
  processes interleaved bytes on the same serial ports. Guards now:
  (1) machine.py claims a flock via utils/single_instance.py — a second
  instance exits with a message; the lock dies with the process, so
  restart loops keep working; (2) every serial open (devices.py
  lefthand, servo_control lunggaze, grbl_utils CNC ×2) passes
  exclusive=True — a second opener fails loudly instead of garbling
  commands, and find_grbl_port's "?" probe now skips ports other
  subsystems hold. The autostart file itself is the operator's to
  remove (`rm ~/.config/autostart/impostor.desktop`) or keep for
  exhibition boot — the flock makes it safe either way. Firmware ruled
  out: the flashed clean variant has no autonomous code (the legacy
  .ino wanderers need a LEFT_ARM_ENABLE nothing live sends — panel
  extras removed July 27; debug/identify_hand_firmware.py verifies a
  flash in doubt). Proof: debug/test_exclusive_ports.py.
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
