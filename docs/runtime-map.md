# Runtime Map — what is actually live

Last verified: June 12, 2026 (branch: rebuild/north-star).
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

## Every line of the SYSTEM prompt and its source

Torn down June 12 (north-star principles 1+2): situation only, no style
rules, no registers, no mood clause. Voice comes from content.

> **CLEAN-ROOM ACTIVE (June 28): `config.BASE_VOICE_DETOX = True`.** While on,
> ALL rows below marked as stored/compressed (felt-state, persona) are stripped
> from the system prompt, and in the user prompt every interior/stored line
> (mode context, introspective, core facts, familiarity/echo, felt delta,
> desire, baseline) is stripped too; the video path drops its "You're seeing
> the last N seconds" wrapper. The model sees only: situation + genre frame +
> mode elicitation (system); situational line + present event + live
> drawing/paper state (user) + image. This isolates the naked base voice from
> months of self-poisoned stores. Set False to restore the full memory prompt.
> Gated in get_monologue_system_prompt + build_simple_caption_prompt (`detox`)
> and the video path (captioner.py).
>
> **Step 0 (June 28): the three detox blind spots are now closed too** — the
> AWAKENING runs time-only (no stored memory) `captioner.generate_internal_awakening`;
> the REFLECTION loop pauses entirely `reflection._should_reflect`; and DRAWING
> is skipped `drawing.should_draw` (its 5-step pipeline + purple step
> system-prompts can't produce a clean drawing yet). So clean-room is now valid
> for the whole running machine.py, not just the caption prompt. NOTE: the
> compression/introspection/self-synthesis/concept GENERATORS still run under
> detox — but their output is gated from the caption prompt, so it's isolated
> (and lets us observe whether plain captions yield plainer stores). See
> docs/memory-redesign-plan.md.

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
| THE STREAM (replaces thread tail, June 12) | last 6 admissible captions ride as the model's OWN assistant turns in the chat (captioner._stream → history param → llama_server messages); "..." user ticks mark time passing; text-only, past images never re-sent | NEW — verify register: one assistant-meta slip in the window would breed, so _stream_admissible gates admission (meta phrases, markdown). Toggle: STREAM_WINDOW=0 reverts to amnesiac captions. Breaks on >180s gaps (STREAM_BREAK_SECONDS) |
| dwell instruction ("Stay with that last thought...") | scene still + 30% chance; cancelled when live | ok — refers to the visible stream now, no quoted tail |
| video motion line | person-angle scene motion (NOT pixel diff) | verify in logs: `[VIDEO] ... scene_motion=` |

## Background consolidation (compression thread + reflection loop)

| Cadence | What | Output |
|---------|------|--------|
| every 8 captions | compression | baseline_context + felt-state |
| every 3rd compression | introspection | desire, belief, discovery |
| every 3rd introspection | self-synthesis | core_facts.self (persona) |
| during introspection | core facts update | core_facts place/people/drawings |
| per compression | concept extraction (LLM, solid objects) | ChromaDB concepts |
| every 30 min + shutdown | journal entry | machine_identity.json journal |
| every ~20 quiet min | REFLECTION LOOP (captioner/reflection.py): long-form thought (600-token budget) on rotating subjects — room / visitor / drawings / time / itself; context = today's compressions + previous reflections + journal + drawings + desire; uses the main model; skipped while drawing; POSTPONED while the store is empty (reflecting on nothing invents a past that would echo forever) | ChromaDB `reflections` collection + REFLECTION log entry; surfaces into quiet captions via echo line |

## Awakening (first caption of a session)

generate_internal_awakening() builds: offline duration + clock time/day +
last thought + desire/belief + journal ("From my diary, last time...") +
core facts + familiar concepts → one LLM call on the MAIN model (was Nemo,
whose cinematic register seeded the whole session's thread — June 12).
COLD START: when there is no past at all (no memory/identity/long-term
context), a separate FIRST_AWAKENING_PROMPT states the truth — first time
online, nothing in memory yet — instead of handing the model empty sections
to fill with priors (the dust-motes register).
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

- utils/ollama.py query path (fallback only; llama-server is primary)
- get_session_greeting, after_perception, build_monologue_prompt etc. —
  removed June 2026 dead-code purge
- subconscious.py — debug scripts only
- generate_awakening_message() — superseded by generate_internal_awakening()
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
