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

| Line | Source | Health |
|------|--------|--------|
| situation ("drawing machine bolted... a thought is a sentence or two") | static `_SITUATION` + monologue clause, prompts.py | ok |
| "You are between drawings at the moment." | state_manager drawing status (gated, never lies; absent while drawing) | NEW June 12 — without it the model narrated drawings that weren't happening. States the fact only; deliberately does NOT say what the machine is doing instead |
| "Right now: {felt}." | compression felt-state, sanitized ≤6 words | ok (often empty by design) |
| persona — quoted as the machine's own words: `What you've come to know about yourself: "…"` | core_facts.self, self-synthesis every 3rd introspection | ok |
| mode addition ("You're aware of someone...") | mode selection | ok |

Mood engine note: the numeric mood vector no longer reaches the system
prompt (mood clause deleted). The engine still runs and feeds servo/hand;
its proper successor is the reflection loop. Watch whether anything is
missed before rebuilding it.

## Every line of the USER prompt and its source

Lines marked [interior] are stripped whenever salience is hot — a live
moment gets the present only (north-star principle 6).

| Line | Source | Health |
|------|--------|--------|
| "Been watching 18 minutes. Looking left." | session clock + gaze | ok |
| "Someone here 5 minutes." | episodic log, visit-clustered | ok |
| "They've come and gone N times." | episodic pairs (debounced 90s) | ok |
| [interior] introspective ctx ("My last drawings were of...") | drawing_memory, completed only | ok |
| [interior] core facts line | core_facts place/drawings | ok |
| [interior] familiarity ("That pink shelf again...") | ChromaDB concept matches, every ~3rd caption | ok; concept near-dups sprawl a bit |
| [interior] reflection echo (`A thought you had earlier today: "…"`) | ChromaDB reflections, relevance-matched, every ~4th caption when no familiarity line | NEW — verify via REFLECTION log entries + echo in prompts |
| drawing/paper state | state_manager | ok |
| felt-state delta | compression | ok |
| [interior] "Preoccupied with: ..." | desire, gated to 3 injections | ok |
| [interior] baseline first sentence | compression (observational/workspace) | ok |
| thread tail "...{last sentence}." | recent_captions; dropout 25-40%; dwell keeps+extends; dwell cancelled when live | ok |
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
core facts + familiar concepts → one LLM call.
COLD START: when there is no past at all (no memory/identity/long-term
context), a separate FIRST_AWAKENING_PROMPT states the truth — first time
online, nothing in memory yet — instead of handing the model empty sections
to fill with priors (the dust-motes register).
HISTORY: until June 12 a 150-char filter rejected nearly every response and
shipped the hardcoded "Coming back online..." fallback. Now trims to
sentence within 300 chars instead of rejecting.

## Motion semantics (June 12 design)

Pixel diff CANNOT separate scene from camera motion — 1° of servo sway moves
every pixel ~10px. Roles now:
- pixel diff: only decides whether to send video frames at all
- ego_motion flag (servo delta >2°/frame): excludes frames from superframe
  pairing (a pan inside a pair encodes the whole room as moving)
- person_angle (camera_pan + bbox offset × FOV): TRUE scene motion — person
  angular movement >4° in window, or person-count change
- dwell trigger: scene_motion is False → 30% chance of 2-caption
  "stay with that thought" development

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
