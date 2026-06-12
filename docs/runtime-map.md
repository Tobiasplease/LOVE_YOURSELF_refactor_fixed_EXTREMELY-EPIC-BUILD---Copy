# Runtime Map — what is actually live

Last verified: June 12, 2026 (branch: experimental/vision-upgrades).
This is the maintenance view: every line the model sees, where it comes from,
and which subsystems are healthy, weak, or dead. Update it when wiring changes.
The audit habit that produced it: features fail SILENTLY here — always check
the event log / state files for evidence a subsystem is producing output,
don't trust that code existing means code running.

## The caption loop (~every 7s)

```
camera frame (~30fps)
  ├─ YOLO + ByteTrack (person bbox, track id, count)     [perception/object_detection.py]
  ├─ Face DNN (face bbox)                                 [machine.py]
  ├─ frame_buffer.push(frame, detection-snapshot)         [machine.py → captioner/frame_buffer.py]
  │    snapshot: face?, person?, count, track_id, pan/tilt, person_angle, ego_motion
  └─ captioner._process_frame                             [captioner/captioner.py]
       ├─ build_simple_caption_prompt  → USER PROMPT      [captioner/prompts.py]
       ├─ get_monologue_system_prompt  → SYSTEM PROMPT    [captioner/prompts.py]
       ├─ video decision + motion framing                 [captioner/captioner.py]
       └─ query_model_video / query_model → caption
            └─ post: concept match → activation → compression.add_caption
                     → observation store (after_monologue)
```

## Every line of the SYSTEM prompt and its source

| Line | Source | Health |
|------|--------|--------|
| identity base ("drawing machine bolted...") | static, prompts.py | ok |
| register (bored/alert/neutral) | activation network boredom+novelty | ok |
| mood clause ("You're content just now.") | mood vector (numeric, loop-safe) | WEAK — engine is keyword-based, barely moves; events (person, novelty) now feed it, rescaled thresholds. Deserves event-driven redesign. |
| "Right now: {felt}." | compression felt-state, sanitized ≤6 words | ok (often empty by design) |
| persona ("I monitor for movement...") | core_facts.self, self-synthesis every 3rd introspection | ok |
| mode addition ("You're aware of someone...") | mode selection | ok |

## Every line of the USER prompt and its source

| Line | Source | Health |
|------|--------|--------|
| "Been watching 18 minutes. Looking left." | session clock + gaze | ok |
| "Someone here 5 minutes." | episodic log, visit-clustered | ok |
| "They've come and gone N times." | episodic pairs (debounced 90s) | ok |
| introspective ctx ("My last drawings were of...") | drawing_memory, completed only | ok |
| core facts line | core_facts place/people/drawings | ok |
| familiarity ("That pink shelf again...") | ChromaDB concept matches, every ~3rd caption | ok; concept near-dups sprawl a bit |
| drawing/paper state | state_manager | ok |
| felt-state delta | compression | ok |
| "Preoccupied with: ..." | desire, gated to 3 injections | ok |
| baseline first sentence | compression (observational/workspace) | ok |
| thread tail "...{last sentence}." | recent_captions; dropout 25-40%; dwell keeps+extends | ok |
| video motion line | person-angle scene motion (NOT pixel diff) | NEW — verify in logs: `[VIDEO] ... scene_motion=` |

## Background consolidation (compression thread)

| Cadence | What | Output |
|---------|------|--------|
| every 8 captions | compression | baseline_context + felt-state |
| every 3rd compression | introspection | desire, belief, discovery |
| every 3rd introspection | self-synthesis | core_facts.self (persona) |
| during introspection | core facts update | core_facts place/people/drawings |
| per compression | concept extraction (LLM, solid objects) | ChromaDB concepts |
| every 30 min + shutdown | journal entry | machine_identity.json journal |

## Awakening (first caption of a session)

generate_internal_awakening() builds: offline duration + clock time/day +
last thought + desire/belief + journal ("From my diary, last time...") +
core facts + familiar concepts → one LLM call.
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
