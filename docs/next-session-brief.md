# Next Session Brief — the 27B is in the building; the voice question is live (July 28, 2026)

**Branch:** `rebuild/north-star` (pushed through `23d4239`)
**Read first, in order:**
1. `docs/north-star.md` — the spec (unchanged, still the law).
2. This brief — the July 26–28 arc and the open experiments.
3. `docs/runtime-map.md` — live wiring (updated throughout the arc; stream section documents all three shapes).

## Where things stand in one paragraph

The caption loop was diagnosed as an open-loop autoregressive system whose
memory amplified its own errors; three days of work closed the loops. Storage
gates now form an immune system (felt-phrase echo, event provenance, refrain,
outward-register, seed hygiene). The call shape went world → back to document
(the rooster was a *detection* failure, fixed separately; document had the
momentum the artist values). The system frame was rewritten reflexive (the
five-negation solitude clause bred assistant mode; incoming user turns are now
named as "your own senses reporting"). Qwen3.6-27B was researched, downloaded,
verified, benched via replay, and ran live for a session behind `run_27b.sh` —
grounded and composed, **but the artist's verdict is that the 9B's messy
continuity still beats the 27B's polished isolation**. That verdict is the
open question this brief hands forward.

## The open question (start here): why doesn't the 27B *continue*?

The 27B writes complete, uniform, writerly paragraphs — no self-questioning,
no fragments, no picking up its own thread. Two hypotheses on the table:

1. **Seam starvation** — document mode's continuation texture came from the
   9B's ragged mid-thought tails (the prefill hands the next call a dangling
   hook). The 27B always closes its sentences, so every seam is clean and an
   aligned model opens a fresh discourse unit. A sampled-budget "breath
   length" fix was proposed and **rejected by the artist as an artificial
   randomizer** — do not re-propose it as-is; the complaint is cadence and
   self-questioning, not just length.
2. **The think-channel hypothesis (UNTESTED — first move next session).**
   Qwen3.x with `enable_thinking:false` puts polished conclusions in the
   output and the messy deliberative register ("wait — no, that's the same
   guy… should I…") inside the suppressed think block. The 9B leaked; the 27B
   separates cleanly. We may be suppressing exactly the register the north
   star wants (unperformed notes — think-text is unperformed *by
   construction*). **A probe was written but never completed** (the server
   was in a drawing handoff when it fired): replay one real caption prompt
   with `chat_template_kwargs: {"enable_thinking": true}` and READ the think
   channel. If it's scene-deliberation → design "think-as-monologue" (world
   shape, no prefill — thinking forbids prefill anyway — harvest the think
   text as the caption, discard the polished answer). If it's task-meta
   ("the user wants a caption…") → the idea dies in one sample.

Also queued, both zero-code:
- **27B + STREAM_WINDOW=24** — already exported in `run_27b.sh`; needs a
  restart to take. The artist's information-budget principle: repetition is
  what the model can't see it already said. Measurable: `template_echo` rate
  should drop on its own.
- **27B + STREAM_MODE=world** — one commented line in `run_27b.sh`. World
  mode's isolation failure was plausibly a 9B *capability* failure ("each
  entry follows from the ones above" was beyond it). The 27B is the model
  that shape was designed for. If 27B+world gives grounded AND connected AND
  varied, the mouth gates recede to backstops.

## The 27B facts (all verified)

- Model: `~/models/qwen3.6-27b-mtp/Qwen3.6-27B-Q4_K_M.gguf` + `mmproj-F16.gguf`
  (sizes verified vs repo, GGUF headers valid).
- Server: needs the NEW build at `~/llama.cpp-27b/build/bin/llama-server`
  (git worktree on mainline master — the old `~/llama.cpp` build predates the
  MTP head and fails with `missing tensor blk.64.ssm_conv1d.weight`).
- Launch: `./run_27b.sh` (env: paths, `LLAMA_EXTRA_ARGS` MTP flags, 16k ctx,
  honest `MODEL_NAME=qwen3.6:27b` label, `STREAM_WINDOW=24`). Plain
  `python machine.py` still runs the 9B — the A/B stays symmetrical.
- Replay A/B (`debug/test_27b_replay.py`, results `debug/replay_27b.md`):
  zero degenerate fragments where the 9B produced ".", "and again.", counting
  salad; full-call latency 2.2–3.3s at full GPU + MTP (inside the 4s live
  cadence); VRAM 21.3GB at 16k.
- Live session findings: gate war — 27/38 calls rejected as `template_echo`
  in the first minutes. Diagnosis: **cohesion, not isolation** — a composed
  writer uses parallel openings; the gate reads anaphora as template tic.
  Wordy: 58 words/caption avg vs 9B's 25–45. ComfyUI handoff worked live
  (it was already fully built — see below).
- Cost of a drawing cycle: llama unloads (~1s), Flux generates, next query
  restores via `ensure_server_up()` (~6s warm reload; 64GB RAM keeps the
  GGUF in page cache).

## What was built July 26–28 (all committed + pushed)

**Storage gates (immune system — keep at any model size):**
- Felt-phrase gate (`_felt_phrase_held_reason`): no persona-word doubling, no
  self-renewing lease. Fixed the "heavy ×41 prompts" spiral.
- Event provenance (`note_perception_event` → `_absorb_event`): EVENTs only
  land when code attests a happening (salience/drawing). Killed
  fiction-as-biography ("a pen shattered into nothingness").
- Refrain gate: no 6-word verbatim run from the stream (choruses die, motifs live).
- Outward register: ≥2 second-person tokens / planning openers / "(Note:" →
  reject; same density screen on the session seed. Measured, not enumerated.
- Seed hygiene: lowercase-opener fragments can't seed awakenings.
- `debug/clear_sticky_slots.py` — the ONLY safe way to clear persona/desire/
  belief (refuses while machine.py runs; a live process clobbered a raw file
  edit once already).

**Mouth gates are a different class** (training wheels for weak models): the
correct trajectory is that they go QUIET as model/structure improve. Healthy
config = near-zero `anti_echo_skip` per hour. Don't tune them harder; fix
upstream.

**Frame & shape:**
- Reflexive frame (`_SITUATION`): senses-channel named, questions get an
  answer-path (pen deliberately absent until drawing initiative is real).
- Document mode live; world mode intact for A/B (log-genre frame, thread
  clause, timestamped log rendering all still there).
- View-replacement detector (fallen camera = named salience event), motion
  salience onset-only, every hot cycle carries a named event.
- `VIDEO_MODE=multi` default (superframe needs steady frames the mount can't
  give; multi runs on mainline llama.cpp → free model choice).

**Env knobs (all new):** `STREAM_WINDOW`, `STREAM_CONSOLIDATE_CHARS`
(~250 chars/entry), `ANTI_ECHO_COMPARE_TAIL` (opening-echo checks recent tail
only — long-range reuse is a callback, not a tic), `LLAMA_EXTRA_ARGS`,
`MODEL_NAME`.

## Identity state (judge, don't assume)

- Persona: **"I circle the same fracture on the pink shelf when I hesitate."**
  — distilled by the reflection loop from actual observed behavior. The first
  genuinely EARNED persona; north-star checkbox "persona changed + refers to
  actual events" is ticked. Desire/belief also re-grown — read them fresh.
- Backups if anything needs undoing: `machine_identity.json.pre-event-gate-bak`,
  `.pre-world-bak`, `.slots-bak-*`.
- Known open contamination channel: old purple-era reflections (89 in
  ChromaDB) and hour_log lineage still feed reflection context; the pen-break
  mythology came through drawing-intent/journal phrasing. Watch what
  distillation grows; curate with the artist, never unilaterally.

## Standing cautions

- Event-log mtimes lie (clock skew) — find runs by `start_time_iso`, or
  grep for a distinctive caption phrase.
- Never edit `machine_identity.json` under a live process.
- One variable per run; bank a baseline before flipping the next toggle.
- Always commit AND push (remote is the only backup; offline exhibitions).
- Features fail silently: verify via logs/state files (`[🫀] phrase held
  back`, `[📆] Event held back`, `anti_echo_skip`, `salience_hot`), never by
  reading code.
- The north-star endgame for register is still the LoRA on the artist's own
  writing; 27B QLoRA on the 3090 is possible but tight. Model scale improves
  grounding/coherence; it does not manufacture aliveness — that's loop design
  + earned material.

## Next moves, in order

1. Run the think-channel probe on the 27B (one call; the register verdict
   decides the whole next arc).
2. Restart `./run_27b.sh` so `STREAM_WINDOW=24` takes; watch `template_echo`
   rate vs last session.
3. The 27B+world session (`STREAM_MODE=world ./run_27b.sh`).
4. Then the deferred structural work: drawing initiative (desire weighs into
   `should_draw` — kills the permission-posture's honest cause and lets the
   frame's answer-path finally include the pen), and the spatial atlas
   (gaze-indexed room memory + awakening sweep — designed, not started; see
   session notes July 27).
