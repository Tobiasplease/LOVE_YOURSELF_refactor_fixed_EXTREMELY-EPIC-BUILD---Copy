# Next Session Brief — The Rebuild (June 2026)

**Branch:** `experimental/vision-upgrades`
**Read first:** `docs/north-star.md` (the spec — all 6 principles), then
`docs/runtime-map.md` (what's live and where every prompt line comes from).

## The task

Implement the north-star rebuild, in this order:

1. **Git archaeology first** (~20 min): walk `prompts.py` history for the
   earlier states the artist remembers working better — what did they do
   that we've since lost? Archaeology before architecture.

2. **System prompt teardown**: reduce to situation-only (drawing machine,
   bolted down, sees through a camera, drawing is how it communicates) +
   the machine's own self-description as a quote ("What you've come to know
   about yourself: '…'"). DELETE all style rules — anti-poetry, registers,
   "let it land", mood clause phrasing rules. Expect wobble for hours;
   do not re-add fences (north-star Principle 1 and 2).

3. **Reflection loop** (the highest-leverage organ): every ~20 min when the
   scene is quiet, a long-form reflection (300+ token output; rich context:
   today's compressions, previous reflections, drawing history, journal) on
   rotating subjects — the room, the visitor, the work, time passing,
   ITSELF. Each reflection sees the thread of previous reflections. Stored
   in ChromaDB as first-class memories, retrieved by relevance into quiet-
   moment captions. Note: a reflection path exists (reason_about_caption,
   REASON_INTERVAL) — audit it, likely absorb/replace it.

4. **Salience gating** (north-star Principle 6): scene_motion / arrival /
   eye-contact strips the caption prompt to the present (no memory,
   familiarity, desire, dwell lines); quiet stretches get the full interior
   context. Caption interval could breathe too (tight when live, stretched
   when quiet). Signals already exist: `_last_scene_motion`, detection
   snapshot face/person counters, episodic arrivals.

Then run a FULL DAY before touching the desire arc (step 5, designed in
north-star Principle 4: persistent desires with state, closed through the
drawing pipeline).

## Current state (end of last session)

- All five consolidation channels verified alive: compression, introspection,
  core facts (patterns-only — never snapshots, people excluded from caption
  prompt), self-synthesis, journal (14+ entries). Observation storage was
  dead for the whole branch (empty-perception bug) — fixed, store will be
  thin at first.
- Scene motion = person-angle (camera-compensated), NOT pixel diff. Pixel
  diff only gates video sending. Ego-motion frames excluded from superframes.
- Awakening: rich path was being discarded by a 150-char filter — fixed;
  now includes offline duration + clock time/day + journal + recognition.
- Mood engine barely moves (keyword-based); events now feed it weakly.
  Real fix deferred — candidate for the reflection loop to absorb.
- Superframe video works (enable_thinking fix); llama-server auto-restarts.
- ByteTrack person IDs live; OSNet re-ID (Tier 2) not started.

## Standing cautions

- Features fail SILENTLY here. After wiring anything, verify via event log /
  state files that it produces output. (See memory: silent-failure audits.)
- Memory must never override perception — would-it-lie test on every
  injected line.
- The artist is not a programmer; keep `docs/runtime-map.md` updated as
  wiring changes — it's their window into the repo.
