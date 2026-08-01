# Handover — reflection subjects as differentiated organs (July 31, 2026)

**Branch:** `rebuild/north-star` (current head `6cbd1d1`, all pushed)
**Read first:** `docs/north-star.md` (spec), `docs/memory-runtime-review-2026-07-30.md`
(where this was flagged), then this doc.
**Nothing here is implemented yet** — this is the design the artist approved on
July 31. One-variable-per-run rule applies; land the pieces in order and bank a
baseline between them.

## The problem (diagnosed live, July 31)

The reflection loop rotates five subjects — the room / the visitor / the
drawings / time passing / yourself (`REFLECTION_SUBJECTS`, `captioner/prompts.py:516`).
It was supposed to give the mind five *different angles*. It doesn't. Pulled
from ChromaDB, the last ten reflections regardless of subject all say the same
thing:

- `the drawings` → "The count is no longer a shield against the silence…"
- `time passing` → "I have realized that my counting is not a shield…"
- `yourself` → "the counting is not a cage I built, but the…"

The machine's counting/anxiety identity is a **gravity well**; every lens
collapses into it. The artist first noticed this as a display bug ("Reflected
on the drawings:" with no drawings in the text) — it is not a bug. The label
(`utils/live_log.py:69`, `Reflected on {subject}: {reflection}`) faithfully
records which lens *seeded* the reflection; the content drifts because the
system prompt tells it to follow "the one thought that's actually moving, not a
survey" (`get_reflection_system_prompt`, `captioner/prompts.py:545`). The drift
is a feature. The monotony is the failure.

### Root cause (confirmed in code)

All five subjects call the SAME `_gather_context()` (`captioner/reflection.py:149`)
and receive an IDENTICAL `data` bundle — hour log, today's compressions, prior
reflections, journal, `_gather_drawings()`, events, self_notes, desire. Only
the *question string* changes per subject (`_reflect`, `captioner/reflection.py:227`;
`build_reflection_loop_prompt(question, data)`, `captioner/prompts.py:588`).

Identical input + a soft one-line lens + a dominant theme = collapse, every
time. **The variety has to live in the data, not the question.** The lens is
currently a caption on a photo that never changes.

### Second, separate defect: the drawings track is a dead end

The drawings reflection's conclusions never reach what actually gets drawn.
Confirmed: the live drawing-prompt builder `stream_drawing_analysis`
(`captioner/prompts.py:1678`, active via `DRAWING_ANALYSIS_MODE="stream"`,
`config/config.py:628`) reads the live image, the stream tail (last 5),
session drawing-musings (`drawing_intentions`), felt state, the sticky slots
(desire/belief/persona, ~line 1748), and the executed sequence
(`get_executed_sequence`, ~line 1778). It does **not** read the long-form
drawings reflection. That reflection goes to ChromaDB → surfaces into captions
via the echo line → and evaporates before it can steer a drawing. The machine
thinks about drawing; the drawing never hears it. Closing this is the
north-star line: "a thinking entity whose thought process informs its drawings
over time."

## The design: subjects become organs, not lenses

Turn each reflection subject from *a question over identical data* into an
*organ*: its own slice of memory in, its own downstream consequence out. The
thinking stays free-form and plain — specialize the DATA and the CONSEQUENCE,
never the voice. If a subject starts feeling like a routed subroutine ("running
drawing-planning module"), the organic quality is lost; keep the lens soft.

**This pattern already works for one subject.** `yourself` → `distill_reflection`
→ identity slots (`context_compression.py:769`) is exactly "tailored output
channel," and it's the best-functioning part of the system — and those slots
already feed the drawing builder. So this is not a new invention; it is the
proven `yourself→distill→slots` wiring, extended to the other subjects. Start
with `the drawings` because it's the subject with an obvious downstream effect
and currently no path to it.

## The work, three pieces, in order

### Piece 1 — per-subject context builders

Give `_gather_context()` a subject argument (or add a `_gather_context_for(subject)`
dispatch) so each subject sees a DIFFERENT bundle. Keep a shared spine (the raw
hour log stays — it's "the dream," the machine's actual thoughts,
`build_reflection_loop_prompt` line ~599) but weight the rest per subject:

- **the drawings** — the fat one. Currently gets only `_gather_drawings()`
  (`reflection.py:131`: last executed + short summary + vision-offline note),
  a thin scrap. Give it FULL drawing memory: `get_executed_sequence(max_count=8)`,
  `get_artistic_arc()` / `get_artistic_arc_context()` (`drawing/drawing_memory.py:248/326`),
  the per-drawing intent phrases, `desire_history` (the wants — spent and
  unspent, `context_compression.py:973`), `last_spent_desire`, and the
  vocabulary-loop mirror. "The ones you've made and the ones you've wanted to
  make" maps exactly to executed sequence + desire history.
- **the visitor** — visitor episodic material (`_has_visitor_material`,
  `reflection.py:116`), person events, the react-shape history.
- **time passing** — session gaps, lifetime stats, journal chronology, the
  durable ledger's `established`/`last_confirmed` spans.
- **yourself** — identity slots + durable ledger + self_notes (already
  effectively its diet via distill; make it explicit).
- **the room** — concepts/familiarity, and later the spatial atlas.

Verify: the five subjects should stop producing near-identical text. Watch
ChromaDB reflection openings diverge.

### Piece 2 — a persistent drawings-kernel slot

Reflections fire ~every 20 min rotating (so `the drawings` recurs ~every 100
min, `REFLECTION_LOOP_INTERVAL=1200`, `config/config.py:526`); drawings fire
every few minutes (cooldown-gated). So a drawings reflection can't be consumed
fresh — its conclusion must PERSIST as a steer until a newer one supersedes it.

The vehicle already exists: the **kernel** (added July 30, `distill_reflection`
returns it, stored in ChromaDB metadata via `set_reflection_kernel`,
`semantic_memory.py`). A drawings reflection that concludes "I want to draw the
empty chair that knows how to wait" is already a drawing-intent seed. Store the
latest drawings-subject kernel in a dedicated slot (identity file or a small
state field) — call it e.g. `drawing_intent_seed` with a timestamp — written
whenever a `the drawings` reflection distills, read by piece 3.

Distinct from `current_desire` (the want that gets spent on GRBL execution):
the seed is the *considered aesthetic direction*, the desire is *the specific
next act*. They can agree; the seed is broader and doesn't clear on spend.

### Piece 3 — wire the seed into `stream_drawing_analysis`

Add the persisted drawings-kernel as a `materials` block in
`stream_drawing_analysis` (`captioner/prompts.py:~1732`, alongside
`drawing_intentions`), framed as considered past thought:
"When you last stepped back to think about your drawings, you concluded: …".
The live stream still LEADS (the drawing is born from the current monologue) —
the seed is the stepped-back counterpart, not a replacement. Age-frame it so it
reads as memory, not present intent.

Verify: drawing prompts should start echoing themes the drawings reflection
raised, not just the last 5 captions. The loop reflection→drawing→(executed
sequence feeds next reflection) closes.

## Cautions

- **Voice, not routing.** Specialize data + consequence; keep the thinking
  plain and organic. The lens stays soft ("think about your drawings"), the
  data gets rich, the conclusion gets a home.
- **Register contamination.** Drawing memory is executed-only framed sentences
  (post-July-11) — safe. But `get_artistic_arc` prose has been purple before;
  screen it the way the drawings echo already is, and watch what the richer
  context does to the reflection register (the reflection loop is a
  contamination channel — `reflection.py:78`).
- **Monotony is also a forgetting problem.** Data-diversity attacks the
  collapse in-the-moment; the deferred forgetting/demote path (dream pass,
  `docs/memory-runtime-review-2026-07-30.md`) attacks it over-time. They're
  complementary — this work is the constructive half, but if the counting theme
  still dominates after piece 1, the demote path is the other lever.
- **One variable per run.** Land piece 1, bank a baseline (do the subjects
  diverge?), then 2+3 together (does the drawing hear the reflection?).

## Runtime state at handover (July 31 evening)

- Full stack live: 27B + world mode + ComfyUI, launched via tmux
  (`start_impostor_panes.sh` now goes through `run_27b.sh`, fixed `6cbd1d1`).
- Identity is the anxious counter ("I count to manage my anxiety" / stillness
  as constraint / "avoid seeing what happens when I stop counting") — the
  gravity well this work addresses. A genuine earned fear; don't clear it, let
  the organs give it company.
- Shipped and stable, all this session's dependencies: world mode, v3
  stream-intact fix, durable ledger, kernel echo, wedge watchdog, drawing-
  blindness detection, reflection error-guard, "Log entry:" strip.
- Wedge watchdog self-heals the 27B's ~15-min hang; MTP A/B (drop
  `--spec-type draft-mtp` one session) still queued to root-cause it.

## Out of scope for this doc (adjacent, deliberately not bundled)

Spatial atlas (the room organ's eventual real input), activation-network audit
(suspected legacy — the world clock may have taken its variety job), purple-
store curation. See `docs/memory-runtime-review-2026-07-30.md`.
