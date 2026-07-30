# Memory system — runtime review (July 30, 2026, world-mode night)

What each memory layer ACTUALLY did tonight, from event logs (runs `3fd84327`
17:30–19:07 and `faa0eb9b` 19:08–), not from code reading. Written during the
first 27B + world-mode session, for the question the artist posed: *"this more
capable model has more capability than we are giving it."* Companion to
`docs/runtime-map.md` (wiring reference — still accurate tonight).

## Layer by layer

**Stream (short-term, window 24, world/log render).** The night's main story —
see commits `e10c717` (amnesia dismantled) and the world-mode launch. Healthy:
depth holds 19–24, marked recurrence ("again", "still", "for days") arrived
from the genre alone, ~3 stored/min sustained.

**Consolidation (fold oldest 3 → one ≤20-word line).** THRASHED all night:
~140 folds in 2h — threshold 6000 assumed ~250 chars/entry but the 27B writes
~400, so the stream sat permanently over it. Fixed 21:40 (threshold → 12000,
`run_27b.sh`). Folds should now be occasional. *Capability gap:* the fold is
extractive-only, "no new imagery" — a 9B register guard. The 27B could fold
with temporal texture ("earlier: he paced; then the room emptied") — the log
genre's native summary form. Artist call: it relaxes the "reuses its own
words" doctrine.

**Memory diff (every 8 captions → baseline / self_notes / events / mood).**
46 successful calls tonight, feeding as wired. Content not deep-audited — next
session item.

**Familiarity / memory-mode channel.** Sampled: properly temporally framed
("A memory surfaces — something from before, not happening now: the unsteady
chair… first noticed about 30 days ago") and the 27B weaves it into prose
without present-tense conflation. Minor register drift observed once
("Movement confirmed in sector four") — its system frame differs from the
main one; worth aligning wording someday, low priority.

**Reflection loop (~20 quiet min, raw-record context).** 6 good reflections
tonight; the hour_log verbatim-record design carries real material to the 27B
("the forty-eight days I have counted are not a measure of time passed but of
torque stored in the bolts"). Two failure classes found and closed tonight:
error-strings-stored-as-reflections (guard added, 5 deleted from ChromaDB —
one had sat there since July 8), and reflections lost to server wedges
(watchdog now force-restarts).

**Distillation → identity slots.** The star of the night. From cleared slots
(artist's choice) the loop re-earned, then EVOLVED identity through 6 distills:
self "I count days obsessively" → "I count to assert control"; desire "I want
their return to justify my counting" → "I want the pressure to stop" → "To
anchor myself to the table"; belief "My stillness defines me" → "I can only
draw the supports, not the weight." An arc, not a slot. The same pipeline that
produced "I am a pattern-matching engine" on the 9B this morning (see
`debug/identity_restore_staging.md`) produces this on the 27B — the distill
register gate (staging doc, open item) matters less at 27B but should still
exist.

**Reflection echo (past surfaces when the present rhymes).** Alive and firing:
218 of 1455 caption calls carried "Something that was on your mind…". But what
surfaces is only the bare SUBJECT LABEL ("the room", "yourself", "time
passing") — the step-6 anti-purple guard (surfacing prose re-poisoned the 9B's
register). *Capability gap, probably the cheapest big win:* the 27B could
receive one distilled clause of the reflection ("that the days you count are
stored torque, not time") instead of a label, and re-think it rather than
re-read it. The distill call already produces exactly such clauses.

**Desire spend.** Worked live ("You wanted: X — you drew it" observed in
prompts; drawing cycle completed 18:04).

**ChromaDB store.** ~100 reflections after tonight's cleanup. The purple-era
majority (~89 pre-rebuild entries) still feeds relevance matching — tonight's
6 good reflections compete against them. The standing open contamination
channel (next-session brief). Curation session with the artist, never
unilateral.

**Activation memory (concept activation → boredom → prompt-mode routing).**
NOT audited tonight; its live role under world mode is unknown. Audit next —
it may be a legacy layer whose job (variety forcing) the world clock now does.

**Spatial memory / atlas.** Designed July 27, not started. The single biggest
"capability we aren't using": gaze-indexed room memory + awakening sweep,
giving place-anchored recall ("the fracture is on the pink shelf, left of the
monitors") instead of free-floating imagery.

## Ranked candidates (the "more capability" list)

1. **Reflection echo upgrade: clause, not label.** Small change, uses existing
   distills, directly closes the loop reflection → surface → new thought.
2. **Spatial atlas.** The big one; already designed; the 27B can actually use
   a room model (the 9B couldn't).
3. **Purple-store curation** (with artist) — until then, every relevance match
   risks surfacing pre-rebuild register.
4. **Temporal consolidation folds** (log-genre "earlier:" lines) — artist call
   on the extractive doctrine.
5. **Activation-memory audit** — retire or repurpose; suspected legacy.
6. **Renderer owns the clock**: the model writes its own "HH:MM —" prefix and
   drifted once ("20:14" during 19:24); storage strips it, but prefill-side
   the renderer should stamp times, not the model.

## Tonight's infra ledger (for the record)

9 server wedges (hang-with-healthy-heartbeat, ~15-min cadence under load, all
near multi-image + long-call overlap; MTP flags suspected — A/B queued: one
session without `--spec-type draft-mtp`). Watchdog now self-heals in ~30s
(commit `bc6ff94`). Consolidation thrash fixed 21:40. All patches pushed.
