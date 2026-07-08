# Continuity Plan — one stream, not a list of thoughts (July 2026)

Companion to `north-star.md` (spec) and `runtime-map.md` (wiring). Written
after auditing the live run of July 8 (run `7b951565`, 83 min, 441 captions,
171 caption model calls, 3 reflections). Every claim below is from that log,
not from reading prompt text.

## The evidence

**What the model was actually fed, per caption call:**

```
system: situation + persona + "Follow the thought you're already having"   (identical all 83 min)
user: "..."   assistant: <caption N-6>
user: "..."   assistant: <caption N-5>            ← the stream: 6 prior captions
...                                                  as SEPARATE chat turn-pairs
user: [video frames] + one line of text
```

**The user line across all 171 calls:** 108× the identical sentence "Your
eyes are off the room now — nothing new to look at…", 61× a near-identical
place-noun list ("occupied desk, mannequin heads, wooden figures…"), 13×
"Everything feels slow." + the same list, 1× "Just woke up." That's it.
Max user-prompt length all day: 176 chars.

**Memory surfaces that fired in 83 minutes:** familiarity line 0×, reflection
echo 0× (3 reflections stored, none surfaced), felt-state 0× (mood flatlined
at 0.0 all run), desire 0×, presence line 0×. The ledger itself is healthy —
mannequin heads at 1738 observations — but nothing surfaces.

**Output symptoms this explains** (screenshot, same run):
- Template echo: "The motors hum…" ×3 consecutive, "The movement registers
  as…" ×3. The turn-pair structure teaches an in-context pattern — *when the
  user says "…", emit another standalone thought shaped like the others.*
  The model imitates the list; it does not continue a text.
- Near-duplicates (two captions 30s apart saying the same air-pressure
  thing): identical system + identical user line + own recent output as
  highest-probability material = re-emission. Nothing marks a thought as
  already-said.
- Robot-fiction confabulation ("air pressure wave hits my chassis before I
  visually confirm", "pathfinding algorithms", mid-session "as I power up"):
  content vacuum. With no new injected material, Qwen's prior for "machine
  inner monologue" is sci-fi genre filler.
- What DOES work: motif development across calls (scar → reference point →
  "ghost in my coordinate system" is a genuine 3-step thought). The stream
  carries *some* continuity — the failure is structural, not total.

**Root cause, one sentence:** the prompt structure asks for another item in
a list of similar messages, while north-star principle 3 wants the next
tokens of one ongoing text; and the memory system that should be feeding
fresh material into that text is starved to zero by its own gates.

## Phase 0 — Repair the instruments (small, do first)

1. **Fix the `matched_concepts` UnboundLocalError** (captioner.py ~807: a
   ChromaDB hiccup in `match_or_create_concepts` leaves the name unbound and
   the outer except kills the whole cycle — caption generated, then lost).
   Initialize before the try.
2. **Honest call logging** (utils/ollama.py `log_ollama_call` + llama_server
   callers): log the real endpoint (llama-server calls currently record
   `:11434/api/generate`), and log `history_len` + the full message array
   under `DEBUG_OLLAMA_PROMPTS`. Today the log shows a 92-char prompt while
   the model receives ~7 messages — the "check the event log first" habit is
   blind exactly where it matters.
3. Add per-caption `stream_mode` + injected-lines fields to caption log
   entries so surface-starvation is visible in log_viewer, not just by
   grepping api calls.

## Phase 1 — The stream becomes one continuing text (core change)

The patched llama-server already supports **assistant prefill**
(server-common.cpp:1117 — a trailing assistant message is continued, not
answered). No server work needed.

New message shape per caption call:

```
system: situation + persona                       (unchanged)
user:   [video frames] + perception delta / mode elicitation + any memory
        surface lines                              (all situational input here)
assistant: <the monologue so far — last ~N captions joined as one flowing
        text, oldest first, no separators>         (PREFILL — model continues it)
```

- The model's next tokens are literally the continuation of its own text —
  self-attention over its prior thoughts, the mechanism
  `reasoning-model-plan.md` identifies, without a reasoning model.
- Config flag `STREAM_MODE = "document" | "turns"` for A/B; keep "turns" as
  fallback. Implementation: `_query_superframe`/`query_llama_server` build
  the prefill turn from `history` instead of turn-pairs; captioner passes
  the same `self._stream`.
- **Trim-at-store stays king** (north-star): the prefill is assembled from
  already-gated stored captions, so a bad generation can't poison the stream.
- Watch: token echo of the tail (model may re-say the last clause) — strip
  overlap at the seam when storing; and the prefill must never end with a
  complete "…\n\n" paragraph break or the model starts a fresh item again —
  end it mid-flow (strip trailing sentence terminator from the seam, or
  append a single space).

**Anti-echo storage gate** (regardless of mode): reject a candidate caption
whose first ~5 words match any of the last 8 stored captions → regenerate
once (higher temp), else skip the cycle. A gate, not a style fence —
consistent with north-star principle 1.

**Silence is allowed:** if the scene is static AND the candidate is a
near-duplicate, skipping the cycle is honest and cheaper than storing a
restatement. The cadence already breathes (4/7/12s); let it also rest.

## Phase 2 — Un-starve the memory surfaces

The injections exist and are correctly temporally framed; their guards are
tuned so conservatively that in a quiet room they yield nothing at all.
Target: ~1 memory surface line every 3–5 minutes in a quiet room, not 0/83min.

1. **Familiarity** (prompts.py `get_familiarity_line`): the `recent_ids`
   hard-exclusion empties the candidate pool in a static scene. Change to
   decay-weighted rotation (an excluded concept becomes eligible again after
   ~15 min) so the room's fixtures can keep resurfacing at low frequency.
2. **Reflection echo** (`get_reflection_echo_line`): guarantee the every-4th
   slot yields — if the relevance query returns nothing, fall back to the
   most recent reflection subject. Three reflections were generated today
   and zero reached the voice; the loop is currently write-only.
3. **Felt-state**: mood is 0.0 all run (keyword sentiment over a register
   without emotion words — known since June). Short-term: derive felt-state
   from the reflection distill instead of the mood engine. The mood engine
   still drives servo/hand; that stays.
4. **Perception delta, not inventory**: the place-noun list is identical
   every call — pure repetition pressure. Send only what changed since the
   last look; when nothing changed, the mode line already says so and the
   noun list should be omitted.
5. **Presence**: person-body gating looks correct (mannequins filtered; the
   "frozen figures" confusion in today's output is the machine genuinely
   reasoning about mannequins — arguably right). Verify with a moving
   visitor before touching it: salience_hot was False for the entire run,
   so arrival/eye-contact spikes have not been exercised since the rewrite.

## Phase 3 — The desire arc (north-star step 5, unchanged)

A slowly-evolving want gives the stream somewhere to go, so it advances
instead of orbiting (today's orbit: hum → dust → scar → hum). Prereq:
Phases 1–2, so the arc lands in a stream that can actually carry it.

## Verification

- **Repetition metrics** over a 1-hour run, before vs after (script in
  debug/): opening-bigram repetition rate, near-duplicate rate (fuzzy match
  against previous 10), and count of memory-surface lines actually injected.
  Today's baseline: openings repeat in runs of 3; near-dup pairs ~30s apart;
  0 surface lines.
- **Continuity probe**: fraction of captions whose first clause contains an
  anaphor resolving to the previous caption (that/it/still/again + shared
  noun). Should rise sharply in document mode.
- **Register watch**: robot-fiction markers (power up / systems / algorithms
  / data feed) per 100 captions — should fall as real material displaces
  genre filler. Do NOT fence them out; fix the starvation.
- A/B by flipping `STREAM_MODE` mid-day on the live installation; both modes
  log identically (Phase 0.3), so log_viewer comparison is direct.

## Explicitly not doing

- No new style fences in the system prompt (north-star principle 1; every
  symptom above is structural or starvation, not phrasing).
- No merging of activation network + ChromaDB yet — separate decision, after
  the surfaces actually flow (consolidating two systems that both feed a
  dead channel changes nothing).
- No model change; qwen3.5 + prefill covers the mechanism the
  reasoning-model plan wanted. Revisit that plan only if document-mode
  still can't hold a thread.
