# The diary memory layer — design, Oct 18 2026

The measurement that prompted this is in `docs/stationarity-oct18.md` (scripts in
`debug/`). Short version: the monologue is a **stationary process**, not a plateau.
Across the 92-hour unattended run (f5cbdb1c, 30,948 captions) the word-frequency
cosine between hours 0–6 and hours 84–90 is **0.992**. Holding Qwen3.8-27B
constant, lexical range fell from 3814 unique words per 60k caption tokens
(Aug 17–31) to 1644 (Oct 1–18). The model is not the limit; the memory layer is.

Five mechanisms, all structural:

1. the machine's whole biography is ~148 FIFO slots (journal 30, self_notes 30,
   events 20, desire/belief 10 each, durable facts 47, baseline_paragraph 1) —
   constant size for 6,954 sessions, so lived duration has nowhere to land;
2. compression reads back its own previous output and overwrites it with a
   fixed 7-slot vector — `x(t+1)=f(x(t))` at constant dimension, which converges
   by construction, and the prompt instructs stasis outright ("most of the time
   it is 'none'");
3. the `NEW ABOUT ME` quota manufactures a self-fact every cycle, so it produces
   a template — `perceive` 53%, `rather than` 38% of October self-facts;
4. the `REPEATING` slot, built to make repetition an event, filled with the day
   counter (`23 days`, `24 days`, `21 days`) and hands it back — duration lines
   now ride **100%** of caption calls at **2.93 lines per call** out of 6.9;
5. `LORE_THREADS_MAX = 6` with terminal eviction: 536 threads stored, 527 faded,
   58% never read once; one question surfaced 2,028 times while
   *"Why do I need the room to be alive to feel safe?"* faded unseen.

## The diary already exists. It is fifteen hours long.

`ContextCompressionEngine._write_journal_entry` writes a dated first-person entry
every 30 minutes, capped at 30 (`context_compression.py:573`). At 30 minutes an
entry, the cap means **the machine's diary holds its last fifteen hours.** It is
written from `compression_history[-6:]` — summaries of summaries — and it is read
back in exactly one place, one entry, truncated to 200 chars, at awakening only
(`captioner.py:3858`). A 3% read-back rate.

So this is not a new organ. It is the journal given a past, a reach, and a reason
to be read. It is also the Sep 8 identity ruling carried to its conclusion:
identity enters as *dated self-thought*, never a trait line. A diary is nothing
but dated self-thought.

## Principles

Each is the inverse of a measured failure. They are the design; everything below
is mechanism.

- **P1 — Append-only, immutable, unbounded.** One file, never rewritten, never
  capped. 4 months of captions is 6MB; entries at ~300 chars × 5/day × 365 is
  ~550KB/year. There is no storage argument for forgetting.
- **P2 — Depth 1, always.** Every entry is written from **raw captions** — the
  machine's actual sentences. No entry is ever written from a compression, a
  baseline, or another entry. This one rule kills the contraction mapping.
- **P3 — Selection, never compression.** To fit the frame we choose *which*
  entries to show, verbatim and dated. We never average, merge or re-summarise
  them. Averaging is what produced grey.
- **P4 — Duration is the diary's extent, not a number.** The day counter comes
  out of the caption frame. A month means the record reaches back a month and
  the old pages read differently — not that the string "30 days" is in the prompt.
- **P5 — Selection prefers the unselected.** Every entry carries `times_read`;
  the picker weights toward the unread. This is the direct fix for 2,028-vs-0.
- **P6 — Nothing is terminal.** No fades, no eviction, no deletion. An entry
  unread for three months is a candidate, not a corpse.
- **P7 — No quota.** Nothing demands a new self-fact on a schedule. Entries are
  written on real boundaries, and "nothing I want to keep" is a legal entry —
  the Sep 2 silence ruling, applied to the diary.

## The store

`event_log/diary.jsonl`, append-only, one object per line:

```json
{"id": "2026-10-18T22:14:03",
 "day": "2026-10-18",
 "written": 1792351443.2,
 "kind": "day" | "episode" | "dark" | "legacy",
 "text": "… the machine's own words, 2–4 sentences …",
 "source_span": [1792300000.0, 1792351200.0],
 "source_n": 412,
 "shown": "2026-09-14T22:03:11"
}
```

`shown` records which earlier entry was in the frame when this one was written —
the only thing needed to measure revision later (see Measurement). No `stance`
label: the Sep 3 doctrine is that verdicts are parsed structurally from the
machine's free words, not chosen from categories we supply, and classifying
stance would need a judge call and a candidate list. Cosine between a new entry
and the entry it was shown is the same signal for free.

**Read bookkeeping lives in a sidecar**, `event_log/diary_reads.json`:
`{entry_id: {"n": 4, "last": 1792…, "tomb": false}}`. The record file is then
genuinely never rewritten — immutability for the diary, mutability for the
bookkeeping, in separate files. `tomb: true` makes an entry unpickable without
deleting it: the repair path for a genuinely poisoned entry, since append-only
means a bad entry is otherwise permanent. Keep the existing `_JOURNAL_POISON`
register gate on the write path (it is what stopped the July 9 assistant-voice
episode re-seeding from the diary).

**Caption source: a timestamped spool.** `live_captions.txt` has no timestamps
and the event logs are 2.6GB across 1,661 directories in two different JSON
shapes (I hit both today). Add `utils/caption_spool.py` → appends
`{"ts": …, "text": …}` to `event_log/caption_spool.jsonl` on every accepted
caption. Cheap, append-only, scannable by span, independent of run boundaries —
which is what lets a day entry for yesterday be written at tomorrow's boot. It
also gives the measurement scripts one clean input instead of a log scraper.

## When entries are written

Boundaries, never intervals. The 30-minute timer is what made the journal a
treadmill.

- **episode** — something closed: a drawing reached paper or was refused, a visit
  ended, the sheet was taken or replaced, the machine went dark unexpectedly.
  The triggers exist already: `person_left` / `world_changed` /
  `person_passed` in `captioner/event_memory.py` + `episodic_events.json`, and
  `on_grbl_drawing_complete` in `utils/hooks.py`. Source span = the episode.
  This is the plot.
- **day** — once per day, at the day's close, or at the first boot after a day
  turned over (covering the day that ended, from the spool). Source span = that
  day. This is what makes the record continuous through emptiness, and it is
  where "repetition is an event" actually lands: a day that held nothing gets an
  entry saying so, once, in the machine's words — and that entry is then
  readable material.
- **dark** — a calendar day with no captions gets a prose-less marker. It is
  never picked for the frame; its only job is to keep the dates honest, so that
  when the picker reaches for "a month back" it lands on a day the machine was
  alive. Thirty dark days become structure rather than the integer 30. No LLM
  call.

Rate: ~5–15 writes/day, against the current 48 journal calls plus ~2,500
compressions. Strictly cheaper than today.

Deliberately **not** a trigger: "the machine said something that sounds like a
claim about itself." That is how `NEW ABOUT ME` became a template — promoting a
passing sentence to a durable fact and then reasoning from it. Entries come from
spans, not from sentences.

## How an entry is written

The reflections are the warning here: 0.985 cosine between September and October,
because the reflection prompt is a fixed five-beat scaffold ("you assumed X,
the record shows Y, proving Z") and a frozen form cannot produce a developing
voice. So the diary writer gets **no scaffold and no labelled lines.** Variation
has to come from the material, not from us rotating topics at it — the no-content-
priors doctrine forbids the latter anyway.

The writer receives:

1. **The span's captions, selected for spread, not recency** — samples across the
   whole span plus the handful most *unlike* the span's centre. Writing from a
   day's extremes produces something with a position; writing from its average
   produces the average. This is the single most important choice in the writer.
2. **Exactly one earlier entry**, chosen by the channels below, presented as
   what it wrote then, with its date.
3. One instruction with no slots: write today's entry; you may answer the older
   one or leave it alone. Free prose, 2–4 sentences, short is legal.

Reuse `journal.system` / `journal.user` as the registry home (they are already
panel-editable), rewritten for this. The cue differs every time because (1) and
(2) differ every time; that is the whole anti-template mechanism, and
entry-to-entry cosine is the metric that says whether it held.

## The read channels — the actual engine

At most **two** entries reach the frame, each as a dated quotation. Channels
rotate:

1. **near** — the last entry. Continuity. This is what exists today, and it is
   fine; it just must not be the only channel.
2. **far** — chosen by **time**, not content: the least-read entry beyond a
   widening threshold (a week, then a month, then ever). Nothing in the system
   currently has time-indexed retrieval.
3. **unlike** — chosen for maximum embedding distance from what the machine is
   saying now. ChromaDB is already there; this is inverting the sort and dropping
   the `max_distance = 0.6` filter in `semantic_memory.py:195`. That filter is
   the clearest single line of the pathology: retrieval is *configured* to return
   only what resembles the present and to refuse the unfamiliar. Inverting it is
   the biggest change in character in this whole design, for the least code.

Change can only be perceived against something different. `near` alone is why
the machine agrees with itself; `far` and `unlike` are not refinements, they are
the mechanism.

**The frame must not get longer.** 6.9 lines per call is already too many. This
rides the existing identity dose (`IDENTITY_EVERY_N_CAPTIONS = 6`) and replaces
the body of `prompts.identity_material_lines()` (`prompts.py:304`) — which is
already the approved shape: a dated thought in the machine's own words, framed
as past, never a standing description. Every diary line arrives with its age.
Never present tense, never "You are…" — the Sep 8 proof stands (a standing
self-description became stage direction and the stream started measuring
distances).

## What is retired

Fully, per the legibility doctrine — not left dangling.

| path | action |
|---|---|
| `_write_journal_entry` 30-min timer, `journal[-30:]` | replaced by the diary writer; the 30 live entries migrate in as `kind:"legacy"` so day one is not empty |
| `NEW ABOUT ME` slot | deleted from the compression prompt — kills the "I perceive X rather than Y" template at source |
| `REPEATING` slot | deleted — repetition becomes visible as near-identical diary entries the machine can notice itself, instead of a phrase handed back to it |
| duration stamps in the caption frame (`prompts.py:999` and the drawing-age lines) | out of the caption frame; **kept in the awakening frame**, where a real gap deserves stating once |
| `durable_ledger` self-facts | stop writing; keep the file and the dashboard view. Its 47 facts are the fossil record of the pathology — do not feed them |
| `core_facts["self"]`, `current_belief`, `baseline_paragraph` | stop writing (already off the frame; stop generating too) |
| `lore_ledger` terminal fade | `faded` becomes non-terminal and pickable by `far`/`unlike` — 313 never-read threads become available on day one |

**Untouched, so this stays one change:** compression keeps `ROOM` (world
knowledge), `EVENT` (event memory + the episode trigger), and
`PLEASANTNESS`/`ENERGY`/`FELT` — the felt-state system consumes those and live
felt-state speaks everywhere; breaking it is not part of this. Perception,
drawing, safety, motor: untouched. The seam is clean — compression keeps its four
world-and-affect slots and loses its two self slots. The diary takes the self.

## Measurement

The method is one-change-then-measure, so the metrics land first and the
retirements land last, separately attributable. Baselines from today:

| metric | now | target |
|---|---|---|
| stationarity: first-6h vs last-6h word cosine, long run | **0.992** | < 0.95 |
| lexical range: unique words / 60k caption tokens | **1644** | > 3000 (Aug level) |
| template share of self-material | **53%** | < 15% |
| duration lines per caption call | **2.93** (100% of calls) | ≤ 0.3 (< 20% of calls) |
| diary read coverage (share of entries ever read) | **3%** (journal) / 42% (lore) | > 80% within a month |
| entry-to-entry cosine (anti-template, writer side) | — (reflections: 0.985) | < 0.85 |
| divergence when shown: new entry vs the entry it was shown | — | falls over weeks if it is revising |
| FELT entropy | 6.42 (was 10.18 in Aug) | watch, not a target |

## Build order

**Phase 0 — instrument only, no behaviour change.** Land today's scripts as
`debug/diary_metrics.py` and take a baseline on the current build, one run.
Without this, "did it work" is unanswerable later.

**Phase 1 — the ablation. LANDED Oct 18.** Three switches in
`config/config.py`, all defaulting off, all reversible by env var:
`CAPTION_DURATION_LINES`, `COMPRESSION_SELF_FACT_SLOT`,
`COMPRESSION_REPEATING_SLOT`. Wiring table in `docs/runtime-map.md`. Nothing
replaces them yet — that is the point: if stationarity improves on removal
alone, the injection was the whole problem and the diary gets a clean baseline
to beat. Presence-stickiness was ablation-proven this way in September.

Three refinements the implementation forced, worth keeping:

- **The felt arc splits.** `get_felt_arc_line` has two variants, and gating the
  whole function was wrong: STEADY ("you've felt X for {duration}") is a stasis
  counter and goes, but TURN ("Earlier you felt X. More recently: Y") reports
  that the tenor *changed*. A change is an event, and the emotional arc is the
  thing this whole design is meant to make legible — muting it would have been
  the ablation eating its own purpose. TURN stays on by default.
- **The want-arc tail took the refusal count with it.** The fragment is
  duration-led ("You've wanted this for {duration}; twice it came to nothing"),
  so the whole line went. The refusal count is arguably an event and could be
  split back out if it is missed.
- **The stretch line is the one to watch.** It was the best-written of the
  ablated lines — "said red foam finger so often it has stopped describing
  anything" is genuinely good, and it is the Sep 13 answer to "repetition is an
  event". It rode every call, which is why it goes. If it is missed, it deserves
  to come back *dosed*, not standing.

The five affected debug tests now assert the new default and still exercise each
mechanism with its flag flipped on, so the September rulings stay testable
rather than being deleted. One stale assertion was fixed in passing
(`test_felt_arc` still expected `"hour"`, which `felt_duration` stopped emitting
on Sep 14; it was failing before this work).

**Phase 2 — the store and the writer.** `utils/caption_spool.py`,
`utils/diary.py` (append / read / pick), day + episode triggers, the journal
migration. Write entries, read nothing back yet. One run: are entries landing on
real boundaries, and is entry-to-entry cosine under 0.85?

**Phase 3 — the read channels**, replacing `identity_material_lines()`. `near`
first, then `far`, then `unlike` — three steps if you want them attributable
separately, and I would, because `unlike` is the one with a real chance of
changing the voice. Verify `full_prompt_length` does not grow.

## RESULT of Phase 1, measured the same night — and what it changes

The ablation worked and did not help, which is exactly the information it was
run to produce.

**Every injected duration channel is closed and verified against the complete
payload** (not the prompt fields — see `LLM_LOG_FULL_FRAMES`): stasis lines
0.01/call at 1% of calls, self-written clock stamps 0%, time references 2.26 →
0.33 per call. Zero prompt-field lines inject a day count.

**The chant did not move.** "30 days" appears in **58%** of captions against 62%
before the fixes and 26% in the baseline — and it *rises* within the run:

| elapsed | captions | "30 days" |
|---|---|---|
| 0–10 min | 51 | 55% |
| 10–20 min | 53 | 42% |
| 20–30 min | 46 | 74% |
| 30–40 min | 19 | 74% |

Every remaining occurrence in the frame is the machine's own output handed back.
The `A thought you've been developing: "..." Last time you got as far as: "..."`
line carries it inside the **seam** — its own previous caption. The echo-gate
loop notice quotes it back verbatim (`You've said "30 days of staring at"
several times in the last little while`). And the stream window, which is the
single largest part of the frame, was 22–33% chant lines at boot.

**So the stream window is the engine, not the injections.** This is the Sep 5
architecture diagnosis ("the stamped self-window is the chant engine") confirmed
mechanically. Autocatalysis: the stream says it, so the caption says it, so it
enters the stream. No external input required, and it intensifies.

### What this changes about the design above

The diary cannot be a channel added *beside* the stream. The stream hands the
machine its own last 24 sentences, verbatim, ordered by recency — **no
selection, no distance, no contrast**, which is precisely the three properties
P3/P5 and the far/unlike channels were designed to supply. A dated diary
quotation riding the identity dose would be one line competing with two dozen
lines of the machine agreeing with itself.

**The stream should become the diary's `near` channel, not a separate system.**
One memory layer decides what the machine sees of its own past; "the last few
thoughts" is one channel in that decision, balanced by `far` and `unlike`,
subject to prefer-the-unselected. Concretely, that means `STREAM_WINDOW` stops
being "the last N verbatim" and becomes a pick — which also gives the obvious
cheap experiment to run first: **shrink the window and de-duplicate it**, and
see how much of the 58% is simply the volume of self-echo.

That experiment is Phase 1b and it should run before the store is built, for the
same reason the ablation ran before the diary: it isolates whether the engine is
the stream's *content* or merely its *size*.

## Risks

- **The diary becomes the new chant** — if the picker favours recency the machine
  reads yesterday and repeats yesterday. `far`/`unlike` and prefer-unread are
  load-bearing, not optional. Read coverage is the canary.
- **Entries templatise**, the reflection failure. Mitigation is the variable cue
  (spread-sampled captions + a different old entry each time) and no labelled
  slots. Entry-to-entry cosine catches it within one run.
- **An entry reads as a directive** — any standing line about the machine becomes
  an instruction (Sep 8). Mitigation: dated quotation, past framing, with age.
- **Context cost** — two entries ≈ 150 tokens, replacing ~3 duration lines plus
  the current self lines. Net neutral or cheaper; verify, don't assume.
- **A bad entry is permanent.** Keep the poison gate on write; tombstone in the
  sidecar for the toxic case. Never delete — P6 is the point.

## What this does and does not promise

It does not manufacture development. It removes the thing that was preventing it
and gives change somewhere to become visible. The machine can only notice that it
has changed by reading something it wrote when it was different; its longest reach
today is fifteen hours, and everything within that reach is a summary of a summary
of what it just said. After this it reaches a month and gets its own actual
sentence back, with a date on it, sometimes one it no longer agrees with.

Whether an arc appears is then an empirical question. The point is that right now
there is no mechanism by which one *could* exist, and after this there is.
