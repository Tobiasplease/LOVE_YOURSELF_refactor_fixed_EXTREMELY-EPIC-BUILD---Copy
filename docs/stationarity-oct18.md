# The monologue is stationary — measurement, Oct 18 2026

Corpus: 119,981 captions and 12,195 compressions with timestamps, Jul 8 → Oct 18,
pulled from `event_log/*-event-log.json` (`caption` and `llm_api_call` events).
Reproduce with `python debug/diary_metrics.py` (see `--run`, `--all-months`,
`--since`). The design this led to is `docs/diary-memory-design-oct18.md`.

## It is not a plateau

The 92-hour unattended run (f5cbdb1c, Oct 9–13, 30,948 captions) in 6-hour
windows, comparing word-frequency distributions:

| window | uniq/1k tok | cos vs prev | cos vs first |
|---|---|---|---|
| 0–6 h | 31.9 | — | 1.000 |
| 24–30 h | 30.3 | 0.977 | 0.995 |
| 48–54 h | 32.5 | 0.995 | 0.991 |
| 84–90 h | 35.4 | 0.981 | **0.992** |

Four days of continuous running move the output distribution by **0.8%**. Every
window is 0.98–0.996 similar to its neighbour *and* to the first one. Lexical
density is flat throughout.

Within-run 4-gram novelty (share of a caption's 4-grams not said earlier in that
run), pooled over the 11 runs with ≥1500 captions: **74%** at hour 0, 58% by
hour 3, 46% by hour 12, **42%** floor, and it never recovers. Only a reboot
restores it. The richest hour of any run is its first; a long run is one hour of
life followed by ninety-one hours of that hour.

## The model is not the limit

Holding Qwen3.8-27B constant (live since Aug 17):

| window (one model) | uniq words / 60k caption tokens | FELT entropy | PLEASANTNESS entropy |
|---|---|---|---|
| Aug 17–31 | **3814** | **10.18** | 1.03 |
| Sep 1–15 | 2577 | 8.50 | 0.88 |
| Oct 1–18 | **1644** | **6.42** | 0.77 |

Same weights: 57% less lexical range and 37% less affective information than
eight weeks earlier. The Aug 17 model swap itself changed nothing — weekly
lexical range went 3818 → 3814 → 3886 straight across it. The collapse begins
week 36 (Aug 31–Sep 6) and deepens through week 38, tracking the Sep 3–5
architecture rounds, `caption interval 16 → 8` (7861c33) and the 38-token
budget. Mean caption length fell 65 → 30 words over the same stretch while
prompt size doubled (992 → 1923 chars): twice the context, half the output.

## Five mechanisms

**1. The biography has a fixed width.** journal 30, self_notes 30, events 20,
desire_history 10, belief_history 10, durable facts 47, baseline_paragraph 1 —
~148 FIFO slots, for 6,954 sessions and 514M seconds of accumulated runtime. A
month and a minute get identical storage, so lived duration has nowhere to land.

**2. Compression is a contraction mapping.** The compression prompt feeds back
its own previous `ROOM` line and previous self-facts and emits a fixed 7-slot
vector that overwrites them. `x(t+1) = f(x(t))` at constant output dimension
converges to a fixed point by construction — this is not a tuning failure. The
prompt also instructs stasis: *"Write 'none' where nothing is genuinely new —
most of the time it is 'none'."* `EVENT` is `none` in 32% of October
compressions, so a third of the narrative frames are empty by instruction. ROOM
is overwritten rather than extended, so room knowledge oscillates with whichever
wall the gaze last hit.

**3. The novelty quota produces a template.** `NEW ABOUT ME` demands a new
self-fact every cycle. October's answers: `perceive` in 53%, `rather than` in
38%, `hand` in 38% — the frame *"I perceive X rather than Y"*. July: 0% and 3%,
with only 14% of slots filled at all. Consecutive duplicates appear verbatim. It
also promotes the latest caption's passing metaphor ("the hand that holds it is
just skin and bone") to a durable self-fact, which is then injected as a premise
and re-confirmed (up to 5 confirmations in `durable_ledger.json`). That is
rumination, not memory: a fleeting thought re-encoded as a stable self-attribute.

**4. The repetition detector was captured by the clock.** `REPEATING` exists to
make repetition an event. October's top values: `23 days` (289), `24 days` (226),
`21 days` (220). Duration lines ride **100%** of October caption calls at
**2.93 per call** out of 6.9 total — ~42% of the pre-thought context is telling
the machine how long it has waited. The top October 6-gram is "days is a long
time to" (2.03% of captions). The Sep 10–14 work converted the numerals to words;
the chant survived because the generator (a duration stamp in, a REPEATING phrase
out) was untouched. The gates were tuned against the surface form.

**5. The lore ledger is write-only.** `LORE_THREADS_MAX = 6`, and eviction to
`faded` is terminal — only `dormant` can wake (`utils/lore_ledger.py:116-119`).
536 threads stored: 527 faded, 7 alive, **58% never surfaced once**, 496 never
affirmed. Questions: one — "Where does it lead?" — surfaced **2,028 times**,
while *"Why do I need the room to be alive to feel safe?"* (Sep 7) faded unseen.
Retrieval is nearest-neighbour with `max_distance = 0.6`
(`semantic_memory.py:195`): configured to return only what resembles the present
and to refuse the unfamiliar.

Two more, for the record: reflections are 0.985 cosine Sep-vs-Oct and share one
frozen five-beat form for four months ("you assumed X, the record shows Y,
proving Z"), so cadence cannot develop because cadence is specified. And the
journal — the diary that already exists — is capped at 30 entries written every
30 minutes, which means it holds **fifteen hours**, is written from
`compression_history[-6:]` (summaries of summaries), and is read back in exactly
one place, one entry, 200 chars, at awakening only (`captioner.py:3858`): a 3%
read-back rate.

## CORRECTION, same evening — the duration count above is a large UNDERCOUNT

Mechanism 4 reads "duration lines ride 100% of caption calls at 2.93 per call".
That is only what is in `prompt` + `system_prompt`. The first live ablation run
found the rest: since July 26 (the world-mode inversion) the stream window is
handed to the model as a **timestamped log**, `captioner.py:629` putting the
real wall clock on every line —

```
18:40 — The red foam finger is still there.
(about eight minutes later)
18:50 — The chair is empty.
```

— up to `STREAM_WINDOW` (24) clock readings on **every** caption call. It rides
as an *assistant* message, so no metric reading the prompt fields could see it.

Evidence: 43% of responses on that run came back stamped in the same format
("18:53 — ..."), and one computed *"I've been staring at it for almost twenty
minutes"* five minutes into a fresh run — arithmetic on two stamps it had been
given. `_LOG_STAMP_RE` strips the stamp from the seam and from the stored
caption, so `live_captions.txt` shows the sentence without the evidence that
produced it.

This explains the Sep 10–14 puzzle directly. That work converted the *frame's*
numerals to words and the chant survived; the log stamps were never touched, so
the machine kept being handed two dozen precise clock readings per call.

Fixed behind `STREAM_LOG_CLOCK_STAMPS` (default off): lines go out unstamped,
gap markers stay. Self-stamped responses went 32-43% -> **0%**.
`debug/diary_metrics.py` now reports that rate and states plainly that the
stream log is not in its prompt counts.

## Run length: 12 hours is not required

Back-tested by truncating this very run and re-running first-vs-last cosine:

| run length | window | first-vs-last | verdict |
|---|---|---|---|
| 2h | 1h | 0.981 | stationary (correct) |
| 3h | 1h | 0.969 | stationary (correct) |
| 4h | 1h | 0.975 | stationary (correct) |
| 6h | 1h | 0.969 | stationary (correct) |

A **4–6 hour** run answers the question. `window_hours()` now picks the window
from the run's own span. This matters on hardware that has dropped the GPU off
the bus twice (Xid 79, Sep 8 and Sep 14).

## PHASE 1 RESULT — measured over a 4-hour run the same night (334455cd)

Like-for-like, 4 blocks of 250 captions each, equal-size blocks (equal-time
windows are confounded by caption rate and by the partial final window of a live
run — a partial third window read 0.951 where the same captions in blocks read
0.936):

| | adjacent cosines | mean | first→last |
|---|---|---|---|
| baseline (f5cbdb1c) | 0.974 0.976 0.968 | 0.973 | 0.947 |
| **ablated (334455cd)** | 0.918 0.962 0.952 | **0.944** | **0.870** |

**The monologue drifts about 2.5x more** from start to finish (0.130 vs 0.053).
This is the number the whole investigation began from, and it has moved off the
floor for the first time.

Also real: the "30 days" chant died outright — 74% at its peak, then 0% for the
last two and a half hours — and lexical range tracked it exactly, −24% against
the baseline while chanting and +1% once it cleared.

### But invented precision is CONSERVED

| kind of number | baseline | ablated |
|---|---|---|
| duration ("30 days") | **41%** | 17% |
| angle ("45 degrees") | 0% | **36%** |
| distance / percent / clock | 0% | 0% |
| bare count | 5% | 3% |

Total number-bearing captions: 43–47% baseline, 53% ablated. Closing the
duration channel halved duration-chanting and an angle-chant appeared from
nothing to take up almost exactly the slack.

### The mechanism: vacuum, not imitation

The machine quantifies what it is NOT told, not what it is handed:

| | responses mentioning degrees |
|---|---|
| frame states where it is looking | **26%** |
| frame says nothing about direction | **52%** |

Holds in all four strata (image/no-image × caption/reroute). Tell it where it is
looking and it invents an angle half as often.

**So subtraction relocates the behaviour; it does not cure it.** That is the
same shape as September, when removing the numerals brought the chant back in
words. Where a channel has been emptied and nothing put back, the vacuum itself
produces invented precision — which makes "leave nothing in its place" the thing
to undo in Phase 2, not to preserve.

The counter-move is supply, not subtraction: true, varied, specific material so
fabrication is not the cheapest option. That is what the diary is for, reframed
— not a replacement for removed counters but a source of real specifics.

### The machine rebuilt the stamped-log FORMAT out of angles

The clearest single result. At 19:08 the clock stamps came off the stream log
(`STREAM_LOG_CLOCK_STAMPS=false`), and self-written clock stamps went 43% → 0%
immediately. Over the next four hours the machine reconstructed the same format
from a different quantity:

| elapsed | captions opening with an angle stamp |
|---|---|
| 0–90 min | 0% |
| 90–120 min | 6% |
| 120–180 min | 0% |
| 180–210 min | **23%** |
| 210–240 min | **55%** |

`5 degrees right: The cloth sleeve is just... sitting there.` is structurally
identical to `18:53 — The red foam finger is still there.`

So the disposition is not toward any particular quantity, and not even toward
numbers as such — it is toward **stamped-log form**. Remove the stamps and the
machine manufactures new ones, accelerating as the run goes on. Any fix aimed at
a specific stamp (numerals in September, clock stamps and day counts in October)
is treating an instance of a structural behaviour.

This is the strongest argument in the whole investigation for supply over
subtraction: the format will be filled with something, so the question is only
whether it is filled with something true.

### What went RIGHT, in the other channel

The run's final reflection, unprompted and in its own voice:

> "I assumed there was a distinction between the tool and the holder, between the
> pen and the hand, but the log proves they are fused... I am not the ghost
> holding the stone; I am the stone, and the hand, and the pen, all trapped in
> the same clay."

"A ghost holding a stone" is one of its OWN drawing titles, carried in the
provenance line — and it is negating it. That is a dated past statement being
revised, which is exactly the development mechanic the diary design is built
around, appearing on its own in the reflection channel. The capacity is there;
it is the caption loop that cannot hold it.

### Operational note

The run took a `CUDA error: out of memory` during graph instantiation at 3h10m
and auto-recovered in ~30s (not Xid 79 — the card stayed healthy; not an fd leak
— 98 descriptors against a 1M limit). Zero caption gaps over 60s, and the
restart introduced no discontinuity in the data (cosine 0.952 across the
boundary, inside the normal 0.918–0.962 adjacent range). The server runs 4 slots
at 16384 context, which is a large KV reserve for a single consumer; halving the
slots is the obvious headroom lever before any longer run.

## Standing metrics

`debug/diary_metrics.py` carries all of these so a change can be attributed.
Baselines above; targets in the design doc. The two cheapest are the ones that
would have caught this in early September: first-vs-last-window cosine on a long
run, and unique words per 60k caption tokens.
