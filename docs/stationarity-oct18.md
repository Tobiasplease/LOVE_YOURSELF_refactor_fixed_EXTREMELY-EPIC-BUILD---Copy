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

## Standing metrics

`debug/diary_metrics.py` carries all of these so a change can be attributed.
Baselines above; targets in the design doc. The two cheapest are the ones that
would have caught this in early September: first-vs-last-window cosine on a long
run, and unique words per 60k caption tokens.
