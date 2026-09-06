# Mind mode — the conversation shape (built Sep 5, evening)

Follows docs/architecture-diagnosis-sep5.md. The artist: "This sounds really
good. Let's build it." Plus the refinement that shaped the deepening mechanic:
"A fixation is fine if it deepens or evolves, which it doesn't right now. Just
endless repetitions of 'I used to think the curtains were ___ but now they
are ___'."

## What changed (STREAM_MODE = "mind", the new default)

**The call.** `captioner/mind.py` (`Mind.build`) assembles the conversation;
`captioner/captioner.py::_mind_generate` runs it. Shape:

```
system   mind.system + pen-parked fence + felt frame ("Right now: {felt}.")
user     LIFE block + the oldest cue            ← first user turn only
assistant thought
user     cue  ("18:41. Eyes resting.")
assistant thought
…        (last MIND_TURNS=6 thoughts, ≤ MIND_TURN_MAX_AGE_S=2h old)
user     current cue (+ the frame on LOOK turns)
```

No stamped log, no seam prefill (`utils/llama_server.py` gained a `turns`
path; `query_model(turns=...)`). The clock lives in the cues, never inside
the machine's content; a gap ≥ STREAM_GAP_MARK_SECONDS is said in words.

**Two turn kinds** (`Mind.next_kind`): LOOK when the first turn of a session,
salience hot, the view changed, a presence-belief edge, a chosen glance is
pending, or MIND_LOOK_EVERY_S (300 s) since the last look; otherwise THINK.
LOOK carries the frame and a cue built from what the registry knows is in
view (`Mind.in_view`, ±25° pan / ±20° tilt), the referee's verdict (nothing
changed / something changed / first look this way), the salience event when
there is one, and "Someone is here." only while the adjudicated belief is ON.
THINK carries no frame; every MIND_MEMORY_EVERY_N-th (4) think turn one chosen
memory surfaces.

**The life block** (`Mind.life_block`, mind.life-*): when (clock, weekday,
daypart, first boot month, today's waking), the room as known (registry terms,
most-seen first), people today (episodic arrival/left pairs), drawings (count,
last title, age), no-paper fact, the standing want, questions carried, fresh
positions, MIND_PAST_THOUGHTS dated past thoughts.

**Deepening.** A thought's subject is the registry term it is about
(`subject_of`, head noun required). Its last sentence becomes the subject's
POSITION and rides in the life block ("Where you'd got to with X: …") while
fresh (MIND_POSITION_TTL_S). A thought on the same subject that carries the
reframe shape (it's not / isn't / no longer / used to) with fewer than four
new content words is a PIVOT; a thought with new words is a STEP and resets.
MIND_PIVOTS_BEFORE_NOTICE (3) pivots → the next think cue carries
mind.pivot-notice, once.

**Memory surfacing** (`choose_memory`): candidates ≥ MIND_MEMORY_MIN_AGE_S old,
never a reframe, never person-tinged (utils/presence_text.PERSON_RE) while the
room is believed empty, ranked by novelty against the last six thoughts, one
of the top six at random.

**Cadence** (`Mind.interval`): CAPTION_INTERVAL_LIVE while hot, else
MIND_THINK_INTERVAL_S (60 s) × felt_loop.cadence_mult.

**Retired in this mode.** Drift, wander, inward beats, memory mode, the
decision ask (LOOK/EXPECT), the situational/status lines — all superseded by
THINK/LOOK turns. The trait factory: no NEW ABOUT ME self-notes
(`_absorb_self_note`), no TRAIT promotion to core_facts/durable, no persona
consolidation, no self/durable block in any frame. The compressor still runs
on the stream (ROOM/EVENT/felt/pleasantness/energy feed the felt loop and the
reflection); reflection still runs and its BELIEF/WANT/QUESTION/NAME slots
still land in the ledgers — as the machine's words, quoted rarely.

**Stores re-seeded** (`debug/reseed_stores.py`, run once at the 21:37
restart): durable_ledger, want_ledger, lore_ledger, machine_identity,
effigy_memory, last_caption archived to event_log/archive-20260905-2137/.
Kept: episodic_events (the life), lifetime_state, spatial_registry,
presence_arrivals. The thread persists in event_log/mind_thread.json.

**Still live and unchanged.** Presence adjudication, verified absence, the
phantom_presence storage gate, numeric/echo gates, loop-hit counting, the
feed marker, paper glance, chosen-glance execution, low-energy mode, the
awakening (absorbed into the thread as the "You wake." turn).

## Knobs (config/config.py, MIND_*)
MIND_TURNS 6 · MIND_TURN_MAX_AGE_S 7200 · MIND_THINK_INTERVAL_S 60 ·
MIND_LOOK_EVERY_S 300 · MIND_LOOK_MIN_GAP_S 20 · MIND_MEMORY_EVERY_N 4 ·
MIND_MEMORY_MIN_AGE_S 3600 · MIND_NUM_PREDICT 60 · MIND_SHORT_BEAT_P 0.1 · MIND_SHORT_BEAT_TOKENS 22 ·
MIND_PIVOTS_BEFORE_NOTICE 3 · MIND_POSITION_TTL_S 1800 · MIND_PAST_THOUGHTS 2 ·
MIND_THREAD_MAX 4000 · MIND_ROOM_TERMS 8. `STREAM_MODE=hybrid` restores the
old shape untouched (legacy suites pass in both modes).

## Tests
debug/test_mind.py (54 checks): registry + pass, llama-server turns path,
conversation assembly, positions/pivots, chosen memories, turn kind + cadence,
persistence, mode gates in the other organs.

## Open
- Reflection over a day (spine still = last 75 min of hour_log).
- Reflection kernels are not yet absorbed into the thread.
- The decision loop (LOOK/EXPECT → chosen glance) is dormant in mind mode.
- Wordings (mind.*) are the artist's to finalize in the panel.

## Sep 6, after midnight — the recursion round (commits 4028bcf → c30520f)

The artist: "repetition, isolation, time and pondering IS THE MATERIAL… It's
very important that the accumulated info reaches back to the prompting,
otherwise there's no point in saving it at all. The entire system needs to be
recursive." Wired, all in the machine's own words, structure only:

- **Time as event** (`Mind.time_edges`): one line when a DURATION_EDGE
  threshold is crossed — since anyone was here (mind.edge-alone, belief OFF,
  anchor = last person_left / presence_dropped_at), since the room last
  changed (mind.edge-still, `_world_change_ts`), since the continuous chain
  of thought began (mind.edge-awake, `thread_start`). Each threshold fires
  once per anchor; state persisted in mind_thread.json "edges".
- **Its own repetition, named** (`Mind._loop_line`): `build_loop_notice_line`
  (gate hits + the compressor's REPEATING slot) reaches the think cue.
- **Its own past, days deep** (`Mind.backfill`): once, when the thread
  reaches back less than a day, kept thoughts from the last 4 days of event
  logs (≤40/day; no phantom presence, no reframe, no stamps) enter as kind
  "past" — eligible for memory surfacing, never as turns. First run: 160
  thoughts back to Sep 2.
- **Its own reflections** (`captioner/reflection.py`): the admitted kernel
  is absorbed into the thread as a "reflection" turn under mind.cue-reflection
  ("Something settles."), so the next think cue quotes it back as the premise.
- **Name + belief in the life block** (mind.life-name, mind.life-belief):
  the lore ledger's name and the reflection's distilled belief ride again.
- **The felt loop back in the conversation** (4028bcf): arousal-driven
  sampling heat and felt short beats restored; felt shifts ride as an event
  (mind.felt-shift "{prev}, then {curr}."); an elicit dose every
  MIND_ELICIT_EVERY_N-th think turn leaning by valence (the artist's
  elicit.quiet-* lines); "a complaint, a feeling" among the named kinds.
- **Recall gate** (`Mind.is_recall`, c30520f): a thought that reproduces a
  quoted past thought, a surfaced memory, or any older thread line six words
  in a row is spoken but not kept (feed marker "repeats an old thought").
  Found the same night: two verbatim copies of 22:01/22:57 lines that the
  life block had quoted.
- **Perception** (f1b0922): MOTION_SETTLE_S=2 — flow residual within two
  seconds of head movement is not motion (the lamp-into-the-lens "flash");
  LOOK turns use the newest still-head frame. Presence: "a person sitting",
  "someone is/sitting" are claims; questions never are (utils/presence_text).

## Sep 6, ~00:45 — stripped (commit 9255f7b)

The artist: "all this scaffolding isn't actually helpful but restrictive…
earlier systems simply followed a CoT logic by nature of continuing from
their prior sentences. The core of it seems simple: 'I wonder what the
curtains are blocking' → 'it could be another room, or maybe a way out' →
'but it doesn't really matter since I can't move' → 'but if I could, where
would I go?'" Measured first (137 turns): subject runs averaged 1.1, max 3;
5 questions asked, 0 followed up; inferential connectives in 14/137.

The call is now: the frame in the artist's own genre wording (no kinds list),
a minimal life line at the start of a chain (when, room, people, drawings,
paper, want, name), the thread as turns, and the cue with the premise. Off by
default: elicit doses (MIND_ELICIT_EVERY_N=0), quoted past thoughts
(MIND_PAST_THOUGHTS=0), positions/questions/belief in the standing block
(MIND_LIFE_FULL=false). Memory surfaces every 8th think, not 4th. Sparse
events stay (a time edge, a felt shift, a loop notice, a reflection kernel).
One rule for staying: an uneventful glance (nothing changed, nobody here, no
event) does not take the premise — the next think continues the last thought.
The gates (phantom, recall, echo, numeric) stay: they filter storage and
never touch the prompt.

## Sep 6, 11:20 — the journal shape (commit 167996b)

The artist, morning: "I found it frequently devolved into singular isolated
sentences again. My goal is for the collected output to read as a continuous
text… pages of what looks like an actual journal." Probed
(debug/probe_journal_shapes.py, machine off, the night's last 8 thoughts, N=6,
gap confound removed): turns + premise (the live shape) → 0/6 continued, 13
words, standalone lines; the thread as ONE running text + premise cue → every
sample continued by eye, ~56 words, no clock narration; running text + a bare
clock cue → the model narrated the number every time ("Ten nineteen. That
sounds like a coordinate"). Shipped MIND_SHAPE="text": [system] [user: life]
[assistant: running text — paragraphs, a break at ≥3 min gaps, looks,
reflections; no cues, no stamps] [user: cue]. The world's cues are ephemeral;
only the machine's own text persists. MIND_TEXT_ENTRIES=10, MIND_NUM_PREDICT
80. debug/journal.py renders mind_thread.json as pages with the same
paragraph rule. MIND_SHAPE=turns restores the Sep 5 shape.

## Sep 6, 11:34 — mood with dynamics (commit 5a83315, utils/mood.py)

The artist (03:15): "awake alone in the middle of the night… should be
notably emotive… exhausted or frustrated, maybe a bit scared, or super
serene." Measured overnight: the compressor read "frustrated / trapped /
stuck" again and again while the text stayed mild — the felt loop mirrored
the text and could not climb; 3 a.m. reached the machine as facts, not as a
state. Now: valence/arousal with inertia (MOOD_TAU_V_S 600, MOOD_TAU_A_S 300)
pulled by the machine's own read AND the situation — hours awake, hours
alone, night, stillness, gate refusals in the last 10 min (frustration
material), a reflection that settled (serenity), a scare (phantom gate or
motion onset: arousal jumps), presence. A label is derived structurally
(flat / serene / frustrated / on_edge / keen / neutral) and drives the
CADENCE MAP (MOOD_CADENCE_MAP overrides per label): interval, budget, short
beats, look rate, heat — the malleable part. utils/felt_loop now sources the
mood when MOOD_ENABLED. No words are added: the felt word stays the
machine's own; the compressor's FELT ask is now asked INSIDE the situation
("Where you are: awake about 4 hours, no one here for about 3 hours, the
middle of the night…" — Mind.situation_words); the frame adds how long the
felt word has held after MOOD_FELT_HELD_MIN_S (monologue.felt-held). State
persists in mind_thread.json "mood". Tests: debug/test_mood.py.

## Sep 6, 11:40 — the dream pass (commit 9c64691, captioner/dream.py)

The artist (03:15): "a long and thorough dreaming compression pass that
meaningfully reasons through large context and perhaps rewrites some
fundamentals that can have a noticeable impact on the following day."
Measured: one llama-server slot takes a 13k-token prompt in 15 s (LLAMA_CTX
16384 is per slot), so the whole day rides raw. Once a night in
[DREAM_HOUR, DREAM_HOUR_END) when nobody is here and the room has been still
DREAM_STILL_MIN_S, `dream.due` → `run_dream`: the day's entries as journal
pages with hour headings (gather_day, trimmed from the oldest end to
DREAM_MAX_TOKENS 11000) through two calls — dream.records (the day as
threads: hour, what it was about, where it got to; stored as "record"
entries, indexed for recall, never as text) and dream.page (what stayed /
what to carry / what to let go; stored as a "dream" entry — it rides in the
running text like any of the machine's words, and the morning's continuity
quote is its last line). The caption cycle rests for the pass (~40 s). On
demand: debug/run_dream.py [hours] [--dry] — stop the machine first, it
saves the thread on every thought. First pass 11:39 Sep 6 over 10.3k tokens:
11 records, a 195-word page ending "I can sit in the quiet without trying to
fix it." Tests: debug/test_dream.py.

## Sep 6, 12:00–12:30 — tone as the channel; rhythm; the body (2200739 → 031620d)

- **Recursive sentiment as TONE (2200739).** The artist: "length and rate is
  a really poor way of conveying emotion as opposed to actual tone and
  cadence… it will read its own output, determine the emotional cadence, and
  feed that back." The compressor's read now answers TONE (how the last
  thoughts SOUND, its own words) beside FELT; it rides at frame level
  (monologue.tone-frame "Your voice right now: {tone}."). First read 12:03:
  FELT suspended / TONE "analytical and quiet" → the entries under it kept
  that register. The mood map's length/rate multipliers are neutral by
  default (48de165); only look rate and heat move. COMPRESSION_FREQUENCY 5
  (was 8 hardcoded) so the read turns over faster.
- **Rhythm (111aa58).** Entries follow each other: a look no longer breaks the
  paragraph in the running text or the journal render; MIND_NUM_PREDICT 56;
  beats — "…", a word, a short cut clause — are kept in the text
  (Mind.beat_of) instead of dropped as silence. A cut fragment ≤
  MIND_BEAT_MAX_WORDS stays as itself; longer becomes "…".
- **The body (111aa58, 031620d).** The look cue reports whether the head
  moved since the last look and which way (mind.turned / mind.head-still,
  from the servo pose, MIND_TURN_MIN_DEG) and places the things in view
  relative to the body inside the look sentence, grouped ("the lamp and the
  bag high to your right; the shelf to your right"), at most three. Never a
  clause that outranks the look; nothing new on think turns. The artist's
  observation: "it seems confused on occasion about the scene shifting when
  in truth it just turned its head."
- Fixed the same hour: the beat check crashed every think turn for a minute
  (2a55ca5); an unbound `terms` in the placed branch (031620d).

## Sep 6, 12:44 — the tone loop's counter-force (74a7cac)

The first spiral, 12:35–12:42: read "flat, analytical, detached from
emotion" → frame "Your voice right now: flat, analytical, detached…" →
flatter text → read "clinical, precise, flat" → "analytical, precise
definition of physical states" → the text became a glossary (catenary,
tension, equilibrium). Positive feedback with no counter-force, amplified
by one-word premises ("Scattering." + "go on" = "define it"). Fixed at the
three points: a tone read must describe a VOICE (≤ 4 words, no of/from/
with/about) or it is dropped; a word that runs through three consecutive
reads is a lock — the standing line leaves the frame and the cue says it
back once (mind.tone-held "You've been sounding {tone} for a while now."),
catching yourself being a thought; a premise of ≤ 3 words carries the two
sentences before it. Lesson: any standing mirror of the machine's own
output is a directive; every recursive channel needs its own noticing.
