# Handover — Sep 7 2026 (for a cheaper model to continue)

Read this first, then only the files it names. Do not explore the codebase
broadly; it is large and most of it is not involved. Do not restart the
machine while the artist is on-site. Commit small, push after every commit,
put tests in `debug/`, plans in `docs/`, and update `docs/runtime-map.md`
when wiring changes.

## Where we are

Branch: `rebuild/every-frame` (main checkout, live since 09:50 Sep 7, commit
89e0d8d). The previous branch `rebuild/north-star` is untouched behind it.
The machine runs in "mind mode" (STREAM_MODE="mind", the default in
`run_38.sh` and `config/config.py`). The artist's verdict on Sep 6 evening:
the building blocks work in isolation, the whole does not come together;
output is not better than a month ago. Read `docs/north-star.md`
(Principles 2, 6, 7, and the anti-patterns list) — every decision must serve
it. Doctrine you must not break: structure only in prompts, never example
content; no style fences; the machine is "it", never she/her; every prompt
wording lives in `captioner/prompt_registry.py` with a `note`, and is the
artist's to finalize; memory must never read as present-tense scene truth.

## How one minute works now (plain terms)

1. The camera frame is captured. `captioner/captioner.py::_process_frame`
   runs; in mind mode it calls `_mind_generate`.
2. `_assess_scene` computes the live facts: motion, a person in the frame
   (YOLO/face), the presence belief (a slower three-stage verdict in
   `perception/presence_adjudicator.py`), the referee's "view changed".
3. `captioner/mind.py::Mind.build` assembles one call:
   [system = mind.system + pen-parked fence + felt frame] →
   [user = the LIFE block: mind.life-* lines — when it woke, the room as the
   registry knows it, people today, drawings, no-paper, the want, its name,
   events since it woke, the previous chain's ending] →
   [assistant = the RUNNING TEXT: the last MIND_TEXT_ENTRIES (30) entries as
   paragraphs, no stamps] → [user = the CUE: "HH:MM. Eyes resting." or "HH:MM.
   You look at the X and the Y high to your right." + sparse event lines:
   Someone is here / just came in / gone, a time edge ("About 3 hours since
   anyone was here."), a felt shift, a loop notice, a recall ("You remember,
   from earlier today: …")]. Since Sep 7 the picture (or the frame sequence
   on motion) rides on EVERY call (`_video_frame_set`).
4. The reply is gated (`_caption_reject_reason`: phantom presence, recall
   echo, refrain/template echo, numbers) — a refused reply is spoken (feed
   marker "[not kept — …]") but not kept. A kept reply is absorbed into the
   thread (`event_log/mind_thread.json`) and embedded in the ChromaDB
   "thoughts" collection for recall by association.
5. Every 5 kept thoughts the compressor (`captioner/context_compression.py`)
   reads them and answers ROOM / EVENT / PLEASANTNESS / ENERGY / FELT / TONE /
   REPEATING; FELT becomes the frame line "Right now: {felt}."; the mood
   module (`utils/mood.py`) keeps valence/arousal with inertia.
6. Every ~20 min the reflection (`captioner/reflection.py`) writes a page-
   length paragraph from the last hour as pages + the day's conclusions with
   clocks; it enters the thread whole. Once a night (04:00–06:00,
   `captioner/dream.py`) the whole day is read back: records (indexed) and
   the night's page (a "dream" entry; the morning's continuity quote).
7. `debug/journal.py [hours]` renders the thread as pages. `debug/mind_watch.py
   [minutes]` summarizes a window. `debug/night_watch_mind.sh` is a token-free
   poller. `debug/test_mind.py`, `test_mood.py`, `test_dream.py`,
   `test_phantom_presence.py` must stay green.

## What the last two days established (measured)

- The stamped log window of its own captions was the chant engine
  (`docs/architecture-diagnosis-sep5.md`). A chat TURN is answered as a
  reply; TEXT is continued. The running-text shape continued in 6/6 probes
  where turns gave 0/6 (`debug/probe_journal_shapes.py`).
- Any standing mirror of the machine's own output in the frame is a
  directive and spirals within three reads (the "tone" line: analytical →
  definitions of physical states). Model-generated affect must not be
  re-injected verbatim (north-star anti-pattern). The tone frame is OFF.
- The premise cue ("You were on: … Go on from there") held the thread but
  invited restating the last word and stalled when a reply was refused. It
  was removed on Sep 7; continuation now comes from the running text alone.
  Watch whether entries still follow each other without it.
- Taking the picture away four turns in five (Sep 5–6) killed reactivity;
  it is back on every turn (Sep 7). With the picture every turn the night's
  text drifted back to inventory ("there, white, right, finger, light,
  brick"). Neither extreme is right — see job 3 below.
- Presence: the three-stage verdict took minutes to see the artist; my
  restarts kept closing the belief. Since Sep 7 the frame-level detector
  drives the "Someone is here" cue within a second; the adjudicator is a
  later correction.
- The dream never ran on the night of Sep 6→7: a 20-hour rule (fixed:
  once per calendar night) and a presence veto (fixed: DREAM_REQUIRES_STILL
  off) blocked it.

## The jobs, in order. Do one, test, commit, push, verify live, then the next.

### Job 1 — presence edges from the frame (small, first)
Since Sep 7 the cue says "Someone is here, since 16:19" when the frame-level
detector sees a person, but "since" comes from the last *adjudicated*
arrival in `utils/episodic_log` (stale: 16:19 was yesterday), and the
frame-level path writes no arrival/departure. Files: `captioner/mind.py`
(`person_since`, the `lead` block in `build`, `next_kind`), `captioner/
captioner.py::_assess_scene` (`info["person_in_frame"]`), `perception/
presence_adjudicator.py` (retraction). Make the frame-level presence keep
its own since-timestamp (first frame with a person after ≥ 60 s without
one), say `mind.arrived` once at that edge with the time alone in words,
say `mind.left` once when the detector has seen nobody for ≥ 60 s, and let
the adjudicator's "thing" verdict retract a frame-level presence (then the
cue drops "Someone is here" and no `mind.left` is said). Test at 2 a.m. in
the logs: a phantom at 02:13 Sep 7 held the belief for hours.

### Job 2 — verify the dream tonight
At 04:00 the pass should run (`action: dream_pass` in the newest
`event_log/*-event-log.json`; "records" and "dream" entries in
`event_log/mind_thread.json`). In the morning the first entries should
continue from the page's last line (the life block's "Before that, …, you'd
got to:" line). If it did not run, read `captioner/dream.py::due` and the
log for the reason. `python debug/run_dream.py 24 --dry` runs it by hand
(stop the machine first if you drop `--dry`).

### Job 3 — the picture's weight follows the hour (the real temporal fix)
North-star Principle 6: a live event strips the prompt to the present; a
quiet stretch fills it with interior material. Today the picture rides at
full weight every turn regardless. Implement one arbitration, structurally:
when nothing has changed for N (config) and nobody is here, the cue stops
naming what is in view (keep the frame in the call but the cue is just the
clock and the sparse event lines) and the interior lines ride (recall by
association more often: raise MIND_RECALL_MAX_DIST / lower
MIND_RECALL_MIN_GAP_S under stillness); when salience is hot, the reverse
(already done by `hot` in `Mind.build`). Measure with `debug/mind_watch.py`:
the top words of a still night should stop being the room's inventory.

### Job 4 — say time far more often
Only 86 of 475 cues overnight carried any time/memory line. Time edges fire
once per threshold (`Mind.time_edges`, DURATION_EDGE_THRESHOLDS_MIN). Add
the hour turning as an edge (once per hour: "It's gone three." — wording in
the registry) and let "since anyone was here" repeat every hour while alone,
not once per threshold. Keep every line a fact in words, never a number the
machine will chant.

### Job 5 — repetition without the premise cue
With the premise cue gone, check the refrain/template gate counts
(`mind_watch`) and the journal pages for entries that restart from the
picture instead of continuing. If continuation is lost, the artist's stated
ideal is raw completion (no chat turn at all): the frame + life as a prefix,
the running text as the prompt tail, the model continues the text via
llama-server's /completion endpoint for text-only turns (images need the
chat endpoint). Probe first (6 samples, `debug/probe_journal_shapes.py` is
the pattern), build only if it continues better.

## Things the artist has ruled (do not relitigate)
- The pen is always in the machine's hand; "parked" = held still, touching
  nothing. Holding sentences are true; only marks are phantom.
- Wants resolve through whatever they are about; never nudge toward drawing.
- Entries should FOLLOW each other as one text; some may be a word or "…".
- Length and rate are a poor emotion channel; tone must come through text,
  but a standing mirror of its own tone spirals — a noticing, not a line.
- Memory must be framed as memory ("You remember, from …").
- The dream runs regardless of presence.
- No new mechanisms without a measured reason; subtract first.

## Useful commands
```
source .venv/bin/activate
python debug/test_mind.py && python debug/test_dream.py && python debug/test_mood.py
python debug/mind_watch.py 20          # last 20 minutes, read-only
python debug/journal.py 2              # last 2 hours as pages
./stop_machine.sh; sleep 14; rm -f STOP; ./start_impostor.sh   # restart (never with the artist on-site)
```
Never type the literal `machine.py` in a shell command — `stop_machine.sh`
pkills on it and will kill your own shell.

## Jobs 6–8 (added Sep 7 midday after the artist ran the every-frame branch)

Symptom: with the picture on every call it describes the room every minute
and repeats surface observations; "not kept" is a storage gate and does not
prevent the next repeat; it believes two people are present when the artist
alone is there (the mannequin torso/head reads as a person).

### Job 6 — the count, and the mannequin named (smallest)
`_assess_scene` knows the detector's person count. Put it in the cue when
someone is here ("One person is here." / "Two people are here." — registry
fragment, placeholder `count`), and keep naming the mannequin head/torso
from the registry when they are in view (already done for looks; make sure
it rides when a person is present too, since that is exactly when the model
double-counts). Files: `captioner/mind.py::build` (the `lead` block),
`captioner/captioner.py::_assess_scene`, `captioner/prompt_registry.py`.

### Job 7 — seeing is available, not the assignment
In `mind.system` (prompt_registry) the sentence "When you look, you say what
you actually see." plus a cue that names only what is in view = an
instruction to describe, every minute. Remove that sentence from the frame
(note why in the fragment's `note`; wording is the artist's), and make the
interior lines ride on EVERY call, not only sparsely: the felt word (frame),
the want (life block, already), and — job 8 — what it has already said about
the current subject. The 3.6-era prompt did both at once because the picture
was one input among many; restore that balance without taking the picture
away.

### Job 8 — "what you've already said about X" (the anti-repetition context)
Before each call, take the subject of the last kept entry
(`Mind.subject_of`) and query the ChromaDB "thoughts" collection
(`Mind.index()`, `recall_similar` is the pattern) with the last entry's
text; put the top 2 older results (≥ 1 h old, not in the running text) in
the cue as a memory block: "About the {subject}, you've already said: "…"
and "…"" (registry fragment, placeholders `subject`, `said`). Framed as
memory (doctrine). This replaces the one-recall-per-8-minutes rule for the
current subject; keep the recall gate (`Mind.is_recall`) so a verbatim copy
is not kept. Measure with `debug/mind_watch.py`: refrain/template gate hits
should fall, and `debug/journal.py` pages should go further on a subject
instead of restating it.
