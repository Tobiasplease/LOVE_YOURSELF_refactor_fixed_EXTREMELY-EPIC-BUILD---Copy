# Presence stickiness after departure — diagnosis + build spec (Sep 4 2026, evening)

**Status: BUILT Sep 4 evening (see "Built" at the end); artist ruling: build now, restart when done.** Original header:

**Status: DIAGNOSED, NOT BUILT.** Artist verdict: "spot on", long-standing issue,
fixes below judged solid. Nothing here has landed in runtime code; the overnight
low-energy run (ca06ddfc, Qwen3.8-27B) is untouched. Build on the next session,
lands on the next restart.

## Symptom

The artist leaves the studio; for 15+ minutes the feed keeps saying "the man in
the grey hoodie is still hunched over the little red thing on his desk" — with
the artist's clothing — while the frames are empty.

## What the logs show (run ca06ddfc, 19:17–19:45)

- **Perception was right.** Check-glance requested 19:27:31 ("Turned to look
  where they were"); belief flipped OFF and "They've gone — the room's quiet
  again." reached the prompt at 19:28:26. No person_arrived / adjudication
  events after that. The Sep 4 verified-absence build (5300147) works.
- **The frames were empty.** Behind three "still hunched" captions: ceiling +
  top shelf (19:34:30), curtain gap (19:36:24), the room with the mannequin
  head (19:37:23). No person, no hoodie.
- **The mouth kept him alive until ~19:41.** From 19:28:30: sighted caption
  calls mentioned him 29/44, blind (inward-beat) calls 7/13. The three loudest
  "I can SEE him" lines (19:32:06, 19:33:43, 19:36:23) were blind calls.
- **19:34:30 was a recitation**, stitched from stream entries 14–16 back
  (19:29:58 / 19:30:05 / 19:30:14 — 5, 8 and 21 shared 6-word runs). The echo
  gate caught it (template_echo, spoken-not-stored) — but the FEED showed it.
  Same for the 19:36, 19:37 and 19:48 hoodie lines. The feed is the mouth, the
  stream is cleaner than the phone suggests.
- **Drift re-seeds him.** 19:45:20 drift (sighted, stored=True): "he's still
  hunched over it…" → into the stream AND the reverie ledger (34/40 reveries
  mention him). drift.presence rides only when belief is ACTIVE; with belief
  OFF and a him-filled stream, drift has no counter-fact.
- **The want was stale for 35 min.** "To stop waiting for the person to leave
  so I can begin" (formed 19:03, replaced by the 19:38 distill) rode as
  "Preoccupied with:" through the departure; no world-check on its premise.
- **Lore is NOT the carrier.** No thread line in the 19:34:30 call; the six
  lore threads contain no "him"; reveries reach only the reflection.

## VLM-alone probes (fresh context, no machine state) — debug/probe_presence_frame*.py

| Frame | Question | Model alone |
|---|---|---|
| 19:37 room, mannequin head at desk height | How many people? | "1 person… seated at a desk in the center" |
| same | What is the head on the middle shelf? | "a mannequin head" |
| 19:34 ceiling/top shelf | How many people? | 0 |
| 19:36 curtain gap | How many people? | none |
| any + ONE prior stream line about the hoodie | "Look again, say what you see" | copies the hoodie line verbatim |

So: one genuine false percept at ONE gaze position (dark-haired head + clutter
+ chair frame = seated person; YOLO+pose is not fooled, the caption VLM is),
plus stream carry-over everywhere else.

## Channel ablation — debug/probe_presence_ablation.py (replays the live
19:34:30 call in hybrid shape; 5 samples/condition; counts hand-judged)

| Condition | present-tense "he's here" | any mention |
|---|---|---|
| as live | 2/5 | 2 |
| durable self-facts removed | 1/5 | 3 |
| "Preoccupied with" want removed | 0/5 | 0 (a 3-sample first pass: 2/3) |
| drawing-provenance line removed | 3/5 | 3 |
| stream minus its 14 "him" entries | 0/5 | 0 |
| standing absence fact ADDED | 0/5 | 4, all past tense ("since he left", "now he's gone", "until he comes back") |

**The stream window is the belief.** Everything else is secondary. A standing
absence fact does not stop the model thinking about him — it fixes the TENSE,
which is exactly the memory/present-conflation doctrine's ask.

## Build spec (ordered by evidence)

1. **Standing absence fact while the stream still carries him.**
   `build_situational_line` is DELTA-only by design (edges, never standing
   state) — keep that, but add ONE corrective standing line that rides only
   while: presence belief OFF **and** any of the last N stored stream entries
   match the person regime (pronoun regex on the machine's own words — structure,
   not content). Text carries time: "He left {duration} ago; the room's been
   empty since." (duration from the belief's last-seen ts via
   casual_time_string; wording the artist's to finalize; registry key
   `caption.absence-standing`, used_by caption + caption_blind). Self-limiting:
   once the stream tail stops mentioning him the line stops. Dose: every call
   while the condition holds (it is a correction, not a restatement).
2. **Drift gets the same fact** in the inverse case (belief OFF + stream
   mentions him) — `drift.presence` currently covers only belief ON + frame
   person-empty. Drift is the organ re-seeding him into stream + reveries.
3. **Inward beat** (`elicit.introspective`, image deliberately dropped) — carry
   the absence fact via (1). Eyes-open for the inward beat is the artist's
   call (drift got eyes Sep 3); the fact is the cheaper first move.
4. **Feed marks spoken-not-stored.** `send_caption_to_display` gets a stored
   flag; the dashboard renders gated lines dimmed/marked. Observation-first,
   no new write-verbs.
5. **Want premise check.** On the belief OFF edge, if the current want mentions
   the person regime, mark it (arc-tail "…he's left since") rather than early
   distill (a distill costs a reflection). Sep 4 evening the want re-formed
   naturally at the next reflection; the mark covers the gap.
6. **The mannequin head.** At desk height among clutter it reads as a seated
   person to the VLM alone. Cheapest fix is physical (move the head or the
   chair). Software: enroll it as an effigy (effigy_memory has one Aug 17 box,
   not this one) and, when gaze sits on it, ride the machine's OWN registry
   word ("mannequin head", it named it itself) as a world-anchored line.
7. (optional) The stitched-recitation class: ANTI_ECHO_COMPARE_TAIL=8 lets
   6-word runs from 14–16 entries back pass. A wider tail at a longer n-gram
   (8) would catch stitches without the July "48% pass rate" cost. Measure first.

## Verification

- Offline: rerun `debug/probe_presence_ablation.py` with the new line in the
  saved prompt (expect ≤1/5 present-tense, mentions in past tense).
- Live: after a real departure, count present-tense him-captions in the 15 min
  after "They've gone". Tonight's baseline: ~10 in 15 min across sighted +
  blind calls; drift stored him at +17 min.
- Watch the feed marker: how much of what the artist reads is gated output.

## Files

debug/probe_presence_frame.py, debug/probe_presence_frames2.py,
debug/probe_presence_ablation.py (inputs in /tmp/probe_sys.txt,
/tmp/probe_user.txt, /tmp/probe_stream.json — regenerate from the event log
for another call). Untracked as of writing.

## Built (Sep 4 evening, commits 075fbfd → this one)

Live finding after the first fresh-stream restart (run 433e323b): two minutes
into a CLEAN window the sighted caption said "His head is down, chin almost
touching his chest, staring at the screen" (frame: the mannequin head at desk
height among clutter + the small red object) — mechanism 2 re-seeding exactly
as the probes predicted — and relational-mode captions had been logged after
verified absence in the earlier run (raw YOLO / gaze aware-tracking routed the
prompt relationally; both fire on the mannequin faces).

1. `build_standing_absence_line` + `caption.absence-standing` (departure
   anchored) + `caption.absence-standing-session` (no departure on record;
   ABSENCE_SESSION_MIN_S settle); `_presence_dropped_at` persisted across
   restarts (state_manager; restored only when the saved belief was OFF).
   Rides for caption, blind beat, drift. Onset/stop logged `absence_standing`.
2. `phantom_presence` storage gate (echo-class: spoken, not stored) — a
   present-tense third-person claim with the belief OFF; absence-marked
   mentions pass. Blocks the stream, the blind beat, drift AND the reverie
   ledger. PHANTOM_PRESENCE_GATE to disable.
3. Relational mode requires the adjudicated belief (`determine_prompt_mode`
   `believed=`).
4. `caption.desire-absent-tail` on person-premised wants after a verified
   departure.
5. Feed marker "[not kept] " on spoken-not-stored lines in live_captions.txt.
6. Clean-boot tooling: fresh_stream.py also clears recent_memory;
   scrub_phantom_presence.py removes present-tense third-person reveries.

Not built: a mannequin-specific line (the gate + the fact cover it; physical
fix still cheapest); the optional stitched-recitation echo widening.

Tests: debug/test_absence_standing.py (20), debug/test_phantom_presence.py (23).
Verify live: after a departure, `absence_standing` onset within a minute of
the first him-caption; `echo_spoken_not_stored` with reason `phantom_presence`
on mannequin-gaze captions; NO `relational` caption mode while the dashboard
shows person=absent.
