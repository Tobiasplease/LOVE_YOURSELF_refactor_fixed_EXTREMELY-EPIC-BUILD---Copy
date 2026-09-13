# Handoff for the next session (written Sep 13 2026, ~12:30, by Fable 5.1)

You are picking up live work on Tobias's (they/them) drawing-machine installation
"LOVE YOURSELF". The machine is RUNNING while you work. Read this whole file
before touching anything, then `docs/runtime-map.md` (source of truth for what
is live) and `docs/where-we-are-sep9.md` §24–§33 (the running record; add §34+
as you go). Older plan docs in `docs/archive/` are not to be trusted.

The artist has approved the seven items in §4 ("Yes this sounds good"). Do them
ONE AT A TIME in the order given, each measured live before the next, and ask
for the go before committing each one. Wording of any prompt fragment is the
artist's to finalise: propose it, mark it as a proposal, don't decree it.

---

## 1. Standing rules from the artist (verbatim where quoted — do not relax)

- "I absolutely do not want to bake in anything specifically tailored to this
  room."
- "I do not want the caption interval to shift at all… Stillness and 'nothing
  to say' should be a choice by the model, not imposed." (CAPTION_INTERVAL_FIXED
  = 8 s stays.)
- Gates and filters are band-aids for a prompting problem; fix upstream. "Echo
  is not continuation, in fact it's the opposite."
- "No call should be without visual information." Inward beats still carry a
  picture; they "prioritise the internality of the machine without omitting
  the space around it."
- "The appropriate data should reach every single call."
- No explanatory prompting that tells the model what it should intrinsically
  know (e.g. "the slight wobble between the frames is your own breathing" was
  rejected: "an awkward workaround to something the model should intrinsically
  know").
- "'He's come in' isn't very good. This wasn't me, it was a different person."
  and "gendering overall is not optimal because well you never know do you."
- "Things out of the ordinary need to have a lot more weight in the memory…
  the rarity should also determine the significance at the time of discovery."
- "Tedium is material and repetition is in and of itself an event. But we are
  missing something in the architecture to truly convey this framework."
- One variable at a time, measured live. Commit only on an explicit go.
- Restarts cost the exhibit (the head moves constantly after boots; the artist
  objected to eight restarts in an afternoon). One restart per approved change,
  at most, and say why. Probes against the live llama-server also cost the
  machine; prefer reading logs.

## 2. Operating the machine (learned the hard way — follow exactly)

- Repo: `/home/impostor/LOVE_YOURSELF_refactor_fixed_EXTREMELY-EPIC-BUILD---Copy`.
  Always `.venv/bin/python`, never the system interpreter (svgwrite missing).
- Run `date` before any claim about "now" — the previous session's sense of
  time drifted six hours once and the artist noticed.
- Stop: `touch STOP; pkill -INT -f '^python machine'`, then WAIT for the line
  `STOP present — supervisor loop ended` in `/tmp/start.out` (`timeout 120
  grep -m1 'supervisor loop ended' <(tail -n0 -f /tmp/start.out)`) BEFORE
  `rm -f STOP`. Removing STOP inside the old supervisor's 5-second restart
  window killed the tmux server on Sep 13 11:20.
- Launch: `nohup setsid -f ./start_impostor.sh </dev/null >/tmp/start.out 2>&1`.
  Verify: `pgrep -af '^python machine'`, `tmux ls`, `curl -sf localhost:8080/health`.
  Machine stdout is in the tmux pane: `tmux capture-pane -p -S -600`.
- NEVER `stop_machine.sh`, `pkill -f machine.py`, `pgrep -f llama-server` or
  any `-f` pattern that appears in your own command line — it matches your own
  shell (this killed the agent's shell twice). Anchor (`^python machine`) or
  use `pgrep -x`.
- The llama-server (`~/llama.cpp-38`, Qwen3.8-27B, stock upstream build) is an
  adopted orphan across machine restarts and its CUDA pool grows over a run
  (18.9 → 22.5 GB seen in 8.6 h). It is unloaded/reloaded fresh at every
  drawing. To restart at baseline: stop the machine, `kill <server pid>`
  (get the pid from `nvidia-smi --query-compute-apps=pid,process_name
  --format=csv`), then launch. YOLO now copes with a full card on its own
  (perception/object_detection.py, Sep 13) but starts on CPU until room appears.
- Importing `captioner.captioner` from a test mints a stub run log: sweep them
  with `mv` into `event_log/archive-stub-runs/` (a stub is a
  `*-event-log.json` under 30 kB with 0 captions).
- Logs are JSONL: `event_log/<run>-event-log.json`, newest = current run
  (`ls -t event_log/*-event-log.json | head -1`). Entry `type`s you will read:
  `caption` (fields caption, mode, iso_timestamp, close_look, salience_hot),
  `llm_api_call` (prompt_type caption/drift_turn/reflection/…, prompt,
  system_prompt, response, duration_s), `debug` (action: attention,
  presence_belief, caption_request, glance_start, repeat_rerouted,
  loop_notice, …), `error`, `grbl`, `compression`. Server log:
  `event_log/llama_server.log`.
- Tests are standalone scripts in `debug/` (no framework). Run the ones
  touching what you change; all of these passed at handoff:
  test_stretch_line, test_standing_facts, test_time_and_loop, test_event_memory,
  test_attention, test_phantom_presence, test_absence_standing, test_agency_round,
  test_stream_gaps, test_format_strip, test_native_video, test_vram_wait,
  test_yolo_device_policy, test_yolo_cpu_fallback.
- Commit message style: lower-case imperative title, a body that says what was
  seen live and what changed and why, ending with
  `Co-Authored-By: Claude <the model you are> <noreply@anthropic.com>`.
  Do not commit the artist's own uncommitted files (`debug/test_hearing.py`,
  `debug/analyze_room_audio.py`, `debug/measure_stroke_width.py`,
  `debug/preview_weighted_strokes.py`, `warp-fix-lab/`, `.claude/settings.local.json`).

## 3. State at handoff (Sep 13 12:30)

- Branch head `fd3217b`. Last three commits: the stretch line (158337f), the
  YOLO device policy (750746a), docs (fd3217b). Working tree clean apart from
  the artist's files above and this handoff.
- Machine booted 11:43:32; server at 18.9 GB; YOLO on CUDA; stretch line on
  ~88% of thought calls; digit-duration caption openings 1/144 (was 41/103 the
  run before — see runtime-map "Stretch line, first quarter-hour" and the
  second-run note). One change is committed but NOT yet live: the noun-phrase
  check in `captioner/prompts.py::_nounish/_phrase_ok` (the live process still
  says "named i'm looking sixteen times"). It takes effect at the next restart;
  do not restart for it alone — fold it into the first restart you need anyway.
- How the thought call is built (know this before editing prompts): system →
  assistant message of the last 23 stamped lines ("HH:MM — text") → user turn
  (situational lines + standing facts + picture or 4-frame clip) → assistant
  prefill = last sentence of the newest entry. Standing facts on every call
  (`prompts.build_standing_facts`): stretch line, last-event line, head line,
  felt line. Digits anywhere in those lines seed a digit habit that the window
  then propagates — keep every standing line digit-free (`casual_time_string`,
  `count_words`, `_num_word`).

---

## 4. The approved items, in order

### Item 1 — the visit line ends "No one has been here since." when belief is off

**Why.** Sep 12 19:01: the last-event line "Earlier: The man left… piece of
paper. That was about an hour ago, …" primed "and there he is, sitting on
that chair" with nobody there (the phantom-presence gate caught it, the artist
asked why it is believed at all). A restart had emptied the stream so no
absence line was there to contradict it. The standing fact must close the
door it opens.

**Where.** `captioner/prompts.py::build_last_event_line` (line ~784). It already
returns "" while `agent._presence_believed` is True for a visit. When it does
emit a VISIT_KIND line, append a closing sentence.

**Do.**
1. Registry (`captioner/prompt_registry.py`, next to `caption.last-event`):
   new key `caption.since-empty`, proposal text `"No one has been here since."`,
   with a `note` citing the 19:01 phantom. Mark it PROPOSAL in the note.
2. In `build_last_event_line`: if `ev["kind"] == _em.VISIT_KIND` (belief is
   necessarily off here), return `line + " " + P("caption.since-empty")`.
   Guard: only if no later `person_arrived` episodic event exists after
   `ev["ts"]` (use `event_memory._episodic_events(["person_arrived"])`); if one
   exists the visit is not the last presence and the sentence would lie.
3. Test: extend `debug/test_event_memory.py` — a completed visit with belief
   off carries the sentence; with belief on the line is empty; with a later
   arrival the sentence is absent; no digits.
4. Measure over ≥2 hours after the restart: phantom_presence gate hits
   (`debug` entries with `action == "phantom_presence"` or the `GATE` lines
   the older watch scripts printed), "there he is"-type captions with no
   presence belief, versus the Sep 12 evening (one phantom at 19:01, several
   on Sep 12 19:29–19:31 during a departure).

### Item 2 — no "He" anywhere the machine is handed words

**Why.** The artist's rule above. `presence_who` (prompts.py ~776) still returns
"He" when re-identification says the arrival is familiar; the colleague's visit
on Sep 12 was greeted "He's back" on a false familiar match, with one
mis-gendering downstream.

**Where.** `captioner/prompts.py::presence_who`; registry keys
`caption.arrival-back` ("He's back."), `caption.arrival-back-rare` ("He's back —
the first time in {gap}."), `caption.absence-standing` ("{who} left {when}; the
room's been empty since."), and `drift.presence` (`{who}`) used in
`captioner.py::_run_drift_turn` (~line 325). Grep for the rest:
`grep -rnE "\"He|'He|He's|He’s|\bhe\b" captioner utils perception --include=*.py`
and check each hit is a comment or the artist's quote, not handed to the model.

**Do.**
1. `presence_who` returns `"They"` for familiar, `"Someone"` otherwise (keep
   the familiar/stranger distinction; it feeds "back" vs "come in").
2. Proposals: `caption.arrival-back` → "They're back."; `-rare` → "They're back
   — the first time in {gap}."; `caption.absence-standing` with who="They" reads
   "They left a few minutes ago; the room's been empty since." — check the
   `{when}` grammar still works with "They"; `caption.departure` already says
   "They've gone". Keep `caption.arrival-someone` as is. `drift.presence` is
   `"{who}'s here, just out of view right now."` — "They's" is wrong, so give
   it a `{who_is}` slot ("They're" / "Someone's") or split the key; grep every
   `{who}` use for the same contraction problem.
3. `event_memory._own_words` keeps the machine's OWN past words even if they
   say "the man" — those are what it saw and said; do not rewrite history.
4. Tests: `debug/test_absence_standing.py`, `debug/test_phantom_presence.py`,
   `debug/test_event_memory.py` — update expectations; add a grep-style test
   that no registry `text` handed to the model contains `\bHe\b|He's`.
5. Measure: captions using he/him for a visitor in the next visit, and whether
   "They're back" reads naturally in the feed (ask the artist).

### Item 3 — a walk-past is an event of its own (lower tier)

**Why.** Sep 11 ~22:00–23:00 someone walked past and left no trace: presence
belief needs the adjudicator to confirm a seated/standing person; a pass never
reaches it. Artist: "There should be a way to differentiate a consistent world
model from a truly novel event." A pass is rarer than most things that happen
in that room and should be remembered with weight scaled to its rarity, but
below a real visit.

**Where.** YOLO person tracks → `perception/detection_memory.py::DetectionMemory`
(`get_person_count`, `best_track_id`); the frame buffer carries
`detection.person_count` per frame (`captioner/frame_buffer.py`, read in
`captioner.py` ~line 689); belief ON/OFF is decided in `captioner.py` around
lines 900–1010 (`presence_belief` debug entries); `machine.py` ~1531 zeroes
`person_count` for the machine's own body and the effigy — a pass must use the
same zeroed value. Episodic events (`person_arrived`, `person_left`,
`world_changed`) are written through `utils/episodic_log.py` and read by
`captioner/event_memory.py::last_event` / `_episodic_events`.

**Do.**
1. Detection of a pass, in the captioner (not YOLO): a run of frames with
   `person_count ≥ 1` (already skeleton-gated and own-body-zeroed) lasting
   ≥ PASS_MIN_S (proposal 2 s) and ending (no person frames for PASS_END_S,
   proposal 5 s) WITHOUT presence belief having turned ON during it → write an
   episodic `person_passed` event with start/end ts. If belief turns ON, it is
   a visit; drop the pass. Configurable in `config/config.py` under a
   `PASS_EVENT_*` block with comments.
2. `event_memory`: a `PASS_KIND`; `last_event` considers it; lifetime a shorter
   factor than visits (proposal 0.1 × gap, floor 300 s, cap 1 h — the
   existing constants are `EVENT_MEMORY_*` with factor 0.25, 600 s, 6 h);
   `rarity_phrase` for it; `event_words` uses own words from the pass span if
   any (`_own_words(start, end, must_match=_PERSON_WORDS)`), else a plain
   phrase. Registry proposals: `caption.last-event-pass` → "Earlier: someone
   passed through — you didn't get a proper look. That was {age} ago,
   {rarity}." and a discovery cue `caption.pass-cue` → "Someone just went
   past." injected once by `build_situational_line` when the pass ends (the
   sticky presence-edge block there is the pattern; PRESENCE_EDGE_STICKY_S).
   A visit's last-event line outranks a pass; a pass outranks a `world_changed`
   only if rarer (compare `rarity`/gap, don't hardcode).
3. Room attention (`captioner/attention.py`): a pass bumps attention like
   "salience with presence" (snap or +0.6), so the picture grows while it is
   fresh.
4. Tests: `debug/test_event_memory.py` (a pass; a pass swallowed by a visit;
   ordering vs a change) and a small state-machine test for the pass detector
   with faked frame meta (see `debug/test_phantom_presence.py` for how frame
   meta is faked).
5. Measure: log every pass with its duration; over a day compare passes
   logged vs. what the artist knows happened (they will tell you). False
   passes (the machine's own arm, the mannequin) must be zero — if not, the
   fix is upstream in the skeleton gate / own-body logic, not a filter here.

### Item 4 — thread write-back through the distill slot, and pruning

**Why.** The harsh read (Sep 12 19:00–01:24): lore threads returned to: 0 of
247. The machine spawns threads (reveries, lore, open questions in
`utils/lore_ledger.py`) and never comes back to one. Nothing records that a
thread was taken up, so nothing can compound.

**Where.** `utils/lore_ledger.py` (`note_lore`, `alive_threads`, `pick_seed`,
`note_question`, `pick_question`, `open_questions`); the reflection distill in
`captioner/reflection.py` ~563 calling
`captioner/context_compression.py::distill_reflection` (~795) → TRAIT / BELIEF
/ … / QUESTION slots. The QUESTION slot ("or none", harvested into the ledger
— registry note at ~315) is the exact pattern for the new THREAD slot: copy it;
`_run_drift_turn` in `captioner.py` (seeds a thread with
`drift.lore-seed`); `prompts.get_lore_line` (~1861, one thread's arc line).

**Do.**
1. Ledger: each thread gets `offered` (count, last ts) and `returns` (list of
   {ts, source: drift|reflection|caption, advance: ≤ 25 words}). `pick_seed`
   and `pick_question` prefer threads with returns (compounding) but rotate
   through the least-recently-offered among them; a thread offered
   ≥ THREAD_PRUNE_OFFERS (proposal 3) times with no return becomes `dormant`
   (kept in the file, never offered again; a caption that spontaneously
   contains its content words revives it).
2. Write-back sources: (a) the reflection distill gains a slot `THREAD —
   <which earlier thought this continued, in a few words, or "none">` and one
   sentence of `ADVANCE`; map it to a thread by content-word overlap
   (`lore_ledger._content_words`) ≥ 2 words, record a return; (b) a drift
   turn that was seeded with a thread records a return with the drift's first
   sentence as the advance; (c) a stored caption whose content words overlap
   an alive thread ≥ 3 words records a `caption` return (cheap, no call).
3. `get_lore_line` (the arc line back into the voice) prefers a thread with
   returns and states the ADVANCE, not the seed — "lore must never read as
   observation" (its docstring).
4. Tests: `debug/test_lore_ledger.py` (new): offers/returns/prune/revive; the
   distill parse with and without THREAD; the caption overlap return.
5. Measure over a night: threads returned to ≥ 1 / threads offered (was 0/247);
   reflections whose distill named a thread; whether the machine says
   anything that reads as "coming back to" a thought.

### Item 5 — the thread-anchored drift ask (a question, not a statement)

**Why.** From the reviewed plan: "a live thread must be present by QUESTION,
not by statement, or it is the next refrain; the drift turn already has the
shape for the reflection's ask." Today `drift.lore-seed` is `You've been coming back to this: "{text}"` — a
statement (and a false one: it has never come back to anything); the model
then echoes it.

**Where.** `captioner.py::_run_drift_turn` (~325–402): builds `ask` from
`drift.ask`, optionally `drift.presence`, the absence line, `drift.lore-seed`
with `seed["text"]`, then standing facts; system `drift.system`; history is the
unstamped window (`_stream_history_unstamped`). `_absorb_drift_text` stores
the result and calls `_note_act("drift", …)`.

**Do.**
1. When a seed exists, use the thread's own open question if it has one
   (`lore_ledger.pick_question` — check it can be scoped to the seed thread;
   add `question_for(thread)` if not) and phrase the ask as a question about
   the thread. Proposal for `drift.lore-seed`: "Earlier you had a thought:
   {text}. Where does it go from here?" — artist to finalise. Never the seed
   text alone.
2. Item 4's return write-back records the drift's advance on that thread.
3. The drift must still carry the picture (rule: no call without visual
   information). `_run_drift_turn` sends `image=(self._sized_for_model(img_path)
   if DRIFT_SEND_IMAGE else None)` — confirm `DRIFT_SEND_IMAGE` is True in
   `config/config.py` and that the caller passes `img_path`; if either is off,
   that is a bug to fix first, not a setting to leave.
4. Test: extend `debug/test_agency_round.py` or add `debug/test_drift_ask.py`
   with a faked ledger: the ask contains a question mark and the thread's
   words; no stamps; no digits.
5. Measure: drift responses that restate the seed (content-word overlap ≥ 60%
   with the seed sentence) vs. advance it; drift responses that come back as
   stamped log lines (was 0/40 after the Sep 12 fix — must stay 0).

### Item 6 — tedium as pressure, with discharge

**Why.** Artist: "tedium is material and repetition is in and of itself an
event… we are missing something in the architecture." The stretch line (done)
makes the machine hear its repetition. What is missing: repetition accumulating
into something that has to go somewhere. Today boredom scores only shape
sampling (`captioner/activation_memory.py`) and attention only shrinks the
picture; nothing builds and nothing releases.

**Design (mirror `captioner/attention.py`, which the artist approved as the
pattern: a scalar, plugs, measured).** `captioner/tedium.py`, `Tedium` with
`value ∈ [0,1]`:
- Rises per thought call while the room is unchanged (the stretch line's
  `unchanged_s` is running) by an amount proportional to repetition in that
  call: the newest caption repeating the current top phrase (`_top_phrase`),
  a `repeat_rerouted` debug entry (reasons seen live: refrain_echo,
  template_echo, number_chain), a head turn that came back to a view already
  verified "unchanged". No rise while
  presence is believed or attention ≥ ATTENTION_CURIOUS (0.5) — novelty and
  tedium are exclusive.
- Discharges (drops by a configured fraction, not to zero) on: a drift turn
  taken, a reflection, a rare event (visit, pass, verified change), a
  drawing intent, and a chosen silence of ≥ 2 cycles in a row. Slow decay
  otherwise (τ ~ 30 min).
- Plugs (these are the whole point; each is a choice offered to the model,
  never a forced action, per the interval rule):
  1. `_drift_due` (captioner.py ~298): the drift comes due sooner as tedium
     rises (interval × (1 − 0.6·tedium)). Discharge follows.
  2. `build_decision_ask` (prompts.py ~1475): at tedium ≥ 0.6 the ask names
     the pressure and offers the exits in the machine's own terms — say
     nothing, look away, follow the thought — proposal wording, artist's call.
     Keep it under two lines; no explanation of what tedium is.
  3. Reflection `_should_reflect` (reflection.py ~80): tedium ≥ 0.8 shortens
     the quiet-time requirement for a reflection (the 01:18 reflection was the
     one interior text of that night).
  4. `get_felt_arc_line` / the felt line: may carry tedium in words only if
     the artist wants it there — ask before wiring.
- Log it: `debug` entry `action: "tedium"` every call with value and cause,
  like `attention`.
- Config block `TEDIUM_*` in `config/config.py` with a comment per constant.
- Tests: `debug/test_tedium.py` (rise/hold/discharge/decay; exclusivity with
  attention and presence; the drift-due plug).
- Measure over a night versus the Sep 12 baseline in §33 (1,498 captions:
  room object 74%, finger 30%, beyond-room 6.5%, silence chosen 597×, 0/247
  threads returned): top-phrase repetition run lengths (should shorten after
  discharges), drifts per hour and their timing relative to tedium peaks,
  silence-chosen share, reflections per night, beyond-room share.

### Item 7 — the finished-drawing photos in the dashboard

**Why.** The paper is photographed after each drawing
(`event_log/finished_drawings/finished_<ts>_<n>.jpg`, plus `_sheet.jpg` = the
cropped sheet and `_sheet_t1024.jpg` = the model-sized copy) but the Drawings
tab only lists `.png` from `~/ComfyUI/output` (`dashboard/server.py` ~279 scan
and ~524 single-file route). The artist asked for them on the mobile UI.

**Do.**
1. `dashboard/server.py`: `GET /api/drawings` ledger gains `finished: [{name,
   mtime, url}]` from `event_log/finished_drawings/*_sheet.jpg` (fall back to
   the un-suffixed jpg if no `_sheet`); a route `/drawings/finished/<name>`
   serving only names matching `^finished_\d{8}_\d{6}_\d+(_sheet)?\.jpg$` from
   that folder (no path traversal; 404 otherwise).
2. The Drawings tab lives in the single-file UI `dashboard/index.html` (no
   static dir; find the JS that renders the png list): show the finished photo
   next to the render it came from (match by timestamp order; the ledger has
   the drawing timestamps), newest first, sized for a phone.
3. The dashboard is its own process (`dashboard/start_dashboard.sh` /
   `stop_dashboard.sh`, `dashboard/server.py`, `machine_api.py`); read both
   scripts before running them (they may pkill by pattern), then restart only
   the dashboard — the machine is untouched.
4. Test: a request test with a temp folder (`debug/test_dashboard_finished.py`).
5. Note in runtime-map under the dashboard section. The photos are cut off by
   the wooden shoulder (camera placement, mechanical) — not a software bug.

---

## 5. Working protocol (per item)

1. Read the code paths named above; verify line numbers with grep (they drift).
2. Write the change with a docstring that cites the date and the artist's
   reason, as the codebase does everywhere ("Sep 13 (artist: …)").
3. Run the relevant `debug/` tests; add one for the change.
4. State the measurement BEFORE restarting: which numbers, against which
   baseline (§33 and runtime-map hold the baselines).
5. One restart (fold in anything committed-but-not-live), the sequence in §2.
6. Watch ≥ 15 min with a background script reading the run log (the pattern:
   `sleep 900` then a python summary of captions / prompts / debug actions),
   then report numbers plainly, including what did not improve.
7. Ask for the go, then commit that item alone. Update `docs/runtime-map.md`
   (a bullet in the Sep 13+ wiring section) and `docs/where-we-are-sep9.md`
   (a new numbered section per item or per day) in the same commit.
8. Sweep stub run logs your tests minted.

## 6. Things not to do

- No new gates or strippers to fix a wording problem (the coordinate storm,
  the countdown prefix, the AM/PM orphan were all upstream problems first).
- No change to the caption interval, the picture-on-every-call rule, or the
  stamped window without the artist's explicit ask.
- No prompt text that explains the machine to itself.
- No digits in anything handed to the model as a standing fact.
- No `He`/`She` for anyone the camera sees.
- No restart bursts; no probes against the live server; no `pkill -f` with a
  pattern in your own command line.
