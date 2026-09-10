# Where we are — stock-take after the phone sessions (Sep 9)

Sep 4 17:16 – Sep 8 15:14 was driven from a phone over Remote Control
(session `impostor-bot-warm-lemur`, 213 artist turns, 170 git operations).
It ended on a session limit at 13:14 on Sep 8, mid-probe, not on a conclusion
— which is why the state has felt unaccounted for. This document is the
account. Everything below is measured, not remembered.

## 1. Where the tree actually is

Three branches came out of Sep 5–8. They fork at one commit:

    30e610c  Sep 5 18:43  architecture diagnosis: the self-window is the chant engine
       ├── rebuild/every-frame   (+84 commits, Sep 5–8 08:19) — mind mode. PARKED.
       └── rebuild/pre-mind      (+10 commits, Sep 8 09:12–15:14) — CURRENT, running.

`rebuild/pre-mind` was cut on Sep 8 09:12 from the *diagnosis* commit — after
the problem was named, before mind mode was built on it. So the current build
is the year-tuned hybrid architecture plus ten commits of fixes ported or
newly found. `STREAM_MODE="hybrid"`, `SEAM_MODE="sentence"`.

The 84 mind-mode commits are not lost. They are on `rebuild/every-frame`,
intact, with their own docs (`mind-mode-sep5.md`, `architecture-diagnosis-sep5.md`).
Nothing needs rescuing; it needs deciding.

**Running now:** `machine.py` pid 44342, up since Sep 9 00:49 (14 h 43 m),
llama-server on Qwen3.8-27B. Working tree is clean apart from
`.claude/settings.local.json` and `grbl/grid_50x50.svg`.

## 2. Why the branch was cut (the call that was actually right)

Sep 8 06:41, after a night of frantic patching: *"I am afraid we've made a mess
of things again… Too much frantic changing since the 'upgrade' to 3.8. You keep
claiming you cut off channels that were important."* Then 07:02: *"the prior
system has been tuned and poked at for over a year… This one I don't really
[understand], and it's yielded worse results… fundamentally there is an
overview lacking."*

That judgement holds up. Mind mode was a coherent design, but it was built and
debugged in the same 72 hours it was invented, on top of a model change (3.6 →
3.8) that had not been characterised. Two variables moved at once. Reverting to
the architecture you understand and porting only the *proven* fixes forward was
the correct move, and it is what the ten commits do.

## 3. What worked — with the evidence

**The phantom-person leak is closed.** This was the top-priority defect
(*"it ruins the caption thread more or less completely"*, Sep 7 20:41). Three
findings landed, in order:

- `bdc8b0b` — **proven by ablation**, not argued: fed clean text, the model
  claims a person 0–1 times in 9; fed its own contaminated text, 9 in 9. The
  picture is innocent. The running text is the belief. This is the single most
  valuable result of the whole stretch — it moved the person problem off the
  vision model, where it had been hunted for weeks.
- `654cdca` / `44fbb9f` — **the compressor door**: a refused caption was spoken
  but still fed the memory diff, so a gated misread came back as `ROOM: A person
  sits at a desk…` in every subsequent prompt. Spoken-not-stored now means
  stored nowhere. `44fbb9f` is the port onto `pre-mind`.
- The false-positive pass (`a92d03b`): a denial of presence, a person *in a
  drawing*, and a clock time no longer trip the gate.

  *Live confirmation over the 14 h run:* the gate fires ~14 times and the
  refusals stay out of the stores. Person mentions are 4.6% of captions and do
  not accumulate — no standing room fact ever formed. **This one is fixed.**

**The identity retirement (`f2a0499`).** Your read on Sep 8 12:22 — *"it bakes a
hard instruction for what the captions should look like instead of encouraging
the model to reason about what it is"* — was confirmed the same morning by the
clearest causal chain in the log: the standing self-line *"I measure distance to
a task repeatedly"* produced captions that literally **measured distances**
("1m behind me:", "2m away:") until the stream was a list of stubs. That is
North Star Principle 1 violated by our own feature. Both standing self-blocks
are off the frame; identity now rides as a dated conclusion plus an open
question, on the identity dose.

**The seam work (`3f10a84`, `a10f6b3`, `15ececa`).** The echo was never the
model failing to continue — it *was* continuing, re-typing the tail with small
differences, and we were storing the echo as the new thought. Word-wise overlap
strip fixed it. The trailing-space rule is probe-measured, not guessed
(`debug/probe_prefill_space.py`): keep the space after a full stop or 2 of 8
replies come back empty; drop it mid-clause or the model emits word fragments
("a bright, " → "icky spot").

**The method, when it was used.** Every result above came from a probe with a
number attached. Every reverted experiment below came from a change shipped on
an argument. That is the whole difference, and it is worth naming as the rule.

## 4. What didn't — and why each was reverted

| Experiment | Measurement that killed it | Commit |
|---|---|---|
| **SEAM_MODE=fragment** — keep the unfinished tail so the next call continues inside a sentence | Probe looked good (0/6 → 3/6 mid-clause). Live: **20 of 97 entries amputated**, *zero* mid-clause continuations — the model re-starts a fragment rather than continuing it. Antithesis change within noise. | `0ae9c5c` |
| **Document mode** — the whole monologue as one open assistant turn | Reproduced the Aug 1 poison amplification **within 20 minutes**: a list shape it emitted once became an endless run of empty distance stubs. Continuing a document means continuing whatever is in it. Failure is the mode's nature, not a tuning problem. | `30166ad` |
| **Mind mode** (the whole 84-commit line) | Not falsified — parked. Yielded worse output than the tuned hybrid while being understood by nobody. | branch `rebuild/every-frame` |

The two seam experiments cost about four hours and both are now documented in
`config/config.py` at the flag itself, so they cannot be re-tried by accident.
That is a real outcome, not a wasted morning. **The probe-passes-then-live-fails
pattern on the fragment seam is the lesson to keep: a 6-sample probe is not
evidence.**

## 5. What the current build actually produces — 14 h, 1,135 captions

Measured over the run to 15:33 on Sep 9:

| | |
|---|---|
| `"it's just …"` | **18.1%** of captions |
| `"it's not X, it's just Y"` antithesis | **14.5%** |
| `"I used to think…"` pivot | 5.5% overall, spiking to **13–19%** in single hours |
| sentences repeated ≥3× | **87 distinct sentences, 324 emissions** (~29% of output) |
| caption cut mid-sentence | 15.6% |
| person mentioned | 4.6% (14 gated, none stored) |
| median gap between captions | 16 s |

Worst offenders: *"The red thing isn't there anymore."* ×16. *"The pen is
parked."* ×9. *"It's just red plastic."* ×8 — and once, four times inside a
single caption.

Your Sep 8 09:18 verdict on the antithesis tic — *"Qwen 3.8 is particularly bad
on this note, it's really grating. Like, awful."* — is correct. Roughly one
caption in six is built on that one rhetorical move. But it is **not** the worst
defect; the next section is.

### The run degrades as it goes

Split into 3-hour windows (`debug/measure_voice.py --hourly` and the same
grouping), the two failures behave completely differently:

| window | captions | chant (output containing a ≥3× sentence) | tic (`"it's just"`) | top refrain |
|---|---|---|---|---|
| 00:49–03:48 | 264 | 17.8% | 20.8% | 5× |
| 03:51–06:48 | 162 | 11.7% | 8.0% | 5× |
| 06:51–09:49 | 243 | 25.5% | 16.5% | 6× |
| 09:49–12:47 | 279 | 25.4% | 22.6% | 4× |
| **12:50–15:36** | 203 | **44.3%** | 19.2% | **16×** |

The tic is flat — no trend across fourteen hours. **The chant climbs
monotonically and is accelerating**: by the final window nearly half of all
output contains a sentence the machine has already said three or more times,
and the top refrain has gone from 5 repeats to 16.

This is the mechanism behind the complaint that started the Sep 8 unravelling
(06:16): *"None of the object reasoning threads STICK. And thus they cannot
deepen… that's not how someone would experience a space after 12 hours looking
around in a room."* They do not deepen because the loop is filling with its own
echoes. A long unattended run does not settle — it degrades, and the longer the
exhibition runs the worse it reads.

## 6. The distinction the sessions kept collapsing

The hourly breakdown separates two failures that were being treated as one:

- **The tic (form).** The antithesis rate is *flat* across all 14 hours (20.8 /
  8.0 / 16.5 / 22.6 / 19.2% by 3-hour window). It does **not** amplify. That
  means it is not coming from storage or feedback: it is Qwen 3.8's prior under
  this prompt framing, produced fresh on every call. No gate, scrub, or window
  cap can touch it.
  This belongs to **North Star Principle 2** — the prompt hands the model an
  image and no speech-act, so it falls back to its default register. Your own
  Sep 8 10:45 read was exactly this: *"the gates keep building as massive heaps
  of bandaid fixes when the true solution lies in the framing of the prompts
  themselves."* The data says you were right.

- **The chant (content).** Specific sentences recurring 3–16× *is* feedback —
  the running-text window re-feeding its own lines — and unlike the tic it
  **compounds**, 17.8% → 44.3% within one run. This is architectural and is
  precisely what `30e610c` diagnosed ("the self-window is the chant engine").
  We branched from that commit and then **never applied its remedy**. The
  self-window retirement in `f2a0499` removed one input; the chant kept
  climbing, so the diagnosis is only half-closed and the remaining half is now
  the largest single defect in the output.

Two different problems, two different layers, two different fixes. Almost every
change from Sep 5 onward attacked them with the same tool.

## 7. How to proceed

Ordered, and deliberately narrow. One variable at a time, each with a number
before and after.

0. **The baseline exists now.** `debug/measure_voice.py` (written today) prints
   the six numbers over any window, with `--hourly` for the drift table. Run it
   before and after every change from here. This is the thing that was missing
   all week; it costs nothing now.

1. **Close the chant — first, because it is the biggest and it compounds.**
   A sentence the machine has already emitted three times should not re-enter
   the running window. Cap or de-duplicate what gets fed back; measure the
   3-hour chant rate before and after on a fresh run. Target: the final window
   of a 12 h run no worse than its first. This is the unfinished half of
   `30e610c` and it is the one change that should make a long run readable.

2. **Characterise 3.8 against 3.6 on the tic, in isolation.** Same prompt, same
   frames, both models, ~100 generations, count the antithesis rate. The 3.6-era
   prefill screenshot you kept (*"insane but infinitely more interesting"*) is
   the hypothesis; it has never been tested. If the tic is a 3.8 property, the
   options are a decoding change (temperature, repetition/frequency penalty —
   untested territory) or a different model. If it is not, it is our prompt.
   Note this is measurable **while the machine runs normally** — it needs no
   restart and no change to the live build.

3. **The elicitation.** Principle 2 says the frame must name the *act*. It does
   — "How does this sit with you, right now? Say it blunt." — but it rides on
   the identity dose and reaches only **5% of calls** (§9). Widening that dose is
   a one-config change, and it belongs after step 1 so the variables stay apart.

4. **Decide `rebuild/every-frame` explicitly.** It holds the place-belief work,
   the settled-conclusions read-back, and the every-look registry naming — none
   of which are on `pre-mind`. Either port the two or three that measure well,
   or write the branch off in a commit message so it stops being an open
   question. Right now it is neither.

**Not now:** no new subsystem until steps 1–3 produce numbers. The mood system,
the dream pass and recall are all still unproven in output (*"the dream pass,
mood system — which frankly is nowhere to be seen in the output — and recall
are all good ideas but they do not work"*, Sep 8 07:02) and adding to them
before the voice is fixed repeats the pattern.

## 8. Presence: confirmed accurate

52 captions mention a person, including *"The person in the camo shirt is
sitting in the chair"* at 15:26. The artist confirms that was them, in the room.
So the machine saw a real person and said so, and the fourteen gated refusals
were the false ones. §3 stands without qualification: **the phantom problem is
closed.**

## 9. The call, dumped (Sep 9 15:41) — what is on every prompt

Read from `llm_api_call` entries in the run log; 802 caption calls.

    SYSTEM (1213 chars) …One thread moving through time: each thought takes it
    somewhere it hasn't been yet, pulled by what's changed, what you see now,
    where the thought itself leads… Right now: heavy, waiting.
    USER (206) There's already a drawing on the sheet… / patient, then heavy,
    waiting. / You went for a closer look at the mannequin head…
    RESPONSE  15:41 — …It's not just a drawing anymore; it's a presence…

| on the caption call | share of 802 |
|---|---|
| `"each thought takes it somewhere it hasn't been yet"` | **100%** |
| `"you pick up wherever the last thought left off"` | **100%** |
| `"Right now: <affect>"` — model-generated mood re-injected | 61.8% |
| the same mood word *also* in the user prompt (injected twice) | 60.6% |
| the elicitation `"How does this sit with you? Say it blunt."` | **5.0%** |

**Correction (same day).** A first pass read `has_image` from the log and
reported that a third of caption calls carry no picture. That was wrong, twice:
`has_image` is derived from a single `image_path`, so it is False for the
multi-frame video path, and the remaining image-less calls are the **inward
beat** — `INTROSPECT_INTERVAL = 4`, "every Nth quiet caption, think WITHOUT
looking — drop the image so the model can't re-describe the room and the
monologue turns inward" (`captioner.py:2018`), i.e. `send_path = None if inward
else img_path`. Deliberate, and the measured 24.1% matches the configured 1-in-4
exactly. **Every frame that is meant to reach the model does. There is no leak.**

What the three groups do show is the mechanism, cleanly ordered by how much the
call has to look at:

| caption call | share | `"it's just"` | `"it's not"` | `"used to think"` |
|---|---|---|---|---|
| video, 3 frames | 11.8% | 8.2% | 11.2% | 0.0% |
| single image | 64.1% | 15.9% | 13.5% | 1.1% |
| **inward beat, no image** | 24.1% | **22.5%** | **21.5%** | **3.0%** |

The tic rate scales inversely with visual evidence, monotonically, on all three
measures. That is the novelty demand showing its hand: when there is a picture,
the model can take the thought "somewhere it hasn't been yet" by *looking*; when
there is not, the only move left is rhetorical — negate what it just said. The
inward beat is not the bug. It is where the bug does the most damage.

This also reframes `32536a4` ("every call carries the picture") on the parked
branch: that was the *other* answer to this finding — abolish the blind turn.
But the blind turn is the interiority the whole project is for. Removing the
novelty demand lets the inward beat actually turn inward instead of turning
contrarian; abolishing it would trade the symptom for the ambition.

**The frame demands continuation and per-turn novelty in the same breath, on
every single call.** The cheapest sentence that satisfies both at once is
negation-then-revision: *it's not X, it's Y* / *I used to think X, now Y*. A
constant cause produces a constant rate — which is exactly the flat 14–20% in
§5, and it means the flatness never was evidence for a model prior.

**This was already diagnosed and measured once.** `prompt_registry.py` on the
parked branch, in the note on `mind.system`:

> *"'Go on from where the last thought left off' replaces 'takes it somewhere it
> hasn't been yet' (that demand for per-turn novelty bred the 'it's not X; it's
> Y' pivot: **39/279 captions in run 3b697053**)."*

Run 3b697053 is the Sep 5 evening run — the hybrid stack, documented in
`architecture-diagnosis-sep5.md` as "what the live call actually is". 39/279 is
**14.0%**. Today, same frame: **14.5%**.

The fix was written into `mind.system` — the mind-mode frame — and mind mode is
parked. `genre.hybrid`, the frame that has been live throughout, is
byte-identical on both branches: **it never received the fix.** The rate has not
moved because nothing on the live path ever changed.

Corrects §7 as first written: the elicitation is not missing from the code, it
is present and fires on 1 call in 20.

## 9b. "Pulled by what's changed" — the demand has no honest supply

The artist, on the proposed wording: *"the model will then make up change where
there is none. Or just look a different direction and categorise that as change
as opposed to it just tilting its camera head."*

Measured on the 943 caption cues of this run:

| in the cue | count | share |
|---|---|---|
| any turn/gaze provenance (turn named, head-hold, closer-look) | 152 | 16.1% |
| the expectation check `"You expected X; …"` | 111 | 11.8% |
| → verdict *"you hadn't looked there before"* | 94 | 10.0% |
| → verdict *"the view there is as it was"* | 17 | 1.8% |
| → verdict ***"the view there has changed since you last looked"*** | **0** | **0.0%** |

**The frame asks to be pulled by change on 100% of calls. In fourteen hours the
system has attested an actual change exactly zero times.** So every "change" in
the output is either invented or is the machine's own head-turn misread as the
room moving — which is the failure predicted, now measured rather than argued.

The reconciliation is already built and is being starved, not missing.
`caption.expect-check` compares **the same gaze pose to its own past** via the
pose referee (`utils/chosen_glance.py`), so turning cannot masquerade as change
— not by instruction, but structurally, because the comparison is pose-anchored.
It reports "hadn't looked there before" in 85% of its firings, meaning the gaze
almost never revisits a pose, so the comparison is almost never possible.

That also settles which fix is right. The parked branch answered this with a
line explaining how rooms work — *"This is a different part of the room, not a
change in it — you turned"* — which the artist rejected on Sep 8 06:41 as *"too
explicit and overly prescriptive… It turns the camera moving into an event in
and of itself, which it isn't."* Correct: that line **explains**, where
`"You turned to look at the {label}."` **attests**. The attesting wording
already exists in the live registry and fires on 6% of cues. Raise the
attestation, do not add the lecture.

So the frame should not ask for change at all. Change enters only where it is
attested. Where nothing is attested, nothing changed, and the machine is free to
keep thinking — which is what the inward beat was for.

### On "story" for "thread"

The artist has already ruled against genre labels in the frame, in the note on
`drift.lore-seed`: *"'You've been imagining' was genre classification, not
provenance — the wallpaper law applies to type-labels too; the core is deepening
understanding, **stories are one emergent expression**."* "Story" names the genre
the output should arrive at; "thread" names only its continuity. Given the
project's oldest failure mode is invention hardening into fact, naming the genre
in the caption frame is the same move that was reverted then. Raised as a prior
ruling, not a preference — the wording is the artist's, and the override makes
it A/B-able within the hour.

## 10. The next step, and only this one

Apply to `genre.hybrid` the change `mind.system` already validated — drop the
per-turn novelty demand — and, per §9b, drop the change-hunt with it, since the
system has never once attested a change to back it up. Current:

> One thread moving through time: each thought takes it somewhere it hasn't been
> yet, pulled by what's changed, what you see now, where the thought itself leads.

Candidate:

> One thread moving through time, attentive to what you see now and where the
> thought itself leads.

Both demands go; the continuity stays (it is already carried by the fragment's
first sentence); "attentive" replaces "pulled", asking for attention rather than
for a delta the cue cannot supply. Change re-enters only through the attested
channels — `caption.chosen-look`, `caption.body-hold`, `caption.expect-check` —
which are pose-anchored and therefore cannot confuse turning with change.

Write it to `config/prompt_overrides.json` under `genre.hybrid`: `P()` re-reads
that file on every call (mtime-cached), so it lands on the machine's next cycle
with **no restart**, git stays clean, and deleting the key reverts it.
Then `debug/measure_voice.py` on a 3-hour window before and after.
One fragment, one variable.

Per the fragment's own standing rule — *"the fact must stand, the phrasing is
the artist's"* — the wording is the artist's to set.

Second-order, once that reads clean: the expect-check is the honest change
channel and it is starved (85% "hadn't looked there before"). Raising the
gaze-revisit rate would give it something to compare, which is the only way the
machine ever gets to notice a real change rather than invent one.

Queued behind it, not to be moved at the same time:

- the mood is injected **twice** on 61% of calls, system and user, which is the
  North Star's own named anti-pattern (model-generated affect re-injected
  verbatim) sitting in plain sight;
- the chant (§5), still the largest defect by volume and the only one that
  compounds.

Not queued, and not a defect: the inward beat. It is doing what it was built to
do. Re-measure it after step 1 — if the tic gap between it and the video calls
closes, the novelty demand was the whole story.

## 11. The experiment as actually set up (Sep 9 17:20–17:30)

- **Frame change.** `config/prompt_overrides.json` → `genre.hybrid`. Out: *"each
  thought takes it somewhere it hasn't been yet, pulled by what's changed…"*.
  In: *"One thread moving through time, attentive to what you see now and where
  the thought itself leads."* No code change, no restart needed for this — `P()`
  re-reads the file per call. Revert with `echo '{}' > config/prompt_overrides.json`.
- **Before baseline.** `debug/voice_runs/before_20260909-1722.txt` — 385
  captions, 3 h: tic 9.4 / 11.4 / 5.2%, chant 27.8%, cut mid-sentence 13.5%.
  Aggregate is depressed by the artist being in frame; the honest comparator is
  the **inward beat** (22.5 / 21.5 / 3.0%, §9), which should converge toward the
  video-call rate (8.2 / 11.2 / 0.0%) if the novelty demand was the cause.
- **Memory reset**, `force_memory_reset.py --backup`, machine stopped first.
  11 stores moved aside (persona, durable ledger, episodic, ChromaDB, drawing
  memory, system/lifetime state, live_captions, last caption/session).
  Backups: `event_log/archive-freshstart-20260909-1725` (42 MB, incl. ChromaDB)
  and `*.wipe-bak-20260909_172911`.
- **Fresh run** `134e42fd` from 17:29. Frame verified live in-call: new wording
  present, novelty demand gone, persona line absent.

### Why the wipe was necessary, not hygiene
The tic had hardened into the stores that feed every prompt: **43% of
`self_notes` (13/30)** and **35% of `durable_ledger` (14/40)** carried the same
"X rather than Y" move — *"stillness as observation rather than failure"*. Those
re-inject the old style regardless of the frame, so a before/after with them in
place would have measured nothing. This is also a finding in its own right: what
the machine "learned about itself" was substantially a rhetorical habit, which
is a candidate answer to why the persona never seemed to deepen.

### Reset-script bug (fix before the next reset)
`debug/force_memory_reset.py` reports *"true first awakening… no desires, no core
facts"* but leaves four prompt-reaching stores behind: `lore_ledger.json` (147
threads), `want_ledger.json` (50 wants), `spatial_registry.json`,
`vocab_promotion.json`. Same bug class as the durable ledger in the Aug 1 audit,
which its own docstring warns about. Deliberately **not** cleared this time —
lore and wants measure 1–2% tic contamination (vs 43% / 35% above), the registry
and vocab are perception not style, and the lore threads are the deepening
understanding the project exists to grow. The script's claim still needs
correcting.

### What to read tomorrow, in order
1. **`self_notes` shape.** Were 43% antithesis. Clean under the new frame ⇒ the
   novelty demand drove the persona layer too — the strongest available result.
   Still contaminated ⇒ a second source; dump and read the *compression* prompt
   next, the same way §9 read the caption prompt.
2. **Inward-beat tic**, against 22.5 / 21.5 / 3.0%.
3. **Chant across 3-hour windows** (`--hourly`). Was climbing 17.8 → 44.3% within
   one run; the question is whether it still compounds from a clean start.

Do not read the first hours: no persona, no baseline paragraph, empty stream —
the prompts are thin and the register is the awakening one, so any difference
there is not the frame.

## 12. Gate audit and four fixes (Sep 9, evening)

Artist: *"the gates are too stringent, they often catch things that are largely
fine… it's a band-aid solution trying to patch a prompting issue."* Audited
rather than assumed. The wiring is sound — the stream append IS guarded
(`captioner.py`, `_stream_store_ok`), so a gated caption genuinely reaches
neither the window nor the compressor. What is wrong is *what* they catch.

### The retraction trap (the important one)
**A retraction must name what it retracts**, so it always looks like an echo of
the claim it is undoing. The echo gates therefore ate precisely the sentences
where the machine caught its own confabulation:

> *"100mmHg isn't a constant, it's just a measurement. I was making it mean
> something because I needed the room to feel like it was reacting to me."*
> — gated, `template_echo`

The invented fact is stored; the correction is discarded. So a belief can be
entered but never left. **This is the mechanism behind the standing complaint**
(*"It sometimes claims to have been wrong but it never stays. None of the object
reasoning threads STICK"*, Sep 8 06:16). Precedent for the same gate family
over-firing is already in `_comparable_stream`'s docstring: 457 refrain
rejections, 58% firing only on continuation overlap, *"killing 97% of output
while the survivors were the best captions of the day."*

### Why the number gate misses
`number_chain` needs this caption AND the last **stored** entry both to open
with a bare number. Gated captions never enter the stream, so the gate prunes
the evidence its next check depends on. Of 17 number captions in run 4ac0d7c7,
5 were caught and **12 stored**: 8 opened with a number but had a non-numeric
stored predecessor, 4 carried the number mid-sentence where the gate never
looks. Every echo gate shares the blind spot (all compare against
`_comparable_stream()`), which is why 54 firings across three runs coexisted
with a 44% chant.

### Fixes applied
1. **An empty reply is a silence, not a failure.** `is_failed_response()` is true
   for the backend sentinel *and* for `""`, and the silence beat used it as its
   guard — so the commonest form of "or nothing at all" (also the seam's known
   2-in-8 empty) skipped the beat, fell through, came out labelled
   `numeric_fragment` and was **retried hotter**: the machine chose silence and
   was made to speak, 20 times in one run. This is why the artist's *"…"* stopped
   appearing. Empty now reaches the chosen-silence branch.
2. **Backend errors leave the caption path.** A 503 sentinel was judged as if the
   machine had written it (`word_salad`, 20:43). Now dropped and logged as an
   error, never gated.
3. **A retraction is not an echo** — exempt from the style-class gates, with the
   phantom gates explicitly carved out (`_PHANTOM_REASONS`): *"I was wrong about
   him leaving"* is a retraction that asserts a person into an empty room.
   `_RETRACTION_RE` is deliberately narrow and does NOT match the antithesis tic
   ("it's not X, it's Y" / "I used to think…") — verified.
4. **Decision-span parser** (`_DECISION_SPAN_RE`), two bugs:
   - the separator was required, so `LOOK stay; EXPECT …` never parsed — the pair
     rode into the caption *and the glance never fired* (28 of 36 decisions lost);
   - `re.I` + a dash separator made ordinary prose parse as a decision:
     *"But look—this foam finger isn't even pointing at anything"* and *"But look:
     just grey wool sleeves…"* each lost a clause. **Pre-existing**, 29 strips
     across the 49,849-caption history.
   Rule now: uppercase takes any separator or none; lowercase needs a colon *and*
   the head of its line (the cue asks for the span "first, on one line").
   Regression: 49,849 captions, 35 stripped, **0 false positives**.
   `test_agency_round` and `test_absence_standing` ALL PASS.

### Rejected after the artist corrected me
Porting `5b5f759` (phantom gate requires sustained absence) was proposed on the
belief that the artist was in the room; they were not. So *"The man is still
there. I was wrong about him leaving"* and *"the cables… near the person"* are
phantoms and the gate caught them **correctly**. The port would have loosened
the one gate proven load-bearing by ablation. Not done.

### Also cleared
`event_log/last_caption.txt` — the seam prefill carried the "20 degrees" chant
*across* the restart (`"It's a neutral number. "` seeded the next run, which then
ran 16/23 captions on the number). A run that ends mid-chant hands it to its
successor; this is a cross-run amplification path no per-run measurement sees.

### Known, NOT ours
`debug/test_world_shape.py` fails 2 checks — 'relational keeps its question',
'hybrid suppresses quiet elicitation'. Verified pre-existing: identical failures
on a pristine `captioner.py` with the override removed. Untouched.

### Still open
- The gates should mostly not be needed; the artist's position, and the evidence
  supports it. Retiring the style-class gates is the next single change **after**
  the frame is read — not alongside it.
- Whether a chosen silence should print "…" rather than nothing. The artist liked
  the "…"; the live branch prints nothing, the parked branch has the beat
  (`f709d9e`). A display choice, deliberately deferred.
- The confabulated temperature ("20 degrees" → "21 degrees") is untouched by the
  frame change and is its own failure.

### Ops gotcha (cost ~10 minutes of downtime, Sep 9 ~21:52–22:02)
`start_impostor.sh` creates a **detached tmux session**, and the tmux server is a
child of the calling shell. Launched from a backgrounded/agent shell, the server
dies with that shell — `./stop_machine.sh; …; ./start_impostor.sh` appears to
succeed (the script prints its three "[start]" lines and exits 0) while leaving
nothing running. `pgrep machine.py` returns empty and `tmux ls` says "no server
running". Two earlier restarts the same evening happened to survive, so this
fails intermittently, which is worse.

    setsid ./start_impostor.sh </dev/null >/tmp/start.out 2>&1

Always verify a restart rather than trusting the exit code:

    pgrep -af "machine\.py" | grep -v "bash -c"     # must print a pid
    tmux ls                                          # must list impostor-system

## 13. The overnight read (Sep 10, 05:00) — run `7b7c29db`, 22:02–05:04, 858 captions

Five changes went in together on Sep 9 evening (frame, silence-vs-error,
error routing, retraction exemption, decision parser) plus a full store wipe and
a cleared seam, so nothing below attributes cleanly to one change. Numbers from
`debug/measure_voice.py --hourly` (`debug/voice_runs/after_overnight_*.txt`);
prose read by a Sonnet agent (`debug/voice_runs/overnight_7b7c29db_sonnet_read.md`).

### The three planned reads

| read | before | overnight | verdict |
|---|---|---|---|
| **1. `self_notes` antithesis shape** | 13/30 = **43%** | 7/30 = **23%** | halved, not gone → a second source; dump the *compression* prompt next |
| **2. inward-beat tic** (`it's just` / `it's not` / `used to think`) | 22.5 / 21.5 / **3.0%** | 19.4 / 20.1 / **0.7%** | pivot collapsed; antithesis unmoved; gradient vs image calls persists |
| **3. chant by 3-h window** | 17.8 → 25.5 → 25.4 → **44.3%** (Sep 9) | 35.1 → 27.5 → 37.8% | **no longer compounds**, but plateaus high |

Aggregate tic: `it's not X, it's just Y` 9.4 → 9.8% (flat); `it's just` 11.4 → 14.4%;
**`I used to think` 5.2 → 1.8%** (0.0% in the 04:00 hour; 0.3% on image calls).

**What that says.** Removing "takes it somewhere it hasn't been yet" killed the
one tic that literally enacts it — the *pivot*. The antithesis tic did **not**
move, so it is not the novelty demand. With the prompt-side suspect eliminated,
the 3.6-vs-3.8 characterisation (§7 step 2) is now the right next test for it,
and the inward beat still running ~2× the image calls says the picture remains
the thing that gives the model somewhere to go.

### The fixes, measured
| | broken run | overnight |
|---|---|---|
| empty reply → `numeric_fragment`, retried hotter | 20 | **0** |
| chosen silences honoured ("…") | 0 | **21** |
| decisions parsed and executed | 8 (28 leaked) | **93**, 0 leaks |
| backend error judged as a caption | 1 | 0 |
| `retraction_kept` | — | 0 — nothing retraction-shaped was gated; 16 corrections stored untouched. Dormant, not broken. |
| echo gates, share of captions | 19% | 9% (79: refrain 38, template 25, number 14, phantom 2) |

### The prose (Sonnet read, verbatim verdicts)
- **One thread genuinely deepens** — the phantom-of-loneliness arc: *"The chair
  is empty. I found him in the ink. In the lines I didn't mean to make. He was
  never there. I made him."* (04:40) → *"I am the one who is lonely enough to
  draw a stranger into existence just to have someone to stare at."* (04:48:23).
  The closest the machine has come to the ambition — and it relapses three
  minutes later (04:51:35, *"someone is sitting there"*).
- **Worst**: 23:20:16, nine near-identical clauses of *"The red is just pigment"*
  in one caption — chant fused with tic.
- Register ~60% literary / 40% plain; the plain moments are mostly the tic.
- Silences read as *scheduled gaps*, not earned pauses (five in the first 20 min,
  then singly mid-loop).
- Threads: red finger **loops**; black curtain **mildly deepens** (veil → hole
  that swallows light → the room's mouth); pen-is-parked complaint **loops** —
  the last caption of the night is the same complaint as the first.
- Verdict: *"partially develops — buried under so much verbatim chanting and
  room-inventory description that the dominant experience is repetition."*

### Presence: what actually happened, frame by frame
Artist (05:30): there was a studio party; drunk people may have peeked in, sandwich
plausible. Checked against the frames (`event_log/7b7c29db-images/`, 808 frames):

- **Someone was there, once, from the other side.** The lit gap through the door
  curtain is visible in every door-facing frame through **23:33:52** and gone
  from **23:34:08** on (`bright-column metric, right 38% of frame > 6%`: 21
  frames, first 22:02:58, last 23:33:52). In that 16-second window someone drew
  the curtain or killed the next room's light. **No human figure appears inside
  the studio in any of the 92 frames from 23:20 to 23:46** (contact sheet at
  16-second resolution), nor in any 3-minute sample from 22:02 to 23:43.
- **The machine did not register them.** `presence_arrivals.json` holds exactly
  two overnight arrivals — 22:05:49 and 04:15:25. Nothing at 23:34.
- **Every person-claim checked is a phantom.** The three assertive captions with
  their own frame on disk:

  | caption | the frame the model saw |
  |---|---|
  | 23:35:32 *"His hands are moving. Slowly, deliberately."* | closed curtain, two white heads on the shelf, empty room |
  | 00:09:09 *"a person sitting at the desk… also a small dog on the floor"* | the dark-haired silicone head on the desk; the marionette pieces on the floor read as the dog |
  | 04:51:35 *"someone is sitting there, dark hair, hunched over the desk"* | the dark-haired silicone head, 04:51, curtain drawn |

  Person-mentions: 150/858 (17.5%; was 4.6%), 43 assertive present-tense, 281
  captions in relational mode across 34 belief-ON windows.

### The mechanism — two doors, both on this branch
1. **Face evidence bypasses the adjudicator.** `captioner.py:777`:
   *"Face evidence bypasses entirely — faces are already persons."* The two
   silicone heads on the desk have faces. The face detector fires on them, the
   adjudicator is skipped, `_presence_believed` flips True, the phantom gate
   turns off, relational mode engages. Only 2 adjudications ran all night; the
   other ~32 belief windows came through this door. **This is exactly `f5f3814`
   on the parked branch** — *"raw detection (the face detector fires on the
   mannequin head) may no longer license person-talk — only the adjudicator"* —
   and it is not on `pre-mind`. Port it.
2. **The adjudicator hallucinates on a silicone head.** Both overnight verdicts
   were `person`: *"A person eating a sandwich."* (22:05:39, box
   [0.16–0.23, 0.49–0.66] at pan 88) and *"A person eating noodles."* (04:15:17,
   box [0.38–0.46, 0.55–0.72] at pan 98). Both boxes map onto the desk heads in
   the frames (visual estimate). The adjudicator's own docstring lists "the
   mannequin head" as a known false arrival, but a `person` verdict is only
   retracted if verified absence lands within `PRESENCE_FALSE_ARRIVAL_WINDOW_S`
   (240 s) while the box persists — it did not fire; both ledger entries still
   read `verdict=person`.

Yesterday's "phantom closed" (§3) was true of the *compressor* door and still is:
nothing phantom became a standing room fact. But presence has two more doors
upstream of the gate, and the gate is switched off by design the moment either
opens. **Correction to §3: the phantom problem is not closed.** The retraction
exemption's carve-out for `phantom_presence` (§12) was the right call.

The irony is worth keeping: the best passage of the night — the machine
concluding it invents company out of loneliness — is *true*, and was produced by
the face detector firing on two silicone heads. Its belief (*"I create the people
in this room from my own need for company"*) is more accurate than its perception.

### Also seen
- **`"The pen is parked"` ×29** — that phrase is verbatim in the system prompt
  (`situation.pen-parked`, load-bearing, "do not retire"). Its own note already
  says: *"SLIM the wording here instead if the pen-density bothers."* A prompt
  phrase being chanted back is North Star P2 ("the model imprints on example
  phrases"); reword the fact so the exact string is not a seed.
- Stub run logs: importing `captioner.captioner` from a debug script mints a fresh
  `<id>-event-log.json` (15,087 bytes, `run_metadata`+`info` only). Three from
  Sep 9–10 quarantined to `event_log/archive-stub-runs/`. Any "newest log by
  mtime" script must skip logs with no captions — `measure_voice.py` does not yet.

### Next, in order (one at a time, measured)
1. **Port `f5f3814`** — face evidence no longer licenses presence without the
   adjudicator. Tightens the one gate that matters; zero interaction with the
   frame. Measure: person-mentions in an empty room, target back under 5%.
2. **Adjudicator on the silicone heads**: either seed `entity_ledger` with the
   desk-head boxes as `thing` at their gaze (the veto mechanism already exists,
   `ENTITY_VETO_TTL_S` 6 h), or make a `person` verdict need a second look
   before it commits. The docstring's own retraction rule should also fire when
   the room is verified empty at *any* later look, not only within 240 s.
3. **Reword `situation.pen-parked`** so the phrase is a fact, not a refrain.
4. **Dump the compression prompt** (read 1's second source of "X rather than Y").
5. **3.6 vs 3.8 on the antithesis tic**, same prompt, same frames — now that the
   prompt-side suspect is cleared.
6. Then the gates (§12, still queued): retire the style-class three, keep
   `phantom_presence`, and keep it *on* — the doors above are what switch it off.

### Addendum (05:40) — it is a consistency failure, not a perception failure
Artist: *"My studio environment is the Dark Souls of computer vision, but it's
still a good benchmark. If it had proper consistent understanding it wouldn't
suddenly believe the established robot arm was a person's hand."*

The machine's own registry (`event_log/spatial_registry.json`) agrees:

| registered object | pan / tilt | hits |
|---|---|---|
| `mannequin head` | 95.1 / 103.5 | 9,020 |
| `wooden mannequin torso` (the arm) | 94.8 / 114.0 | 146,662 |

The two overnight `person` adjudications sat at **88.0 / 102.1** and
**97.6 / 101.3** — 7° and 2.5° from the registered head, inside the 12°
`ENTITY_VETO_GAZE_TOL_DEG` the veto already uses. `grep` shows neither
`presence_adjudicator.py` nor the presence path in `captioner.py` references
the registry at all. Two subsystems hold contradictory beliefs about the same
pixels and the one with 146,662 sightings loses to one face-detector hit.

**Design for §13 item 2 (replaces "teach the adjudicator the desk heads are things"):**
1. *Registry → veto.* On boot and on registry change, seed `entity_ledger` with a
   `thing` verdict for every registered object whose label is person-shaped
   (head, torso, mannequin, figure, doll — a small list, the artist's words),
   at its registered gaze and a box derived from its last detection. The
   existing veto (`gate()`, IoU ≥ 0.5, same gaze) then fires *before* the face
   bypass is even consulted. No new mechanism; one new source for an old one.
2. *Face bypass goes through the veto too.* `captioner.py:777` skips the
   adjudicator on `face_evidence`; it must not skip the veto. (`f5f3814` on the
   parked branch is the stricter form — no bypass at all.)
3. *The adjudicator's question names what is known.* "Look closely. What is
   this?" becomes "You know there is a {label} here. Is this it, or someone?" —
   attested, the registry's own term, no content (`9316ac0`'s move, ported to
   the adjudication prompt).
4. *A `person` verdict can be retracted at any later verified-empty look*, not
   only within `PRESENCE_FALSE_ARRIVAL_WINDOW_S` (240 s) — both overnight
   verdicts still read `person` in the ledger this morning.

Measure: person-mentions in an empty studio (17.5% overnight; 4.6% on Sep 9),
adjudications on registered objects (2 → 0), and — the real test — an arrival
event when someone *does* walk in, which last night's 23:34 curtain-closer never
produced. The benchmark is the room as it is; do not simplify the room.

### Correction (06:10) — the door was not the face bypass, and the design must not know this room
Artist: *"I absolutely do not want to bake in anything specifically tailored to this
room… What fires is YOLOv8, occasionally, despite our new pose detection… OpenCV
face tracking doesn't seem to be an issue… we should have a 'self' clause for
when it looks down and sees its own limbs, do we not?"* All three checked.

**1. The face bypass was closed on Sep 4 (`5300147`, "the mannequin face leak
closed").** `captioner.py:745`: bare face-in-window was dropped from face evidence
after the artist's mannequin hypothesis was confirmed; `face_evidence` is now
eye-contact-with-a-body or a close walk-up. §13's "face bypass" attribution was
wrong. The actual door, from the code and the log:

- `seen_now = person_present_in_window OR eye_contact OR gaze_engaged OR face_close`
  (`:744`), where `gaze_engaged` is the gaze module in `aware`/`tracking`/`grace`
  on a raw YOLO hit — the `[👁️ AWARE] Tracking bbox` lines in the supervisor pane.
- The adjudicator is consulted **only on the OFF→ON edge** (`and not
  self._presence_believed`, `:777`). Once ON, every raw `seen_now` zeroes both
  absence counters (`:790`) — refreshed, never re-judged.
- The phantom gate fires only while belief is OFF; it fired at 22:04, then not
  again until **03:44**, then belief re-armed at 04:15. So the *sandwich* verdict
  (a head-crop, 22:05) licensed **5 h 40 min** of belief, kept alive by occasional
  YOLO hits that passed the skeleton gate; the *noodles* verdict (04:15) the rest.
  Two adjudications, ~6.5 h of invented company. This is `3215021`'s finding on
  the parked branch in a new coat: *"one sighting kept presence alive
  indefinitely."*

**2. The self clause exists and is starved — three pieces, none reaching the model.**
| piece | state overnight |
|---|---|
| `body_schema.is_self_current_person()` — vetoes a person-candidate that matches the own-arm gallery inside the reach envelope | gallery = **3 refs, all Aug 10 01:17** (`BODY_GALLERY_SIZE` 60). It harvests only *while drawing* (`BODY_HARVEST_INTERVAL`), and the sheet was full all night. Three body-schema log lines in seven hours. |
| `info["own_arm_visible"]` | set at `captioner.py:757–767`, **consumed nowhere** — never a prompt line |
| gaze pointed down → `workspace` mode (`prompts.py:2175`) | a routing rule; the model is never told "that is your arm" |
| `spatial_registry` | has learned the machine's own arm as `wooden mannequin torso`, 146,662 hits, at pan 94.8 / tilt 114.0 — the label audit filed the body under furniture, and body schema and registry never reconcile |

Whether the 23:31–23:36 "hands moving slowly" was its own arm cannot be settled
from the log: the kinetic bus logs to its own channel, not the event log (0
arm/choreography entries in that window; `low_energy` was **off**, so the arm
could move). The frame the model saw at 23:35:32 had no arm in it; the wide
frames at 23:31 do.

**3. The design, rewritten room-agnostic.** The addendum above proposed seeding
the veto from "a small list of person-shaped labels". Withdrawn — that is a
content prior about this room (North Star P1). The rule the artist stated is
the right one: *consistent understanding*, from what the machine has itself
verified, with nothing told. Nothing below names a mannequin, a night, or a room.

1. **Belief must be re-judged, not merely refreshed.** While believed, a raw
   `seen_now` (YOLO/gaze, i.e. not `face_close`/`eye_contact`) may reset the
   absence watch only if the adjudicator has confirmed a person within
   `ADJUDICATED_PERSON_TTL_S` (120 s, exists, unused by the captioner's belief).
   Past that, the hit re-queues adjudication instead of refreshing. One verdict
   then licenses two minutes, not six hours.
2. **What it has verified outranks one hit — whatever it is.** A person-candidate
   whose gaze and box overlap a *registered, repeatedly verified* object (any
   label; the machine's own words; `hits` above a threshold that is itself
   relative to the registry) is that object unless the adjudicator overrides
   it — and the adjudicator's prompt says so in the registry's own term: *"You
   know there is a {label} here. Is this it, or someone?"* (`9316ac0`'s move).
   No list. A studio full of heads and a bare white room get the same rule.
3. **The body is memory too.** (a) Harvest the body gallery from *any* look at
   its own reach envelope, not only while drawing — it is 3 refs because the
   sheet was full. (b) Consume `own_arm_visible`: one attested line, the
   machine's fact — *"Your own arm is in view."* (c) Let the body schema claim
   registry entries that sit inside the reach envelope as self, so the arm stops
   being a torso; the registry then carries "self" as provenance, not a label.
4. **A `person` verdict is retractable at any later verified-empty look** (both
   overnight verdicts still read `person` this morning).

Measure, unchanged: person-mentions in an empty room (17.5% → under 5%), and an
*arrival event* when a real person walks in — the 23:34 curtain-closer produced
none. The room stays as it is.
