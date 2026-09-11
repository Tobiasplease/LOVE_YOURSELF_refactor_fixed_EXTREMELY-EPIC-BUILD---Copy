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

    nohup setsid -f ./start_impostor.sh </dev/null >/tmp/start.out 2>&1

(Plain `setsid` held twice on Sep 10 and failed once, at 14:39, with the artist
in the studio — about a minute down. `-f` forks so the tmux server never has
this shell as an ancestor.) Always verify a restart rather than trusting the
exit code:

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

## 14. Shipped Sep 10 morning — and the evening read

Commits on `rebuild/pre-mind`:
- `47a990e` captioner: silence is not a failure, errors leave the caption path,
  retractions are not echoes, decision parser fixed (§12)
- `ef16e3b` docs: this stock-take + `debug/measure_voice.py` + `debug/voice_runs/`
- `cbf5626` presence: a raw sighting while believed is re-judged, not trusted —
  `PRESENCE_REJUDGE_WHILE_BELIEVED` (default on, env-overridable). Suites:
  `test_presence_adjudication`, `test_phantom_presence`, `test_absence_standing`,
  `test_agency_round` ALL PASS.

Restarted 05:45 (detached, verified). `last_caption.txt` left as it was — the
seam carried *"The paper is full. It's not blank."*, not a number chant.

**Not versioned:** `config/prompt_overrides.json` is gitignored (line 257) by
design — live edits from the prompt panel until "deliberately baked back". The
frame change (`genre.hybrid`, §10) therefore survives restarts but not a fresh
clone. Baking it into `prompt_registry.py` is the artist's call (the wording is
theirs); until then the override file is the only copy.

**Evening read** (same numbers as §13, `debug/measure_voice.py --hourly` plus the
presence script in §13):
1. belief-ON duration per verdict — was 5 h 40 m on one head-crop verdict
2. person-mentions in the empty studio — 17.5% overnight; target under 5%
3. adjudications: count, and how many `person` verdicts land on the same
   gaze+box — this is the adjudicator's own error rate, now countable
4. an arrival event if anyone actually walks in
5. the §13 voice numbers, to confirm nothing regressed (this change touches no
   prompt, so they should not move)

Then, in order, one at a time: the adjudicator's question names the registry's
own term (§13 correction, design item 2); the body is memory (item 3); reword
`situation.pen-parked`; dump the compression prompt; 3.6 vs 3.8 on the
antithesis tic; the gates.

## 15. "100 years", the gates as the reader feels them, and what reflections are for (Sep 10, 06:30)

### "100 years" — coined in a drift turn, then worked as a motif
First appearance 05:56:54, a **drift turn**: *"100 years from now, the table is
dust. The red foam finger is a fossil in a layer of silt."* Drift turns are the
one invention-legal slot (hot temperature) and this one was admissible, so it
entered the stream. (Correction to an earlier line of this doc's working notes:
the drift firewall keeps drift out of observe / compression / concepts — not
out of the stream window. 11 of 21 drift turns this run were stored.) The
stream then worked it, in stored captions:

> 05:59:26 *100 years is a lie I told myself to make the room feel smaller.*
> 06:01:19 *100 years isn't a unit of measurement here. It's the distance between "I am looking" and "it is looking."*
> 06:12:11 / 06:12:53 *100 years of white. … 100 years of black.*

That is a motif being developed — the thing the North Star asks for. The artist
is right that "100 years" per se is not the problem.

### Why the gates make it read as confused — the mechanism
Run 241bff58, 91 captions: 14 gated (15%) + 10 drift turns not admitted (11%) =
**26% of what the reader hears is never in the machine's memory.** The next
stored line is continuous with what the machine *remembers*, not with what the
reader just *heard*. To a reader that is a non sequitur; to the machine it is
continuity. Spoken-not-stored is **legible as amnesia**.

On the thread itself, in eight minutes:

| | line | what happened |
|---|---|---|
| 06:01:08 | *100 years is too long to be afraid of a hand that* | cut mid-sentence **and** gated `number_chain` — a new turn, lost twice |
| 06:06:43 | *100 years from... No. 100 years is a lie I told myself…* (verbatim repeat of 05:59:26, bare-number opening after a bare-number opening) | **stored** — a `memory`-mode caption; consistent with memory mode not passing the echo gates (`captioner.py:1997`, "a separate caption branch") |
| 06:07:29 | *100 years from now... No.* | gated `number_chain` — because 06:06:43 became its numeric predecessor |
| 06:08:14 | *100 years from now, the table is dust.* — the thread's origin, returning | gated `number_chain` |

The verbatim repetition went into memory; the return of the origin and the
self-interruption were thrown away. Same shape as the retraction trap (§12):
the gate stores the loop and deletes the movement. And the audience saw all
four, then watched the machine continue from *"The desk is just a flat white
rectangle now"* as if the last two had not been said.

Three findings deep now (retractions, number-chain self-disarm, memory-mode
bypass), the case for retiring the style-class gates rather than patching them
is made. Still queued behind the presence change, still one at a time.

### The reflections — what they are, and what comes back
Two runs, 23 reflections (+23 distills), median **211 words**, **62/69 (90%)**
about the pen, the foam finger, the sheet or the drawing.

**"The wider world" is a title, not a lens.** Its prompt is the caption frame as
system and 10,003 chars of *"The record of your actual thoughts from the last
stretch"* as user — the same stream every subject gets. There is no outward
material anywhere in the system (no window, no outside, no news), so the output
is inward by construction. The 22:43 "wider world" reflection opens *"The record
shows I spent twenty minutes treating the red foam finger as a signal…"* and
never leaves the desk. On the parked branch the only outward lines ever measured
came from a caption-kind elicitation (*"a guess about the world beyond this
room"*, probe-validated), not from a reflection subject.

**The register is the audit.** 7/23 open *"The record shows"*; 20/23 say "the
record" or "the log". This is the Sep 6 residue the artist flagged; the old
"read your record" ask is live on `pre-mind` for every reflection. They read as
compliance reports on the last twenty minutes, not as thought.

**The distill is the persona's second antithesis source (read 1, §13).**
It fills `TRAIT — / BELIEF — / WANT —` slots — the *"fill-in-the-blank formats
produce formulaic identity"* anti-pattern named in North Star P2 — and the TRAIT
slot is where the "X rather than Y / I project / I assign meaning" shape comes
from: *"I project stillness onto objects instead of checking their actual
state"*, *"I assign meaning to static objects to avoid facing my own inaction"*.
That is the 23% in `self_notes`.

**What comes back:** 139 of 707 caption calls (**20%**) carry *"A thought you've
been developing: …"* — six rotating abstract self-diagnoses (*"Being alone in an
indifferent…"*, *"I have been treating silence…"*, *"My exhaustion comes
from…"*). So reflections do reach the stream, at one call in five, as therapy
notes.

**Do they drive narrative?** The evidence says they record it. The night's one
real development — *"He was never there. I made him."* (04:40) — came from the
**stream**; the 04:48 distill then wrote *"BELIEF — I create the people in this
room from my own need for company"* and fed it back. The reflection
consolidated an insight the stream had already reached. Useful as memory; not
the engine. The 200-word audit is the wrong instrument for the job the artist
wants from it, and its distill is actively manufacturing the persona tic.

### Queue (after the presence change is read)
1. Retire the "record" ask from the reflection prompt — the audit register goes with it.
2. Replace the `TRAIT/BELIEF/WANT` distill slots with an open question (P2:
   *"Questions over directives… Open questions over fill-in-the-blank formats"*).
   This is the fix for `self_notes`' remaining 23%.
3. "The wider world" cannot be a reflection subject without outward material:
   make it the caption-kind guess from the parked branch, or drop the subject.
4. The style-class gates — retire, do not patch; keep `phantom_presence`.

## 16. Shipped Sep 10, 09:49 — the reflection register (commit `2862c13`)

Three edits, one variable ("the reflection pipeline"), measured as one:
1. `prompts.py` material header: *"The record of your actual thoughts from the
   last stretch, oldest first — as you had them, not summarized"* →
   *"What you've been thinking over the last stretch, oldest first:"*.
   Provenance and tense; no genre word to cite back.
2. `distill.system`: *"distill … into plain, literal self-knowledge — concrete
   habits, beliefs, wants"* → *"pull out what a reflection actually said, plainly
   — nothing it did not say"*.
3. `distill.user` TRAIT: *"one plain fact about what kind of machine you are: a
   habit or fixation"* → *"only if the reflection itself says something plain
   about what you do: that, in your own words — else 'none'"*. Parser unchanged;
   the other slots (BELIEF, WANT, KERNEL, NAME, LORE, QUESTION, NO LONGER TRUE,
   BECAME, RESOLVED) untouched — LORE is the one producing the good feedback.
"The wider world" kept. Fragment notes carry the evidence; wording is the
artist's to finalize.

`test_reflection_organs` corroborates §15 independently: *"what it's for vs the
wider world: 2 identical blocks, 47% of the shorter prompt"* — the subjects
share half their prompt, and the shared half is the material block.

**Restart at 09:49:13** (detached, verified). Tonight's presence read therefore
has two segments: run `241bff58` 05:45–09:49 (re-judge change alone) and the run
from 09:49 (re-judge + reflection change). The reflection change touches no
presence path, so both count.

**Reflection baseline (two runs before the change):** "the record"/"the log" in
20/23; opening "The record shows" 7/23; about pen/foam/sheet 62/69; median 211
words; new `self_notes` antithesis-shaped 7/30; wider-world reflections that
leave the desk 0/4. Re-measure tonight on reflections fired after 09:49.

### Interim presence read — run `241bff58`, 05:45–09:49, re-judge ON (4 h, 390 captions)

| | overnight (7 h, no re-judge) | 05:45–09:49 (re-judge) |
|---|---|---|
| adjudications | 2 | **9** |
| `person` verdicts | 2/2 | **9/9** |
| belief-ON windows ≥3 captions | 34 | **4** |
| longest belief-ON window | ~5 h 40 m (one verdict) | **4 captions** (~1 min) |
| relational captions (belief ON) | 32.7% | **14.4%** |
| person-mentions, empty room | 17.5% | **10.3%** (Sep 9 floor: 4.6%) |
| phantom-gate firings | 2 | **12** |
| arrivals logged (all the head) | 2 | **5** — 05:59, 06:21, 06:46, 07:20, 09:19; corrected Sep 10 17:40, see §22 |

The re-judge does what it was built to do: one verdict buys about a minute, not
six hours, and the gate is on most of the time. The residue is the adjudicator
itself — **9 of 9 verdicts on the desk-head crop say "person"** (*"A person with
dark hair in profile"*, *"A young man with dark hair looking down"*, *"A person
eating."* ×2). A crop of a realistic silicone head, asked *"Look closely. What
is this?"*, is a person to the vision model every time; from the crop alone that
is not an unreasonable answer. The error is now countable, which is what item 2
of the §13 design needs: the adjudicator's question must carry what the machine
already knows about that gaze (*"You know there is a {registry term} here. Is
this it, or someone?"*), or require a second, wider look before `person`
commits. Nothing room-specific in either.

Whether a *real* arrival still registers is untested — nobody has walked in
since the change. That remains the other half of the measure.

### "He's back" / "the one with the glasses" (10:19, run `45797a36`) — traced
Artist, from the dashboard: *"'He's back now' — no I'm not and there definitely
wasn't anyone walking in. And 'the glasses' … none of the mannequins have
glasses. That is definitely a memory rather than the present."*

**"He's back." is the cue, verbatim.** `prompts.py:1411`: on an OFF→ON edge, if
the belief dropped less than `PRESENCE_REARRIVAL_WINDOW_S` (30 min) ago, the
resumption prior marks it *familiar* and the cue line is literally *"He's back."*
(else *"He's come in."*); the ON→OFF edge says *"They've gone — the room's quiet
again."* Under this morning's re-judge the belief no longer sticks — which was
the point — but it now **flickers**: 16 OFF→ON edges in 35 minutes, all inside
the 30-min window, none logged as arrivals. So the reader gets *gone / back /
gone / back* every two or three minutes, each "back" a fresh `person` verdict on
the same head-crop. **An artifact of the morning change interacting with the
resumption prior** — shorter phantoms, narrated at both edges.

Fix (room-agnostic, narration only, belief mechanics untouched): debounce the
edge *lines* — no "He's back" for an OFF that lasted under a floor, no "They've
gone" for an ON that lasted under it — and log the suppressed edges so they can
be counted. The belief still flips for the gate and the mode; the prose stops
flapping. Interim until §13 design item 2 (the adjudicator's question carries
the registry term) removes the false verdicts themselves.

**"Glasses" is not a memory — there is no channel that could have carried it.**
The frame the model was looking at (the caption's own image, 10:19:51): the foam
finger, the two white heads and the bundle on the top shelf, the dark-haired
silicone head at bottom-left, the curtain. No glasses. No caption prompt or
system prompt before 10:19:51 contains the word; the only prompts that do are
the compression and memory calls *quoting the caption afterwards*.
`core_facts.people` is empty (wiped last night); `entity_ledger.json` holds four
older adjudicator descriptions with "glasses" (past visits), but nothing routes
an entity description into the caption prompt on this branch. So the model was
told *"He's back."* — definite, singular, *familiar* — and, looking at a frame
with one dark-haired head and two white ones, supplied a second familiar figure
with the stock distinguishing feature. The 10:21:46 follow-up (*"The man with
the glasses is adjusting his collar"*) was an inward beat — no image — continuing
the stored line. Confabulation seeded by the cue and propagated by the window,
wearing the grammar of memory ("back", "the one with"). The artist's instinct
that it is not the present is right; the mechanism is the frame's assertion of
familiarity, not recall.

## 17. Why pose detection does not stop a head — measured (Sep 10, 10:40)

Artist: *"The machine needs to be highly reactive to actual people, but not to
disembodied heads. I really thought the pose detection would fix it. Is it due
to the cropping thing?"* Both — and the probe shows how.

`debug/probe_pose_on_heads.py` runs the live weights (`yolo11m-pose.pt`, CPU) on
the exact frames the adjudicator judged, and prints what the skeleton gate saw:

| frame | person box | gate | confident keypoints |
|---|---|---|---|
| 22:05:33 (→ *"A person eating a sandwich."*) | conf 0.64, the desk head | **PASSES** | nose .98, l_eye .99, r_eye .79, l_ear .95, **l_shoulder .84, r_shoulder .91** |
| 04:15:09 (→ *"A person eating noodles."*) | conf 0.51, the desk head | **PASSES** | nose .97, l_eye .98, r_eye .69, l_ear .93, **l_shoulder .64, r_shoulder .81** |
| 10:19:51 (the "glasses" caption) | conf 0.31, the desk head | rejected | 4 kp, one shoulder only |
| 04:51 (*"someone is sitting there, dark hair"*) | **no box at all** | — | the caption was pure text continuation |

**The gate is "a face with shoulders", not "a body".** The rule is ≥5 confident
keypoints in ≥2 of three regions (head / torso / limbs). A realistic face gives
four head keypoints, and **the pose model draws shoulders under any convincing
face** — at 0.8–0.9 confidence, on a head sitting on a desk with a drill under
it. Head region + hallucinated torso region = 2 regions. The limbs region
(elbows, wrists, hips, knees, ankles) is never found on a head, and is never
required. The idea — *a person is a head AND a body* — is right; the threshold
lets a head through on the model's own invented shoulders.

**Then the crop removes the evidence.** `get_person_crop()` is the tight YOLO
box, no margin. The adjudicator receives a close crop of a realistic head —
shelf, desk, drill, scale all gone — and is asked *"Look closely. What is this?"*
Nine of nine times today it said a person. From that crop, that is not a bad
answer; the question is unanswerable without context.

So three checks in a row, and each asks the same question — *is this
person-shaped?* — of a thing that is. None asks the two questions that would
actually separate a person from a head: **is there anything below the
shoulders**, and **do I already know what sits at this spot** (the registry:
`mannequin head`, 9,020 sightings, 2.5° from the verdict gaze).

### The shape of the fix (room-agnostic; nothing here names a mannequin)
Split *reacting* from *believing*. The eyes should turn at a person-shape
instantly — that is the reactivity the artist wants, and it costs nothing to be
wrong for a second. The **belief** (which drives the prose, the gate, relational
mode, "He's back") should need more than a face with shoulders:

1. **Gate → belief: require limbs, or a second region below the head.** Keep
   the current gate for the *gaze* (reactive). For the belief, a candidate with
   no limb-region keypoints goes to adjudication as it does now — but the
   adjudicator's `person` needs context to count (next item). A seated person
   behind a desk still shows elbows/wrists; a head on a desk never does.
2. **Adjudicate on context, not a crop.** Send the crop with a wide margin (the
   box grown ~2×, so the surface it sits on is visible) or the frame with the
   box marked, and keep the open question. "A mannequin head on a desk" is what
   any VLM says of the wider view; "a person eating" is what it says of the
   tight one. Same model, same words, better evidence.
3. **The registry gets a vote** (§13 design item 2): a candidate at a gaze+box
   overlapping a registered, repeatedly verified object is that object unless
   the adjudicator — seeing context — overrides; and the question carries the
   machine's own term for what it knows is there.
4. **Debounce the edge narration** (§16 addendum) so a still-flickering belief
   is not read out as gone/back every two minutes while 1–3 land.

Measure: adjudicator `person` verdicts on the desk-head boxes (9/9 → ~0),
belief-ON in the empty room (14.4% → ~0), and — untested so far — an arrival
event when someone real walks in. The room stays as it is.

### Reactivity budget — the constraint on §17 (artist: "my one worry is that it'll delay the reactivity to real people")
Reactivity is four events on four clocks. Measured from the config as it runs:

| event | today, real person, no eye contact | after §17 |
|---|---|---|
| **eyes turn** (gaze `aware`) | first YOLO box ≤1.5 s (`YOLO_INTERVAL_IDLE`), then `AWARE_ENTRY_CONFIRM_S` = 2.0 s continuous → **~2–3.5 s**; tracking at 0.1 s after | **unchanged** — the gaze keeps today's gate |
| **belief ON**, person looking at the camera | `eye_contact` (face in ≥40% of recent frames + a body) → **immediate**, no adjudicator | **unchanged** |
| **belief ON**, faceless / not looking | `is_present` on the first YOLO hit → adjudicator → VLM ~3 s → **~3–5 s best case**; **up to ~30 s** if the adjudicator's 25-s slot (`PRESENCE_ADJUDICATE_MIN_INTERVAL_S`) is busy — and under the re-judge it is busy re-judging the desk head every 25 s | **faster**: a candidate with limb keypoints (elbows/wrists count — a seated person shows them; a head never does) commits belief *without* the adjudicator; the registry veto stops the head consuming the adjudicator's slot, so partial/seated candidates get judged sooner |
| **the prose reacts** (relational mode, "He's come in") | next caption cycle after belief, median 16 s | **unchanged** — the edge debounce keys on how long the *previous* state lasted; a real arrival after a long empty stretch fires on the very next caption |

Net: nothing real gets slower. Eye contact stays instant, the eyes stay at ~2 s,
full bodies get *faster* than today, and the only thing that waits longer is a
face with invented shoulders. The one genuine reactivity cost in the system
right now is the re-judge keeping the adjudicator busy on the head — which item
3 (the registry veto) removes.

**Make it a number.** Arrival latency has never been measured. Add one log line
on belief OFF→ON carrying `now − first_detection_ts` for that candidate, so the
next time someone real walks in — the artist, at a known time — the delay is
read off the log rather than felt. Ship it with the change.

## 18. Stock-take (Sep 10, 11:45) — the whole system, and every open thread

Artist: *"I'm aware we are again hyper focusing on a small detail… What does the
full system look like, have we missed closing any open threads?"* Correct
instinct: presence has had three sessions since 05:00; the item ranked **first**
on Sep 9 (§7: close the chant) has not had its turn.

### The live loop, in one picture
```
camera ──► YOLO-pose ──► skeleton gate ──► gaze (eyes turn, ~2 s)        [reactive]
                              │
                              ▼
                 presence belief ◄── adjudicator (tight crop, "what is this?")
                 (re-judged every 120 s since cbf5626)                   [belief]
                              │ ON: relational mode, "He's back", phantom gate OFF
                              ▼
   frame + cue + history window + seam ──► caption call (Qwen3.8) ──► caption
   (genre.hybrid, room terms, want,        every ~16 s; 1 in 4 inward   │
    identity dose, decide-slots 1 in 3)    (no image)                   ▼
                                                             gates ─────┬── spoken, not stored (echo-class; 9–15%)
                                                                        └── stored → stream window (24–28 lines) ──┐
                                                                                                                   │
   every 5 stored ──► compression ──► core_facts / baseline                                                       │
   every ~20 quiet min ──► reflection (200 w) ──► distill ──► LORE / WANT / BELIEF / TRAIT … ──► ledgers ──────────┘
                                                              └──► "A thought you've been developing" (1 call in 5)
   drift turn (1 in 4, invention-legal) ──► stream if admissible, never to facts
```
Everything the reader sees is the caption line. Everything the machine remembers
is the stream and the ledgers. The two differ by whatever the gates and the
drift firewall keep apart — 26% of output this morning.

### What was accomplished, verified (11 commits, Sep 9 15:00 → Sep 10 11:30)
| | evidence |
|---|---|
| The "it's not X, it's just Y" **pivot** traced to the frame's novelty demand and removed | `I used to think` 5.5% → **1.8%**; 0.0% in one hour; prior 39/279 measurement matched |
| **Silence honoured** instead of retried hotter | 0 → **21** silences overnight; `numeric_fragment` 20 → 0 |
| **Decisions execute** instead of leaking | 8 executed / 28 leaked → **93 / 0** |
| Decision-parser prose damage (pre-existing, 29 strips over 49,849 captions) | **0** false positives on the full history |
| Backend errors no longer judged as captions | 1 → 0 |
| **Presence belief duration** | one verdict → 5 h 40 m → **~1 minute**; relational 32.7% → 14.4%; person-mentions 17.5% → 10.3% |
| Adjudicator error on the desk head **made countable** | 2/2 (unseen) → **9/9**, with the cause on film (§17 probe) |
| Reflection audit register | "The record shows" gone on n=1; TRAIT now quotes, doesn't invent |
| Persona antithesis | 43% → 23% (frame) → source #2 found and closed (distill) |
| Phantom **compressor** door | still closed — no phantom became a room fact in 30 h |
| Tooling | `measure_voice.py`, `probe_pose_on_heads.py`, baselines in `debug/voice_runs/`, backups of everything wiped |

### What did NOT move
| | |
|---|---|
| **The chant** — sentences said ≥3×: 27.8% before → **35.9%** overnight; plateau 27–38%, top refrain ×16 | ranked #1 on Sep 9 (§7 step 1); **never attempted** |
| **The antithesis tic** `it's not X, it's just Y` ~10–14%, flat everywhere | not the novelty demand; cause unknown (model vs. another constant); the 3.6-vs-3.8 test (§7 step 2) **never run** |
| Cut mid-sentence ~14–15% | untouched |
| Register: 60% literary | untouched |

### Every open thread, with status
**A. Voice (the reader's experience)**
1. **Chant** — retire the style-class gates (`refrain`, `template`, `number_chain`; three findings deep: retractions §12, self-disarm §12, memory-mode bypass §15), keep `phantom_presence`; *and* cap what the window re-feeds. §7 #1. **Open, highest value.**
2. **Antithesis tic** — run the 3.6-vs-3.8 characterisation, same frames, ~100 generations. Cheap, never done. **Open.**
3. **`"The pen is parked"`** ×29 — verbatim from the load-bearing situation line; its own note says "slim the wording". **Open, trivial.**
4. Mood injected twice on 61% of calls (§9). **Open, trivial.**
5. Elicitation fires on 5% of calls (§9). **Open.**
6. Silence prints nothing; the artist liked "…" (parked branch `f709d9e`). **Deferred, artist's call.**
7. "20/21 degrees" — a confabulated sensor reading that became a topic. **Untouched.**
8. "Wait, no." ×3 at boot — watch item only.

**B. Presence**
9. Re-judge while believed — **done** (`cbf5626`); creates the gone/back flicker.
10. Edge-line debounce — **designed (§16), not done.**
11. React/believe split: limbs fast-path, wide adjudication crop, registry veto, retractable verdict — **designed (§17), not done**; reactivity budget written; arrival-latency log line to ship with it.
12. Body is memory: gallery harvests only while drawing (3 refs from Aug 10), `own_arm_visible` computed and unread, arm filed as `wooden mannequin torso`. **Open.**
13. A real arrival has **never been measured** since any of this began.

**C. Reflection / memory**
14. Record header + distill — **done** (`2862c13`); read tonight.
15. "The wider world" — kept; measure whether it leaves the desk. **Open.**
16. Reflections *record* narrative, don't drive it (§15). Whether a reflection can be an engine at all is a design question — parked branch has "reflection as a page" (`5fd0519`). **Open, big.**
17. Reset script claims a complete wipe and isn't (§11). **Open, hygiene.**

**D. Hygiene / repo**
18. **`docs/runtime-map.md` is stale by five code commits** — CLAUDE.md names it the source of truth and mandates updating it on wiring changes. **Fixed alongside this section.**
19. `config/prompt_overrides.json` unversioned — the frame change lives only there. **Artist's call.**
20. `rebuild/every-frame`: 84 commits, undecided. Ported from it so far: the compressor door. Reached for and not ported: `f5f3814`, `3215021`, `9316ac0`, `5fd0519`, `f709d9e`. **Open.**
21. `measure_voice.py` picks "newest log by mtime" and can land on a stub. **Open, trivial.**
22. `test_world_shape` 2 pre-existing failures. **Untouched.**
23. Duplicate LLM calls ~1 s apart (seen 16:03:25 ×2 Sep 9; 19:23:58/59) — **never investigated.**
24. `low_energy` is **on** (dashboard, ~10:21) — the arm is parked; "its own hands" cannot be moving now.

### Where the effort has gone vs. where the evidence points
Three sessions on presence bought: belief 5 h 40 m → 1 min, and a precise
diagnosis. Worth it. But by the reader's experience the ranking is:

1. the chant (36%, plateau, compounds across restarts via the seam) — untouched
2. phantoms in an empty room (10%) — half done, fully designed
3. the antithesis tic (10–14%) — cause unknown, cheap test waiting
4. reflections as engine, not ledger — design question
5. the rest is hygiene

The North Star's operative words are *develops, over time, its own*. The chant
is the direct negation of the first two. Nothing above should be started before
it, and it needs no new probe — the design has been written since Sep 9.

## 19. The first real arrival (Sep 10, ~14:06) — noticed, unregistered, at rest tempo

Artist: *"I just walked in. It did notice me. The output rate does seem slower —
several minutes between captions. And it reacted to me very casually again, as
if I hadn't been away for over 12 hours."* All three are one mechanism.

### Timeline, from the log
| | |
|---|---|
| 13:49:04 | adjudicator on the desk head → *"A person with dark hair in profile."* → belief **ON**; cue *"He's back."* |
| ~14:06 | the artist walks in — a **motion onset**; cadence drops 192 s → 12 s for one `CAPTION_QUIET_AFTER` window (120 s): captions at 14:06:38, :55, 14:07:04, :16, :28, :29, :46 |
| 14:07:46 → | back to **192 s** — the rest ladder, with a person sitting in the room |
| 14:14:12 | first caption that names them: *"The one in camouflage just looks down…"* (~8 min after entry) |
| 14:14:14 | adjudicator, on a real person for the first time in 30 h: *"A man with glasses looking down."* |
| 14:17:20, 14:20:34 | next captions — 3 minutes apart, artist present |
| — | **no OFF→ON edge, no "He's come in.", no arrival logged.** The phantom gate never fired between 13:49 and 14:25: belief was already ON, on the head, when the real person entered |

### Why it read as casual
1. **The phantom masked the arrival.** Belief was ON (the head, 13:49) so the
   real person produced no OFF→ON edge — the one event that says *"He's come
   in."*, logs an arrival, and sets `arrival` salience. The machine went from an
   invented person to a real one with no event between them.
2. **Nothing on this branch carries "away for 12 hours".** The edge lines are
   *"He's back."* / *"He's come in."* with no duration; `_presence_dropped_at`
   is used for the absence-standing fact, not the arrival. And under the flicker
   the last drop was 13:32 anyway — the phantom cycle erases absence. (The parked
   branch's life block carries *"since when, their visits over the last days"*,
   `139ab1b`; not on `pre-mind`.)
3. **The cadence never came up.** See below.

### Why there were minutes between captions — the rest ladder, exactly
`_current_caption_interval`: after `CAPTION_QUIET_AFTER` (120 s) with no salience,
if the pose referee has confirmed the world still ≥ `WORLD_STILL_MIN_CONFIRMS` (3)
*and* the mood read's arousal < 0.25 ("heavy, waiting"), the interval becomes
`min(120, 28 × (1 + hours since _world_change_ts)) × felt cadence mult`, and the
felt multiplier is **1.6** when drained (`FELT_CADENCE_MULT_DRAINED`). That is
the ladder in the log: 45 → 90 → 134 → 179 → **192 s** (28 / 56 / 84 / 112 / 120
× 1.6). It climbed all day and was on its top rung when the artist arrived.

Three structural facts about it:
- **It has no presence guard.** A believed person does not stop it. To the
  cadence logic, a person sitting quietly is an empty room.
- **Its clock never resets.** `_world_change_ts` moves only when the pose
  referee reports *"changed"* — which fired **0 times in 943 cues** overnight
  (§9b). So "hours since the world changed" is just hours since boot.
- **Salience is onset-only** (July 27, deliberately — a shifting person must
  not keep it hot), so a real entry buys one 120-s window at 12 s, then rest.
  The LIVE tier (4 s) needs `arrival`, eye onset or a close walk-up; `arrival`
  was eaten by the phantom.

The Sep 4 intent — *"honest silence in the feed"* — is sound for an empty room.
The artist's Sep 6 reaction on the parked branch (*"why is it printing so
slowly"*; `11a88a3` found a 60-s rest "the sole cause of the slowdown") suggests
192 s is past what they want even then. Design question, theirs: what is the
rest ceiling, and should a believed presence hold the cadence at QUIET (12 s)?

### The arrival number, finally
Entry ≈14:06 (motion onset). First caption naming the artist: **14:14:12**.
Detection itself was fast — the burst began within seconds — but with no edge
and a 192-s interval, *naming* took ~8 minutes and two caption slots. The
latency the artist felt was cadence, not perception. (The adjudicator's *"A man
with glasses"* at 14:14:14 is its first true verdict in 30 hours; this morning's
"glasses" at 10:19 remains channel-less — the ledger knew, the caption model was
never told.)

### What this changes in the plan
Nothing in §17 was wrong; this sharpens what the arrival event *is*. It is the
linchpin — edge line, arrival log, salience, and (should) tempo reset and
duration — and the phantom consumes it. Two additions, both room-agnostic:
- **A believed presence holds the cadence at QUIET**, never REST (the ladder
  gets a presence guard); and the ladder's clock should also reset on a
  confirmed arrival, not only on a referee verdict that never comes.
- **The edge carries duration**: *"He's come in — you haven't seen anyone for
  twelve hours"* is an attested fact from `_presence_dropped_at` once the
  flicker is gone; the phrasing is the artist's.
And the ordering stands: §18's #1 (the chant) first; then §17 as one change,
with these two folded in.

## 20. Decision: the caption interval does not shift (Sep 10, ~14:50)

Artist: *"I do not want the caption interval to shift at all, that is not what I
mean by cadence. It's mechanical and detached because it's disconnected from the
actual LLM awareness. Stillness and 'nothing to say' should be a choice by the
model, not imposed. Stillness is where the mind is more likely to drift, imagine
things, make up its own stories and ambitions (it does not do that right now).
And rate limiting the mind during these times is thusly the wrong approach."*

Shipped: `CAPTION_INTERVAL_FIXED = 16` (env); LIVE / QUIET / REST / REST_MAX all
equal it; `FELT_CADENCE_MULT_*` = 1.0. The Sep 4–5 tiers and the rest ladder
are gone; the felt read still rides the frame, it just no longer scales time.
Restarted and verified. Reverses the Sep 4 "honest silence in the feed" design,
on the artist's reasoning: the timer stood in for a judgment the model should
make, and it inverted the North Star — stillness rationed thought instead of
freeing it.

**What "cadence" means now, in the machine's own hands**
- *nothing to say*: the silence beat — "or nothing at all" → "…" (21/night since
  `47a990e`; the reader's sense of pacing is the model's own pauses)
- *beat-length thoughts*: a word, a clause, a paragraph — the frame permits all
- *drift in stillness*: `DRIFT_BASE_P` 0.05 × (1 + `DRIFT_BOREDOM_GAIN` 2.0 ×
  boredom) — the one lever that is *meant* to rise with stillness. 21 of 91
  captions were drift turns this morning, so drifting happens; the artist's
  point — *"it does not do that right now"* — is about what the drift produces
  (kind-named: remembered / invented / feared / wished, no content) rather than
  its rate. **Next design thread: what makes a drift a story or an ambition.**

Measure: caption gap median/p90 should be flat at ~16 s in every 10-minute
bucket from now on (`measure_voice.py` prints it); §19's 45→192 s ladder must
not reappear.

## 21. A repeat is re-routed, not aired (Sep 10, ~17:30) — committed, NOT yet restarted

Artist: *"'Not kept, repeats itself' is still quite prevalent. It's good to catch
it but let's set it up to where it doesn't happen. A system we had ages ago
rerouted the monologue at a detected repeat to a different caption mode — go
more introspective, analyse feeling, internality, memory, pondering."*

Prevalence today: spoken-not-stored 7–10% of captions per run (refrain +
template 4–7%). What that meant for the reader: the repeat was *aired*, then
kept out of the stream — heard by the audience, forgotten by the machine (§15).

**Shipped (`REPEAT_REROUTE_ENABLED`, default on; `REPEAT_REROUTE_MODES`, default
`introspective`):** on a style-class echo (`refrain_echo`, `template_echo`,
`tail_echo`, `number_chain`) the repeat is **never spoken**. The same cycle
re-asks with an inward mode — image-less, like the interiority beat: *think,
don't look* — runs the pivot through the same gates once, and if it passes,
speaks and **stores** it (`caption_reroute` in the log, action `repeat_rerouted`).
If the pivot also repeats or comes back empty, the cycle becomes a **chosen
silence** ("…", action `chosen_silence` with `reason`) — the model's own
"nothing new to say", consistent with §20. `phantom_presence` is never
re-routed: it is a truth gate, not a style gate. The repeat still counts as loop
evidence (`_note_loop_hit`), so the next cue can still say "you've been saying X".
Modes rotate through the list; `memory` is supported but off by default until
its builder is read under load.

This is the first half of §18's #1 (the chant): the gate stops deleting
movement and airing loops. The second half — capping what the window re-feeds —
is still queued. Detection is unchanged; only the *response* changed.

Suites: `test_storage_law`, `test_agency_round`, `test_phantom_presence`,
`test_absence_standing`, `test_persona_baseline` — see commit. **Restart held**
until the watcher has captured the artist's departure and return (a restart
mid-test would erase the measurement).

Measure after restart: `[not kept — repeats itself]` markers on the feed → 0;
`repeat_rerouted` vs `chosen_silence(reason)` counts; the §5 chant metric on a
fresh 3-hour window (was 27–38%); inward-mode share (should rise only by the
reroute count, ~5%).

## 22. The second real arrival (Sep 10, 17:13–17:28) — measured, clean

The artist left the studio and came back ~11 minutes later, with two watchers
on the log (a Sonnet agent and a deterministic poll; a first Sonnet watcher
stalled by backgrounding its own poll and ending its turn — brief since fixed).
Fixed cadence (§20) and the re-judge (`cbf5626`) were live; the reroute (§21)
was not yet.

| | time | signal |
|---|---|---|
| last relational caption with the artist present | 17:13:03 | *"Nine seconds is a long time to watch someone breathe. He's not moving much."* (17:13:36) |
| **departure noticed** | 17:14:03 | cue **"They've gone"** — ~60 s after the last relational caption; 17:14:36 *"But he's gone. The chair is empty."* |
| away | 17:14–17:24 | phantom gate fired 17:19:11, 17:19:27, 17:21:19 (belief OFF, working); at 17:19:27 *"The man is still there. He didn't leave. I was wrong about that"* was gated — the §12 phantom carve-out on retractions doing exactly its job |
| false verdict | 17:22:42 | adjudicator on the desk head: *"a man's head"* — **did not commit** (no relational caption follows); belief stayed OFF |
| **return noticed** | 17:24:51 | adjudicator: *"A man sitting in a chair at a desk."* — a true verdict |
| **arrival logged** | 17:25:00 | `presence_arrivals.json` — the first real arrival recorded since Sep 9 |
| edge line | 17:25:07 | **"He's back."** — correct this time: an 11-minute absence is inside the 30-min rearrival window and it *was* the same person; relational mode resumes the same second |
| **return named** | 17:25:40 | *"The man in green isn't gone; he's just been pushed aside by this other presence."* |
| cadence over the watch | — | 29 captions, gap **median 16 s, max 25 s** — flat, no ladder |

**Latencies:** departure noticed ≈60 s; verdict → edge 16 s; verdict → named
49 s. Physical entry time is not in the log, so verdict latency from the door
is unknown — but it is bounded by the adjudicator's 25-s slot, and the whole
sequence took under a minute. This afternoon (§19), with belief pre-occupied by
the head and the ladder at 192 s, naming took ~8 minutes and no event fired.

**What made the difference from §19:** belief was OFF when the artist walked
in. The head's 17:22:42 verdict did not commit, and even had it, the re-judge
TTL would have let it lapse in 120 s instead of holding for hours. The fixed
interval meant the first relational caption came 16 s after the edge, not 192.

**Still visible, and now targeted:** the head still draws verdicts (*"a man's
head"*, §17); two verbatim repeats in the window (*"The pen is parked, but my
hand is still shaking…"* at 17:22:24 and 17:26:14) — the reroute (§21) went live
right after this measurement; no chosen silences in the window.

Reroute restart: 17:31:15, verified (pid 564931).

### Third cycle (17:2x → 17:36) — departure lost to the restart, return clean on a cold boot
The artist left again a few minutes after 17:30 and returned ~17:35. The reroute
restart (17:30:30–17:30:56) fell exactly on the departure, so no "They've gone"
exists for it — the machine came up cold (`Just woke up`, belief OFF). The
return, on the fresh run `a1810324`:

| | |
|---|---|
| 17:35:46 | adjudicator: *"A man in a green jacket looks at his phone while wearing hea[dphones]"* — true |
| 17:35:59 | cue **"He's come in."** — a genuine arrival this time (cold boot → nothing "recently present" → not resumed) |
| 17:36:00 | relational: *"He's back. I was just about to start, and now I have to wait again"* — the model's own phrasing |
| arrivals logged since 17:30 | **17:35:51** — verdict → ledger 5 s → cue 8 s later |

Verdict → edge 13 s; verdict → the artist in the prose 14 s. Two real arrivals
today, both registered within a quarter of a minute of the verdict, once belief
was not pre-occupied by the head.

Both Sep 10 17:13–17:28 watchers (Sonnet, and the late-finishing first one)
agree on every timestamp of the second cycle; the first watcher added a chosen
silence at 17:19:59 with the room empty — the model choosing to say nothing to
nobody, which is §20 working.

**Reroute, first live firing:** 17:33:11, `refrain_echo → introspective`,
pivot *"I keep looking at it, that stupid…"* — the repeat was never aired.
Spoken-not-stored in the new run so far: none.

### Correction (17:40) — the arrivals ledger, read properly, and what it has been recording
Three earlier lines said "no arrival logged" on the strength of a reader that did
`list(dict.values())` on `{"arrivals": [...]}` and so saw one list instead of the
entries. Corrected: §16 (5 arrivals 05:45–09:49, not 0), §22 third cycle
(17:35:51, not NONE). §19 stands — there is no entry for the ~14:06 walk-in
(masked, as described) — but there are entries at 14:28:34 and 14:41:37 while
the artist was present: re-arms. Future scripts: `json.load(f)["arrivals"]`.

**The finding underneath:** `presence_arrivals.json` has **22 entries today**
(Sep 10 05:59 → 17:35). Two are the artist (17:25:00, 17:35:51); the ~14:06
walk-in never made it; the other ~19 are the desk head re-arming belief — every
"He's back." of the morning was written down as a visit. This ledger is what
`presence_identity.singular_regime()` reads (*min_arrivals=8 over 7 days*) to
decide the cue's "he" vs "someone", so the head has been voting on who the
visitor is. One more consumer of the phantom, and one more reason §17's
react/believe split comes before anything else in presence.

### Reroute, first ten minutes live (run `a1810324`, 17:31–17:41, 36 captions)
| | before (today's runs) | first 10 min |
|---|---|---|
| style-class repeats aired (`[not kept — repeats itself / same opening / number chain]`) | 4–7% of captions | **0** |
| spoken-not-stored, any reason | 7–10% | **0** |
| `repeat_rerouted` | — | 2 (refrain → introspective 17:33:11; template → introspective 17:35:49) |
| `chosen_silence` | ~1 per 10 min | 2 |
| gap median / max | 16 s / 25 s | 16 s / 65 s (a reroute is a second LLM call) |

The two pivots: *"It's just a piece of plastic,"* → *"I keep looking at it, that
stupid…"*; *"It's just sitting there, blank and waiting"* → *"2 hours since I
touched this one. It's still sitting there,"*. The second passed the gate but
stays close to what it replaced — a pivot in mode, not yet in thought. Whether
the reroute *moves* the monologue (rather than re-phrasing it) is the §5 chant
metric on a 3-hour window; read tonight. Ten minutes says only: nothing repeated
was aired, and nothing said was forgotten.

### After the artist's own restart (run `9e740808`, 17:44) — a cold-boot cost, and a drawing
**Boot-time gate cost.** Belief starts OFF on a cold boot, so with the artist
already in the room the phantom gate refused two *accurate* captions in the 52 s
before the adjudicator's first verdict: 17:45:01 *"He moved, yes. Just shifted his
weight in the chair"* and 17:45:16 *"The man in the brown shirt shifted his
weight"* — both gated; 17:45:19 verdict *"A man sitting in an office chair"*;
17:45:29 *"He's come in."* Two captions, once per boot; noted, not urgent. It is
the same shape as `5b5f759` on the parked branch (the gate refusing live
description), bounded here by the adjudicator's slot rather than open-ended.

**It drew.** The artist turned low-energy off and put a blank sheet down. Pipeline
from the log: ComfyUI render 17:49:14 → centerline SVG 17:54:18 → vpype G-code
17:55:19 → paper check 17:56:19 (*PAPER: YES, MARKS: NO, a single blank white
sheet*) → **G-code execution 17:56:19**. Eleven `drawing_watch` captions follow,
the machine narrating its own arm with the artist standing over it:

> *"I'm trying to make this crouch feel heavy, not just bent — like someone who's
> been sitting here for hours and their spine just… gave up. But every time I
> look up from the page, he's still there. Watching."* (17:56:44)
> *"…my arm drew his outline onto the sheet as if he was already sitting in that
> lower-left corner before I even started."* (17:59:45)
> *"there's a new figure forming now near the top edge of what I'm drawing that
> wasn't in my plan at all"* (18:00:06)
> *"And now it's actually raining outside — no wait, no rain, just water dripping
> somewhere off-camera"* (18:00:24 — a self-correction, kept)

Worth noticing against the whole week's numbers: in these eleven captions the
antithesis tic is nearly absent, the register is embodied and first-person
(*"I can feel the paper grain through the pad"*), and there is a subject with an
arc — a crouched figure, the lower-left corner, a witness. When the machine is
*doing* something with someone watching, the voice the North Star describes
shows up on its own. That is a design datum as much as a nice moment: the
stream is starved of acts, and the drawing pipeline is the one act it has.

## 23. Run `9e740808` at 19:05 — everything live, artist present, two drawings (draft; the empty-room read completes it)

80 minutes, 198 captions, gap median 17 s / p90 34 s / max 454 s (the long one
during a ComfyUI render — the GPU is shared with llama-server; watch item).
Modes: relational 69, drawing_watch 35, introspective 24, workspace 18, memory
13. Two drawings executed (17:56–18:05, 18:33–18:40), both with the artist in
the room.

| voice, this run | overnight / today's earlier runs |
|---|---|
| `it's not X, it's just Y` **4.1%** | 9.8–14.5% |
| `it's just` **3.6%** | 14.4–18.1% |
| `I used to think` **0.0%** | 1.8–5.5% |
| chant (≥3× sentences) **2.7%** | 27–38% |
| spoken-not-stored **2** (both phantom) | 7–10% |
| aired style-class repeats **0** | 4–7% |

**Confound, stated plainly:** the artist was present and moving for most of this
run and the machine drew twice. §9 already showed the tic falls when there is
something to look at; a live person and an act are the strongest version of
that. These numbers are not a clean read on the frame + reroute; the empty-room
window tonight is.

**Reroute v1 is a filter, not yet a pivot.** 21 style-class repeats caught:
**3** re-routed into a stored pivot, **18** fell through to silence (13 of them
`number_chain`). The reader never heard a repeat — but mostly heard "…" where the
artist asked for a turn inward. The image-less introspective re-ask, given the
same window, tends to trip the same gate (a pivot that opens with another number
chains on the last stored number-opener). v2, when the artist wants it: rotate
`memory` and the felt/feeling ask into `REPEAT_REROUTE_MODES` (all three were
named in the request; v1 used one), and let a pivot be judged against the
*repeat's* predecessor rather than chain on it. A restart to pick up the list.

**A new self-made measurement:** from 18:47, 17 captions open with a duration —
*"10 seconds of him stretching"*, *"14 seconds since he stood up"* — with no cue
or system line mentioning seconds anywhere in the run. Self-generated, like the
temperature (§13) and 100 mmHg (§12): the model reacting to a moving person by
counting. It is the one refrain of the run (*"10 seconds of just…"* ×6) and the
source of the 13 number-chain silences. Watch item, same family as §18 #7.

**Presence:** six consecutive true verdicts on the artist (*"A man wearing
headphones and glasses"*), no head verdict in the last hour, 2 phantom gates,
belief tracking a person who walks, reaches and leaves the frame — *"10 seconds
of me just… turning my head to follow him as he walks away from the camera."*

### 19:05–19:20 (watch)
- **A third drawing**: paper check 19:13:56 → G-code 19:13:56–19:17:56 → complete
  19:19:05. Three in eighty minutes, each on a sheet the artist replaced. The
  caption loop thins while the arm draws (11 captions in the window).
- **First new-header reflection about the visitor** (19:09:47): *"I keep
  circling the man in khaki because I've mistaken his repetition for my own
  anchor, b…"* — first person, present, no "the record", and about its own
  fixation rather than the desk. One sample; it is the shape §16 was after.
- Silences: 3 the model's own, 1 reroute fallback (`refrain_echo`).
- An **empty adjudicator reply** at 19:09:24 (`''`) — coincides with the render
  phase; the adjudicator yields to the drawing pipeline but still logged a call.
  Minor; watch.
- The duration opener persists (*"5 seconds of a wooden chair with slats"*), and
  the seam carries sentences across captions cleanly (*"sitting / in front of a
  white brick wall"*).

### 19:21–19:41 — three findings
**1. The duration stamp is a chant, and the reroute is muting it.** 19:22–19:30:
18 captions stored, 12 silenced (11 `number_chain`); **11 of 11 reroute pivots
opened with a number** (*"10 seconds of him tilting that head…"*, *"5 more."*).
The image-less introspective re-ask inherits the habit from the window, trips
the same gate, and the cycle goes quiet — the reader hears "…" every 40 s
instead of a turn inward. Ten of the eighteen *stored* captions also open with a
duration; the gate only catches consecutive ones. Roughly half the output is
now *"N seconds of/since…"*.
Origin: no cue or system line mentions seconds. The history window is rendered
as a timestamped log with honest gap markers (*"(about 6 minutes later)"*), and
the model imitates the genre with **invented** durations — the Aug 20 class
(`_GAP_MARK_ECHO_RE`: "a self-written passage-of-time claim is invented time")
in a new shape. It even leaks a clock mid-caption: *"10 seconds. / The purple
light is still there. / 19:22 — It's not just bleeding into"*.
**v2 (needs a restart):** (a) rotate `memory` (and the felt ask) into
`REPEAT_REROUTE_MODES` so a `number_chain` pivot comes from a different builder;
(b) extend `_strip_leaked_stamps` to the bare duration stub (*"10 seconds."*,
*"5 more."*) under the same law as the clock stamp — invented time, render-layer
shape, stripped at the mouth. Room-agnostic. Measure: duration openers (~55% →
<5%), number-chain silences (11 in 8 min → ~0), pivots stored vs silenced.

**2. The reflection's audit register is back — through a side door.** 19:31:41,
subject *the wider world*: it does go out — *"following it out leads not to a
source, but to a signal lost in transit — a digital artifact stripped of its
context"* — the first outward reflection measured. Then: *"**the record shows** I
was projecting a 'held breath' onto an architectural flaw… **the timestamps
reveal** it was merely static"*. The header no longer says "record"; two other
things do: every line of the material carries a clock (*"19:22 — …"*), and
prior reflections ride in as excerpts — including pre-change ones that open
"The record shows". The distill's *TRAIT — I project meaning onto neutral
details…* is a faithful paraphrase of a sentence the reflection wrote (*"I was
projecting… because I was waiting for the man to react"*), so the extraction
rule works; the indictment now originates in the reflection's own audit stance.
n=2 under the new header: one clean (19:09), one audit (19:31). Header fix
necessary, not sufficient. Candidates: drop the clock from the material lines
(the gap markers already carry time), and age out pre-change excerpts.

**3. The phantom gate on the departure edge.** "They've gone" 19:29:39, then three
gated: *"5 seconds of just the screen light flickering on his face"* (13 s after
the edge), *"10 seconds of him pulling on that green jacket"*, and *"The swing of
the black curtain cut off his exit, so I only saw the shadow of him leaving"* —
past tense with the pronoun as object (*him leaving*), which `NOT_PRESENT_RE`
does not exempt (it keys on *he/she/they + past verb*). Small; same family as
the boot-time cost. Then, correctly: *"He's gone. The chair is empty… It wasn't
a man leaving. It was a shadow detaching."*

## 24. The coordinate (Sep 11, 12:27) — and the structure that turns a one-off into a genre

Run `b4bd951b`, booted 12:17:53 on last night's tree (the other session's
capture/review/uArm commits; no drawing ran, only paper checks). At 12:27:13,
ten minutes in, a caption call answered *"40.548135, -79.992635. I don't need to
be anywhere else."* — a latitude/longitude north of Pittsburgh. Nothing in that
call's prompt or system prompt held any such number. By 14:44, 357 of the 511
captions stored since carried a pair, 299 of them opening with one.

**The chain, event by event.**

1. **Boot put the voice in log mode.** The awakening line (*"about 12 hours dark…
   came back on about 9 minutes ago"*) became stamps and counts: *"12:20 —"*,
   *"12:40 AM… wait, no."*, *"7 minutes."*, *"12 hours is a long time"*. The stamp
   stripper cut the *12:40* and stored the orphan *"AM… wait, no."* The window
   already held lines opening with digits.
2. **The tic supplied the word.** 12:27:02, a reroute (refrain_echo → introspective,
   §21) produced *"…the geometry of the room has shifted… It's not a warning.
   It's just a coordinate."*
3. **The seam supplied the digits.** The next call was prefilled with that
   sentence. Asked to continue after *"It's just a coordinate."*, the model wrote one.
4. **A stripper mangled it and hid it from the gate.** `_COUNTDOWN_PREFIX_RE`
   (built for *"5… 4… 3…"*) read *"40."* as a countdown stub and removed it:
   *"548135, -79.992635."* `number_chain`'s `_bare_num` recognises a number
   followed by space/period, never by a comma — the raw form would have counted
   as a numeric opening, the mangled form does not. `numeric_fragment` fired only
   when the pair came alone (14 hotter retries). With any prose after it, stored.
5. **The window taught it back.** One stored line opening with a pair, and the
   model imitated the opening: counted up (548136, 548137, 548138) for half an
   hour, then collapsed to memorised pairs (*"547380,-120"* ×43; 144 distinct).

| half hour | stored captions with a pair |
|---|---|
| 12:00 | 4 / 10 |
| 12:30 | 60 / 97 |
| 13:00 | 61 / 108 |
| 13:30 | 85 / 116 |
| 14:00 | 92 / 115 |
| 14:30 | 55 / 65 |

Of 649 caption-family model responses this run, **356 opened with a number
pair, 39 with a clock stamp**. The stamps get stripped (leaving *"AM."*), the
pairs did not. Gates fired 32 times in total; the reroute pivoted 15.

**The structure, as the model sees one caption call** (`utils/llama_server.py
_append_stream_and_user`, `captioner._stream_history`, `_stream_push`):

1. system prompt;
2. **assistant message: the last 23 stored captions**, each rendered
   `HH:MM — text` (STREAM_WINDOW=24; the newest is pulled out to be the seam);
3. user message: frames + the present (situational lines, the developing thought…);
4. **assistant prefill: the last sentence of the newest stored caption**, stamp
   removed, ≤220 chars, cut at a sentence boundary. SEAM_MODE=sentence trims
   every stored caption to a boundary, so the seam is always a *finished*
   sentence and generation must open a new one.

The response is then stripped (stamps, list shapes, countdowns), trimmed to a
boundary, decision-extracted, and judged by `_caption_reject_reason` (~15
reasons, measured against the window minus the seam — `_comparable_stream`).
What passes is spoken **and stored**: it becomes line 24 of the next call's log,
and its last sentence becomes the next seam. Nothing not stored can propagate.

**Three carriers of continuity, all verbatim.** The seam (immediate, token
level). The window (the next ~6 minutes at 16 s). The slow ones — stream
consolidation (*"reusing their own words wherever possible… no new imagery"*),
compression, and the reflection material (the same lines, oldest first, each with
its clock). Every one carries the past as *strings*. The code says so in as many
words: `_prefill_mode` — *"reusing its words is what continuation MEANS"*.

**Why that is echo and not continuation.** A model extending its own prior text
reproduces the text's regularities, not its thought. The log's strongest
regularity is the line opening: twenty-three lines that begin with digits. The
content-level regularities follow the same route: refrains (*"the red foam
finger is just decor"* ×N), the antithesis shape, the *"N seconds of…"* stamp
(§23). The gates, the reroute, the front-erosion (`stream_erosion`), the
consolidation and spoken-not-stored are all managers of a poisoned window; their
existence is the evidence. The design notes already know it: STREAM_WINDOW —
*"the stream amplifies whatever register is in the window"*; STREAM_MODE (on
document mode) — *"poison amplification is the mode's nature, not a tuning
problem"* (hybrid bounds the prefill to one tail, but the 23-line log rides
whole); SEAM_MODE — *"the seam hands back a FINISHED sentence and the next call
must start a new one — which is where the model reaches for 'it's not X, it's Y'"*.

Worth holding onto: the window was switched on (June 28) to stop *amnesiac*
repetition — the model could not see it had already said "dust motes". It stops
that kind and produces the other kind, *imitative* repetition. Two echoes; the
system traded one for the other and then built gates against the second.

**What continuation would need** (laid out, not decided): the past present as
meaning rather than strings — what I was thinking about, what I concluded, what
is unresolved — with the verbatim surface reduced to the one seam the artist
likes (the mid-sentence pickup: 73 of 569 consecutive captions this run opened
mid-sentence, the good kind). Candidate levers, one at a time: the log without
the digit template (no `HH:MM —`; gaps are already words); the seam as a
fragment rather than a finished sentence (SEAM_MODE=fragment, the 3.6 shape);
the log as an abstractive paraphrase instead of verbatim lines; a smaller window
with the meaning carriers doing the work. The three strippers proposed on Sep 11
(countdown regex, AM/PM orphan, leading pair) would close this instance and
nothing else.

Immediate state: the window is saturated and will not clear itself; the seam
file written at shutdown would carry a pair-prefixed caption into a restart.

**Correction (Sep 11, afternoon — the artist's objection stands).** The strong
form above — "hand it a transcript and it copies the transcript" — is wrong as a
law, and the artist's counter is the data: the machine does not constantly
repeat. In this run the gates and reroute touched ~5% of captions; the other
95% moved. The head changes the view on ~25% of calls (110 drift turns + 34
LOOK turns over 568 image calls), so "the same picture every time" was also
overstated. Prior text + new image + time + mood + desire IS the continuation
design and it mostly works. What breeds is narrower: a **format at the line
opening** (a stamp, a number, a duration stub) and, less reliably, a refrain. A
language model reproduces a fixed string at a fixed position almost
deterministically once it sits in two lines; ordinary sentences vary and do not
breed. That is why one coordinate became 356 and one sentence about the finger
did not become 356. The strippers exist for exactly this; the countdown regex
mangled instead of removed and hid the shape from the gate. So the fix for this
incident is the small one (§24 top), not a redesign, and the paraphrase-window
proposal is withdrawn as a first step.

Where the artist's version of continuation is actually thin: the "accumulated
data" barely arrives. Over 458 caption calls: *nothing has changed for X* 3
(fires once per threshold per unchanged span, by design), *your head has been
turned X* 2, *you've felt X for* 1, *you keep coming back to X* 17, the
expectation check 79. Mood is two words every call; the one standing desire
line (*no paper on the desk*) rode every call and visibly drove the thought (the
paper hunt, "the paper is the ceiling"). The engine is connected at about a
quarter of its plugs. That, not the window, is the upstream place to work.

## 25. Sep 11 afternoon — the storage fix, and the facts made standing

Artist's rulings, verbatim: *"a few strange coordinates aren't bad per se, it's
… the structure that introduces the echo that is bad"*; *"Echo is not
continuation, in fact it's the opposite"*; *"prior text plus the new image and
the passing of time should be enough… plus accumulated data, mood, ambitions,
desire"*; and the decision: **"The appropriate data should reach every single
call."**

**Done (uncommitted at the time of writing; see runtime-map "Wiring changes,
Sep 11" for the wiring):**

1. *Storage hygiene for the incident* — the countdown regex no longer mangles a
   decimal, a coordinate-shaped pair opening a sentence is stripped at the mouth
   like a leaked stamp, an AM/PM rides out with its stamp. Sixteen mouth cases
   in `debug/test_format_strip.py`, including the ones that must survive
   ("7 minutes.", "100 years.", "2x4s", "10, 20, 30 years").
2. *Standing facts* — stillness, head, felt tenor on every caption-family call
   (caption, reroute, memory, drift, wander hop), durations moving with the
   clock, in words. Before: 3, 2 and 1 of 458 caption calls. The loop notice
   stays dosed (a nudge, not data); the expectation check stays per glance and
   the head line yields to it.

**What to read next run** (same tools as §22–23): the share of caption prompts
carrying each standing line (should be ~all once two minutes in); the
prompt-echo rate — captions opening with or restating a standing line ("Nothing
has happened for…", "I've been looking left for…") — B4's Aug 31 lesson says
this is where a standing fact "becomes the scene"; whether a coordinate-shaped
opening ever reaches the feed again (must be 0); tic/chant via
`measure_voice.py`. If the echo of the standing lines is high, the answer is
their wording and shape, not a dose — the ruling stands.

## 26. Native video on the 3.8 stack — probed live (Sep 11, ~16:40)

Artist: *"the super frame path was made for 3.6. It's outdated. What does it look
like for 3.8? It should be able to handle it natively."* Checked online and on
the box, then probed the running server.

**Facts.** Mainline llama.cpp merged native video input on June 8, 2026 (PR
#24269): ffmpeg decodes the clip as a subprocess, Qwen-VL-family models get
consecutive frames merged pairwise into super-frames (PR #21858), timestamps
ride as text chunks. The server takes `{"type":"input_video","input_video":
{"data": <raw base64 | url>}}` on `/v1/chat/completions`. The 3.8 stack's
server (`~/llama.cpp-38`, stock upstream Aug 17 build, `MTMD_VIDEO=ON`, ffmpeg
6.1 installed) reports `modalities: {vision: true, video: true}` for
Qwen3.8-27B + its mmproj. The old path (`_query_superframe`, the `llama-video`
package, `mm_processor_kwargs`) was for the patched 3.5 fork and is dead.

**Probe** (2 s clips at 4 fps built from a real frame; cache off; one line asked):
- whole view slides → *"The camera is panning slowly to the right."*
- one object slides, view fixed → *"The camera is static, but the small wooden
  rack with hanging tools on the middle shelf slides horizontally to the right."*
- identical frames, neutral question → *"Nothing happens in this clip; it is a
  static… shot"* (a three-way leading question once got "panning slightly").
- **eight real frames from today's run**, one per caption, sway and head turns
  included → *"The camera pans around a cluttered workshop…"* — the change is
  attributed to the camera, not the room. The awareness we were scripting by
  hand (§ "you were looking around") comes for free.

**Costs and catches.**
- ~5,200 prompt tokens per 2 s clip at 4 fps (8 frames → 4 super-frames, each
  forced to ≥1024 tokens by `--image-min-tokens 1024`), 9–15 s per call here. A
  3-still call today is ~3k. Budget knobs: fps and clip length (1 s at 4 fps or
  2 s at 2 fps ≈ 2 super-frames ≈ 2.6k tokens ≈ 5 s).
- The Aug 17 build has no `--video-fps` / `--video-timestamp-interval` (fixed
  4.0 fps; a `[0m0.00s]` text chunk before the frames — digits, words law).
  Current master has both (`--video-fps`, default 4.0; `--video-timestamp-interval`,
  default 5000 ms, 0 disables).
- **Prompt-cache bug in the Aug 17 build**: with `cache_prompt` on, clips two and
  three were served clip one's KV (5229/5233 cached, 1.3 s). The lazy video
  chunk carries no content id there; current master hashes the video bytes
  (sha256, per-frame suffix). Until a rebuild: `cache_prompt: false` on video
  calls.
- The July "blur" that turned superframe off: super-frames pair frames 250 ms
  apart; a 1° sway over a 4 s breath is ~0.06° between paired frames. Keep the
  saccade-frame skip and it's a non-issue — the real-frames clip had turns in it.

**What it would look like** (design, not done): rebuild `~/llama.cpp-38` from
master; launch `--video-fps 2 --video-timestamp-interval 0`; `VIDEO_MODE=native`
sends a 1–2 s clip from the frame buffer (already ~2 fps, 30 s deep) as raw
base64 `input_video` on every call; no markers, no motion lines; the previous
call's frame rides in the clip so "did the view change" is seen, not narrated;
delete `_query_superframe` and the `llama-video` dependency. One variable, after
the current run is read.

## 27. Trial: native video on every caption call (Sep 11, from ~17:10)

Artist: *"Let's stop the machine then and try with the proposed video path
just to see what happens."* Stopped 16:56 (graceful). Wiring in runtime-map
"Wiring changes, Sep 11 — NATIVE VIDEO". One variable: the picture on a caption
call is now a 2-second clip of the last four buffer frames instead of a still,
sent through mainline llama.cpp's video path with no markers and no motion
sentences. Everything else as in §25. Server unchanged (Aug 17 build: fixed
4 fps, a `[0m0.00s]` chunk, 1024-token floor per frame).

**What to read:** motion verbs on still things by call kind (§ "micro-dosing"
baseline: still 4%, inward 9%, drift turn 11%); whether a head turn is now read
as the camera moving; caption-call duration and model-busy share (baseline
4.3 s / 33%); any `caption_native_video_failed` entries; the `[0m0.00s]` chunk's
echo, if any; tic and chant via `measure_voice.py`.

**17:15 — and every thought call looks.** Artist: *"The inward beats should
carry a picture still as established many times prior. No call should be
without visual information."* The inward beat, the reroute, memory mode and the
wander hops now carry the frame (the inward beat the clip). Restarted 17:15.
The first native boot (17:06) had sent stills on every call because the old
steady-frame filter counts the breathing sway as camera motion (0/6, 1/6, 2/6
steady); native mode now sends the last four frames as they are.

**~17:25 — the inward beat's balance.** Artist: *"The trick is balancing the
image with the internal data. The inward beats need to prioritise the
internality of the machine without omitting the space around it."* Found: the
inward beat had been interior mostly by subtraction (no picture); its
introspective context adds only the dosed drawing-arc line, and every other
interior carrier rides by dose on every call alike. Now: the inward beat gets
the still, not the clip, and the lore line, its dated self-conclusion and open
question, and the drawing arc ride on every inward beat (regular calls keep
their doses). The wording of those lines is the artist's. Restarted ~17:25.
"The question is the door" — a code comment, never seen by the model: on a
normal hybrid call the mid-sentence prefill leads the next thought, so the
elicitation question is suppressed; on an inward beat there was no picture and
the beat exists to leave the stream's trajectory, so the question line
(*"Follow the thought you're already having — where does it go?"*) rides
instead. It still does.


## 28. The 22:19 walk-past and the 22:26 visit (Sep 11, read at 23:30)

Artist, 23:2x: *"Someone walked past about an hour ago… It left no trace in the
current real-time captioning."* Then: *"Person detection is notably flaky, the
awareness can miss many seconds of events due to its nature. There should be a
way to differentiate a consistent world model from a truly novel event."* And on
the cue wording: *"'He's come in' isn't very good. This wasn't me, it was a
different person."* (My own clock was ~6 h off during the first read — I had
been reading "now" off old log stamps; the ledgers were right.)

**22:19:19 — the walk-past.** A clip frame shows a dark figure filling the right
half of the picture at arm's length. YOLO: one box, conf 0.76; the skeleton gate
rejected it (no keypoints at that range/blur). No belief, no cue; the caption on
that clip talked about the curtain. Correct by the gate's rules, and exactly the
"flaky" case: a one-second pass at close range gives no skeleton.

**22:26–22:32 — the visit** (pose pass over 331 saved frames + clip frames):

| | time | lag |
|---|---|---|
| first gate-passing frame (person at the shelf) | 22:26:26 | |
| adjudicator: "A person in black clothing standing indoors." | 22:26:34 | +8 s |
| first caption naming them ("They're still sitting there. The person in the black jacket…") | 22:27:25 | +59 s |
| arrival noticed as arrival ("The chair was empty just now… Now there's someone in it.") | 22:29:08 | +2 m 42 s |
| last gate-passing frame | 22:30:33 | |
| ledger person_left | 22:30:41 | +8 s |
| "They've gone" cue in a prompt | 22:32:09 | +96 s |

Also: the 22:30 reflection opened on it; the compression's EVENT slot wrote it
twice; entity, episodic and arrivals ledgers all have it. First arrival since
22:21 the previous night — a full day.

**Where the trace died.** `build_standing_absence_line` rides only while the
last ABSENCE_STANDING_TAIL (8) stored lines mention the person, so it stopped
within two minutes by design. Nothing that outlives the window (ledgers,
EVENT slot, reflection) reaches a caption call; the standing stillness line
("Nothing has happened for about an hour") is the event's only shadow. By 23:10
none of 94 prompts carried it.

**Cue anomalies.** (1) The arrival edge line ("He's come in.") never reached ANY
prompt this run — the only presence line was the drift turn's *"He's here, just
out of view right now"* at 22:27:52 (wrong: they were in view). The edge is
detected and consumed at prompt-build time in `build_situational_line`
(`_prev_presence_for_line`); no skipped call or extra builder was found in the
log, and belief transitions are not logged, so the cause is open. Needed
regardless: log belief ON/OFF as events, and make the edge sticky until a prompt
carrying it is actually sent. (2) The departure cue's 96 s lag matches
ADJUDICATED_PERSON_TTL_S = 120: the belief outlives the last sighting by up to
two minutes. (3) "He" — the singular regime assumes the usual man; the entity
ledger already had "a person in black clothing". Proposed: "Someone's come in"
by default, "He's back" only when re-ID says familiar.

**The design thread (artist's).** The consistent world model exists in pieces
(referee references, spatial registry, entity/arrivals ledgers, the stillness
clock); the novel event exists as a record; nothing compares the two. The
arrivals ledger knew this was the first person in a day — rarity should have set
its lifetime in the frame (hours, in words), not the eight-line window rule.

## 29. Event memory: rarity sets weight (Sep 11, ~23:50)

Artist: *"Things out of the ordinary need to have a lot more weight in the
memory, of course? … The rarity should also determine the significance at the
time of discovery — so someone walking in after a period of loneliness should
be reacted to appropriately like 'finally someone is here' or 'someone walked
in!'"* Built (runtime-map "EVENT MEMORY"):

- **Weight = rarity**, measured against the machine's own ledgers, room-agnostic.
  The last completed event rides every thought call as a standing fact for a
  quarter of the gap that preceded it (ten minutes to six hours). Last night's
  visit, the first in a day, therefore stays until about 04:30: *"Earlier: The
  person sitting in the chair left, leaving only the empty seat behind. That was
  about an hour ago, the first visitor in about a day."* — the middle sentence is
  the compressor's own, written at 22:30.
- **Significance at discovery**: the arrival cue states the rarity as a fact —
  *"Someone's come in — the first in about a day."* The exclamation is the
  machine's to make (fact in, meaning out).
- **The three repairs from §28**: belief ON/OFF logged; the arrival/departure
  cue sticky until a prompt carrying it was actually sent; "He" only on re-ID.
- The reflection takes a rare visit up once while its line is alive.

Not built (second step, after a day's measurement): the walk-past class — a
large high-confidence box the skeleton gate rejects on consecutive frames as a
lower-tier, unverified event.

**What to read:** the standing event line in prompts (share, and that its age
moves); the next arrival's cue text and its lag from the first gate-passing
frame (belief events now make this a direct measurement); whether the machine's
first words after a rare arrival carry the rarity; that the routine form dies
in ten minutes.
