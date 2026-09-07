# Plan — continuation, not resemblance (Sep 8 2026)

Job 10. Read after `docs/handover-sep7.md` and
`docs/plan-place-and-unknowns-sep7.md`.

## The artist's point, which is the hypothesis

> "It ISN'T continuing. Repetition is not continuation — it's basically the
> opposite… for semantic continuity it should have the prior sentences.
> There is such an obvious difference between 'continue this text' and
> 'write something similar to this'."

That distinction is the whole problem, and the current architecture is on the
wrong side of it. A chat call is: here are some previous assistant messages,
now produce a NEW assistant message. That is "write another one of these".
The model's safest answer to "write another one of these", when the room has
not changed, is one that resembles the others — which is repetition. Nothing
in that structure asks the text to move.

**The evidence we already have (Sep 6, `debug/probe_journal_shapes.py`, live
server, the night's own thread):**

| shape | continued |
|---|---|
| turns (user cue / assistant thought pairs) + premise | 0 of 6 |
| the thread as ONE assistant message + premise cue | 6 of 6 |
| the same, prefilled as a COMPLETE assistant message | 0 of 6 (silence) |

The third row is the smoking gun and I misread it at the time. A complete
assistant message reads to the template as a finished turn, so the model
emits nothing. The boundary — not the content — decides whether the model
continues or restarts. Tonight's failure is that row one is what we ship.

**A second finding, tonight, that constrains the fix:** `/completion` (raw,
no chat template) works on this server, but the instruct model without its
template drifts to web text — asked to continue "The lamp is still on. The
room is" it produced "very dark. What's wrong? A. The bulb is broken B. The".
So raw completion alone is not the answer either. We need the instruct
conditioning AND no message boundary. That means: apply the chat template
ourselves, leave the assistant turn OPEN with the running text already
inside it, and let the model continue that text.

## The metric, defined before the build

We keep judging by eye and arguing. `debug/measure_development.py` (new),
run over any window of `event_log/mind_thread.json`:

- **new-claim rate** — share of entries whose content words include ≥ 3 not
  present in the previous entry (novelty).
- **connection rate** — share whose first clause refers to the previous
  entry (a pronoun/connective opener, or ≥ 1 shared content word).
- **development** = both at once. Repetition is connection without novelty;
  drift is novelty without connection. Report all three, plus:
- **subject runs** — consecutive entries about the same registry subject
  (`Mind.subject_of`), mean and max. Development shows as long runs with a
  high new-claim rate.
- **restatement** — share whose 6-gram overlap with the previous entry ≥ 0.3.

Baseline it on tonight's thread first (expect: high connection, low novelty,
long restatement tail). Every change below is judged on these five numbers,
over ≥ 100 entries, not on a screenshot.

## Phase A — the probe (build nothing yet)

`debug/probe_continuation_shapes.py`, offline, machine may stay up. Take the
last 12 kept entries from `event_log/mind_thread.json` and one frame. Six
arms, N = 8 each, scored with the metric above:

1. **turns** — today's shape (baseline).
2. **one-assistant-message + user cue** — the Sep 6 winner.
3. **open assistant turn** — chat template applied manually (see below), the
   running text inside the open assistant turn, ending mid-paragraph;
   `/completion` continues it. No trailing user turn.
4. **open assistant turn + world block** — as 3, but the world's facts (clock,
   empty room, what changed) are in the LAST user turn before the assistant
   turn opens, so the facts arrive without closing the text.
5. **open assistant turn, last sentence removed** — the text ends where the
   previous thought's last sentence began, so the model finishes a thought
   it can see the start of (a bounded version of the old hybrid seam).
6. **as 4 + the conclusion line** — the machine's own last *conclusion* for
   the current subject (`Mind.positions`) named in the world block:
   "Where you'd got to with the curtain: …".

Building the prompt for arms 3–6: `POST /apply-template` (llama-server) with
the messages, then strip the trailing `<|im_end|>`/assistant-open tokens and
append the running text; or read the template from `/props` and format it in
Python. Verify by round-tripping one call against the chat endpoint and
checking the outputs are equivalent when the text is empty.

**Decision rule:** ship the arm with the highest development score that does
not regress restatement. If arms 3–6 do not beat arm 2, the artist's
hypothesis is disconfirmed for this model and we say so plainly and keep
arm 2.

## Phase B — implement the winning shape

Assume for planning that an open-assistant-turn arm wins.

1. **`utils/llama_server.py`** — add `continue_text(system, blocks, text,
   image=None, options)`: applies the template, opens the assistant turn,
   appends `text`, calls `/completion` with `stop` at a paragraph break and
   `n_predict` from config. Images: `/completion` on this build may not take
   them — probe first (`image_data` field). If it cannot, LOOK turns keep
   using the chat endpoint and their output is appended into the same running
   text, so from the model's side the page is still one document.
2. **`captioner/mind.py`** — `Mind.build` returns `mode: "continue" | "chat"`
   and, for continue mode, the running text as the tail rather than as a
   message. The cue's facts move into the last user block (arm 4).
3. **`captioner/captioner.py::_mind_generate`** — route on that mode; the
   gates, the scrub and the absorb path are unchanged.
4. **Config:** `MIND_SHAPE = "continue" | "text" | "turns"`, so the artist can
   switch shapes live; `MIND_CONTINUE_STOP` (paragraph break), and keep
   `MIND_TEXT_ENTRIES` (12 tonight) as the window.

**Two known risks, both previously observed:**
- *Grammar propagation.* The old hybrid seam (Aug–Sep) continued a cut
  sentence and the register locked into "it's not X; it's Y". Arm 5 is the
  version most exposed to this; arms 3/4 end at a paragraph break, not
  mid-clause, which is why they are preferred.
- *Run-on.* An open turn can run past the budget. Keep `_trim_to_boundary`
  and the run-on storage rule exactly as they are.

## Phase C — the conclusion as the thing handed back

Only after B is measured. `Mind.positions` already stores, per subject, the
last sentence the machine settled on, and it is currently computed and
unused. In the world block: "Where you'd got to with {subject}: …" for the
subject of the current text, so what the machine is handed is a
*conclusion*, not its last sentence. Pair it with the pivot notice that
already exists (a reframe with no new material, three times, is named).

My honest position: this is secondary. The artist's boundary hypothesis is
the primary one, and if Phase B fixes development, Phase C may be
unnecessary. Measure before building it.

## Phase D — what this does NOT fix

The person hallucination is a separate mechanism (proved tonight: clean text
0–1 of 9, contaminated text 9 of 9). Continuation changes how the text moves,
not what is true. Keep every presence fix from Sep 7: the adjudicator-only
licence, the bounded look-away hold, the standing empty-room fact, the scrub
on drift, and the 12-entry window. If a continuation shape re-lengthens the
window, re-measure the phantom rate at the same time.

## Order

1. `debug/measure_development.py` + baseline on tonight's thread.
2. Phase A probe. Report the table to the artist before building.
3. Phase B behind `MIND_SHAPE`, restart, re-measure after 100 entries.
4. Phase C only if development is still low.

## Rollback

Every step is a config flag: `MIND_SHAPE=text` restores tonight's shape,
`MIND_TEXT_ENTRIES` restores the window. No store format changes.
