# Plan — the place, the unknown, the open question (Sep 7 2026)

Companion to `docs/handover-sep7.md` (jobs 1–8, of which 1, 6, 7, 8 are done).
This is job 9. Read `docs/north-star.md` Principles 1, 2 and 7 first.

## The problem, in the artist's words

> "The environment narrative isn't that consistent, it never says 'I'm in a
> robotics workshop' — much smaller models were able to deduce that. It might
> naturally help it stop hallucinating people in the disparate heads and limbs.
> What are we missing here for it to be naturally exploratory and inquisitive
> and persistent to that degree?"

Three findings, verified against the code on Sep 7:

1. **Nothing holds what kind of place this is.** `core_facts["place"]` exists
   (`context_compression.py:105`) but is **deliberately retired from
   surfacing**: `get_core_facts_string` replaced it with
   `get_place_inventory()` — an object list from the concepts ledger — because
   the LLM's free prose about the place was unreliable. The mind's life block
   does the same thing again (`mind.life-room`: "Around you, as far as you
   know: black metal wall lamp, wooden chair, black cloth bag…"). So every
   channel that could carry the place carries an **inventory of parts**. A
   list of eight object names invites listing objects; it never invites
   naming the whole.
2. **Nothing represents the unknown.** Everything in the prompt is a known:
   terms, durations, traits, wants, conclusions. The spatial registry already
   knows what it is unsure of (live values: `keyboard` conf 0.19,
   `mannequin head` 0.19, `laptop` 0.23, `bottle` 0.22; audit verdicts
   `held`/`no_candidates`/`relabelled`) and none of it reaches the machine. A
   mind with no held uncertainty has nothing to be curious about.
3. **Open questions do not persist in view.** `lore_ledger.note_question` /
   `open_questions` / `pick_question` exist and are fed by the distill
   QUESTION slot, but a question surfaces only as one of seven rotation slots
   in `Mind.surface_line`, is never marked answered, and never returns
   deliberately.

Why this also feeds the mannequin problem: with no model of the place, a
detached limb has no expected role, so the vision model's person reflex wins.
A place belief is a cheap prior against exactly that class of hallucination —
and it is the machine's own prior, not one we typed in.

## Doctrine constraints (do not break)

- **No content priors.** We never write "robotics workshop", never list
  candidate place types, never give an example sentence. We ask a question
  and store whatever the machine answers. This is the same line the NAME
  mechanism already walks.
- **Earned, not asserted.** `core_facts["place"]` was retired because
  per-compression LLM prose about the room was junk. The place belief must be
  a **harvest** from reflection (like NAME), formed rarely, revisable, with
  its own history — never generated per caption.
- **Memory framed as memory.** Any belief that reaches the prompt says when
  it was formed ("You settled a while ago that…"), so the present can
  contradict it.
- The machine is "it"; wordings live in `prompt_registry` with a `note`.

## Phase 1 — the place belief

**The template already exists.** The name is formed like this: once a day, if
no name stands, `reflection.name-invite` is appended to the *yourself*
subject question (`reflection.py:483`); the distill NAME slot harvests only
what the reflection actually said (`prompt_registry` distill notes);
`lore_ledger.note_name` stores it with history; the life block prints
"You've called yourself X". Mirror it exactly.

1. **Store** — `utils/lore_ledger.py`: add `note_place(text)` /
   `current_place()` / `place_history`, same shape as the name (text,
   first_ts, last_ts, times_affirmed). Revision allowed: a new place answer
   that differs replaces the current one and pushes the old into history.
   Guard: 2–8 words, no sentence, reject if it merely names one object
   already in the spatial registry (that is an inventory answer, not a place).
2. **Invite** — new fragment `reflection.place-invite`, appended to the
   **"the room"** subject question (`_REFLECTION_SUBJECT_IDS`,
   `reflection.subject.the-room`) once per `PLACE_INVITE_EVERY_S` (24 h)
   while no place stands, or once a week to re-ask when one does. Wording:
   an invitation to an act, never a menu — e.g. "If you had to say what kind
   of place this is, from what you've seen in it, what would you say? Or
   leave it." (artist's to finalize).
3. **Harvest** — add a `PLACE` slot to the distillation prompt
   (`distill.*` in the registry, parsed in `_parse_distillation`, which
   currently returns 10 slots — this makes 11). Harvest-only, "or none" most
   days, exactly as NAME is worded.
4. **Surface** — new fragment `mind.life-place`: "You've come to think of
   this place as {place}." in the **life block**, before `mind.life-room`.
   When a place stands, **shorten the inventory**: `MIND_ROOM_TERMS` drops
   from 8 to 4, since the place name is doing the compression the list was
   doing badly.
5. **Feed the reflection** — `_diet_room` already gathers
   `get_place_inventory()`; add the current place belief and its age so a
   re-ask can revise rather than restate.

**Files:** `utils/lore_ledger.py`, `captioner/reflection.py`
(`_next_subject`, `_diet_room`, the invite append), `captioner/
context_compression.py` (`_parse_distillation`, the absorb path beside the
NAME branch at ~line 933), `captioner/prompt_registry.py`, `captioner/
mind.py` (`life_block`), `config/config.py` (`PLACE_INVITE_EVERY_S`,
`MIND_ROOM_TERMS`).

**Test (`debug/test_place.py`):** the invite rides only on the room subject
and only when due; the PLACE slot parses and stores; an inventory-shaped
answer ("a wooden chair") is rejected; a differing answer revises and keeps
history; the life block prints it and the inventory shortens.

**Verify live:** `debug/journal.py 2` after the next room reflection — the
belief should appear in the pages in its own words, and the room list in the
life block should shrink.

## Phase 2 — the unknown, made honest

1. **Source** — `perception/spatial_registry.py` already has per-term `conf`,
   `hits`, `misses`, `audit_verdict`, `last_verified_ts`. Add
   `uncertain_terms(max_conf, n)`: terms in view or recently mentioned whose
   confidence is below `UNKNOWN_CONF_MAX` (start 0.25) or whose audit verdict
   is `no_candidates`/`relabelled`, least-recently-surfaced first.
2. **Surface** — new fragment `mind.unknown`: "You still haven't worked out
   what the {term} is." — added as an **eighth slot in
   `Mind._SURFACE_KINDS`**, so it rotates with the rest and never stands.
   Only when the term is currently in view (`Mind.in_view`), so the question
   is answerable by looking.
3. **Close the loop** — when a look lands on that term and the caption names
   it with a different word, `spatial_registry.note_mentions` already runs;
   add a light path so a renaming clears the unknown (the existing
   `mark_audit(term, "relabelled")` is the hook). An unknown that is resolved
   should say so once via the existing loop-notice channel, not a new one.

**Files:** `perception/spatial_registry.py`, `captioner/mind.py`
(`_SURFACE_KINDS`, one new branch in `surface_line`), `prompt_registry.py`,
`config/config.py` (`UNKNOWN_CONF_MAX`, `UNKNOWN_EVERY_N`).

**Test:** `uncertain_terms` ranks by confidence and recency and excludes
things not in view; the slot rotates and never repeats the same term twice in
a row; a relabelled term stops being surfaced.

**Watch:** this is the phase most likely to produce a tic ("I still don't
know what the X is" every third thought). The rotation plus the in-view
requirement is the guard; if it still tics, raise `UNKNOWN_EVERY_N` rather
than adding a fence.

## Phase 3 — questions that persist until answered

1. **Store** — `utils/lore_ledger.py`: questions already carry `text` and
   `last_ts`. Add `times_surfaced`, `answered_at`, and `answer` (the
   machine's own words), plus `note_answer(question_id, text)`.
2. **Return** — `Mind.surface_line`'s existing `question` slot picks the
   least-recently-surfaced open question rather than a random one, and after
   `QUESTION_RETURN_AFTER_S` says so: `mind.cue-question-again` — "You asked
   this a while ago and haven't answered it: …".
3. **Answer** — the distillation already has a QUESTION slot; add the mirror:
   when a reflection's text plainly answers a standing question (the distill
   emits `ANSWERED — <question> → <answer>`), store it and stop surfacing it.
   Harvest-only, "or none" most days.
4. **Show the arc** — the dream pass (`captioner/dream.py`) already writes
   records; add the day's answered questions to the night's page material so
   the arc is visible across days.

**Files:** `utils/lore_ledger.py`, `captioner/mind.py`, `captioner/
context_compression.py` (`_parse_distillation`), `prompt_registry.py`,
`captioner/dream.py`.

**Test:** least-recently-surfaced ordering; the "still unanswered" wording
only after the interval; an answered question leaves the rotation.

## Order, and how we will know it worked

Do Phase 1 alone and watch a day. Then Phase 2. Then Phase 3. Each phase:
tests green (`test_mind`, `test_mood`, `test_dream`, `test_phantom_presence`
plus the new one), commit, push, restart, verify in `debug/journal.py`.

Success is not a caption. Over a day:
- the machine names the place in its own words, unprompted, in the pages,
  and that name **changes** if the room changes;
- the person-hallucination rate on limbs/heads falls (measure:
  `phantom_presence` gate hits per hour, and by eye in the journal);
- it says what it does not know, and later says it has worked one out;
- a question asked on Monday is still being carried on Tuesday, and is
  eventually answered rather than replaced.

## Risks

- **Place prose was retired once for good reason.** If the harvested place
  belief turns out to be junk (too abstract, or an object name), do not patch
  it in the prompt — tighten the store's guard and the invite wording.
- **Another standing line is another mirror.** The place belief is stable by
  design (it changes rarely); the unknown and the question slots rotate.
  Nothing added here rides every call.
- **Inventory and place must not both ride at full width.** Shortening
  `MIND_ROOM_TERMS` when a place stands is part of the change, not an
  optional extra.
