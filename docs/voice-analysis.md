# Voice Analysis — where the monologue actually is (June 2026)

A shared map so we stop losing the thread. Written after a full audit of the
live code on branch `rebuild/north-star`. Companion to `north-star.md` (the
spec) and `runtime-map.md` (the wiring). This doc is the *diagnosis*: of the
three systems that shape the voice, which are fine and which is broken, and
why the voice keeps defaulting to "shitty poet."

## The one-line conclusion

Cadence and the perception/introspection balance are basically fine now. The
problem is **tone**, and tone has two fixable causes — both in our control,
neither requiring a fine-tune or model swap:

1. We **deleted the speech-act framing** that used to make the voice react and
   feel, when we tore out the old "emotional register" prompts as style fences.
2. A **feedback loop quietly poeticizes everything**: the machine's own purple
   output is compressed into felt-state / baseline / persona and re-injected
   into the next prompt as its established voice.

## The three systems

### 1. Modes — collapsed, and we did it on purpose

Five modes exist (relational / observational / workspace / introspective +
a separately bolted-on memory mode). But the difference between them is now
*one 6–8 word clause* in the system prompt ("you're aware of someone near
you," "your attention is on the desk below," "your attention has drifted
inward") plus a slightly different fact injected into the user prompt. They no
longer change *how the machine thinks* — only which fact gets shown, and even
that leaks across modes (introspective context is injected into all modes in
quiet moments).

**What was lost:** an earlier build's modes carried real register prompts —
literally *"Express raw emotions — curiosity, frustration, fascination,
boredom"* and *"Think in fragments, use pauses, talk about how YOU feel, not
analytical observations."* Those were removed in the north-star teardown as
"style instructions." That prompt is what produced reactions like *"it's so
messy in here, what is even going on?"* — and we deleted it, replacing it with
one weak word: *plain*.

### 2. Perception vs introspection balance — actually fine

After the salience fix (June), interiority is no longer suppressed by mere
presence. When a person is quietly in the room it runs ~90% interior;
perception only dominates in the ~10s arrival window and during big motion.
So "it only ever watches" is fixed. The balance is now: it attends to you
when you're active, and thinks about itself when it's quiet — which is roughly
correct. Cadence (4/7/12s breathing) is fine too. **Neither cadence nor
balance is the bug.**

### 3. Tone — the whole problem, and it's a feedback loop

The machine's own poetic output gets compressed into three things that are
then re-injected into the very next prompt *as its established voice*:

- **felt-state** → the system prompt literally says `Right now: settled in a
  loop of small details` (LLM-generated at temp 0.7, no plainness discipline)
- **baseline context** → a compressed sensory-prose sentence
- **persona** → its poetic self-description, quoted back

So every prompt effectively reads: *"Write plain notes to yourself. — Right
now: settled in a loop of small details. What you've come to know about
yourself: [poetic line]. You're seeing the last 9 seconds. Someone is moving
in the room."* It's saying "be casual" while quoting Keats. We lowered the
*caption* temperature but left the *compression / felt-state / persona*
generators at 0.7–0.8, so they keep manufacturing poetry and feeding it back.
The genre frame is one line against four lines of re-injected purple.

Top tone drivers, ranked:
1. **Video wrapper** — `"You're seeing the last 9 seconds. Someone is moving
   in the room."` wraps ~70% of captions in a cinematographic frame (camera
   narration, not thinking).
2. **Felt-state re-injection** — LLM poetry quoted back in the system prompt.
3. **Baseline context** — compressed sensory prose injected.
4. **Persona** — poetic self-description quoted back.
5. **Caption temperature** (0.6–0.7) — already lowered; the compression-side
   generators were not.

## The pattern behind the year of back-and-forth

The project oscillates between two failure modes:

- **Over-authored** (the old "express raw emotions, think in fragments" era):
  alive, but feels like *we* wrote it.
- **Under-framed** (the north-star rebuild: strip all instruction, identity
  from memory): authentic in principle, but with no speech-act framing the
  model defaults to its trained prior — and for "machine inner monologue" that
  prior is *literary description*. The feedback loop then poeticizes the rest.

The north star's "no instructions, identity from memory" kept pulling us toward
under-framed. The thing that gets us unstuck is a distinction the spec
conflated (now added to north-star Principle 2):

- A **fence** polices the mouth — "no metaphors," "don't say dust motes."
  Primes the bad thing, reads as manipulation. *Correctly banned.*
- An **elicitation** frames the *act* — "what do you make of this? blunt, the
  way you'd mutter it to yourself." Polices nothing; it tells the model **what
  kind of thought to have** — a reaction, not a description.

We threw out the elicitations along with the fences. But the prompt currently
hands the model an image and context and **never says what to do with it**, so
it describes. Reactions come from asking for a reaction, plainly. The image
stays central — the machine is perceptive and reactive — it just reacts
instead of narrating a film.

## Where we need to be (no fine-tune, image stays central)

1. **Restore the modes as speech-act framing, positively.** Each mode frames
   *what kind of thought* — relational: react to the person; introspective:
   follow your own thought; a real reactive/emotional register ("how does this
   sit with you — say it blunt"); memory: what does this bring back. Revives
   what the teardown removed, as elicitation (legitimate) not fences (banned).
2. **Fix the feedback loop at the storage gates.** Lower the compression /
   felt-state / persona-synthesis temperatures and frame *those* generators
   for plainness, so the material fed back stops being Keats. Plain captions →
   plain compression → plain re-injection. (North-star-correct: fix what's
   stored, not the mouth.)
3. **Reframe the video wrapper** off "you're seeing the last 9 seconds"
   (camera narration) toward a plain present cue.
4. **Keep the stream (CoT continuity) off** until the base tone is healthy —
   it amplifies whatever voice exists, so it goes back on last.

## Status (updated June 28 — most of this doc is now history)

The diagnosis above drove the work; here's where it landed. See
`memory-redesign-plan.md` for the full ledger migration and `next-session-brief.md`
for the live handoff.

- **Stream is ON** (`STREAM_WINDOW=6`) + DRY sampling — the verbatim-repetition
  ("dust motes" loop) cause turned out to be sampling, not framing: `repeat_last_n`
  default 64 < the stream length, so prior captions had zero repetition penalty.
- **The feedback loop is dismantled.** Every store that fed the prompt is now a
  ledger (concepts/place/drawings/felt/desire/belief/reflection/persona) — items
  1–2 of "where we need to be" are done, at the storage gates as the north-star
  requires. Validated cold-start: after 1.7h with memory on, no store re-grew
  purple.
- **Detox is OFF** (`BASE_VOICE_DETOX=False`); kept as a regression harness.
- **Identity development** moved off the inert introspection onto the reflection
  loop's distillation (Reflect→Become).
- **Open**: the residual base-voice purple is the model prior (Principle 7 —
  likely the fine-tune); the drawing pipeline is still un-cleaned; place-inventory
  injected every caption flattens the relational voice.
