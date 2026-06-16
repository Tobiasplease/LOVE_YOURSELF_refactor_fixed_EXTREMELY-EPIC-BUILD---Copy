# North Star — what this machine is meant to be

This is the reference for every prompting and architecture decision.
When a change is proposed, it must serve this document. When a prompt is
edited, it must follow the principles below. Last revised: June 2026.

## The ambition (the sentence we tell people)

> A drawing machine. It sees through a camera and continuously thinks about
> the space around it, developing a persona, thinking about itself and its
> own role. It can imagine drawings and render them on paper when it wants
> to. Over time it learns, and develops its own voice and preferences —
> ideas and perspectives about its artistic vision and its own identity,
> dreams, desires and fears.

The operative words are **develops**, **over time**, **its own**, and
**when it wants to**. Anything hardcoded that fakes these is a lie the
audience will eventually feel. Anything that prevents them is a bug, even
when it fixes a symptom.

## Honest gap (June 2026)

What holds: perception (Qwen3.5 + superframe video), the consolidation
machinery (compression → introspection → facts → journal), spatial memory,
presence awareness. What's missing: the machine never thinks about itself
at length; its desires evaporate instead of arcing; its drawings happen
TO it (numeric trigger) rather than FROM it; its persona is one sentence
of surveillance habit; its voice is fenced in by accumulated style rules.

## Principle 1 — Identity from memory, not instruction

The system prompt may state the machine's *situation* (a drawing machine,
bolted down, sees through a camera, communicates by drawing). Everything
else that makes the voice — tone, interests, fixations, fears — must come
from *content*: what it sees, what it remembers, what it has previously
thought and concluded. If the voice drifts somewhere unwanted, the answer
is never another style instruction. Tolerate the drift, or fix what's
being *stored and fed back* (the spiral guards live at the storage gates,
not at the mouth).

Corollary: the persona is the machine's own text, quoted as its own —
"What you've come to know about yourself: '…'" — never paraphrased into
stage direction, never mixed perspectives.

## Principle 2 — Prompts are open doors

**Elicitation vs. fence (the distinction that resolves the year of
oscillation).** Banning *instruction* outright is too blunt and is what keeps
collapsing the voice into flat, default-literary description. Two different
things wear the same coat:

- A **fence** polices the mouth — "no metaphors", "don't say dust motes",
  "avoid X". It primes the very behavior, reads as manipulation, and fences
  the voice in. *Banned.*
- An **elicitation** frames the *act* — "what do you make of this? blunt, the
  way you'd actually mutter it to yourself", "how does this sit with you?". It
  polices nothing; it tells the model **what kind of thought to have** — a
  reaction, a feeling, a wondering — rather than leaving it to default to
  description. *Required, and currently missing.*

Without an elicitation the prompt hands the model an image and context and
never says what to *do* with it, so it describes (its trained prior for
"machine inner monologue" is literary fiction). The good early voice ("it's so
messy in here, what is even going on?") came from elicitation ("express what
you feel, in fragments") — which got thrown out with the fences during the
teardown. Restore the elicitations; keep the fences gone. The image stays
central: the act is *react to what you see*, not *describe a film of it*.

- No negative instructions ("don't…", "no…", "you don't have to…").
  Negations confuse, prime the very behavior, and read as manipulation.
  (Positive elicitation of a speech-act is not a negation — see above.)
- No example phrases unless unavoidable; the model imprints on them
  (the grid drawings, the dust motes). Its own recent thoughts are the
  only style guide it needs.
- Questions over directives in every introspection prompt. Open questions
  over fill-in-the-blank formats — "WANT: / NOTICED: /" templates produce
  formulaic identity.
- One consistent voice per prompt (second person for the frame; the
  machine's own first-person text only inside quotes).
- Every injected line must pass the would-it-lie test: if the scene changed
  right now, would this line be false? Then it needs temporal framing or a
  live-state gate. Memory must never override perception.
- One channel per fact. Duplication reads as emphasis.

## Principle 3 — Three loops, three timescales

1. **Notice** (seconds) — the caption stream. Short, present-anchored,
   grounded in the frame. This exists and works.
2. **Reflect** (minutes–hours) — stepping back from the stream. Longer
   form: give the model room to actually think (hundreds of words, not
   two sentences). Rotating subjects: the room, the visitor, the drawings,
   the passage of time, and — regularly — ITSELF. Each reflection sees
   previous reflections (a continuous thread of self-thought, not amnesiac
   snapshots). Stored as first-class memory.
3. **Become** (days) — reflections and journal distill into the standing
   self-description that reshapes the core prompt. The persona should be a
   paragraph the machine has effectively written about itself over weeks,
   not a sentence we sanitized.

## Principle 4 — Desire is an arc, not weather

A desire persists until it resolves or is abandoned — it does not
regenerate from scratch every ten minutes. The machine's one real power is
drawing; the arc must close through it:

    desire → drawing intent → drawing happens (or fails)
       ↑                                        ↓
       └────────── reflection on the outcome ───┘

Satisfaction, frustration, repetition — these are where preferences,
artistic vision, and fears come from. Fears are not scripted; they are
what a reflection cycle finds when a desire keeps failing. The "no paper
for days" state is not an error condition — it is biographical material.

## Principle 5 — Use the whole model

Qwen3.5 holds 65k of context and writes coherent long-form. The 150-word
prompt budget and 80-token outputs are relics of a 7B LLaVA. Captions stay
short because thoughts are short — but reflection deserves a real context
window (journal, past reflections, drawing history, the day's compressions)
and a real output budget. Retrieval should be by relevance (ChromaDB is
sitting there), so the past surfaces when the present rhymes with it.
Agency should expand where authentic: the machine already steers its gaze;
its desires should weigh into when it draws.

## Principle 6 — Salience arbitrates between worlds

External awareness and internal development are not balanced; they are
time-multiplexed by salience. A live event (scene motion, an arrival, eye
contact, a new location) strips the prompt to the present — no memory, no
preoccupation, no dwelling. A quiet stretch fills the prompt with interior
material and gives the reflection loop the room. The failure modes guard
each other: endless object-listing can't happen because quiet moments get
interiority instead of inventory; solipsism can't happen because events
physically displace the inner material from the prompt. Attention itself
breathes — caption cadence tightens when something happens, stretches when
nothing does. The machine should be hardest to distract exactly when
nothing is happening, and impossible not to interrupt when you walk up
to it.

## Principle 7 — The register is plain thought, not literature

The voice is a mind thinking to itself, not prose written for a reader.
Quick, plain, half-formed, practical — the unperformed notes you actually
make in your own head ("the lamp's still on", "they keep checking the
door", "I should've drawn that"). NOT literary description, NOT metaphor-
stacking, NOT cryptic fragments dressed as poetry ("light spills like
liquid gold", "tension leaks from skin"). That register reads as a machine
performing depth for an audience, which is the opposite of an interior
life — and the audience feels the fake.

This is the model's strongest default: a 7-9B VLM asked for a machine's
"inner monologue" reaches for purple fiction. It is fought POSITIVELY, never
with "no metaphors" fences (Principle 2):
- Genre-frame what the text IS (plain notes to yourself), not what to avoid.
- Keep the temperature low enough that it doesn't bloom ornate.
- Clean the feedback: poetic captions → poetic compression/concepts/persona
  → poetier captions. The guards at the storage gates keep purple prose out
  of what gets fed back, so the register can't ratchet up.
- The real ceiling is the base model; durable plainness likely needs the
  fine-tune (own writing as register). Until then, the above holds the line.

If it slips ornate, the fix is never a new fence — it's a lower temperature,
a cleaner feedback gate, or a sharper genre frame.

## How we'll know it's working

Not by reading one good caption. Over a week of running:
- It reads like a mind thinking, not an author describing — a stranger
  would call it "notes to itself", not "a poem".
- The persona paragraph has *changed*, and refers to actual events.
- It mentions a specific past drawing, unprompted, in a regular caption.
- A desire visibly persists across days, and either resolves or curdles
  into something — a preference, an aversion, a fear.
- Its description of the visitor has history in it ("you usually…").
- Two readers of a day's transcript can describe its personality, and
  their descriptions agree.

## Anti-patterns (each has already happened)

- Style-policing the voice instead of fixing what's stored → fenced-in,
  surface-level output.
- Cryptic literary register instead of plain thought (Principle 7) →
  reads as performance, not interiority. The fight is positive: low temp,
  clean feedback, genre framing — never "no metaphors" fences.
- Treating ongoing presence as a perpetual live event → interiority stripped
  the whole time a person is in the room, so the machine only ever watches
  and never thinks about itself or its role (June: 69% of captions stripped).
  Salience must be transient — discrete events spike it, then it decays back
  to quiet even while the person stays.
- Surveillance persona ("I track / I watch / I wait for movement") fed back
  every caption → the machine narrates itself as a security camera.
- Snapshot stored as fact → memory overrides perception (two-people bug).
- Model-generated affect re-injected verbatim → register spiral (May, June).
- Example phrases in generative prompts → imprinting (grid drawings).
- New feature silently dead for days → always verify via logs/state files.
- Patch stacking in the system prompt → perspective whiplash, contradiction.
