# Next Session Brief — Reground the base voice (June 2026)

**Branch:** `rebuild/north-star`
**Read first, in order:**
1. `docs/north-star.md` — the spec. Pay attention to Principle 2 (the
   **elicitation vs. fence** distinction) and Principle 7 (plain thought).
2. `docs/voice-analysis.md` — the diagnosis. What's actually broken (tone)
   and what isn't (cadence, balance).
3. `docs/runtime-map.md` — the wiring.

Do NOT start by changing code. Read the three docs, then read the actual
current prompts in `captioner/prompts.py` (the system prompt assembly
`get_monologue_system_prompt`, `_SITUATION`, `_MONOLOGUE_CLAUSE`, the mode
additions, `build_situational_line`, and the mode context functions).

## The objective

Restore the grounded, embodied base voice. We have had it before, many times
— this is achievable through prompting alone. It is NOT a fine-tune question
and NOT a model-ceiling question. Do not raise either; they are out of scope
and unhelpful with current resources.

The voice we want is a mind thinking to itself from inside a body: present,
plain, reactive to what it sees, with an attitude and curiosity of its own.
Not a flunked-out poetry major ("dust motes dance in streams of light"). Not
a surveillance camera ("I track the movement, capturing every detail"). Not
stilted telemetry ("your gaze is pointed upwards").

## The hard constraint — establish embodiment, do not prescribe a character

This is the line to walk carefully. We establish the *conditions* for an
embodied voice; we do NOT script the voice itself.

- **Yes:** structural embodiment — it is a body bolted to a table; it sees;
  it can turn its look; it draws; it is situated in a specific room over
  time. Framing the *act* as responding to its situation rather than
  describing a picture.
- **No:** prescribing the output. Do not seed example phrases ("it's so messy
  in here"), do not mandate a mood or attitude, do not write its personality
  for it. The artist will reject anything that reads as the machine acting out
  a character we handed it. It must have room to grow and evolve into its own
  voice — that growth is the whole point (north-star: *develops, over time,
  its own*).

The mechanism for this is **elicitation, not fence and not script**: frame
the kind of act (respond to where you are, in your own plain words) and then
get out of the way. An elicitation opens a door; it does not walk through it
for the model.

## The method — the whole prompt sets the register, so the whole prompt must be clean

The model mirrors the register of *every line it receives*, not just the
explicit instruction. A single stilted or purple injected line ("your gaze is
pointed upwards", "settled in a loop of small details") pulls the output with
it. So:

- **Every injected line must be clear, clinical, and precise.** Clinical here
  means clean and exact — register-neutral, no fluff, no awkward stiffness, no
  poetry. The prompt is a clean substrate; it states the situation precisely
  and invites a response. It must not carry a voice of its own that the model
  will copy.
- **Audit every surface, not just the system prompt:** the situational line,
  the mode additions, the mode context fragments, the video wrapper
  ("You're seeing the last N seconds…"), the felt-state and baseline and
  persona injections. Each one is teaching the model how to sound. Any that
  read awkward or literary are actively corrupting the voice.
- **The system prompt needs the most work.** It is the foundation and is read
  on every call. Get it clean, precise, embodied, and register-neutral first.

## Where to start

1. **The system prompt**, then **the awakening** (the seed, and the most
   isolated single call — one prompt, no superframe, no telemetry pile). Get
   these two clean and embodied. The base register is judgeable in *minutes* —
   if the awakening still comes out purple, the framing is still wrong; iterate
   on the framing, not by adding fences.
2. **The register audit** of every other injected line (above). Rewrite the
   stilted/literary ones to be clean and precise.
3. **Keep the feedback channels clean.** The compression / felt-state /
   persona-synthesis generators run at their own temperatures and feed their
   output back into the prompt. If they produce purple, the voice re-poisons
   itself within the hour (this is why a bad base compounds rather than
   settles — see voice-analysis.md). Their register must match the target too.

## Standing cautions

- **Base register is judgeable fast; the developmental arc is not.** Judge the
  awakening + system prompt in minutes. But do NOT spiral into minute-by-minute
  reactive patching of every downstream symptom — that is what derailed the
  prior session. Fix the base cleanly, then stop and let it run.
- **The stream (CoT continuity, `STREAM_WINDOW`) stays at 0** until the base
  voice is healthy. It amplifies whatever register exists, so it goes on last.
- **Never re-add fences** when the voice wobbles (no "no metaphors", no
  example phrases). Fix the framing or the feedback, per north-star Principle
  1, 2, 7.
- Features fail silently here — verify via the event log / `live_captions.txt`
  that changes produce output.
- Keep `docs/runtime-map.md` updated as wiring changes; it is the artist's
  window into the repo.

## State at handoff

- Stream OFF. Salience transient (interiority returns during quiet presence).
- Surveillance persona gated (read + write) and cleared; will re-form from new
  captions — so it will re-form from whatever the base voice produces next.
  Get the base right before it synthesizes a new persona.
- Caption temp 0.6/0.7. Compression/felt-state/persona temps NOT yet lowered.
- Many real perception bugs fixed this session (numpy crash, phantom motion,
  false arrivals, embodied vision, own-arms). The machinery is sound; the
  voice is the open problem.
