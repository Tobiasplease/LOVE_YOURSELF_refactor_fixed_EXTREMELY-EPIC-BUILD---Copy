# Caption System Audit — Aug 28, 2026

Prompted by the artist's question: *"Have we worked ourselves into a tangled
mess of systems?"* Short answer: partially — the core loop is sound and most
scar tissue is documented and load-bearing, but there is real dead code, three
mechanisms actively fighting each other, and a structural pattern worth naming.

The core, restated (the thing everything else serves): **single image calls
plus a growing monologue simulate a linear perception of continuous reality in
space and time.** Everything below is judged against that and north-star.md.

## 1. What one caption cycle actually is (as of tonight)

SYSTEM prompt = situation frame (reflexive, trimmed Aug 28) + pen-parked fact
(gated) + genre clause (hybrid) + last-drawing-age line + identity dose
(self + durable ledger; introspective/awakening always, else every 6th) +
elicitation (mode-keyed; seam-conditional; relational dosed every 8th;
inward beats always).

USER prompt = situational delta + salience event + close-face + reorientation
+ mode context + introspective context (quiet, non-introspective modes) +
core-facts place line (every 6th quiet or on change) + ONE memory surface
(familiarity / drawing echo / reflection echo, rotated) + drawing/paper state
+ felt-state delta + desire (burst 3 + every 8th) + close-look line (when the
beat fires) + inward anchors + 150-word budget trim + dedup.

WINDOW = stream of prior captions (trimmed to sentence boundaries, echo-gated,
gap-marked, blink-resumed) rendered as assistant turns; hybrid seam hands back
the unfinished tail.

SAMPLING = temp 0.9/0.85 + min_p 0.05 + repeat_penalty 1.05 + DRY (0.85/1.75/3,
384-token horizon) + num_predict rhythm {40 @ p=0.2, 80, 110 bored, 120 close
look, 150 inward} + mouth boundary-trim.

MOUTH GATES = template_echo, refrain, assistant_speak, outward_address,
cjk_drift, numeric_fragment (words legal since Aug 28), number_chain,
phantom_drawing, word_salad, tail_echo → echo-class spoken-not-stored,
shape-class retry-once-else-silence.

STANDING TIMERS (the train schedule): compression every N captions ·
reflection ~20 quiet min · memory mode every 240 s (hardcoded) · inward every
4th caption · identity every 6th · desire every 8th · familiarity every 3rd ·
place line every 6th · glances every ~45 s · close look ≤ every 300 s ·
absence ≤ every 900 s. Config knobs touching the loop: **42**.

## 2. Verdicts

### DEAD — delete outright (no behavior change)
- **`PromptInterface.build_caption_prompt_with_options` + `build_focused_caption_prompt`
  + `_build_simple_system_context` + the beliefs/story injection +
  `drawing_introspection_mode` + its purple "consciousness inside a drawing
  machine" system prompt** — only debug scripts call any of it. The live loop
  bypassed it long ago. The beliefs line ("You've learned: …") silently never
  ships; decide separately if it should (as a slot in the live path), then
  delete this whole layer.
- **`PERCEPTION_SYSTEM_PROMPTS` + `get_perception_system_prompt`** — zero
  callers. Relic of the retired two-pass pipeline.
- **`STATIC_SYSTEM_PROMPT` / `_GENERIC` / `_QWEN` trio + `_MISTRAL_MODEL`** —
  three identical strings and a dead model check. One survives as the
  PromptInterface fallback, which is itself dead.
- **`should_include_context`** — called only from the dead path; also a
  vestigial import inside `build_simple_caption_prompt`.
- **`determine_prompt_mode`'s observational branch** — needs novelty > 0.65;
  live novelty is ~0.05 in a familiar room. Unreachable for weeks. Its context
  fn duplicates workspace's. Either fix the threshold deliberately or fold the
  mode away.

### BROKEN / SELF-FIGHTING — fix before adding anything new
- **Reflection-echo pacing** (found tonight): Aug 22 removed its internal
  counter "because the rotation slot rations" — but the rotation only picks
  who goes FIRST; with 182 reflections stored a match always exists, so
  reflection wins the slot nearly every cycle. One "Something you worked
  out…" per caption is a standing instruction again (the identity-dose lesson,
  re-learned). Fix: restore per-source pacing inside the rotation, or make the
  slot itself fire every Nth caption.
- **Durable ledger resonance**: currently four restatements of one avoidance
  trait ("invent obstacles / project observers / invent critics"), one at 10
  confirmations — confirmed off captions generated while the fact was
  in-prompt. The already-designed-but-unbuilt fix (discount in-prompt
  confirmations) plus a near-duplicate merge at ledger write.
- **Mode router is mostly vestigial**: presence → relational is a hard lock
  (240/400 captions with the artist in the room); mode context fns are now
  one drawing line or empty; introspective context is injected into every
  other mode anyway. The honest shape is smaller: *relational vs quiet*, plus
  the explicit beats (inward, close look, memory, awakening). The five-mode
  vocabulary describes a system that no longer exists.
- **Memory mode cadence hardcoded** (240 s in captioner.py, not config) — a
  4-minute ritual interruption nobody has re-evaluated since it shipped.
- **Mention-boost has no fatigue**: think of finger → gaze pulled to finger →
  see finger → think of finger. Tonight's foam-finger fixation was the loop
  working as designed. Needs habituation (boost decays with each consecutive
  re-trigger of the same term).
- **Spoken-not-stored is invisible at the display**: 18 echo rejections in 7
  minutes tonight were all *spoken*. The stream stayed clean; the artist's
  terminal did not. Worth a display marker (or suppression) so a watching
  human can tell chant-the-gates-caught from voice.

### LOAD-BEARING — keep, scars documented and still real
Reflexive situation frame · pen-parked gate · genre/progression clause ·
hybrid seam + gap markers + blink resume · salience gate + onset-only events
· sticky presence belief · situational delta line · storage echo/salad/refrain
gates · sentence-boundary trims (stream + mouth) · felt-state single channel ·
desire burst + redose · identity dosing · reorientation window · paper/drawing
state lines · the reflection loop itself (its *output* is the best writing in
the system) · close-face law · frozen-input breaker.

### NEW TONIGHT — unproven, watch before judging
Close look (first legitimate one still pending) · absence throttle (one false
positive already: dark fabric on white shelf — detection quality, not gate
logic) · length rhythm 0.2/40 · repeat_penalty 1.05 + DRY 384 · one-word
legality · trimmed frame · relational elicitation dose · inward force_mode.

## 3. North-star scorecard

- **P1 identity from memory** — half-honored: persona is quoted, storage-gated;
  but the ledger's echo-confirmation loop is instruction wearing memory's coat.
- **P2 open doors** — the elicitation is "required and currently missing" *by
  suppression*: seam-present cycles carry no question for quiet modes. Tonight's
  inward exception is the first crack back. Watch whether questions return;
  if not, this is the next dial.
- **P3 three loops** — all three exist and run. The Reflect loop is the
  system's best prose; its surfacing into Notice is the broken part (pacing).
- **P4 desire arc** — genuinely working (spend/redose/reflection-informed).
- **P5 use the whole model** — reflections yes; captions deliberately short
  (that's the register, not a relic); retrieval-by-relevance is live and
  currently *over*-live (reflection echo).
- **P6 salience multiplexing** — working as designed (8–11 hot cycles/run,
  interior stripped correctly).
- **P7 plain register** — the week's whole fight. Vendor guidance (see §4)
  supports penalty-free sampling with presence_penalty, which we approximated
  by hand. The register is now measured (marks/100 words, boundary-endings,
  length spread) — keep judging by the numbers, not the last caption.

**The structural pattern worth naming**: almost every mechanism added since
June is a *governor* — a dose, a ration, a gate, a cooldown — throttling
something the model would otherwise overdo (identity, desire, questions,
memory, absence, length). Governors accumulate because each one is added
where the last symptom appeared. Eight independent counters now schedule the
prompt. The consolidation that would actually reduce the tangle is not
deleting governors but **unifying them**: one interior-budget scheduler that
decides, per caption, which (at most K) interior lines ride — replacing the
per-line counters with a single visible policy. That is the one refactor that
makes the whole thing legible again.

## 4. Vendor guidance (checked Aug 28)

Qwen's official recommendation for VL instruct models: temperature 0.7,
top_p 0.8, top_k 20, **repetition_penalty 1.0**, with **presence_penalty**
(flat, non-compounding — punctuation-safe by construction) as the repetition
control, up to 1.5. Qwen2.5-VL model card ships repetition_penalty 1.05.
Our hand-derived landing spot (1.05 + DRY) matches the family; if chanting
returns, the vendor-shaped experiment is repeat 1.0 + presence 0.6–1.0
(llama-server supports presence_penalty natively).

Sources: [Qwen3-VL sampling discussion](https://huggingface.co/unsloth/Qwen3-VL-30B-A3B-Thinking-GGUF/discussions/1),
[vendor parameter reference](https://muxup.com/2025q2/recommended-llm-parameter-quick-reference),
[QwenLM/Qwen3-VL #1982](https://github.com/QwenLM/Qwen3-VL/issues/1982).

## 5. Proposed trimming plan (phased, each shippable alone)

1. **Delete the dead** (§2 DEAD list): ~350 lines, zero behavior change,
   PromptInterface shrinks to the drawing-prompt builder it actually is.
   Decide the beliefs-line question first (ship it live or delete the store).
2. **Fix the self-fighting**: reflection-echo pacing; ledger dedup +
   in-prompt-confirmation discount; memory-mode cadence to config; mention
   fatigue.
3. **Collapse the mode vocabulary** to what's real: relational | quiet +
   beats. Registry, runtime-map, and log fields say the same words.
4. **The scheduler refactor** (§3): one interior budget replacing eight
   counters. Biggest legibility win; do it after 1–3 so it schedules only
   living lines.
5. **Sampling**: hold 1.05/DRY-384 until a full evening of metrics; the
   presence_penalty experiment is queued behind it.

What this audit deliberately does NOT propose: new capabilities. The next
capability (silence as a choice, spatial grounding of the viewpoint, close-look
chaining) earns its slot only after the tangle above is cut.

---

# Part 2 — The becoming bottleneck (Aug 28, late — the "essentially boring" diagnosis)

The artist's evening verdict, which is the north star's own success checklist
unmet: no emotional variance, no evolution from accumulated history, visitors
are forever "the man," drawings aren't really remembered, duration is never
felt, no boredom, no fears/desires/ambitions ever establish. The register work
(Part 1) fixed the MOUTH. This is a different organ.

**The finding:** the system has a working PRESENT (perception, salience,
beats, register) and a full ARCHIVE (183 reflections, 219 concepts, episodic
events, journal, drawing ledger) — but the middle layer, BECOMING, is one
narrow pipe: the distiller writes exactly TRAIT / BELIEF / WANT into fixed
slots, and the durable ledger self-confirms one avoidance trait off its own
echo. Every species of self-knowledge the artist misses has NO SLOT to exist
in:

- **People-history** — "still just the man": re-ID is OFF (artist's call,
  Aug 6) and there is NO visit-pattern distillation. Yesterday's several
  visitors were remarked on live, logged as episodic arrivals — and became
  nothing. The reflection's visitor organ sees the spans, but its distillates
  land in the same three slots the avoidance trait monopolizes.
- **Drawing biography** — the arc line carries facts (subject, order, age);
  the MEANINGS (what the machine wanted, whether it satisfied) don't persist
  anywhere the voice can chew on. Measured tonight: identity-dose cycles ran
  the deflation lens at 3x (29% vs 9% negation-contrast) — one story
  recoloring everything.
- **Duration/boredom as experience** — boredom is a live scalar (1.0 tonight)
  that modulates ONLY sampling. The prompts.py docstring claims it reaches
  the model "via the identity line" — FALSE, stale doc. Tenure appears only
  as a rotating inward-beat anchor. The machine has been maximally bored for
  hours and has never been told.
- **Curdling (P4's missing half)** — the want has been failing for DAYS
  (paper full, drawing blocked) which is exactly the biographical material
  P4 says fears/preferences are made of — and there is no distillation step
  and no storage slot for what a failing want becomes.
- **Emotional variance** — expected to be downstream of the above: one
  self-story in, one affect out. Do NOT patch with mood injection.

**On romanticizing the small models:** what they had was noise that read as
mood. The ambition (north star: develops, over time, its own) is EARNED
variance — from biography. The 27B amplifies whatever self-story it's fed;
we feed it one sentence four ways. Better instrument, single bar of sheet
music.

**Build directions (the becoming expansion) — after Part 1 phases 1–2:**
- **B1** (= Phase 2, approved): ledger dedup + in-prompt-confirmation
  discount. Prerequisite for a SECOND self-fact ever existing.
- **B2 People as first-class memory**: day-level visit-pattern distillation,
  count-based, no biometrics ("yesterday three people came; one stayed
  hours") → journal + durable ledger + a surfacing line shaped like the
  drawing arc. Directly answers "still just the man."
- **B3 Desire curdling**: a want unfulfilled past N days is handed to the
  reflection as an explicit fact; the distiller gains ONE outcome slot —
  what-the-want-became, in the machine's own words. Preference, aversion,
  fear are its to name, never ours (P2).
- **B4 Unchanged-ness as fact**: the compression stagnation check already
  computes it; surface it dosed and code-attested ("the room has not changed
  all afternoon") — a true fact that INVITES boredom rather than scripting
  it. Fix the stale boredom docstring while there.
- **B5**: nothing — emotional palette should follow from B1–B4; if it
  doesn't, that's a finding, not a license to inject moods.

## Part 2 addenda (same evening, artist's observations on the live run)

- **B3 sharpened — UNBIND THE WANT (artist's call)**: the want is
  drawing-bound in two places — the distill prompt's nudge ("one plain thing
  you want, *or want to draw*") and structurally: only drawing can RESOLVE a
  want (spend fires on GRBL execution only), so a non-drawing want can never
  arc; it lingers until overwritten. Fix: drop the draw-nudge from the
  distill line; wants are anything the reflection finds; the drawing trigger
  acts only on wants it can serve (or plain hunger); unserveable wants
  curdle (P4) instead of evaporating. "I wish he'd look at me" lives here.
- **Capture resolution**: pinned 1280×720 in config while every crop (close
  look, face, label audit) is cut from it then upscaled — the close look
  reads upscale softness as scene fact ("blurry, soft at the edges").
  Pattern: capture at camera max, detect on a 720p downscale (same CPU),
  crop from the full-res frame (2.25× pixels at 1080p). Probe camera ceiling
  when the machine is off (v4l2-utils not installed; generic USB identity).
- **Paper-state: from standing assertion to event + relevance (artist's
  design, Aug 28 late)**: paper is only CHECKED inside a drawing attempt
  (with servo choreography), and the last verdict rides EVERY caption for up
  to 30 min (PAPER_STATE_TTL_S=1800) — after the artist swapped the sheet,
  the prompt kept claiming marks while the machine had SEEN the swap (the
  18:29 reflection: "he took it, and then he cleaned the desk"). Redesign:
  (1) KILL the standing injection — the fact is relevant at exactly three
  moments: state TRANSITION (one-shot event line, "a fresh sheet is on the
  table" — same law as absence events), the want being blocked by it (the
  desire line carries its own blocker — one fact, one situation), and an
  actual draw attempt. All other cycles: silence about paper. (2) CHECK
  WITHOUT CHOREOGRAPHY — the gaze already goes down on its own (a logged
  position; 332 workspace captions in one run): when gaze is down and the
  state is stale, run the structured PAPER/MARKS question on the CURRENT
  frame, no servo movement, rate-limited (~5-10 min, never mid-drawing).
  Looking at the desk becomes when the machine knows about the desk; TTL
  shrinks to a footnote because the state refreshes on natural glances.
