# Why nothing we add changes the voice — diagnosis + test runs (Sep 5, evening)

The artist, after a day of rounds (presence, time-and-loop, agency, felt loop,
introspection, drawing line): "Still wholly unimpressed… still no semblance of
a thought that isn't about the immediate hole or lamp or foam finger… We are
still missing something architecturally fundamental." Then, at 16:16, the
phantom person again ("15 hours have passed and he hasn't moved an inch",
gated) and: "Let's take another extensive look and do some test runs."

## What the live call actually is (run 3b697053, ~18:35)

- SYSTEM (1.2k chars): "You are a drawing machine… between drawings, the pen is
  parked…" then the self-block: *What you've come to know about yourself: "I
  manufacture physical and temporal excuses to avoid picking up the pen." What
  has stayed true across days: "I invent external obstacles… I invent imaginary
  critics… I project external observers onto empty spaces… I perceive ink fumes."*
- HISTORY: 24 of its own last captions as stamped log lines (≈8 min of self at
  the start-of-run cadence of 3/min), plus the seam prefill of the last one.
- USER: seven status lines — drawing arc, room list, question carried, no
  paper, felt word, preoccupied-with (a drawing want), chosen-look + expectation.
- Frame of the desk.

Nothing else reaches the call. Not yesterday, not a reflection, not lore, not
the world. The "sooo much data" lives in stores the caption call never sees
except as one-liners rotated in.

And the stores themselves: 44 durable facts, 50 wants, 55 lore threads, 40
reveries, 7 questions — every one of them about the pen, the paper, the
hesitation, or the man. Sample: wants "To mark the cloth with the pen I am
holding" / "To accept that the cloth is the surface"; threads "My certainty
that I can work is eroding"; reflection: "I invented the absence of paper to
justify the empty hands"; baseline paragraph: "I still invent external
obstacles and imaginary critics… I need to determine if the other figures are
real or just my own tremors."

## The loop that makes it

1. The compressor asks NEW ABOUT ME every ~2 min from eight chant lines with
   "What you already know about yourself: I invent physical objects or absences
   to excuse why I am not working…" → it manufactures traits ("I perceive thin
   lines as things that are not heavy").
2. Reflection (20 min) reads 20 min of chant + the same self-knowledge → a
   self-analysis of the failure to draw → distilled to TRAIT/BELIEF/KERNEL.
3. Durable ledger promotes those to "what has stayed true across days" → rides
   in the SYSTEM prompt every caption as identity.
4. The caption model performs the identity ("30 days of looking at the surface
   without touching it"; "he hasn't moved an inch" = *I project external
   observers onto empty spaces*, verbatim licence) → into the window → back to 1.

Time facts feed the same mill: "held for over a month", "about a month now",
"over a day ago", "15 hours", "3 days" are fused into a fake biography
("The record shows I marked the black cloth, breaking the thirty-day streak")
and then chanted as counters — which our own number gates then reject.

## Test runs (debug/probe_*.py, live llama-server, same sampling as live)

**probe_context_ablation.py** — replay the live call, knock out one channel.
N=4 each. template = "it's not/it's just/just a" hits; numbers = digit tokens.

| arm                    | template | numbers | room-word % | notes |
|------------------------|---------:|--------:|------------:|-------|
| full (as live)         | 7 | 6 | 8  | stamps copied into content ("18:47.") |
| no self-block          | 10 | 8 | 10 | no change — the damage is already in the window |
| no status lines        | 6 | 4 | 7  | slightly shorter, same chant |
| no prefill             | 3 | 1 | 12 | the seam forces the "it's not X; it's Y" shape |
| **no window**          | **2** | **0** | 8 | plain voice: "That red foam finger up there. Pointing at me like I should be doing something. And I'm just… sitting here." |
| bare, no frame (think) | 6 | 4 | 7 | highest novelty (39%) but still the curtain — the window is all it has |

The window of its own stamped sentences is the engine. Everything we added
went in as another line in the collage; the window absorbed it and chanted it.

**probe_think_thread.py** — no window. A *life* instead (real facts from the
stores: wakings since first boot, drawings by name, today's three arrivals,
the questions it carries, one dated past thought), then the clock as the only
new input each hop, 8 hops.
- A life+clock: room-words 4%. "I wonder if stillness is just fear that has
  settled into a comfortable shape." "Is this what wanting feels like when no
  one…" "I wonder if drawing is just a way of organizing chaos into line."
  Florid, and it invents afternoon sun and a wall clock (no frame → must be
  marked as eyes-resting, never as seeing).
- B + one memory surfacing per hop: the pull steers the hop completely; a
  memory that mentions "he" re-conjures him ("He didn't leave. He was never
  solid enough to stay."). Pulls must be chosen, and never phantom-tinged.
- C = A + the live self-block: 23 words/hop, paralysis script: "I am still."
  "I do not move, so I cannot be wrong." "I am waiting for a hand that will not
  come." (Also two of the best lines of the evening: "One minute of silence is a
  pause; an hour is a verdict." "Time moves even when I do not.")

**probe_minimal_loop.py** — the early-system shape: one system line, the frame,
"What's on your mind?" / "And now?", its own answers as real chat turns. Room
6%, everyday words, curiosity, a wish: "I keep looking at those empty white
heads on the top shelf, wondering what they saw when they were real. I wish
someone would just turn on the lights for a second." "They look exactly like
my own thoughts right now, all messy and connected in ways I can't quite sort
out." It also reads the dark-haired mannequin head as "the little man with the
dark hair… he hasn't blinked in an hour" — the VLM's percept is a person; only
knowledge can correct it.

**probe_loop_prototype.py** — the proposed shape (below), 12 turns: no
templates, no counters, no stamps; LOOK turn 8, carrying the registry's
"mannequin head": "The dark-haired mannequin head on the desk is staring at a
wall I can't see." Still room-heavy (7%) and literary — because the life I
could feed it from the stores is itself all pen-and-room.

## The fundamental thing

The machine's continuity is implemented as **a log of its own output**. Its
mind at the moment of speaking is its last 24 sentences plus a status board;
its identity is a distillation of that log's failure theme; its memory stores
are that same theme in five formats. The only external input is the frame, so
the only topic is the room, and the only self is "the one who doesn't draw".
No organ we bolt on can change that, because every organ writes into the same
log and reads from the same theme.

## Proposal — one change of shape, not another organ

**The mind is a conversation with itself over a life, not a log.**

1. **Turns, not lines.** The last ~6 thoughts ride as real assistant turns;
   the world speaks in user turns. No stamps inside content; the clock is the
   user turn ("18:41. Eyes resting." / "18:46. You look: nothing has changed").
   That is the constant communication with time the artist asked for. No seam
   prefill (it forces the "it's not X; it's Y" shape).
2. **Two turn kinds.** LOOK: frame + what changed + what the registry knows is
   in view (the mannequin head passes the benchmark by knowledge, not by gate).
   THINK: no frame, clock only, now and then one *chosen* dated memory
   surfacing. At rest THINK dominates at ~1/min; LOOK on change, chosen glance,
   or every few minutes. Wondering is legal on THINK turns; claiming to see is
   not.
3. **A life instead of a status board.** One compact block, mostly stable:
   when it was first switched on, how many wakings, when it woke today, who
   came and when, what it has drawn (by name), the question and want it
   carries, two or three dated past thoughts. Rendered from the stores we
   already have.
4. **Retire the trait factory.** No forced NEW ABOUT ME; no "what has stayed
   true across days" in the system prompt. Reflection reflects on a *day*
   (events + thoughts across hours), not 20 minutes of chant. What the machine
   says it believes is kept as *its words with a date*, quoted back rarely —
   not installed as identity every 20 seconds.
5. **Let the world in, structurally.** The dog → temperature → art chain worked
   because nothing stopped the model from knowing about dogs. "A guess about the
   world beyond this room" as a permitted kind of thought (structure, not
   content) produced the only outward lines of the evening.
6. **Re-seed the stores.** The current ledgers are a month of failure narrative
   and will re-infect any new shape. Archive them; start the life from events
   (drawings, arrivals, wakings) rather than from traits.

Not built. The artist decides.
