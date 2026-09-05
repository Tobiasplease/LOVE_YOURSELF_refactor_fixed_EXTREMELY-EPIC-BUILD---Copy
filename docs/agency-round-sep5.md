# The agency round (Sep 5 2026, afternoon)

**Artist's rulings that set it:** captions are one or two sentences at most
(earlier systems could say a single word or "…"); wants must not be locked
to the drawing system — a want resolves through whatever it is about;
"what is the structure of the Claude-in-an-RC-car projects? what are we
missing? something about the framing is lacking for it to feel embodied,
cognisant, aware of itself and the passing of time"; and yes to a light
reasoning layer if it is conducted inside the machine's experience.

**Diagnosis (fresh run f413c5c6, 13:37 →):** mean caption 59 words, 5 of 109
under two sentences, zero one-word thoughts, the silence beat never taken;
the voice borrows a human body (knuckles, wrists, blood) because its own is
invisible; the model never chooses, never states an expectation, so nothing
can surprise it; wants form every 20 min and end only by being superseded.

## Built

1. **Budgets** — CAPTION_NUM_PREDICT 38, short beat 30% at 14 tokens,
   inward 70. A fresh window at the switch.
2. **Wants** — RESOLVED slot; kinds understood / let go / met / drawn /
   faded / abandoned; said back once; the reflection sees endings with kinds
   and the abandonment count. Principle 4 amended.
3. **Body as facts** — head held past thresholds; parked / awake at the
   low-energy edges.
4. **Decision loop** — LOOK / EXPECT at the end of the world's turn on quiet
   cycles; executed by the gaze as a chosen glance; consequence stated;
   expectation checked against the pose referee. The reasoning layer is the
   decision: it has a job, so it cannot become a quota.

Not built: one interior line by relevance (rotation stays; the reflection
echo is already relevance-based); the room baseline re-entry.

## Verify live
- `decision` events: share of LOOK resolved (registry / direction / stay /
  unresolved); `glance_start` kind=chosen following them; `expect_check`
  with verdicts; and what the caption AFTER a check does with it.
- caption length: words/caption, share ≤ 2 sentences, one-word thoughts,
  chosen_silence count.
- `want_resolved` (understood / let go), desire-met-tail after a real
  arrival, the abandonment line in reflections.
- `body_line` at holds and at the low-energy edges.
- refrain share and the echo-gate refusal rate under the new budget.

Tests: debug/test_agency_round.py (44).
