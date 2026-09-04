# The attention round (Sep 4) — curiosity, questions, and honest absence

Queue #2-of-the-week from the Sep 3 evening diagnosis (memory:
attention-permanence-round): the artist's three complaints — dull output,
random-feeling gaze, "the man is gone" — were one system: attention feeds
the voice, belief holds what attention can't see, and neither had a channel
for choice. Their examples that shaped it: "What is that, actually?" /
"What else is in here?" / "Wonder what he's working on now. The robot arm?"
Ruling: the choosing must surface as WANT, never mechanism.

## Built

**1. Investigate glances — the familiar strangers.**
`spatial_registry.pick_glance_target` gains a third kind: with
INVESTIGATE_WEIGHT (0.25), the gaze commits to a high-hits, low-confidence
entry (conf ≤ 0.35, hits ≥ 500 — the live map carried a wall lamp at 783k
sightings, conf 0.20: seen thousands of times, never resolved). Uncertainty
is the pull (least-sure weighted); 15-min per-term cooldown. The cycle then
carries the attested fact — "You're looking at the {label} — you've seen it
many times without ever being sure of it" (`caption.investigate`) — and the
close look now accepts investigate glances (the crop goes where the
uncertainty is). Discernment verification covers them too, so an
investigate look that finds nothing feeds the absence ladder. "What IS
that?" stays the machine's move; the fact is the door.

**2. Open questions — wonders that outlive the stream window.**
The distiller gains a fourth harvest slot: QUESTION ("one question you're
still carrying, as you'd ask it — or 'none'"). Harvested questions live in
the lore ledger (dedupe by content overlap, cap 8, oldest fades) and
re-enter as the fifth memory-surface source — "A question you're still
carrying: '{text}'" — least-recently-surfaced first. Before this, "wonder
what he's working on" evaporated in 20 minutes structurally.

**3. Honest absence edges.**
- PRESENCE_ABSENCE_LOOK_TOLERANCE 30°→18°: 30° was the frame half-width, so
  a person at the frame edge (where the skeleton gate rightly refuses
  partial bodies) counted as looked-for-and-absent — belief died on
  evidence never collected. The day-one "man is gone" complaint.
- The drift call carries the presence fact when the belief is active and no
  person is in the frame ("{who}'s here, just out of view right now") — it
  was the one prompt with no presence line, structurally inviting phantom
  departures.

**4. Observability**: every glance choice and check is a logged fact now
(`glance_start` / `glance_check` actions) — the selector was random AND
unaccountable before.

Same day, same direction: the genre clause took the artist's SPOKENNESS
ruling ("Said the way you'd say it to yourself, not written for anyone")
and lost its two remaining "plain"s.

## Deliberately not built

- B2 day-level visit patterns — needs its own design conversation (memory:
  design wanted, no biometrics). The DetectionMemory latent bug folds in
  there too.
- Vocabulary growth for "what else is in here?" — vocab promotion exists;
  explore stays a coverage roll for now.
- Question closure mechanics — a question fades at cap or gets re-asked;
  whether an answer "closes" it is the machine's business in the voice, not
  a ledger state, until live behavior argues otherwise.

## Verify

`python debug/test_attention_round.py` (19 checks) + all prior suites
green. Live watches: `glance_start` kind mix (investigate should appear at
~25% of glances), the first QUESTION harvest, whether investigate close
looks resolve any conf-0.2 entries (label audit + conf rising = a
relationship deepening), and whether "the man is gone" recurs at the frame
edge (if yes, tighten further or require pose-gate viability).
