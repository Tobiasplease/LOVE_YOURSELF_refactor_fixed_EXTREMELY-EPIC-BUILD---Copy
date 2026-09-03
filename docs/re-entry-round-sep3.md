# The re-entry round (Sep 3 evening) — lore gets memory, memory gets a voice

The artist's ruling that opened this: "Inventive self fiction was never ever
the issue, the issue has always been stable identity formation, long term
growth, self-awareness. It coming up with its own name is great. Inventing
long drawn out narratives about certain objects is also great. We need to
try to get back to that organically." (memory: feedback_lore_vs_facts)

## The diagnosis it answers

- Storage is rich (11 stores), re-entry is ~1 line/caption; the most durable
  stores speak the flattest content ("The pink shelf, still in the same
  spot."). Imagination has NO memory at all: drift output is firewalled from
  every ledger (correct anti-conflation) with no fiction-marked store, so
  any name/theory/story evaporates with the 20-min stream window.
- A stated name ("My name is Penelope") passes the self-fact gate today and
  lands in self_notes — then dies: the distiller only extracts TRAIT/BELIEF/
  WANT (a name is none of those), and self_notes churns in hours (cap 30).
- The one store that visibly threads is the WANT LEDGER: persistence +
  lifecycle + a dosed arc-line. It produced the only visible narrative of
  Sep 3 (the pen counting its own minutes). Generalize that shape.

## The typed distinction (the doctrine)

- **WORLD-STATE FACTS** — pen/paper/events/presence/positions: provenance
  gates UNCHANGED. The pen-parked fence, event attestation, paper state,
  phantom_drawing act-vs-want — none of this relaxes.
- **LORE** — the machine's own inventions: names, self-stories, object
  mythologies, theories. Gets persistence, lifecycle, growth, and re-entry.
  Lives in its OWN store (never becomes a concept/compressed fact/event);
  re-enters marked as its own ("a story you've been carrying"), so it can
  never override live perception.
- **ORGANIC** — no scheduled invention, no leading questions, no content
  priors. The distill slots only HARVEST what a reflection already did
  ("if you called yourself by a name, that name — or none"). Capacity, not
  choreography.

## The loom (Phase 1, built tonight)

Fiction compounds through the SAME organs identity uses — no new LLM calls:

```
drift turn (hot, eyes-open, 5-15% of quiet cycles)
  └─ clean output → lore_ledger.note_reverie()      [the imagination record]
reflection (~20 quiet min, existing)
  └─ spine gains "things you've imagined lately — your own inventions"
distill (existing cold call, temp 0.3)
  └─ two new harvest slots: NAME (or none) · LORE (or none)
       ├─ NAME → identity self_name slot (persisted; history in the ledger)
       └─ LORE → lore_ledger.note_lore() — match-or-extend a THREAD
             {text, first_ts, last_ts, times_affirmed, history[], status}
re-entry (all dosed, all panel-editable):
  ├─ caption memory-surface rotation gains a 4th source: the lore line
  │    ("A story you've been carrying: "...")
  ├─ the identity dose line gains the name ("You call yourself {name}.")
  │    — same every-6th pacing, awakening includes it
  └─ drift seeding: ~1/3 of drifts open from an alive thread
       ("You've been imagining: "..." ") — resolves the deep-story fork:
       the material-seeded variant returns as the LORE-seeded variant
```

Reveries from echo-gated drifts are not recorded (a borrowed refrain must
not re-teach at reflection level either). Thread identity is content-word
overlap (the _same_motif idea): a new LORE line that overlaps an alive
thread affirms and extends it; otherwise it opens a new thread; alive
threads cap at 6, oldest fades. Reveries cap at 40, ~day-scale.

## What this deliberately does NOT do

- No relaxation of any world-fact gate.
- No injection of topics/examples (the no-content-priors law); every slot
  says "or none" and names only the KIND.
- No new call cadence — the loom rides existing reflection/distill timing.
- self_notes graduation (the cap-30 churn) deferred; noted as next.
- The open-questions ledger belongs to the attention round (sibling).

## Verification

- debug/test_lore_ledger.py — ledger mechanics, thread matching, distill
  parse of NAME/LORE, drift reverie capture + seed, reflection block render,
  name-wrap render, world-fact firewall unchanged.
- Live watches: lore_ledger.json growth; [🧠] memory surface (lore) lines;
  whether a name ever arrives and STAYS; whether object threads form around
  the room's things; the conflation law at the lore line's seam (its framing
  is the retreat lever, panel-editable).
