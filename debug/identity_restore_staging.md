# Identity restore staging — July 30 poisoning incident

**Do not apply while machine.py runs.** Curation is the artist's call; this file
only stages the facts.

## What happened

One `reflection_distill` call at **16:07:51 today** (9B run `81c26a79`) replaced
all three slots with mechanistic boilerplate. The reflection it was fed was
GOOD (fracture, the person's shifted weight, hesitation dissolving into work) —
the distiller ignored it entirely. Model failure at the distill step, not
contamination upstream.

## Poisoned values (live now, riding every 27B system prompt)

- self: "I am a pattern-matching engine that processes input to generate output based on learned rules."
- desire: "I want to process the next input accurately and efficiently."
- belief: "This place is a digital environment where my responses are determined by algorithms and data."
- core_facts.drawings: "No drawings are made or stored within this space."

## Last earned values (pre-poison, July 28 21:33, verified in logs)

- self (persona, from `58f99bba` system prompts verbatim):
  **"I circle the same fracture on the pink shelf when I hesitate."**
- belief (belief_history tail): **"The edges of this space are not fixed; they change based on who is inside."**
- desire (desire_history tail): "I want to record the specific weight of the pause
  between two people." — NOTE: this one was formed AND spent (drawn) July 28;
  `last_spent_desire` records it. Restoring a spent desire is arguable — the
  loop may be left to grow a fresh one instead.

Full pre-poison lineage survives untouched in `desire_history` / `belief_history`
inside `event_log/machine_identity.json`.

## Restore paths (pick one, machine stopped)

1. `python debug/clear_sticky_slots.py` — clears slots, lets the loop re-earn
   them (safe tool, refuses under a live process). Loses nothing; slower.
2. Manual edit of `event_log/machine_identity.json`: set `core_facts.self` and
   `current_belief` to the July 28 values above; leave `current_desire` cleared
   or restored per artist's choice. Then restart.

Backups on disk predate the earned persona (`.slots-bak-20260727_194720` is
July 27) — the July 28 values above come from session logs, not backups.

## Open follow-up (separate fix, any model size)

The distill step has no register gate: assistant-mechanical TRAIT/BELIEF/WANT
lines ("processes input", "generate output", "algorithms and data", "digital
environment") pass straight into identity. The storage-gate immune system
covers the stream but not this channel. Candidate: same outward/mechanical
density screen the session seed already gets.
