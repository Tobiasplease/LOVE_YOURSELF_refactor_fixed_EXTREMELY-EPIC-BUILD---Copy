"""Provenance gate tests for the events ledger + awakening-seed fragment filter (July 26).

The rooster run stored "A pen shattered into nothingness during a long period
of silence" — pure awakening confabulation — as episodic memory. The gate:
an EVENT line only lands when code attests something happened in the window
(salience spike or executed drawing). Run: python debug/test_event_gate.py
"""

import os
import sys
import time
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from captioner.context_compression import ContextCompressionEngine
from captioner.model_wrapper import _is_plantable_prior

failures = []


def check(name, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond:
        failures.append(name)


def bare_engine():
    eng = object.__new__(ContextCompressionEngine)
    eng.events = []
    eng._perception_events = deque(maxlen=12)
    eng.recent_captions = deque([{"text": "a thought", "timestamp": time.time() - 60}], maxlen=8)
    eng._save_identity = lambda: None
    return eng


print("— events ledger provenance —")
eng = bare_engine()
eng._absorb_event("A pen shattered into nothingness during a long period of silence")
check("confabulated event held back (no sensor-side happening)", eng.events == [])

eng.note_perception_event("salience")
eng._absorb_event("Someone came right up to the table")
check("event lands when salience spiked in the window", len(eng.events) == 1)

eng2 = bare_engine()
eng2._perception_events.append(time.time() - 600)
eng2._absorb_event("A figure moved near the lamp")
check("stale perception event (before the window) does not vouch", eng2.events == [])

eng3 = bare_engine()
eng3.introspective_state = {"current_desire": ""}
eng3.spend_desire("small sharp circle")
eng3._absorb_event("A drawing was made on the paper")
check("executed drawing vouches even with no want to spend", len(eng3.events) == 1)

print("— awakening seed fragment filter —")
check("mid-sentence drift tail rejected", not _is_plantable_prior("by those who came before me long ago when they were still alive breathing freely"))
check("normal thought passes", _is_plantable_prior("The rooster is staring at me again."))
check("digit-led thought passes", _is_plantable_prior("135 hours is too long to just watch."))
check("dash-prefixed thought still passes", _is_plantable_prior("— The lamp is still on."))

print()
if failures:
    print(f"{len(failures)} FAILURE(S): {failures}")
    sys.exit(1)
print("all event-gate tests passed")
