"""Storage-gate tests for the felt-state phrase (July 26).

The mood read's own phrase is the documented May/June anti-pattern (model
affect re-injected verbatim) unless bounded. Two bounds, both storage-side:
no channel-doubling with the persona, no lease renewal for the same
vocabulary. Metaphor itself stays legal. Run: python debug/test_felt_gates.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from captioner.context_compression import ContextCompressionEngine


def bare_engine(persona=""):
    eng = object.__new__(ContextCompressionEngine)
    eng.core_facts = {"self": persona}
    return eng


PERSONA = "I vibrate when the silence gets too heavy."
failures = []


def check(name, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond:
        failures.append(name)


print("— channel doubling with the persona —")
eng = bare_engine(PERSONA)
eng._absorb_mood({"pleasantness": "unpleasant", "energy": "settled", "felt": "heavy, hesitant"})
check("'heavy, hesitant' held back (persona carries 'heavy')", eng.last_mood_read["felt"] == "")
check("numbers kept while phrase held", eng.last_mood_read["valence"] == -0.5 and eng.last_mood_read["arousal"] == 0.3)
eng._absorb_mood({"pleasantness": "neutral", "energy": "settled", "felt": "vibrating quietly"})
check("'vibrating quietly' held back (stem of 'vibrate')", eng.last_mood_read["felt"] == "")
eng._absorb_mood({"pleasantness": "neutral", "energy": "settled", "felt": "restless, waiting"})
check("'restless, waiting' accepted (no persona overlap)", eng.last_mood_read["felt"] == "restless, waiting")
check("accepted phrase becomes the standing felt-state", eng.get_felt_state() == "restless, waiting")

print("— lease renewal —")
eng._absorb_mood({"pleasantness": "neutral", "energy": "settled", "felt": "restless, waiting"})
check("identical re-read held (no re-lease)", eng.last_mood_read["felt"] == "")
eng._absorb_mood({"pleasantness": "neutral", "energy": "settled", "felt": "still waiting, restless"})
check("reworded same vocabulary held too", eng.last_mood_read["felt"] == "")
eng._absorb_mood({"pleasantness": "neutral", "energy": "stirred", "felt": "restless dread rising"})
check("genuinely shifted feeling accepted ('dread', 'rising' are new)", eng.last_mood_read["felt"] == "restless dread rising")
eng._last_accepted_felt["timestamp"] = time.time() - ContextCompressionEngine.FELT_REBORE_SECONDS - 60
eng._absorb_mood({"pleasantness": "neutral", "energy": "stirred", "felt": "dread rising"})
check("same vocabulary may return after FELT_REBORE_SECONDS", eng.last_mood_read["felt"] == "dread rising")

print("— fallback path —")
eng2 = bare_engine(PERSONA)
eng2._absorb_mood({"pleasantness": "unpleasant", "energy": "drained", "felt": "heavy again"})
check("poisoned phrase held on fresh engine", eng2.last_mood_read["felt"] == "")
eng2.set_felt_state("a little flat", source="vector")
check("vector translation NOT blocked while phrase is held", eng2.get_felt_state() == "a little flat")

print("— metaphor stays legal —")
eng3 = bare_engine(PERSONA)
eng3._absorb_mood({"pleasantness": "pleasant", "energy": "stirred", "felt": "a low hum of wanting"})
check("figurative phrase accepted when it doubles no channel", eng3.last_mood_read["felt"] == "a low hum of wanting")

print()
if failures:
    print(f"{len(failures)} FAILURE(S): {failures}")
    sys.exit(1)
print("all felt-gate tests passed")
