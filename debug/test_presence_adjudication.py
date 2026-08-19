"""Adjudicated presence, phase 1 — offline checks.

The ontology parser (the machine's free words -> person/thing/no-verdict)
and the gate ladder (pending -> verdict commits or vetoes, place veto
persists, person grace expires). No camera, no model.

    python debug/test_presence_adjudication.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from perception.presence_adjudicator import PresenceAdjudicatorThread, parse_ontology

CASES = [
    ("A man in a black shirt bending over a desk.", "person"),
    ("a wooden mannequin torso on the floor", "thing"),
    ("A small robot with cables attached.", "thing"),
    ("a child sitting on the floor", "person"),
    ("A mannequin of a man, seated.", "thing"),
    ("a chair against the wall", "thing"),
    ("Someone standing near the shelf.", "person"),
    ("A seated figure.", None),
    ("a pale silhouette in the corner", None),
    ("A doll wearing a sweater.", "thing"),
    ("papier-mache arm holding a pencil", "thing"),
    ("", None),
]

for reply, want in CASES:
    got = parse_ontology(reply)
    status = "ok" if got == want else f"WRONG (want {want})"
    print(f"  {str(got):8s} <- {reply!r}  [{status}]")
    assert got == want, (reply, got, want)

# gate ladder with a synthetic ledger
adj = PresenceAdjudicatorThread.__new__(PresenceAdjudicatorThread)
import threading

adj.lock = threading.Lock()
adj.ledger_path = "/tmp/entity_ledger_test.json"
adj._entities = []
adj._pending = None
adj._person_until = 0.0
adj._last_call = 0.0
adj._current_candidate_box = staticmethod(lambda: (0.1, 0.6, 0.3, 0.95))

# no verdict yet: gate holds (None) and queues a request (request() finds no crop here — fine)
assert adj.gate() is None
# a thing-verdict at that place vetoes
adj._entities.append({"desc": "a wooden torso", "verdict": "thing", "box": [0.1, 0.6, 0.3, 0.95], "pan": 90, "tilt": 90, "ts": time.time()})
assert adj.gate() == "thing"
# an expired thing-verdict no longer vetoes
adj._entities[0]["ts"] -= 10**6
assert adj.gate() is None
# person grace commits, then expires on presence drop
adj._person_until = time.time() + 60
assert adj.gate() == "person"
adj.notify_presence_dropped()
assert adj.gate() is None
print("gate ladder: pending -> veto -> expiry -> person grace -> drop, all correct")
