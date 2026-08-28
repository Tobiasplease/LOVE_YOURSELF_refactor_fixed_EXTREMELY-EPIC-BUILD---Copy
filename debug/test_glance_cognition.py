"""The cognizant-gaze loop, offline: mention boost (thought leads gaze),
glance verification ladder (missed -> absence event -> forgotten), and the
situational-line events the caption loop receives. No camera, no model.

    python debug/test_glance_cognition.py
"""

import os
import sys
import time
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from perception.spatial_registry import SpatialRegistry

reg = SpatialRegistry(state_path="/tmp/sr_glance_test.json")
reg.entries = {
    "rooster figurine": {"pan": 70, "tilt": 100, "conf": 0.6, "hits": 20, "first_seen": time.time() - 9000, "last_seen": time.time() - 3000},
    "pink shelf": {"pan": 120, "tilt": 110, "conf": 0.5, "hits": 8, "first_seen": time.time() - 9000, "last_seen": time.time() - 3000},
}

reg.note_mentions("The pink shelf keeps catching the light near the window.")
picks = [reg.pick_glance_target(explore_weight=0.0)["term"] for _ in range(300)]
print(f"mention boost: pink shelf picked {picks.count('pink shelf')}/300 (rooster {picks.count('rooster figurine')})")
assert picks.count("pink shelf") > 180

# Absence gates (Aug 28): a low-hit term decays SILENTLY — no event. Tested
# before any event is minted so the global gap can't be what suppressed it.
reg.entries["dust bunny"] = {"pan": 90, "tilt": 90, "conf": 0.3, "hits": 2, "first_seen": time.time() - 900, "last_seen": time.time() - 400}
reg.note_glance_result("dust bunny", False)
reg.note_glance_result("dust bunny", False)
assert reg.pop_absence_event() is None
print("low-hit term missed silently (min-hits gate)")

print("miss 1:", reg.note_glance_result("rooster figurine", False))
print("miss 2:", reg.note_glance_result("rooster figurine", False))
ev = reg.pop_absence_event()
assert ev and ev["term"] == "rooster figurine" and reg.pop_absence_event() is None
print("absence event popped once:", ev["term"])

# Global gap: a second established term missing right after mints nothing.
reg.note_glance_result("pink shelf", False)
reg.note_glance_result("pink shelf", False)
assert reg.pop_absence_event() is None
print("second absence within the global gap suppressed")

# Per-term cooldown: gap cleared, but the term announced absence recently.
reg._last_absence_global = 0.0
reg.entries["wire basket"] = {
    "pan": 80,
    "tilt": 70,
    "conf": 0.5,
    "hits": 9,
    "first_seen": time.time() - 9000,
    "last_seen": time.time() - 3000,
    "last_absence_ts": time.time() - 60,
}
reg.note_glance_result("wire basket", False)
reg.note_glance_result("wire basket", False)
assert reg.pop_absence_event() is None
print("re-announcement within the term cooldown suppressed")

print("miss 3:", reg.note_glance_result("rooster figurine", False))
assert reg.note_glance_result("rooster figurine", False) == "gone" and "rooster figurine" not in reg.entries
print("miss 4: gone, entry forgotten")
assert reg.note_glance_result("pink shelf", True) == "seen" and reg.entries["pink shelf"]["misses"] == 0
print("re-seen resets misses")

import vision.gaze as gaze

gaze._glance_active = True
gaze._glance_label = "pink shelf"
gaze._glance_kind = "revisit"
gaze._glance_started = 123.0
import perception.spatial_registry as sr_mod

sr_mod.spatial_registry = reg
reg._absence_events.append({"term": "rooster figurine", "ts": time.time()})

from captioner.prompts import build_situational_line

agent = SimpleNamespace(_prev_presence_for_line=False, _presence_believed=False)
line = build_situational_line(agent)
print("situational line:", repr(line))
assert "Turned to look where the pink shelf should be." in line
assert "The rooster figurine isn't where it was." in line
assert build_situational_line(agent) == ""
print("second call empty (edge noted once) — all good")
