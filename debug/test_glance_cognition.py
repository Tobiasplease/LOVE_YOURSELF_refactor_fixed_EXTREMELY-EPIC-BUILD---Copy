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

print("miss 1:", reg.note_glance_result("rooster figurine", False))
print("miss 2:", reg.note_glance_result("rooster figurine", False))
ev = reg.pop_absence_event()
assert ev and ev["term"] == "rooster figurine" and reg.pop_absence_event() is None
print("absence event popped once:", ev["term"])
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
