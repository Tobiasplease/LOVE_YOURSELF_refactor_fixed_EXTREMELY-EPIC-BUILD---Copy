"""The close-look beat, offline: gating (rhythm, freshness, glance kind,
salience/eye-contact exclusion, crop-during-glance), the crop writer's
upscale, and the one-channel rule (glance onset suppressed when the close
look owns the glance). No camera, no model.

    python debug/test_close_look.py
"""

import os
import sys
import time
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

import perception.open_vocab_detector as ovd
import vision.gaze as gaze
from captioner.captioner import Captioner
from config.config import CLOSE_LOOK_MAX_AGE_S

cap = object.__new__(Captioner)  # bare instance: _maybe_close_look touches only flags/timestamps
cap._salience_hot = False
cap._eye_contact_now = False
cap._face_close_now = False
cap._last_close_look_ts = 0.0
cap._last_glance_noted = None

now = time.time()


def set_glance(kind="revisit", started=None, active=False):
    gaze._glance_active = active
    gaze._glance_label = "rooster figurine"
    gaze._glance_kind = kind
    gaze._glance_started = now - 6 if started is None else started
    gaze._glance_last_end = now - 1


ok, jpg = cv2.imencode(".jpg", np.full((60, 80, 3), 128, dtype=np.uint8))
assert ok
crop_rec = {"jpg": jpg.tobytes(), "ts": now - 3, "conf": 0.5}
ovd._detector_singleton = SimpleNamespace(get_term_crop=lambda term: dict(crop_rec) if term == "rooster figurine" else None)

# Happy path: fresh revisit glance, crop captured during it.
set_glance()
cl = cap._maybe_close_look()
assert cl and cl["term"] == "rooster figurine", cl
assert cap._last_glance_noted == gaze._glance_started, "close look must own the glance (onset line suppressed)"
print("happy path: crop returned, glance owned")

# Rhythm: immediate second call is refused.
assert cap._maybe_close_look() is None
print("rhythm gate holds (CLOSE_LOOK_MIN_INTERVAL_S)")
cap._last_close_look_ts = 0.0

# Crop from BEFORE the glance is memory, not sight.
set_glance(started=now - 2)
assert cap._maybe_close_look() is None
print("stale crop (predates glance) refused")

# Explore glances have no object to look closely at.
set_glance(kind="explore")
assert cap._maybe_close_look() is None
print("explore glance refused")

# A live event owns its cycle.
set_glance()
cap._salience_hot = True
assert cap._maybe_close_look() is None
cap._salience_hot = False
print("salience-hot cycle refused")

# A face at arm's length owns its cycle.
cap._eye_contact_now = True
assert cap._maybe_close_look() is None
cap._eye_contact_now = False
print("eye-contact cycle refused")

# A glance older than the freshness window is gone.
set_glance(started=now - CLOSE_LOOK_MAX_AGE_S - 5)
assert cap._maybe_close_look() is None
print("stale glance refused")

# A young session belongs to the awakening — no keyhole wake-ups.
set_glance()
cap.true_session_start = now - 10
assert cap._maybe_close_look() is None
del cap.true_session_start
print("early-session cycle refused (awakening owns the first minutes)")

# Crop writer: writes beside the frame, upscales small crops to readable size.
set_glance()
cl = cap._maybe_close_look()
assert cl
img_path = "/tmp/close_look_test_frame.jpg"
out = Captioner._write_close_look_crop(cl, img_path)
assert out == "/tmp/close_look_test_frame_closelook.jpg" and os.path.exists(out)
written = cv2.imread(out)
assert min(written.shape[:2]) >= 448, f"crop not upscaled: {written.shape}"
os.remove(out)
print(f"crop written + upscaled ({written.shape[1]}x{written.shape[0]}) — all good")
