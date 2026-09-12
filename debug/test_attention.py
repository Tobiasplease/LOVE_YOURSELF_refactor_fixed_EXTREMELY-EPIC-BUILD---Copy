#!/usr/bin/env python3
"""Sep 12 — room attention (captioner/attention.py) and the client-side image
token sizing (utils/image_tokens.py). Pure functions; no server, no run log."""
import math
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402

from captioner.attention import RoomAttention, curious, decide_every  # noqa: E402
from utils.image_tokens import dims_for_tokens, sized, sized_copy, sized_jpeg, tokens_for_attention, tokens_of  # noqa: E402

fails = 0


def check(name, ok, got=""):
    global fails
    print(("PASS  " if ok else "FAIL  ") + name + (f"   → {got!r}" if not ok else ""))
    fails += 0 if ok else 1


# --- token math
check("1280x720 encodes at ~880", tokens_of(1280, 720) == 880, tokens_of(1280, 720))
w, h = dims_for_tokens(1280, 720, 1024)
check("full attention → ~1024 tokens, multiples of 32, aspect kept", 950 <= tokens_of(w, h) <= 1100 and w % 32 == 0 and h % 32 == 0 and abs(w / h - 16 / 9) < 0.1, (w, h, tokens_of(w, h)))
w, h = dims_for_tokens(1280, 720, 256)
check("floor → ~256 tokens", 220 <= tokens_of(w, h) <= 300, (w, h, tokens_of(w, h)))
w, h = dims_for_tokens(200, 400, 1024)
check("a small crop is upscaled to ~1024", 950 <= tokens_of(w, h) <= 1100, (w, h, tokens_of(w, h)))
check("attention 1 → max, 0 → min, half → between", tokens_for_attention(1.0) == 1024 and tokens_for_attention(0.0) == 256 and 600 <= tokens_for_attention(0.5) <= 680)
img = np.zeros((720, 1280, 3), dtype=np.uint8)
check("sized array", tokens_of(*sized(img, 256).shape[1::-1]) <= 300)
jb = sized_jpeg(img, 512)
check("sized jpeg bytes", isinstance(jb, (bytes, bytearray)) and len(jb) > 100)
d = tempfile.mkdtemp()
p = os.path.join(d, "mood_1.jpg")
import cv2  # noqa: E402

cv2.imwrite(p, img)
q = sized_copy(p, 256)
check("sized copy written beside the original, named by tokens", q.endswith("mood_1_t256.jpg") and os.path.exists(q), q)
check("sized copy reused", sized_copy(p, 256) == q)
check("missing path passes through", sized_copy("/nonexistent.jpg", 256) == "/nonexistent.jpg")

# --- attention dynamics
T = time.time()
a = RoomAttention(T)
check("boot is a fresh look", a.value == 1.0)
a.update({}, T + 600)
check("ten minutes of sameness → ~0.46 (floor 0.15, tau 600)", abs(a.value - (0.15 + 0.85 * math.exp(-1))) < 0.02, a.value)
a.update({}, T + 3600)
check("an hour → at the floor", a.value < 0.17, a.value)
check("below curious", not curious(a.value))
a.update({"scene_motion": True, "ego_count": 4}, T + 3609)
check("motion while the camera itself moved earns nothing", a.value < 0.2 and a.last_reason == "own motion", (a.value, a.last_reason))
a.update({"scene_motion": True}, T + 3610)
check("motion alone adds 0.4, does not reset", abs(a.value - (0.15 + 0.4)) < 0.03 and a.last_reason == "motion", a.value)
a.update({"scene_motion": True, "salience_hot": True, "presence_believed": True}, T + 3611)
check("a person snaps to 1.0", a.value == 1.0 and a.last_reason == "someone")
a.update({}, T + 4211)
a.update({}, T + 4212, view_verdict="changed")
check("the referee's 'changed' bumps +0.6", abs(a.value - min(1.0, 0.15 + 0.85 * math.exp(-1) + 0.6)) < 0.02, a.value)
b = RoomAttention(T)
b.update({}, T + 3600)
b.update({}, T + 3601, new_view=True, unseen_share=1.0)
check("a first look this way (a new 20° cell) bumps +0.3", abs(b.value - (0.15 + 0.3)) < 0.03, b.value)
b2 = RoomAttention(T)
b2.update({}, T + 3600)
b2.update({}, T + 3601, new_view=True, unseen_share=0.1)
check("a new cell in a mostly-seen room pays only +0.03", abs(b2.value - 0.18) < 0.03, b2.value)
c = RoomAttention(T)
c.update({}, T + 3600)
c.update({"presence_believed": True}, T + 3601)
check("someone present holds attention at 0.8", c.value == 0.8, c.value)

# --- the LOOK ask on the dial
check("cadence: full attention → every 3", decide_every(3, 1.0) == 3)
check("cadence: half → every 6", decide_every(3, 0.5) == 6)
check("cadence: floor → every 12", decide_every(3, 0.15) == 12)

print("\n" + ("ALL PASS" if not fails else f"{fails} FAILED"))
sys.exit(1 if fails else 0)
