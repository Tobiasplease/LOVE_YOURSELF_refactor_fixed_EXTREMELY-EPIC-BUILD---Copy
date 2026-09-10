#!/usr/bin/env python3
"""
Finished-drawing capture check (Sep 10 2026) — drawing/finished_capture.py.

Runs the Step 2.5 capture with a stub camera and stub kinetic hooks, so the
choreography and the file/state/log plumbing can be verified without the arms,
the gantry or a real table. Asserts the order the ritual depends on:

    get-clear (on_paper_check_start) -> gaze parked -> frames -> gaze released
    -> release (on_paper_check_done)

Run:  python debug/test_finished_capture.py
"""

import os
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config.config as cfg  # noqa: E402

cfg.FINISHED_CAPTURE_SETTLE_S = 0.2  # keep the test quick
cfg.FINISHED_CAPTURE_FRAMES = 2
# Write into a throwaway dir, NEVER the live event_log: the real captures land
# in <MOOD_SNAPSHOT_FOLDER>/finished_drawings/ under the same finished_*.jpg
# naming, and cleaning up after this test once took four real drawings with it.
cfg.MOOD_SNAPSHOT_FOLDER = tempfile.mkdtemp(prefix="finished_capture_test_")
print(f"[test] capture output redirected to {cfg.MOOD_SNAPSHOT_FOLDER}")

import safety.paper_detection as pd  # noqa: E402
import utils.hooks as hooks  # noqa: E402
import vision.gaze as gaze  # noqa: E402
from drawing.finished_capture import capture_finished_drawing  # noqa: E402
from utils.state_manager import state_manager  # noqa: E402

events = []


class StubCamera:
    def read_frame(self):
        # a recognisable frame: mid-grey with a white rectangle for the "sheet"
        f = np.full((480, 640, 3), 60, dtype=np.uint8)
        f[120:360, 160:480] = 235
        return f


# the aruco shared frame would win over the camera; force the camera path
pd.paper_detector._grab_frame = lambda camera: camera.read_frame()

hooks.on_paper_check_start = lambda: (events.append("clear"), 0.3)[1]
hooks.on_paper_check_done = lambda: events.append("release")

_real_set = gaze.set_paper_search_mode


def _traced(active, **kw):
    events.append(f"gaze:{'park' if active else 'free'}")


gaze.set_paper_search_mode = _traced

state_manager.set_hardware_refs(StubCamera(), None)

print("=== running capture ===")
path = capture_finished_drawing()

print("\n=== results ===")
print("returned path :", path)
print("event order   :", events)
print("state path    :", state_manager.last_finished_drawing_image)
print("state ts      :", state_manager.last_finished_drawing_ts)

failures = []
if not path or not os.path.exists(path):
    failures.append(f"no image written (path={path})")
elif os.path.getsize(path) < 1000:
    failures.append(f"image suspiciously small: {os.path.getsize(path)} bytes")

expected = ["clear", "gaze:park", "gaze:free", "release"]
if events != expected:
    failures.append(f"choreography order wrong:\n    expected {expected}\n    got      {events}")

if state_manager.last_finished_drawing_image != path:
    failures.append("state_manager.last_finished_drawing_image not filed")

d = os.path.join(cfg.MOOD_SNAPSHOT_FOLDER, "finished_drawings")
written = sorted(f for f in os.listdir(d) if f.endswith(".jpg")) if os.path.isdir(d) else []
if len(written) < cfg.FINISHED_CAPTURE_FRAMES:
    failures.append(f"expected {cfg.FINISHED_CAPTURE_FRAMES} frames on disk, found {len(written)}")
print("frames on disk:", written[-cfg.FINISHED_CAPTURE_FRAMES :])

print("\n=== disabled-path check ===")
cfg.ENABLE_FINISHED_DRAWING_CAPTURE = False
events.clear()
if capture_finished_drawing() is not None or events:
    failures.append("disabled flag did not short-circuit the capture")
else:
    print("OK — returns None, body never moves")
cfg.ENABLE_FINISHED_DRAWING_CAPTURE = True

print("\n=== no-camera path check ===")
state_manager.set_hardware_refs(None, None)
events.clear()
if capture_finished_drawing() is not None or events:
    failures.append("missing camera did not short-circuit before moving the body")
else:
    print("OK — returns None, body never moves")

print("\n=== raising-hook check (must not propagate) ===")
state_manager.set_hardware_refs(StubCamera(), None)


def _boom():
    raise RuntimeError("kinetic bus is down")


hooks.on_paper_check_start = _boom
try:
    r = capture_finished_drawing()
    print(f"OK — survived a raising get-clear hook, returned {os.path.basename(r) if r else None}")
    if r is None:
        failures.append("a failing get-clear hook lost the photo entirely")
except Exception as e:
    failures.append(f"exception escaped into the ritual: {e}")

print("\n=== two-halves wait (arms vs gantry) ===")
# the shutter must wait for the SLOWER half of the body, not just the arms
hooks.on_paper_check_start = lambda: 0.2
timings = []
for arms, gantry in ((0.2, 1.0), (1.0, 0.2)):
    hooks.on_paper_check_start = lambda a=arms: a
    t0 = time.time()
    capture_finished_drawing(extra_clear_s=gantry)
    waited = time.time() - t0
    timings.append((arms, gantry, waited))
    print(f"  arms={arms}s gantry={gantry}s -> waited {waited:.2f}s")
    if waited < max(arms, gantry):
        failures.append(f"shot before the body stopped: arms={arms} gantry={gantry} waited={waited:.2f}")

print("\n=== gantry plan from the live paper take ===")
from grbl.paper_gantry import paper_gantry_plan, paper_take_name  # noqa: E402

take = paper_take_name()
moves, dur = paper_gantry_plan()
print(f"  take   : {take or '(none)'}")
print(f"  moves  : {len(moves)}  recording {dur:.1f}s")
if take and not moves:
    failures.append(f"paper take {take} yields no gantry moves — the x/y track is missing or unreadable")
if moves:
    bad = [m for m in moves if not (100 <= m[2] <= 3000)]
    if bad:
        failures.append(f"{len(bad)} moves outside the feed clamp, e.g. {bad[0]}")

print("\n" + "=" * 50)
if failures:
    print("FAILURES:")
    for f in failures:
        print("  -", f)
    sys.exit(1)
print("ALL CHECKS PASSED")
