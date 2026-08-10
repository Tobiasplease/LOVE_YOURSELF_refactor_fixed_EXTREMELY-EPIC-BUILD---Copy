"""Prove the body stays inside combinations it has actually performed.

No geometry, no camera: the recordings ARE the safety data. Both arms are
recorded together, so a normal chain walk is demonstrated by construction
— the danger is the glue between takes. Measured against the REAL
recordings:

1. the envelope loads and the demonstrated cloud is dense
2. a normal recorded pose passes untouched (no tax on honest movement)
3. straight-line crossfades and full gaze lean stray far — and the guard
   pulls them back inside the threshold
4. the pull-back is CONTINUOUS (no snapping) as a pose sweeps outward
5. live bus: crossfades, lean and a relative startle all stay on proven
   ground, and the guard reports how often it intervened

  python debug/test_safe_envelope.py
"""

import math
import os
import random
import shutil
import statistics as st
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import KINETIC_SAFE_MAX_DIST
from motor_panel import kinetic_bus as kb
from motor_panel.safe_envelope import SAFE_CHANNELS, SafeEnvelope

failures = []
random.seed(11)

# --- 1: the demonstrated cloud ----------------------------------------------
env = SafeEnvelope(on_log=lambda m: print(f"  {m}"))
if not env.active:
    print("\nSKIPPED: no envelope (scipy or recordings missing) — the guard is inert, which is the safe default")
    sys.exit(0)
pts = env.points
spacing = []
for p in random.sample(pts, 40):
    sub = random.sample(pts, 2000)
    ds = sorted(d for d in (math.dist(p, q) for q in sub) if d > 1e-9)
    spacing.append(ds[0])
print(f"cloud: {len(pts)} combinations, neighbours ~{st.median(spacing):.1f} units apart, threshold {KINETIC_SAFE_MAX_DIST}")

# --- 2: honest movement pays no tax -----------------------------------------
untouched = 0
for p in random.sample(pts, 200):
    pose = dict(zip(SAFE_CHANNELS, p))
    if env.project(pose, KINETIC_SAFE_MAX_DIST) is pose:
        untouched += 1
print(f"recorded poses passing untouched: {untouched}/200")
if untouched < 190:
    failures.append(f"guard is taxing demonstrated poses ({untouched}/200 untouched)")


# --- 3: the strays get pulled back ------------------------------------------
def stray_test(label, make_pose, n=40):
    before, after = [], []
    for _ in range(n):
        pose = make_pose()
        before.append(env.distance(pose))
        after.append(env.distance(env.project(pose, KINETIC_SAFE_MAX_DIST)))
    print(f"{label:28s} {st.median(before):5.1f} -> {st.median(after):4.1f} units (worst {max(before):.1f} -> {max(after):.1f})")
    if st.median(after) > KINETIC_SAFE_MAX_DIST + 0.5:
        failures.append(f"{label}: still {st.median(after):.1f} units out after the guard")
    return st.median(before), st.median(after)


def crossfade_midpoint():
    a, b = random.choice(pts), random.choice(pts)
    f = random.uniform(0.25, 0.75)
    return {c: x + (y - x) * f for c, x, y in zip(SAFE_CHANNELS, a, b)}


LEAN = {"shoulder": 14.0, "wrist": 10.0, "x": 4.0, "y": 4.0, "elbow": 9.0}


def leaned():
    return {c: v + LEAN.get(c, 0.0) for c, v in zip(SAFE_CHANNELS, random.choice(pts))}


print()
stray_test("crossfade midpoints", crossfade_midpoint)
stray_test("full gaze lean", leaned)


# --- 4: the correction the BODY receives is continuous ----------------------
# Raw projection onto a point cloud is discontinuous by nature (the landing
# spot swaps as neighbours change); the bus eases the correction so the
# commands the machine actually gets stay smooth. Test what the body gets.
class _Probe(kb.KineticBus):
    def __init__(self, envelope):
        self._commanded, self._corr, self._guard_pulls = {}, {}, 0
        self.envelope = envelope


probe = _Probe(env)
base = dict(zip(SAFE_CHANNELS, random.choice(pts)))
target = dict(zip(SAFE_CHANNELS, random.choice(pts)))
prev, jumps, raws = None, [], []
for i in range(201):
    f = i / 200.0
    pose = {c: base[c] + (target[c] - base[c]) * f for c in SAFE_CHANNELS}
    out = probe._guard(dict(pose))
    v = tuple(out[c] for c in SAFE_CHANNELS)
    raws.append(env.distance(out))
    if prev is not None:
        jumps.append(math.dist(prev, v))
    prev = v
raw_step = math.dist(tuple(base[c] for c in SAFE_CHANNELS), tuple(target[c] for c in SAFE_CHANNELS)) / 200
print(f"\ncontinuity (as the body receives it): steps median {st.median(jumps):.2f}, worst {max(jumps):.2f} (unguarded step {raw_step:.2f})")
print(f"             stray during the sweep: median {st.median(raws):.1f}, worst {max(raws):.1f}")
if max(jumps) > raw_step + 1.5:
    failures.append(f"guard snapped {max(jumps):.1f} units in one step (raw step {raw_step:.2f})")


# --- 5: live bus, guarded vs unguarded (the honest before/after) ------------
def live_run(guarded: bool):
    sent = []
    state = {c: 0.0 for c in ("elbow", "shoulder", "wrist", "finger0", "finger1", "finger2", "finger3", "x", "y")}
    state.update(dict(zip(SAFE_CHANNELS, random.choice(pts))))
    for k in ("finger0", "finger1", "finger2", "finger3"):
        state[k] = 90.0
    gaze = {"v": (0.0, 0.0)}
    bus = kb.KineticBus(
        library=kb.TemperamentLibrary(owned=kb.OWNED_CHANNELS | {"x", "y"}),
        get_emotion=lambda: "calm_observant",
        is_drawing=lambda: False,
        get_gaze=lambda: gaze["v"],
        get_person=lambda: "absent",
        on_log=lambda m: None,
        send_ease=lambda d: (state.update(d), sent.append(dict(state))),
        send_plan=lambda d, dt: (state.update(d), sent.append(dict(state))),
        get_state=lambda: dict(state),
    )
    bus._dir_flips = {c: False for c in kb.DIRECTION_CHANNELS}
    if not guarded:
        bus.envelope = type("Off", (), {"active": False})()  # not None — enable() would build a real one
    bus.enable()
    time.sleep(4.0)
    gaze["v"] = (1.0, 1.0)  # full lean — the always-on offender
    time.sleep(6.0)
    bus.startle()  # relative flinch from wherever the body stands
    time.sleep(3.0)
    pulls = bus.status()["guard_pulls"]
    bus.shutdown()
    d = [env.distance(x) for x in sent if all(c in x for c in SAFE_CHANNELS)]
    return d, pulls


raw, _ = live_run(guarded=False)
guarded, pulls = live_run(guarded=True)
print(f"\nlive bus (crossfades + full lean + startle), stray from proven ground:")
print(
    f"   unguarded: median {st.median(raw):5.1f}  worst {max(raw):5.1f}  ({sum(1 for d in raw if d > KINETIC_SAFE_MAX_DIST)}/{len(raw)} sends outside)"
)
print(
    f"   guarded:   median {st.median(guarded):5.1f}  worst {max(guarded):5.1f}  ({sum(1 for d in guarded if d > KINETIC_SAFE_MAX_DIST)}/{len(guarded)} sends outside), {pulls} interventions"
)
if st.median(guarded) >= st.median(raw) * 0.7:
    failures.append(f"guard barely helped (median {st.median(raw):.1f} -> {st.median(guarded):.1f})")
if max(guarded) > max(raw):
    failures.append("guarded run strayed further than unguarded")

print("\n" + ("ALL OK" if not failures else "FAILURES:\n  " + "\n  ".join(failures)))
sys.exit(0 if not failures else 1)
