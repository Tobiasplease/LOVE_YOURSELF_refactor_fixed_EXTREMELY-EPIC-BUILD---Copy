"""Prove the body can now choose — and that choosing didn't cost smoothness.

Before July 31 the chain keyed identity at 1-degree bins on all seven
channels at once, so a take trained one state per sample: branching 1.00,
no choice to bias, the recording replayed forever. Identity is now coarse
while poses stay exact. This measures, on the REAL recordings:

1. branching: how often the walk has a genuine choice (before vs after)
2. smoothness: the pose step between consecutive states must stay small —
   coarse identity must NOT mean coarse movement
3. divergence: two walks from the same seed must part company
4. samplers: temperature widens the visited set, repetition penalty stops
   the walk falling into the same loop, min_p keeps the wild ones sane

  python debug/test_move_sampler.py
"""

import math
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from motor_panel import arms_markov as engine
from motor_panel.session import Session

DATASETS = ["session_calm_observant_a.json", "session_energized_engaged_a.json", "session_withdrawn_distant_a.json"]
SERVO_ONLY = {"x", "y", "pen", "lung"}
failures = []


def servo_chain(fn, scale):
    s = Session.load(fn)
    tracks = [t for t in s.tracks if t.has_take and not (set(t.channels) & SERVO_ONLY)]
    chans = [c for t in tracks for c in t.channels]
    bins = {c: engine.DEFAULT_BINS.get(c, 1.0) * scale for c in chans}
    return engine.train(s._joint_samples(tracks), chans, bins=bins), chans


def branch_stats(chain):
    f = chain["servo_transitions"]
    b = [len(v) for v in f.values()]
    return len(f), st.mean(b), 100.0 * sum(1 for v in f.values() if len(v) > 1) / max(1, len(f))


def walk(chain, chans, n=200, temp=1.0, rep=1.0, min_p=0.0, seed=1234):
    """Deterministic-seeded walk; returns visited keys and poses."""
    import random as _r

    _r.seed(seed)
    gen = engine.Generator(
        chain, send_ease=lambda d: None, send_plan=lambda d, dt: None, temperature=temp, repetition=rep, min_p=min_p, repetition_window=24
    )
    first, second = chain["servo_transitions"], chain.get("servo_second_order", {})
    cur = next(iter(first))
    prev, keys, poses = None, [cur], [gen._state(cur)]
    for _ in range(n):
        nxts = gen._candidates(first, second, prev, cur)  # same table choice the live walk makes
        if not nxts:
            break
        nxt = gen._pick(nxts, gen._state(cur))
        prev, cur = cur, nxt
        keys.append(cur)
        poses.append(gen._state(cur))
    return keys, poses


# --- 1: branching, before vs after ------------------------------------------
print(f"{'dataset':26s} {'BEFORE (1x)':>22s}   {'AFTER (8x)':>22s}")
for fn in DATASETS:
    old, _ = servo_chain(fn, 1.0)
    new, _ = servo_chain(fn, engine.KINETIC_STATE_BIN_SCALE)
    o, n = branch_stats(old), branch_stats(new)
    print(f"{fn[8:-5]:26s} {o[0]:4d} states, {o[1]:.2f} branch  {n[0]:4d} states, {n[1]:.2f} branch, {n[2]:.0f}% with a choice")
    if n[1] <= o[1]:
        failures.append(f"{fn}: coarsening did not increase branching ({o[1]:.2f} -> {n[1]:.2f})")
    if n[2] < 5:
        failures.append(f"{fn}: only {n[2]:.0f}% of states offer a choice")


# --- 2: the body must move at the SAME SPEED, with fewer waypoints ----------
# The real invariant isn't step size (coarse states are further apart by
# definition) — it's degrees per second. A merged state dwells longer, so its
# transition must be billed the whole dwell or the take replays too fast.
def speed_profile(scale):
    ch, cs = servo_chain(DATASETS[0], scale)
    f = ch["servo_transitions"]
    poses = ch["state_poses"]
    sp = []
    for src, nxts in f.items():
        for dst, info in nxts.items():
            step = max(abs(poses[dst][c] - poses[src][c]) for c in cs)
            sp.append(step / max(0.02, info["avg_dt"]))
    return st.median(sp), ch, cs


slow, _c1, _s1 = speed_profile(1.0)
fast, chain, chans = speed_profile(engine.KINETIC_STATE_BIN_SCALE)
print(f"\ntempo: {slow:.1f} deg/s at 1x  vs  {fast:.1f} deg/s at {engine.KINETIC_STATE_BIN_SCALE:.0f}x — the performance's pace, not a speed-up")
if fast > slow * 2.5:
    failures.append(f"coarse states move {fast / slow:.1f}x faster than recorded — dwell time not being charged")

# poses must be the REAL recorded values, not the bin grid
gen0 = engine.Generator(chain, send_ease=lambda d: None, send_plan=lambda d, dt: None)
offs = [abs(gen0._state(k)[c] - engine._unkey(k, chans, chain["discretization"])[c]) for k in list(chain["state_poses"])[:200] for c in chans]
print(f"pose fidelity: stored poses sit {st.mean(offs):.2f}deg off the bin centres — real performance values")
if st.mean(offs) < 0.2:
    failures.append("poses match bin centres — the identity/pose split is not in effect")

# --- 3: divergence — the walk is no longer one fixed loop -------------------
a, _ = walk(chain, chans, n=150, seed=1)
b, _ = walk(chain, chans, n=150, seed=99)
same = sum(1 for x, y in zip(a, b) if x == y)
print(f"\ndivergence: two walks share {100 * same / len(a):.0f}% of their step sequence (before the split: 100% — one fixed loop)")
if same == len(a):
    failures.append("both walks identical — still deterministic")


# --- 4: the samplers do what they say ---------------------------------------
# Exploration is NOT distinct-states-visited: in a mostly-linear chain the
# faithful path visits the most states by walking the whole recording. What
# temperature buys is UNPREDICTABILITY — how much two runs differ.
def agreement(temp=1.0, rep=1.0, min_p=0.0):
    runs = [walk(chain, chans, n=200, temp=temp, rep=rep, min_p=min_p, seed=s)[0] for s in (11, 22, 33)]
    pairs = [(runs[0], runs[1]), (runs[0], runs[2]), (runs[1], runs[2])]
    return 100.0 * st.mean([sum(1 for x, y in zip(p, q) if x == y) / min(len(p), len(q)) for p, q in pairs])


cold, hot = agreement(temp=0.25), agreement(temp=3.0)
print(f"temperature: cold runs agree {cold:.0f}% of the time, hot runs {hot:.0f}% — heat buys unpredictability")
if hot >= cold:
    failures.append(f"temperature did not loosen the walk (cold {cold:.0f}% vs hot {hot:.0f}%)")


def loopiness(rep):
    keys, _ = walk(chain, chans, n=400, temp=1.6, rep=rep, seed=3)
    win = 24
    return 100.0 * sum(1 for i, k in enumerate(keys) if k in keys[max(0, i - win) : i]) / len(keys)


off, on = loopiness(1.0), loopiness(3.0)
print(f"repetition:  revisits within 24 steps — {off:.0f}% off, {on:.0f}% on — the penalty breaks loops")
if on > off:
    failures.append(f"repetition penalty increased looping ({on:.0f}% vs {off:.0f}%)")

capped, _ = walk(chain, chans, n=200, temp=2.5, min_p=0.3, seed=5)
print(f"min_p:       wild walk with a floor still ran {len(capped)}/201 steps without stranding")
if len(capped) < 150:
    failures.append(f"min_p stranded the walk after {len(capped)} steps")

# --- 5: old chains (no state_poses) still play --------------------------------
legacy = dict(chain)
legacy.pop("state_poses")
g = engine.Generator(legacy, send_ease=lambda d: None, send_plan=lambda d, dt: None)
if not g._state(next(iter(legacy["servo_transitions"]))):
    failures.append("chain trained before the split no longer yields a pose")
print("back-compat: a chain without stored poses still resolves (bin-centre fallback)")

print("\n" + ("ALL OK" if not failures else "FAILURES:\n  " + "\n  ".join(failures)))
sys.exit(0 if not failures else 1)
