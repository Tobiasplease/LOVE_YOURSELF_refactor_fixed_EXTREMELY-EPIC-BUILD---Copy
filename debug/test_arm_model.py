"""Prove the arm-studio fit math on a synthetic ground-truth arm.

A known planar arm generates 'dragged' observations at 4 poses with ±1.5
unit noise (a careful human eyeballing an oblique camera view). The fit
must recover forward kinematics accurate to a few units on HELD-OUT
poses, and the separation helper must respect geometry.

  python debug/test_arm_model.py
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from motor_panel.arm_model import ArmModel, _dist

failures = []

TRUTH = ArmModel(
    base=(85.0, 55.0), l1=28.0, l2=24.0, shoulder=(math.radians(-1.0), math.radians(195.0)), elbow=(math.radians(0.9), math.radians(-15.0))
)

# deterministic pseudo-noise (no random: repeatability)
NOISE = [(1.1, -0.8), (-1.3, 0.6), (0.4, 1.2), (-0.9, -1.1), (1.0, 0.2), (-0.5, 1.4), (0.7, -1.2), (-1.2, 0.3)]


def observe(servo_s, servo_e, k):
    elbow, hand = TRUTH.fk(servo_s, servo_e)
    ne, nh = NOISE[k % len(NOISE)], NOISE[(k + 3) % len(NOISE)]
    return {
        "servo_shoulder": servo_s,
        "servo_elbow": servo_e,
        "base": (TRUTH.base[0] + 0.5, TRUTH.base[1] - 0.4),
        "elbow": (elbow[0] + ne[0], elbow[1] + ne[1]),
        "hand": (hand[0] + nh[0], hand[1] + nh[1]),
    }


calib_poses = [(70, 70), (70, 110), (110, 90), (100, 120)]
captures = [observe(s, e, i) for i, (s, e) in enumerate(calib_poses)]
model, resid = ArmModel.fit(captures)
if resid > 3.0:
    failures.append(f"fit residual {resid:.2f} too high")

holdout = [(80, 80), (95, 100), (105, 75), (75, 115)]
errs = [_dist(model.fk(s, e)[1], TRUTH.fk(s, e)[1]) for s, e in holdout]
if max(errs) > 4.0:
    failures.append(f"held-out hand error {max(errs):.2f} units (want <4)")
print(f"fit: residual {resid:.2f}, held-out hand errors {[round(e, 1) for e in errs]} (truth L1=28/L2=24 -> {model.l1:.1f}/{model.l2:.1f})")

# separation geometry: pen far from the arm is positive, pen ON the hand is negative
(ex, ey), (hx, hy) = model.fk(90, 90)
far = model.separation(90, 90, hx + 60, hy + 60)
on_hand = model.separation(90, 90, hx, hy)
mid_fore = model.separation(90, 90, (ex + hx) / 2, (ey + hy) / 2)
if far < 40:
    failures.append(f"far separation {far:.1f} too small")
if on_hand > -3:
    failures.append(f"pen on hand should be well negative, got {on_hand:.1f}")
if mid_fore > 1.0:
    failures.append(f"pen touching forearm midpoint should be ~0, got {mid_fore:.1f}")
print(f"separation: far {far:.0f}, on-hand {on_hand:.1f}, on-forearm {mid_fore:.1f}")

# persistence round-trip
tmp = "/tmp/arm_model_test.json"
model.save(tmp)
back = ArmModel.load(tmp)
if back is None or _dist(back.fk(90, 90)[1], model.fk(90, 90)[1]) > 1e-6:
    failures.append("save/load round-trip drifted")
os.remove(tmp)
print("persistence: round-trip exact")

# --- gantry arm: affine + elbow IK fit -------------------------------------
from motor_panel.arm_model import GantryArmModel, arms_separation

R_TRUTH = GantryArmModel(matrix=((0.4, 0.05), (-0.03, 0.45)), offset=(25.0, 55.0), base=(22.0, 92.0), l1=34.0, l2=30.0, elbow_sign=-1.0)
r_caps = []
for i, cmd in enumerate([(5, 5), (60, 10), (30, 40), (90, -5)]):
    elbow, hand = R_TRUTH.fk(*cmd)
    ne, nh = NOISE[i], NOISE[i + 4]
    r_caps.append({"cmd": cmd, "base": (22.4, 91.7), "elbow": (elbow[0] + ne[0], elbow[1] + ne[1]), "hand": (hand[0] + nh[0], hand[1] + nh[1])})
r_model, r_resid = GantryArmModel.fit(r_caps)
r_errs = [_dist(r_model.hand(*c), R_TRUTH.hand(*c)) for c in [(20, 20), (75, 5), (45, 30)]]
if r_resid > 3.0 or max(r_errs) > 4.0:
    failures.append(f"gantry fit poor: resid {r_resid:.2f}, held-out {max(r_errs):.2f}")
if r_model.elbow_sign != R_TRUTH.elbow_sign:
    failures.append("gantry elbow side chosen wrong")
print(f"gantry fit: residual {r_resid:.2f}, held-out {[round(e, 1) for e in r_errs]}, elbow side {r_model.elbow_sign:+.0f}")

# --- arm-vs-arm separation ---------------------------------------------------
apart = arms_separation(model, r_model, 90, 90, 5, 5)
lh = model.fk(90, 90)[1]
import numpy as np

cmds = [(x, y) for x in range(0, 101, 5) for y in range(-10, 51, 5)]
best = min(cmds, key=lambda c: _dist(r_model.hand(*c), lh))
touching = arms_separation(model, r_model, 90, 90, *best)
if apart < 5:
    failures.append(f"arms far apart but separation {apart:.1f}")
if touching > apart - 5:
    failures.append(f"hands driven together but separation barely dropped ({touching:.1f} vs {apart:.1f})")
print(f"arm-vs-arm: apart {apart:.1f}, gantry hand driven at servo hand -> {touching:.1f}")

print("\n" + ("ALL OK" if not failures else "FAILURES:\n  " + "\n  ".join(failures)))
sys.exit(0 if not failures else 1)
