"""Calibrated planar model of the left servo arm — fit from human drags.

The arm studio (arm_studio.py) shows the machine's camera view beside a
top-down canvas. At a handful of commanded poses the operator drags a
skeleton (base, elbow, hand) until it matches what the camera sees; each
capture pairs servo values with dragged positions. The fit is closed-form:

  base   = mean of dragged base positions
  L1, L2 = mean joint-to-joint distances
  joint maps = per-joint linear fit  angle = a * servo + b
               (upper arm's world angle vs the shoulder servo; forearm's
               angle RELATIVE to the upper arm vs the elbow servo)

The gantry arm is an elbowed arm too — its hand rides the CNC, so its
model is an AFFINE map from gantry command (x,y) to studio space plus a
base and link lengths for the elbow (two-link IK, side chosen from the
drags). Both arms land in one shared STUDIO frame (an abstract square —
the calibration decides all real limits), so separation is honest
arm-vs-arm segment distance, not point-vs-arm. This module is
deliberately free of Tk and serial: the kinetic bus imports it later for
the collision floor.
"""

import json
import math
import os
from typing import Dict, List, Optional, Tuple

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "arm_model.json")
HAND_RADIUS = 6.0  # command units — covers the wrist sweep's footprint


def _mean(xs):
    return sum(xs) / len(xs)


def _wrap(a: float) -> float:
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


def _linfit(xs: List[float], ys_angles: List[float]) -> Tuple[float, float]:
    """Least-squares line through (servo, angle) pairs. Angles are
    unwrapped around the first sample so the fit never straddles ±pi."""
    ref = ys_angles[0]
    ys = [ref + _wrap(a - ref) for a in ys_angles]
    n = len(xs)
    if n == 1:
        return 0.0, ys[0]
    mx, my = _mean(xs), _mean(ys)
    denom = sum((x - mx) ** 2 for x in xs)
    if denom < 1e-9:
        return 0.0, my
    a = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom
    return a, my - a * mx


class ArmModel:
    """base, link lengths, and two linear servo->angle maps."""

    def __init__(self, base=(0.0, 0.0), l1=30.0, l2=30.0, shoulder=(math.radians(-1.0), math.radians(180.0)), elbow=(math.radians(1.0), 0.0)):
        self.base = tuple(base)
        self.l1 = float(l1)
        self.l2 = float(l2)
        self.shoulder = tuple(shoulder)  # (a, b): world angle of upper arm = a*servo + b
        self.elbow = tuple(elbow)  # (a, b): forearm angle RELATIVE to upper arm

    # --- kinematics -----------------------------------------------------------
    def fk(self, servo_shoulder: float, servo_elbow: float) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """(elbow_xy, hand_xy) in gantry command units."""
        t1 = self.shoulder[0] * servo_shoulder + self.shoulder[1]
        ex = self.base[0] + self.l1 * math.cos(t1)
        ey = self.base[1] + self.l1 * math.sin(t1)
        t2 = t1 + self.elbow[0] * servo_elbow + self.elbow[1]
        hx = ex + self.l2 * math.cos(t2)
        hy = ey + self.l2 * math.sin(t2)
        return (ex, ey), (hx, hy)

    def separation(self, servo_shoulder: float, servo_elbow: float, gx: float, gy: float) -> float:
        """Distance from the gantry pen point to the arm (both links treated
        as segments, hand as a disc). Negative = interpenetrating."""
        elbow, hand = self.fk(servo_shoulder, servo_elbow)
        d = min(
            _seg_dist((gx, gy), self.base, elbow),
            _seg_dist((gx, gy), elbow, hand),
        )
        return min(d, _dist((gx, gy), hand) - HAND_RADIUS)

    # --- fitting --------------------------------------------------------------
    @classmethod
    def fit(cls, captures: List[dict]) -> Tuple["ArmModel", float]:
        """captures: [{"servo_shoulder", "servo_elbow", "base": (x,y),
        "elbow": (x,y), "hand": (x,y)}, ...] — one per dragged pose.
        Returns (model, mean residual of hand positions)."""
        if len(captures) < 2:
            raise ValueError("need at least 2 captured poses (3+ recommended)")
        base = (_mean([c["base"][0] for c in captures]), _mean([c["base"][1] for c in captures]))
        l1 = _mean([_dist(c["elbow"], base) for c in captures])
        l2 = _mean([_dist(c["hand"], c["elbow"]) for c in captures])
        t1s = [math.atan2(c["elbow"][1] - base[1], c["elbow"][0] - base[0]) for c in captures]
        sh = _linfit([c["servo_shoulder"] for c in captures], t1s)
        t2rels = [_wrap(math.atan2(c["hand"][1] - c["elbow"][1], c["hand"][0] - c["elbow"][0]) - t1) for c, t1 in zip(captures, t1s)]
        el = _linfit([c["servo_elbow"] for c in captures], t2rels)
        model = cls(base=base, l1=l1, l2=l2, shoulder=sh, elbow=el)
        resid = _mean([_dist(model.fk(c["servo_shoulder"], c["servo_elbow"])[1], c["hand"]) for c in captures])
        return model, resid

    # --- persistence ----------------------------------------------------------
    def save(self, path: str = MODEL_PATH, extra: Optional[Dict] = None):
        d = {"base": self.base, "l1": self.l1, "l2": self.l2, "shoulder": self.shoulder, "elbow": self.elbow}
        if extra:
            d.update(extra)
        with open(path, "w") as f:
            json.dump(d, f, indent=1)

    @classmethod
    def load(cls, path: str = MODEL_PATH) -> Optional["ArmModel"]:
        try:
            with open(path) as f:
                d = json.load(f)
            return cls(base=d["base"], l1=d["l1"], l2=d["l2"], shoulder=d["shoulder"], elbow=d["elbow"])
        except Exception:
            return None


class GantryArmModel:
    """The CNC-driven arm: hand = affine(command x,y); base + link
    lengths give the elbow via two-link IK for display and collision."""

    def __init__(self, matrix=((1.0, 0.0), (0.0, 1.0)), offset=(0.0, 0.0), base=(25.0, 92.0), l1=30.0, l2=30.0, elbow_sign=1.0):
        self.matrix = tuple(tuple(r) for r in matrix)
        self.offset = tuple(offset)
        self.base = tuple(base)
        self.l1 = float(l1)
        self.l2 = float(l2)
        self.elbow_sign = float(elbow_sign)

    def hand(self, cx: float, cy: float) -> Tuple[float, float]:
        m, t = self.matrix, self.offset
        return (m[0][0] * cx + m[0][1] * cy + t[0], m[1][0] * cx + m[1][1] * cy + t[1])

    def fk(self, cx: float, cy: float) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        h = self.hand(cx, cy)
        return _two_link_elbow(self.base, h, self.l1, self.l2, self.elbow_sign), h

    @classmethod
    def fit(cls, captures: List[dict]) -> Tuple["GantryArmModel", float]:
        """captures: [{"cmd": (x,y), "base": (x,y), "elbow": (x,y),
        "hand": (x,y)}, ...]. Affine needs 3+ non-collinear commands."""
        if len(captures) < 3:
            raise ValueError("need at least 3 captured gantry positions")
        import numpy as np

        A = np.array([[c["cmd"][0], c["cmd"][1], 1.0] for c in captures])
        hx = np.array([c["hand"][0] for c in captures])
        hy = np.array([c["hand"][1] for c in captures])
        sx, _res, rank, _sv = np.linalg.lstsq(A, hx, rcond=None)
        sy = np.linalg.lstsq(A, hy, rcond=None)[0]
        if rank < 3:
            raise ValueError("gantry captures are collinear — spread the positions out")
        base = (_mean([c["base"][0] for c in captures]), _mean([c["base"][1] for c in captures]))
        l1 = _mean([_dist(c["elbow"], base) for c in captures])
        l2 = _mean([_dist(c["hand"], c["elbow"]) for c in captures])
        crosses = []
        for c in captures:
            v1 = (c["hand"][0] - base[0], c["hand"][1] - base[1])
            v2 = (c["elbow"][0] - base[0], c["elbow"][1] - base[1])
            crosses.append(v1[0] * v2[1] - v1[1] * v2[0])
        sign = 1.0 if sum(1 for x in crosses if x > 0) >= len(crosses) / 2 else -1.0
        model = cls(matrix=((sx[0], sx[1]), (sy[0], sy[1])), offset=(sx[2], sy[2]), base=base, l1=l1, l2=l2, elbow_sign=sign)
        resid = _mean([_dist(model.hand(*c["cmd"]), c["hand"]) for c in captures])
        return model, resid


def _two_link_elbow(base, hand, l1, l2, sign) -> Tuple[float, float]:
    """Elbow position for a 2-link chain base→elbow→hand; if the hand is
    out of reach the chain stretches straight (display never breaks)."""
    d = max(_dist(base, hand), 1e-6)
    if d >= l1 + l2:
        f = l1 / (l1 + l2)
        return (base[0] + (hand[0] - base[0]) * (l1 + l2) * f / d, base[1] + (hand[1] - base[1]) * (l1 + l2) * f / d)
    a = (l1 * l1 - l2 * l2 + d * d) / (2 * d)
    h = math.sqrt(max(0.0, l1 * l1 - a * a))
    ux, uy = (hand[0] - base[0]) / d, (hand[1] - base[1]) / d
    return (base[0] + a * ux - sign * h * uy, base[1] + a * uy + sign * h * ux)


def arms_separation(left: ArmModel, right: GantryArmModel, servo_shoulder: float, servo_elbow: float, gx: float, gy: float) -> float:
    """Min distance between the two arms' link segments (hand discs
    subtracted). Negative = interpenetrating. Studio units."""
    le, lh = left.fk(servo_shoulder, servo_elbow)
    re_, rh = right.fk(gx, gy)
    lsegs = [(left.base, le), (le, lh)]
    rsegs = [(right.base, re_), (re_, rh)]
    d = min(_seg_seg_dist(a1, a2, b1, b2) for a1, a2 in lsegs for b1, b2 in rsegs)
    d = min(d, min(_seg_dist(lh, b1, b2) for b1, b2 in rsegs) - HAND_RADIUS)
    d = min(d, min(_seg_dist(rh, a1, a2) for a1, a2 in lsegs) - HAND_RADIUS)
    return min(d, _dist(lh, rh) - 2 * HAND_RADIUS)


def save_models(left: Optional[ArmModel], right: Optional["GantryArmModel"], path: str = MODEL_PATH, extra: Optional[Dict] = None):
    d = dict(extra or {})
    if left is not None:
        d["left"] = {"base": left.base, "l1": left.l1, "l2": left.l2, "shoulder": left.shoulder, "elbow": left.elbow}
    if right is not None:
        d["right"] = {
            "matrix": right.matrix,
            "offset": right.offset,
            "base": right.base,
            "l1": right.l1,
            "l2": right.l2,
            "elbow_sign": right.elbow_sign,
        }
    with open(path, "w") as f:
        json.dump(d, f, indent=1)


def load_models(path: str = MODEL_PATH) -> Tuple[Optional[ArmModel], Optional["GantryArmModel"]]:
    try:
        with open(path) as f:
            d = json.load(f)
    except Exception:
        return None, None
    left = right = None
    if "left" in d:
        L = d["left"]
        left = ArmModel(base=L["base"], l1=L["l1"], l2=L["l2"], shoulder=L["shoulder"], elbow=L["elbow"])
    if "right" in d:
        R = d["right"]
        right = GantryArmModel(matrix=R["matrix"], offset=R["offset"], base=R["base"], l1=R["l1"], l2=R["l2"], elbow_sign=R["elbow_sign"])
    return left, right


def _dist(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _seg_dist(p, a, b) -> float:
    ax, ay, bx, by = a[0], a[1], b[0], b[1]
    dx, dy = bx - ax, by - ay
    den = dx * dx + dy * dy
    if den < 1e-9:
        return _dist(p, a)
    t = max(0.0, min(1.0, ((p[0] - ax) * dx + (p[1] - ay) * dy) / den))
    return _dist(p, (ax + t * dx, ay + t * dy))


def _seg_seg_dist(a1, a2, b1, b2) -> float:
    if _segs_intersect(a1, a2, b1, b2):
        return 0.0
    return min(_seg_dist(a1, b1, b2), _seg_dist(a2, b1, b2), _seg_dist(b1, a1, a2), _seg_dist(b2, a1, a2))


def _segs_intersect(a1, a2, b1, b2) -> bool:
    def ccw(p, q, r):
        return (r[1] - p[1]) * (q[0] - p[0]) > (q[1] - p[1]) * (r[0] - p[0])

    return ccw(a1, b1, b2) != ccw(a2, b1, b2) and ccw(a1, a2, b1) != ccw(a1, a2, b2)
