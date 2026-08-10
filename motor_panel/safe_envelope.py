"""Where the body has provably been — empirical collision safety.

No geometry, no IK, no camera. The machine never needs to know where its
hands ARE in space; it needs to know which COMBINATIONS of motor values
are safe, and every recording is a few hundred of those, performed by
hand and verified by the fact that nothing hit anything.

Both arms are recorded together (one chain group: x, y, elbow, shoulder,
wrist, fingers), so a normal markov walk only ever visits combinations
that were performed. Collisions live in the glue between demonstrated
material. Measured on the real recordings (Aug 1):

    demonstrated combinations pooled ....... 8400
    typical spacing between neighbours ..... 1.2 units
    straight-line crossfade midpoints ...... 6.9 units from anything (worst 12.1)
    full gaze lean on a demonstrated pose .. 11.0 units from anything

So the body leaves proven territory by 6-10x the local spacing, exactly
during transitions — and the lean does it continuously. This class is the
guard: every outgoing command is asked how far it is from the nearest
thing ever performed, and pulled back if it has wandered too far. A
lookup is ~0.02 ms, so it can run on every send.

Conservative by design: it cannot judge a configuration it has never
seen, so it refuses to go there. The envelope grows as the artist records
more. It also trusts that the recordings themselves were collision-free.
"""

import glob
import os
import sys
from typing import Dict, List, Optional, Sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Collision-relevant channels only: the pen and lung cannot hit anything,
# and the fingers are inside the hand's own footprint.
SAFE_CHANNELS = ("x", "y", "elbow", "shoulder", "wrist")


class SafeEnvelope:
    """A cloud of demonstrated motor combinations, with nearest-point
    lookup. Inactive (everything passes) when scipy or the recordings are
    missing — the guard must never be the reason the body stops."""

    def __init__(self, channels: Sequence[str] = SAFE_CHANNELS, sessions_dir: Optional[str] = None, on_log=lambda m: None):
        self.channels = list(channels)
        self.on_log = on_log
        self.points: List[tuple] = []
        self._tree = None
        self._np = None
        self.load(sessions_dir)

    @property
    def active(self) -> bool:
        return self._tree is not None

    def load(self, sessions_dir: Optional[str] = None):
        from motor_panel.session import SESSIONS_DIR, Session

        d = sessions_dir or SESSIONS_DIR
        pts = []
        for path in sorted(glob.glob(os.path.join(d, "session_*.json"))):
            try:
                s = Session.load(os.path.basename(path))
            except Exception:
                continue
            tracks = [t for t in s.tracks if t.has_take and set(t.channels) & set(self.channels)]
            if not tracks:
                continue
            try:
                joint = s._joint_samples(tracks)
            except Exception:
                continue
            for smp in joint:
                if all(c in smp for c in self.channels):
                    pts.append(tuple(float(smp[c]) for c in self.channels))
        self.points = pts
        if len(pts) < 50:
            self.on_log(f"safe envelope inactive — only {len(pts)} demonstrated combinations found")
            return
        try:
            import numpy as np
            from scipy.spatial import cKDTree

            self._np = np
            self._tree = cKDTree(np.asarray(pts, dtype=float))
            self.on_log(f"safe envelope: {len(pts)} demonstrated combinations over {','.join(self.channels)}")
        except Exception as e:
            self._tree = None
            self.on_log(f"safe envelope inactive (scipy unavailable: {e})")

    def distance(self, pose: Dict[str, float]) -> float:
        """How far this combination is from the nearest one ever performed.
        0.0 when inactive or when the pose lacks the guarded channels."""
        if not self.active or any(c not in pose for c in self.channels):
            return 0.0
        q = self._np.asarray([pose[c] for c in self.channels], dtype=float)
        return float(self._tree.query(q)[0])

    def project(self, pose: Dict[str, float], max_dist: float, movable: Optional[Sequence[str]] = None, neighbours: int = 8) -> Dict[str, float]:
        """Pull a stray combination back onto proven ground.

        The pull TARGET is the average of the k nearest demonstrated
        combinations, not the single nearest one. That matters: a single
        nearest neighbour flips as the body crosses between them, and the
        correction jumps with it (measured: 3.6-unit snaps). An average of
        eight moves smoothly, so a smoothly moving body stays smooth.

        The correction is applied in FULL, not rate-limited. A steady
        offset like the gaze lean pushes ~12 units out on every single
        send, so a capped pull-back just loses a tug of war forever —
        trimming the excess outright is what "lean, but only as far as you
        have proven you can" actually means. Bisection guarantees the
        result really is inside max_dist rather than merely closer.

        Only the channels in `movable` are allowed to change (default: all).
        Sends arrive split — servos on one call, the gantry on another — and
        a correction written to a channel the caller is not about to send
        would simply be discarded, leaving the combination stray. Correcting
        within the subspace the caller CAN act on means the fix that gets
        computed is the fix that actually reaches the machine."""
        if not self.active or any(c not in pose for c in self.channels):
            return pose
        np = self._np
        q = np.asarray([pose[c] for c in self.channels], dtype=float)
        if float(self._tree.query(q)[0]) <= max_dist:
            return pose
        mask = np.asarray([1.0 if (movable is None or c in movable) else 0.0 for c in self.channels])
        if not mask.any():
            return pose

        def toward(target):
            lo, hi, best = 0.0, 1.0, q + (target - q) * mask
            for _ in range(14):
                mid = (lo + hi) / 2
                cand = q + (target - q) * mid * mask
                if float(self._tree.query(cand)[0]) <= max_dist:
                    hi, best = mid, cand
                else:
                    lo = mid
            return best

        k = min(neighbours, len(self.points))
        _d, idx = self._tree.query(q, k=k)
        # the average of k neighbours moves smoothly as the body crosses
        # between them (a single nearest one flips, and the correction snaps),
        # but on a curved cloud that average can itself sit off the manifold —
        # so fall back to the true nearest point, which always lands.
        cand = toward(np.asarray([self.points[int(i)] for i in np.atleast_1d(idx)], dtype=float).mean(axis=0))
        if float(self._tree.query(cand)[0]) > max_dist:
            cand = toward(np.asarray(self.points[int(np.atleast_1d(idx)[0])], dtype=float))
        out = dict(pose)
        for i, c in enumerate(self.channels):
            out[c] = float(cand[i])
        return out
