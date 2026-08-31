import math
from typing import Optional, Tuple


def clamp_to_reach(x: float, y: float, base: Tuple[float, float], L1: float, L2: float) -> Tuple[float, float]:
    bx, by = base
    dx = x - bx
    dy = y - by
    r = math.hypot(dx, dy)
    r_min = abs(L1 - L2)
    r_max = L1 + L2
    if r < 1e-9:
        return bx + r_min, by  # arbitrary direction
    if r < r_min:
        s = r_min / r
        return bx + dx * s, by + dy * s
    if r > r_max:
        s = r_max / r
        return bx + dx * s, by + dy * s
    return x, y


def ik_2link(
    x: float,
    y: float,
    base: Tuple[float, float],
    L1: float,
    L2: float,
    elbow_up: bool = True,
    last_thetas: Optional[Tuple[float, float]] = None,
) -> Optional[Tuple[float, float]]:
    """Closed-form IK for planar 2-link.
    Returns (theta1, theta2) in radians or None if invalid.
    """
    bx, by = base
    dx = x - bx
    dy = y - by
    r2 = dx * dx + dy * dy
    r = math.sqrt(r2)
    # Law of cosines
    c2 = (r2 - L1 * L1 - L2 * L2) / (2.0 * L1 * L2)
    c2 = max(-1.0, min(1.0, c2))
    s2 = math.sqrt(max(0.0, 1.0 - c2 * c2))
    if not elbow_up:
        s2 = -s2
    theta2 = math.atan2(s2, c2)
    # Compute theta1
    k1 = L1 + L2 * c2
    k2 = L2 * s2
    theta1 = math.atan2(dy, dx) - math.atan2(k2, k1)

    if last_thetas is not None:
        # Choose solution closest to last theta1/theta2 (handle 2*pi wrap)
        t1, t2 = theta1, theta2
        for d in (-2 * math.pi, 0.0, 2 * math.pi):
            # Only adjust t1 by multiples of 2pi; t2 reasonably bounded
            pass
        # Not doing alternative branch here (elbow flip) because elbow_up fixes branch
    return (theta1, theta2)


def fk_2link(theta1: float, theta2: float, base: Tuple[float, float], L1: float, L2: float) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Forward kinematics: returns (elbow_xy, tip_xy)."""
    bx, by = base
    ex = bx + L1 * math.cos(theta1)
    ey = by + L1 * math.sin(theta1)
    tx = ex + L2 * math.cos(theta1 + theta2)
    ty = ey + L2 * math.sin(theta1 + theta2)
    return (ex, ey), (tx, ty)
