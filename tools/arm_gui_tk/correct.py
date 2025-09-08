from typing import Iterable, List, Sequence, Tuple

import numpy as np

from .calibration import apply_homography

Point = Tuple[float, float]


def prewarp_points(points: Sequence[Point], H_inv: np.ndarray) -> List[Point]:
    """Pre-warp ideal points using the inverse homography so that after the
    physical skew (modeled by H) they land where intended.
    """
    return apply_homography(H_inv, points)


def simulate_print(points_to_send: Sequence[Point], H: np.ndarray) -> List[Point]:
    """Simulate the printed result by applying the forward homography H
    (useful for preview/debug).
    """
    return apply_homography(H, points_to_send)
