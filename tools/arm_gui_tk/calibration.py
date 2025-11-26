import json
import math
import time
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np

Point = Tuple[float, float]


def _normalize_points(pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Hartley normalization: translate to mean 0, scale so mean dist = sqrt(2).

    Returns (normalized_pts, T) where T is the 3x3 normalization matrix.
    """
    if pts.shape[1] != 2:
        raise ValueError("Expected Nx2 points array")

    mean = pts.mean(axis=0)
    shifted = pts - mean
    dists = np.sqrt((shifted[:, 0] ** 2) + (shifted[:, 1] ** 2))
    mean_dist = np.mean(dists) if len(dists) > 0 else 1.0
    scale = math.sqrt(2) / mean_dist if mean_dist > 0 else 1.0

    T = np.array(
        [
            [scale, 0.0, -scale * mean[0]],
            [0.0, scale, -scale * mean[1]],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )

    # Convert to homogeneous, apply T, convert back
    ones = np.ones((pts.shape[0], 1))
    homo = np.hstack([pts, ones])
    norm = (T @ homo.T).T
    norm = norm[:, :2] / norm[:, 2:3]
    return norm, T


def compute_homography(src_pts: Sequence[Point], dst_pts: Sequence[Point]) -> np.ndarray:
    """Compute a 3x3 homography H such that dst ~ H * src using normalized DLT.

    - src_pts and dst_pts: sequences of 4 points (x, y), ordered consistently.
    - Returns H as a float64 3x3 matrix with H[2,2] normalized to 1.
    """
    if len(src_pts) != 4 or len(dst_pts) != 4:
        raise ValueError("Homography requires exactly 4 point pairs")

    src = np.asarray(src_pts, dtype=float)
    dst = np.asarray(dst_pts, dtype=float)

    src_n, T_src = _normalize_points(src)
    dst_n, T_dst = _normalize_points(dst)

    A = []
    for (x, y), (X, Y) in zip(src_n, dst_n):
        A.append([0, 0, 0, -x, -y, -1, Y * x, Y * y, Y])
        A.append([x, y, 1, 0, 0, 0, -X * x, -X * y, -X])
    A = np.asarray(A)

    # Solve Ah=0 via SVD
    U, S, Vt = np.linalg.svd(A)
    h = Vt[-1, :]
    Hn = h.reshape(3, 3)

    # Denormalize: H = T_dst^{-1} * Hn * T_src
    H = np.linalg.inv(T_dst) @ Hn @ T_src
    # Normalize so H[2,2] = 1
    if abs(H[2, 2]) > 1e-12:
        H = H / H[2, 2]
    return H


def invert_homography(H: np.ndarray) -> np.ndarray:
    """Invert homography, normalized so H[2,2]=1 if possible."""
    Hin = np.linalg.inv(H)
    if abs(Hin[2, 2]) > 1e-12:
        Hin = Hin / Hin[2, 2]
    return Hin


def apply_homography(H: np.ndarray, points: Sequence[Point]) -> List[Point]:
    """Apply 3x3 homography to a list of (x, y) points."""
    P = np.asarray(points, dtype=float)
    ones = np.ones((P.shape[0], 1))
    homo = np.hstack([P, ones])
    out = (H @ homo.T).T
    out_xy = out[:, :2] / out[:, 2:3]
    return [(float(x), float(y)) for x, y in out_xy]


def save_calibration(path: str | Path, *, H: np.ndarray, H_inv: np.ndarray, ideal_pts: Sequence[Point], measured_pts: Sequence[Point]) -> None:
    data = {
        "type": "homography",
        "timestamp": time.time(),
        "H": H.tolist(),
        "H_inv": H_inv.tolist(),
        "ideal_points": [list(map(float, p)) for p in ideal_pts],
        "measured_points": [list(map(float, p)) for p in measured_pts],
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_calibration(path: str | Path) -> dict:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    # Convert lists back to numpy
    data["H"] = np.asarray(data["H"], dtype=float)
    data["H_inv"] = np.asarray(data["H_inv"], dtype=float)
    return data
