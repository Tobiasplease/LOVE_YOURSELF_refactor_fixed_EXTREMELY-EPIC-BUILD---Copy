"""Measured warp calibration — the model-free successor to map_to_quad.

The arm's distortion is a smooth, position-dependent field (2-link ~300mm
kinematics behind fictional gear constants — see WARP_TRANSFORM_README).
A 4-corner bilinear quad can pin corners but cannot bend edges or fix the
interior, which is why a decade of PRE_ROTATION / PRE_SCALE / NUDGE band-aids
never produced a good square.

This module skips modeling entirely and measures the field:

  1. generate_calibration_gcode(): the machine dots an n×n grid of KNOWN
     command coordinates onto paper (sent raw — no transforms).
  2. The operator photographs the sheet; debug/warp_calibrate.py turns
     clicks into paper-mm positions for each dot.
  3. fit(): a thin-plate spline is fitted INVERSE (paper -> command):
     "to put ink here, send this." Tilt, scale, offset and curvature are
     all just part of the fitted field — no tuning constants remain.

warp_transform.warp_transform_line() uses this automatically whenever
grbl/warp_calibration.json exists; delete the file to fall back to the
legacy quad transform.
"""

import json
import math
import os
import re
from typing import List, Optional, Tuple

import numpy as np

CALIBRATION_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "warp_calibration.json")
SURVEY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "warp_survey.json")

# Command-space domain the dots are laid over (raw GRBL coords the machine
# accepts today — same box the current pipeline commands into).
DEFAULT_DOMAIN = (0.0, 70.0, 0.0, 40.0)  # x0, x1, y0, y1

# Reachable-envelope boundary, measured by hand-guided exploration July 20
# 2026 (pen-position command coords; soft limits are OFF on this fork, so
# these are physical truth, ±a few units). Convexity verified: straight
# edges between these points lie inside true reach. Order matters (polygon).
MEASURED_BOUNDARY = [
    (0.0, 0.0),      # home corner (bottom-left of reach)
    (0.0, 47.0),     # upper-left: shoulder at travel limit
    (13.4, 50.8),    # top apex: full arm aligned with +Y
    (66.0, 20.9),    # upper-right: table edge
    (110.2, -13.9),  # bottom-right: forearm mech limit, pen at table edge
    (44.9, -7.0),    # bottom-center: forearm mech stop
]


def _point_in_polygon(x: float, y: float, poly: List[Tuple[float, float]]) -> bool:
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            xin = x1 + (y - y1) / (y2 - y1) * (x2 - x1)
            if x < xin:
                inside = not inside
    return inside


def _inset_polygon(poly: List[Tuple[float, float]], factor: float = 0.92) -> List[Tuple[float, float]]:
    cx = sum(p[0] for p in poly) / len(poly)
    cy = sum(p[1] for p in poly) / len(poly)
    return [(cx + (x - cx) * factor, cy + (y - cy) * factor) for x, y in poly]


def polygon_grid(boundary: List[Tuple[float, float]] = None, spacing: float = 10.0,
                 inset: float = 0.92) -> List[Tuple[float, float]]:
    """Survey dots: a spacing-mm lattice clipped to the inset reach polygon,
    serpentine-ordered row by row for sane clicking."""
    poly = _inset_polygon(boundary or MEASURED_BOUNDARY, inset)
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    rows = []
    y = min(ys)
    row_i = 0
    while y <= max(ys) + 1e-9:
        row = []
        x = min(xs)
        while x <= max(xs) + 1e-9:
            if _point_in_polygon(x, y, poly):
                row.append((round(x, 2), round(y, 2)))
            x += spacing
        if row:
            rows.append(row if row_i % 2 == 0 else row[::-1])
            row_i += 1
        y += spacing
    return [p for row in rows for p in row]


def save_survey(points: List[Tuple[float, float]], meta: dict = None) -> str:
    with open(SURVEY_PATH, "w") as f:
        json.dump({"points": points, **(meta or {})}, f, indent=1)
    return SURVEY_PATH


def load_survey() -> Optional[List[Tuple[float, float]]]:
    try:
        with open(SURVEY_PATH) as f:
            return [tuple(p) for p in json.load(f)["points"]]
    except Exception:
        return None


def grid_points(n: int = 5, domain: Tuple[float, float, float, float] = DEFAULT_DOMAIN) -> List[Tuple[float, float]]:
    """n×n command-space grid in serpentine order (matches the click tool)."""
    x0, x1, y0, y1 = domain
    pts = []
    for iy in range(n):
        y = y0 + (y1 - y0) * iy / (n - 1)
        xs = range(n) if iy % 2 == 0 else range(n - 1, -1, -1)
        for ix in xs:
            pts.append((x0 + (x1 - x0) * ix / (n - 1), y))
    return pts


def generate_calibration_gcode(n: int = 5, domain=DEFAULT_DOMAIN,
                               pen_up: int = 34, pen_down: int = 52,
                               feed: int = 1500, points: Optional[List[Tuple[float, float]]] = None) -> List[str]:
    """Dot the grid + an orientation dash next to dot #1. Send these lines
    RAW (no warp transform) — they define command space itself."""
    pts = points if points is not None else grid_points(n, domain)
    lines = ["G21", "G90", f"M3 S{pen_up}", "G4 P0.3"]
    for i, (x, y) in enumerate(pts):
        lines.append(f"G0 X{x:.2f} Y{y:.2f}")
        lines.append(f"M3 S{pen_down}")
        lines.append("G4 P0.25")
        lines.append(f"M3 S{pen_up}")
        lines.append("G4 P0.2")
        if i == 0:  # orientation dash: identifies dot #1 and grid direction
            lines.append(f"G0 X{x + 2:.2f} Y{y:.2f}")
            lines.append(f"M3 S{pen_down}")
            lines.append("G4 P0.2")
            lines.append(f"G1 X{x + 5:.2f} Y{y:.2f} F{feed}")
            lines.append(f"M3 S{pen_up}")
            lines.append("G4 P0.2")
    lines.append("G0 X0 Y0")
    return lines


def best_rect_rotated(paper: np.ndarray, aspect: float,
                      angles=range(-90, 91, 2)) -> Tuple[float, float, float, float, float]:
    """Largest aspect-true rectangle inscribed in the measured footprint,
    searching position, size AND rotation. The reach envelope is a tilted
    band — an axis-aligned window wastes most of it (the July 20 '54% of A4'
    underestimate); aligning the window to the band is free, because the
    physical sheet is taped at the window's angle. Returns (cx, cy, w, h,
    angle_deg) in survey-sheet mm."""
    best = None
    for ang in angles:
        t = math.radians(ang)
        c, s = math.cos(t), math.sin(t)
        rot = np.array([[c, s], [-s, c]])  # rotate points INTO window frame
        pts = paper @ rot.T
        (rect, h) = best_rect(pts, aspect)
        if best is None or h > best[0]:
            x0, y0, x1, y1 = rect
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            # window center back in sheet frame
            back = np.array([[c, -s], [s, c]]) @ np.array([cx, cy])
            best = (h, back[0], back[1], h * aspect, ang)
    h, cx, cy, w, ang = best
    return cx, cy, w, h, ang


def window_corners(cx: float, cy: float, w: float, h: float, angle_deg: float):
    """The four corners of a rotated window, sheet mm, BL/BR/TR/TL order."""
    t = math.radians(angle_deg)
    c, s = math.cos(t), math.sin(t)
    out = []
    for dx, dy in ((-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)):
        out.append((cx + dx * c - dy * s, cy + dx * s + dy * c))
    return out


def best_rect(paper: np.ndarray, aspect: float) -> Tuple[Tuple[float, float, float, float], float]:
    """Largest rectangle of the given aspect (w/h) inscribed in the measured
    footprint, searching both position and size. Returns (rect, achieved_h).

    This is how survey findings become paper placement: calibrate on a big
    survey sheet, then ask for the largest A4-aspect window the measured
    region can honestly serve — and tape the real sheet exactly there."""
    from scipy.spatial import Delaunay
    tri = Delaunay(paper)
    x0, y0 = paper.min(axis=0)
    x1, y1 = paper.max(axis=0)

    def fits(cx, cy, h):
        hw, hh = h * aspect / 2, h / 2
        xs = np.linspace(cx - hw, cx + hw, 5)
        ys = np.linspace(cy - hh, cy + hh, 5)
        border = [(x, ys[0]) for x in xs] + [(x, ys[-1]) for x in xs] \
               + [(xs[0], y) for y in ys] + [(xs[-1], y) for y in ys]
        return bool((tri.find_simplex(np.array(border)) >= 0).all())

    best = None
    for cx in np.linspace(x0, x1, 15):
        for cy in np.linspace(y0, y1, 15):
            if tri.find_simplex(np.array([[cx, cy]]))[0] < 0:
                continue
            lo, hi = 0.0, max(x1 - x0, y1 - y0) * 1.5
            for _ in range(22):
                mid = (lo + hi) / 2
                if fits(cx, cy, mid):
                    lo = mid
                else:
                    hi = mid
            if best is None or lo > best[0]:
                best = (lo, cx, cy)
    h, cx, cy = best[0] * 0.96, best[1], best[2]  # 4% safety inset
    w = h * aspect
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2), h


def _inscribed_rect(paper: np.ndarray) -> Tuple[float, float, float, float]:
    """Largest-ish axis-aligned rectangle INSIDE the measured footprint.

    The TPS is an interpolator: queries must stay inside the convex hull of
    the measured dots, or it extrapolates garbage. The footprint of an arm
    is curved (banana-shaped), so its bounding box is mostly OUTSIDE the
    hull — using it as the drawing area was the first bug this module had.
    Binary-search a centered rectangle (bbox aspect) against hull
    containment, then inset 5% for safety."""
    from scipy.spatial import Delaunay
    tri = Delaunay(paper)
    cx, cy = paper.mean(axis=0)
    hw = (paper[:, 0].max() - paper[:, 0].min()) / 2
    hh = (paper[:, 1].max() - paper[:, 1].min()) / 2

    def inside(s: float) -> bool:
        xs = np.linspace(cx - hw * s, cx + hw * s, 5)
        ys = np.linspace(cy - hh * s, cy + hh * s, 5)
        border = [(x, ys[0]) for x in xs] + [(x, ys[-1]) for x in xs] \
               + [(xs[0], y) for y in ys] + [(xs[-1], y) for y in ys]
        return bool((tri.find_simplex(np.array(border)) >= 0).all())

    lo, hi = 0.0, 1.0
    for _ in range(24):
        mid = (lo + hi) / 2
        if inside(mid):
            lo = mid
        else:
            hi = mid
    s = lo * 0.95
    return (cx - hw * s, cy - hh * s, cx + hw * s, cy + hh * s)


class WarpCalibration:
    """Thin-plate-spline inverse map: paper mm -> command coords."""

    def __init__(self, command_pts: List[Tuple[float, float]],
                 paper_pts: List[Tuple[float, float]],
                 paper_area: Tuple[float, float, float, float],
                 paper_window: Optional[Tuple[float, float, float, float, float]] = None):
        from scipy.interpolate import RBFInterpolator
        self.command_pts = np.asarray(command_pts, dtype=float)
        self.paper_pts = np.asarray(paper_pts, dtype=float)
        self.paper_area = paper_area  # legacy axis-aligned fallback (x0, y0, x1, y1)
        self.paper_window = paper_window  # rotated drawing window (cx, cy, w, h, angle_deg)
        self._rbf = RBFInterpolator(self.paper_pts, self.command_pts,
                                    kernel="thin_plate_spline", smoothing=1e-3)

    # --- fitting ---------------------------------------------------------------
    @classmethod
    def fit(cls, command_pts, paper_pts, paper_area=None) -> "WarpCalibration":
        paper = np.asarray(paper_pts, dtype=float)
        if paper_area is None:
            paper_area = _inscribed_rect(paper)
        return cls(command_pts, paper_pts, paper_area)

    def residuals_mm(self) -> Tuple[float, float]:
        """Round-trip check in command space: rms, max (should be ~0.1mm)."""
        back = self._rbf(self.paper_pts)
        err = np.linalg.norm(back - self.command_pts, axis=1)
        return float(np.sqrt((err ** 2).mean())), float(err.max())

    # --- application -----------------------------------------------------------
    def paper_target(self, u: float, v: float, aspect: float = 1.0) -> Tuple[float, float]:
        """[0,1]² -> paper mm, aspect-true. Uses the rotated paper_window when
        set (the band-aligned A4 window); otherwise the largest aspect-true
        rectangle centered in the legacy axis-aligned paper_area."""
        if self.paper_window is not None:
            cx, cy, W, H, ang = self.paper_window
        else:
            x0, y0, x1, y1 = self.paper_area
            W, H = x1 - x0, y1 - y0
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            ang = 0.0
        if W / H > aspect:
            w, h = H * aspect, H
        else:
            w, h = W, W / aspect
        t = math.radians(ang)
        c, s = math.cos(t), math.sin(t)
        dx, dy = (u - 0.5) * w, (v - 0.5) * h
        return cx + dx * c - dy * s, cy + dx * s + dy * c

    def to_command(self, paper_x: float, paper_y: float) -> Tuple[float, float]:
        out = self._rbf(np.array([[paper_x, paper_y]]))[0]
        return float(out[0]), float(out[1])

    def apply_to_line(self, gcode_line: str, max_x: float, max_y: float) -> str:
        """Drop-in for warp_transform_line: ideal gcode -> command gcode."""
        xm = re.search(r"X([-+]?\d*\.?\d+)", gcode_line, re.IGNORECASE)
        ym = re.search(r"Y([-+]?\d*\.?\d+)", gcode_line, re.IGNORECASE)
        if not (xm and ym):
            return gcode_line
        u = float(xm.group(1)) / max(1e-9, max_x)
        v = float(ym.group(1)) / max(1e-9, max_y)
        px, py = self.paper_target(min(1, max(0, u)), min(1, max(0, v)), aspect=max_x / max(1e-9, max_y))
        cx, cy = self.to_command(px, py)
        line = re.sub(r"X[-+]?\d*\.?\d+", f"X{cx:.3f}", gcode_line, count=1, flags=re.IGNORECASE)
        line = re.sub(r"Y[-+]?\d*\.?\d+", f"Y{cy:.3f}", line, count=1, flags=re.IGNORECASE)
        return line

    # --- persistence -------------------------------------------------------------
    def save(self, path: str = CALIBRATION_PATH) -> str:
        with open(path, "w") as f:
            json.dump({
                "format": "warp_calibration_tps_v1",
                "command_pts": self.command_pts.tolist(),
                "paper_pts": self.paper_pts.tolist(),
                "paper_area": list(self.paper_area),
                "paper_window": list(self.paper_window) if self.paper_window else None,
            }, f, indent=1)
        return path

    @classmethod
    def load(cls, path: str = CALIBRATION_PATH) -> Optional["WarpCalibration"]:
        try:
            with open(path) as f:
                d = json.load(f)
            pw = d.get("paper_window")
            return cls(d["command_pts"], d["paper_pts"], tuple(d["paper_area"]),
                       tuple(pw) if pw else None)
        except Exception:
            return None
