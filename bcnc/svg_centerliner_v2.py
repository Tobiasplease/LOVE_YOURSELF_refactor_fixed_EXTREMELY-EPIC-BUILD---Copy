"""Centerline tracer v2 — skeleton GRAPH walking instead of contour tracing.

Why v1 had to go (measured July 21 2026 on live output):
  - v1 ran cv2.findContours AROUND the 1px skeleton: every stroke became a
    closed loop traced out-and-back. 32/32 paths closed on a sample piece —
    every line in every drawing was drawn TWICE, and the return pass's
    ±1px disagreement is part of the machine's "wobble".
  - Global threshold (180) drops faint strokes and bloats dark regions.
  - No junction handling, no spur pruning: fused hatching skeletonizes
    into sawtooth zigzags.

v2: adaptive threshold -> skeleton -> graph (endpoints/junctions as nodes,
degree-2 chains as edges) -> walk each edge ONCE -> prune spurs shorter
than the local stroke width -> simplify+smooth -> polyline SVG with the
same conventions as v1 (px coordinates, svgwrite polylines) so downstream
vpype/gcode tooling needs no changes.

STANDALONE: nothing in the runtime imports this until it is proven on
paper. A/B against v1 with debug/test_centerliner_v2.py.
"""

import cv2
import numpy as np
import svgwrite
from skimage.morphology import skeletonize

NEIGH = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))


def binarize(img: np.ndarray, contrast_alpha: float = 2.0, adaptive: bool = True,
             threshold_value: int = 180, min_component: int = 12) -> np.ndarray:
    """Ink mask. Union of adaptive and global thresholds: adaptive keeps
    faint fine strokes that v1's global 180 erased; global keeps broad soft
    strokes that look locally uniform and fool the adaptive pass."""
    f = img.astype(np.float32) / 255.0
    pivot = float(f.mean())  # torch adjust_contrast pivots at the mean, and so did v1
    f = np.clip((f - pivot) * contrast_alpha + pivot, 0, 1)
    img8 = (f * 255).astype(np.uint8)
    if (img8 < 128).mean() > 0.7:  # mostly-dark output: invert (v1 behavior kept)
        img8 = 255 - img8
    b = img8 < threshold_value
    if adaptive:
        b |= cv2.adaptiveThreshold(img8, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 51, 12) > 0
    n, lab, stats, _ = cv2.connectedComponentsWithStats(b.astype(np.uint8))
    keep = np.zeros_like(b)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_component:
            keep |= lab == i
    return keep


def skeleton_paths(skel: np.ndarray) -> list:
    """The heart of v2: extract the skeleton's graph and walk every edge
    exactly once. Nodes = pixels of degree != 2 (endpoints, junctions);
    edges = the degree-2 chains between them; isolated cycles walked too."""
    ys, xs = np.nonzero(skel)
    pix = set(zip(ys.tolist(), xs.tolist()))

    def neighbors(p):
        return [(p[0] + dy, p[1] + dx) for dy, dx in NEIGH if (p[0] + dy, p[1] + dx) in pix]

    deg = {p: len(neighbors(p)) for p in pix}
    nodes = {p for p in pix if deg[p] != 2}
    used = set()

    def key(a, b):
        return (a, b) if a <= b else (b, a)

    paths = []
    for n0 in nodes:
        for nb in neighbors(n0):
            if key(n0, nb) in used:
                continue
            path = [n0, nb]
            used.add(key(n0, nb))
            prev, cur = n0, nb
            while cur not in nodes:
                nxt = [q for q in neighbors(cur) if q != prev and key(cur, q) not in used]
                if not nxt:
                    break
                q = nxt[0]
                used.add(key(cur, q))
                path.append(q)
                prev, cur = cur, q
            paths.append(path)
    # isolated cycles (rings with no junction anywhere)
    for p in pix:
        if p in nodes:
            continue
        for nb in neighbors(p):
            if key(p, nb) in used:
                continue
            path = [p, nb]
            used.add(key(p, nb))
            prev, cur = p, nb
            while cur != p:
                nxt = [q for q in neighbors(cur) if q != prev and key(cur, q) not in used]
                if not nxt:
                    break
                q = nxt[0]
                used.add(key(cur, q))
                path.append(q)
                prev, cur = cur, q
            paths.append(path)
    return paths, nodes, deg


def prune_spurs(paths: list, nodes: set, deg: dict, width_map: np.ndarray,
                spur_factor: float = 1.6, spur_min: float = 6.0) -> list:
    """Drop stub edges (one free end) shorter than the local stroke width —
    they're skeletonization noise (the sawteeth), not drawn marks."""
    out = []
    for path in paths:
        a, b = path[0], path[-1]
        is_spur = (deg.get(a, 0) == 1) != (deg.get(b, 0) == 1)  # exactly one free end
        if is_spur:
            free = a if deg.get(a, 0) == 1 else b
            local_w = float(width_map[free[0], free[1]]) * 2.0
            if len(path) < max(spur_min, spur_factor * local_w):
                continue
        out.append(path)
    return out


def _chaikin(pts: np.ndarray, iterations: int = 1) -> np.ndarray:
    for _ in range(iterations):
        if len(pts) < 3:
            return pts
        q = pts[:-1] * 0.75 + pts[1:] * 0.25
        r = pts[:-1] * 0.25 + pts[1:] * 0.75
        mid = np.empty((len(q) + len(r), 2))
        mid[0::2] = q
        mid[1::2] = r
        pts = np.vstack([pts[:1], mid, pts[-1:]])
    return pts


def simplify(path: list, epsilon: float = 1.2, smooth_iterations: int = 1) -> np.ndarray:
    """Douglas-Peucker (kills pixel-stair vertices) then Chaikin (rounds
    the survivors) — pen-friendly curves instead of raster staircases."""
    pts = np.array([(x, y) for y, x in path], dtype=np.float32)
    if len(pts) > 2:
        approx = cv2.approxPolyDP(pts.reshape(-1, 1, 2), epsilon, False)
        pts = approx.reshape(-1, 2).astype(np.float64)
    return _chaikin(pts.astype(np.float64), smooth_iterations)


def raster_to_centerline_svg(input_path: str, output_path: str,
                             contrast_alpha: float = 2.0,
                             adaptive: bool = True,
                             threshold_value: int = 180,
                             min_component: int = 12,
                             spur_factor: float = 1.6,
                             simplify_epsilon: float = 1.2,
                             smooth_iterations: int = 1,
                             min_path_px: int = 5,
                             scale: float = 1.0,
                             save_steps: bool = False):
    """v1-compatible entry point: PNG in, polyline-SVG out."""
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to load image from {input_path}")

    binary = binarize(img, contrast_alpha, adaptive, threshold_value, min_component)
    if save_steps:
        base = output_path.rsplit(".", 1)[0]
        cv2.imwrite(f"{base}_v2_step1_binary.png", binary.astype(np.uint8) * 255)

    skel = skeletonize(binary)
    if save_steps:
        cv2.imwrite(f"{base}_v2_step2_skeleton.png", skel.astype(np.uint8) * 255)

    # local stroke half-width for width-aware pruning
    width_map = cv2.distanceTransform(binary.astype(np.uint8), cv2.DIST_L2, 3)

    paths, nodes, deg = skeleton_paths(skel)
    paths = prune_spurs(paths, nodes, deg, width_map, spur_factor)
    polylines = [simplify(p, simplify_epsilon, smooth_iterations)
                 for p in paths if len(p) >= min_path_px]

    h, w = skel.shape
    dwg = svgwrite.Drawing(output_path, size=(f"{w * scale}px", f"{h * scale}px"))
    for pl in polylines:
        pts = [(float(x) * scale, float(y) * scale) for x, y in pl]
        if len(pts) > 1:
            dwg.add(dwg.polyline(points=pts, stroke="black", fill="none", stroke_width=1))
    dwg.save()
    print(f"[v2] {len(polylines)} single-pass strokes -> {output_path}")
    return len(polylines)


if __name__ == "__main__":
    import sys
    inp = sys.argv[1] if len(sys.argv) > 1 else "input.png"
    raster_to_centerline_svg(inp, inp.rsplit(".", 1)[0] + "_center_lined_v2.svg", save_steps=True)
