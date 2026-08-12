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


def _end_direction(path: list, end: int, span: int = 6) -> np.ndarray:
    """Unit vector pointing INTO the path from the given end (0=start, 1=end)."""
    if end == 0:
        a, b = np.array(path[0], float), np.array(path[min(span, len(path) - 1)], float)
    else:
        a, b = np.array(path[-1], float), np.array(path[max(-span - 1, -len(path))], float)
    v = b - a
    n = np.linalg.norm(v)
    return v / n if n else v


def merge_through_junctions(paths: list, angle_thresh_deg: float = 35.0) -> list:
    """Rejoin strokes that a junction chopped up: where two edges meet at a
    node and continue in nearly the same direction (a hatch line crossing
    another), stitch them into one polyline. Cross-hatching keeps its long
    directional strokes instead of becoming thousands of stubs = pen lifts."""
    ends = {}
    for i, p in enumerate(paths):
        if p[0] == p[-1]:
            continue  # closed loops stay as they are
        ends.setdefault(p[0], []).append((i, 0))
        ends.setdefault(p[-1], []).append((i, 1))

    cos_limit = -np.cos(np.radians(angle_thresh_deg))  # directions must be near-opposite
    partner = {}
    for node, incident in ends.items():
        if len(incident) < 2:
            continue
        dirs = {(i, e): _end_direction(paths[i], e) for i, e in incident}
        cands = []
        for a in range(len(incident)):
            for b in range(a + 1, len(incident)):
                ia, ib = incident[a], incident[b]
                if ia[0] == ib[0]:
                    continue
                c = float(np.dot(dirs[ia], dirs[ib]))
                if c < cos_limit:  # near-antiparallel: straight continuation
                    cands.append((c, ia, ib))
        cands.sort()
        for _, ia, ib in cands:
            if ia in partner or ib in partner:
                continue
            partner[ia] = ib
            partner[ib] = ia

    def oriented(i, start_end):
        return paths[i][:] if start_end == 0 else paths[i][::-1]

    merged, consumed = [], set()
    for i, p in enumerate(paths):
        if i in consumed:
            continue
        if p[0] == p[-1]:
            merged.append(p)
            consumed.add(i)
            continue
        # walk backwards to the chain's true start
        cur, end = i, 0
        seen = {i}
        while (cur, end) in partner:
            nxt, nend = partner[(cur, end)]
            if nxt in seen:
                break  # cycle: start anywhere
            seen.add(nxt)
            cur, end = nxt, 1 - nend
        # walk forward stitching
        chain = oriented(cur, end)
        consumed.add(cur)
        pos = (cur, 1 - end)
        while pos in partner:
            nxt, nend = partner[pos]
            if nxt in consumed:
                break
            chain += oriented(nxt, nend)[1:]
            consumed.add(nxt)
            pos = (nxt, 1 - nend)
        merged.append(chain)
    return merged


def split_fills(binary: np.ndarray, width_map: np.ndarray, fill_factor: float = 2.2):
    """Separate thin strokes from solid blobs. Skeletonizing a filled region
    yields honeycomb mush (the mesh between white holes), so blobs get
    outline+hatch treatment instead. Returns (stroke_mask, fill_mask, w_half)."""
    skel = skeletonize(binary)
    on_skel = width_map[skel]
    if on_skel.size == 0:
        return binary, np.zeros_like(binary), 1.0
    w_half = float(np.median(on_skel))
    core = width_map > fill_factor * w_half
    if not core.any():
        return binary, np.zeros_like(binary), w_half
    r = max(2, int(round(2 * w_half)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    fill = (cv2.dilate(core.astype(np.uint8), kernel) > 0) & binary
    # a fat spot on a stroke is not a fill region — keep only sizeable masses
    min_area = (5 * w_half) ** 2
    n, lab, stats, _ = cv2.connectedComponentsWithStats(fill.astype(np.uint8))
    keep = np.zeros_like(fill)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            keep |= lab == i
    fill = keep
    if fill.sum() < 0.01 * binary.sum():
        return binary, np.zeros_like(binary), w_half
    return binary & ~fill, fill, w_half


def hatch_fills(fill_mask: np.ndarray, w_half: float, angle_deg: float = 45.0,
                spacing_factor: float = 3.0) -> list:
    """Fill regions the way a pen does: contour outlines plus serpentine
    hatch lines. Consecutive hatch rows are linked into zigzags where they
    overlap, so a dark mass costs a handful of pen lifts, not hundreds."""
    polylines = []
    mask8 = fill_mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask8, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    for c in contours:
        if cv2.arcLength(c, True) < 6 * w_half:
            continue
        approx = cv2.approxPolyDP(c, 1.2, True).reshape(-1, 2).astype(np.float64)
        if len(approx) >= 3:
            polylines.append(np.vstack([approx, approx[:1]]))
    polylines += _serpentine(fill_mask, max(4, int(round(spacing_factor * w_half))), angle_deg)
    return polylines


def _serpentine(fill_mask: np.ndarray, spacing: int, angle_deg: float) -> list:
    """The serpentine hatch core, shared by the legacy uniform fill and the
    tone-aware fill: rows at `spacing` along `angle_deg`, linked into zigzags."""
    polylines = []
    mask8 = fill_mask.astype(np.uint8)
    spacing = max(3, int(spacing))
    h, w = fill_mask.shape
    diag = int(np.ceil(np.hypot(h, w)))
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle_deg, 1.0)
    M[0, 2] += (diag - w) / 2.0
    M[1, 2] += (diag - h) / 2.0
    rot = cv2.warpAffine(mask8 * 255, M, (diag, diag), flags=cv2.INTER_NEAREST)
    Minv = cv2.invertAffineTransform(M)

    def unrotate(x, y):
        return (Minv[0, 0] * x + Minv[0, 1] * y + Minv[0, 2],
                Minv[1, 0] * x + Minv[1, 1] * y + Minv[1, 2])

    chains, open_chains = [], []  # open: (last_x0, last_x1, points)
    for y in range(spacing // 2, diag, spacing):
        row = rot[y] > 127
        idx = np.flatnonzero(row)
        runs = []
        if idx.size >= 2:
            splits = np.flatnonzero(np.diff(idx) > 1)
            runs = [(int(r[0]), int(r[-1])) for r in np.split(idx, splits + 1) if r.size >= max(2, spacing // 2)]
        next_open = []
        used = set()
        for x0, x1 in runs:
            best = None
            for k, (px0, px1, pts) in enumerate(open_chains):
                if k in used or min(x1, px1) - max(x0, px0) <= 0:
                    continue
                if best is None or abs((x0 + x1) - (open_chains[best][0] + open_chains[best][1])) > abs((x0 + x1) - (px0 + px1)):
                    best = k
            if best is not None:
                used.add(best)
                pts = open_chains[best][2]
                near_x1 = abs(pts[-1][0] - x1) < abs(pts[-1][0] - x0)
                pts.append((x1, y) if near_x1 else (x0, y))
                pts.append((x0, y) if near_x1 else (x1, y))
                next_open.append((x0, x1, pts))
            else:
                next_open.append((x0, x1, [(x0, y), (x1, y)]))
        for k, ch in enumerate(open_chains):
            if k not in used:
                chains.append(ch[2])
        open_chains = next_open
    chains.extend(ch[2] for ch in open_chains)

    for pts in chains:
        polylines.append(np.array([unrotate(x, y) for x, y in pts], dtype=np.float64))
    return polylines


def _principal_angle(mask: np.ndarray) -> float:
    m = cv2.moments(mask.astype(np.uint8), binaryImage=True)
    if m["mu20"] + m["mu02"] < 1e-3:
        return 45.0
    return float(np.degrees(0.5 * np.arctan2(2 * m["mu11"], m["mu20"] - m["mu02"])))


def tone_fill_polylines(img: np.ndarray, region: np.ndarray, w_half: float) -> list:
    """Tone-aware fill rendering (Aug 12 2026, prototyped in
    debug/tone-centerliner-proto/). The uniform 45° screen flattened every
    mass into wallpaper; this renders a fill the way a pen builds tone:

      - ACCENTS: small features darker than their local surround (eyes,
        knots) leave the tone system and become outlined, densely-filled
        marks; the hatch yields around them.
      - BANDS: quantiles of the source gray under the region (adaptive per
        image); one hatch direction whose DENSITY carries the tone —
        collinear layers deepen the mids, cross-hatch only in the darkest
        band. Angles follow the region's own principal axis.
      - One contour pass around each mass — the pen edge a hand would give it.

    Known limit (kept knowingly): fine features inside smooth tone (a face's
    eyes) still band away more than a hand would allow — the artist's verdict;
    long-term this wants a stroke-native approach, not more filtering."""
    polylines = []
    imgf = img.astype(np.float32)
    k_loc = max(3, int(8 * w_half) | 1)
    local_mean = cv2.boxFilter(imgf, -1, (k_loc, k_loc))
    accents = ((local_mean - imgf) > 25) & (imgf < 170) & region
    na, laba, sta, _ = cv2.connectedComponentsWithStats(accents.astype(np.uint8))
    acc_keep = np.zeros_like(accents)
    for i in range(1, na):
        if (1.5 * w_half) ** 2 <= sta[i, cv2.CC_STAT_AREA] <= (8 * w_half) ** 2:
            acc_keep |= laba == i
    accents = acc_keep
    acc_contours, _ = cv2.findContours(accents.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for c in acc_contours:
        approx = cv2.approxPolyDP(c, 1.0, True).reshape(-1, 2).astype(np.float64)
        if len(approx) >= 3:
            polylines.append(np.vstack([approx, approx[:1]]))
    if accents.any():
        polylines += _serpentine(accents, int(round(1.6 * w_half)), _principal_angle(accents))

    g = cv2.GaussianBlur(img, (0, 0), max(1.5, w_half / 3.0))
    grow = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(2 * w_half) | 1, int(2 * w_half) | 1))
    tone_region = region & ~(cv2.dilate(accents.astype(np.uint8), grow) > 0)
    if not tone_region.any():
        return polylines
    vals = g[tone_region]
    q_dark, q_mid = np.percentile(vals, [40, 75])
    dark = tone_region & (g <= q_dark)
    midplus = tone_region & (g <= q_mid)

    contours, _ = cv2.findContours(tone_region.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for c in contours:
        if cv2.arcLength(c, True) < 12 * w_half:
            continue
        approx = cv2.approxPolyDP(c, 1.5, True).reshape(-1, 2).astype(np.float64)
        if len(approx) >= 3:
            polylines.append(np.vstack([approx, approx[:1]]))

    base_angle = _principal_angle(tone_region)
    for mask, spacing, ang in (
        (tone_region, 5.0 * w_half, base_angle),
        (midplus, 2.4 * w_half, base_angle),
        (dark, 3.0 * w_half, base_angle + 90.0),
    ):
        if mask.any():
            polylines += _serpentine(mask, int(round(spacing)), ang)
    return polylines


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
                             merge_angle_deg: float = 35.0,
                             hatch_angle_deg: float = 45.0,
                             hatch_spacing_factor: float = 3.0,
                             simplify_epsilon: float = 1.2,
                             smooth_iterations: int = 1,
                             min_path_px: int = 5,
                             scale: float = 1.0,
                             save_steps: bool = False,
                             tone_fills: bool = None,
                             engine: str = None):
    """v1-compatible entry point: PNG in, polyline-SVG out.

    tone_fills: None reads CENTERLINE_TONE_FILLS from config (default True);
    False forces the legacy uniform 45° hatch.
    engine: None reads CENTERLINE_ENGINE from config — "v2" (skeleton walk)
    or "dsv_hybrid" (stroke layer through Deep Sketch Vectorization, masses
    through the tone renderer; any DSV failure falls back to v2)."""
    if tone_fills is None:
        try:
            from config.config import CENTERLINE_TONE_FILLS as tone_fills
        except ImportError:
            tone_fills = True
    if engine is None:
        try:
            from config.config import CENTERLINE_ENGINE as engine
        except ImportError:
            engine = "v2"
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to load image from {input_path}")

    binary = binarize(img, contrast_alpha, adaptive, threshold_value, min_component)
    if save_steps:
        base = output_path.rsplit(".", 1)[0]
        cv2.imwrite(f"{base}_v2_step1_binary.png", binary.astype(np.uint8) * 255)

    width_map = cv2.distanceTransform(binary.astype(np.uint8), cv2.DIST_L2, 3)
    strokes, fills, w_half = split_fills(binary, width_map)

    # Engines: "dsv_hybrid" = masses -> tone renderer, stroke layer -> DSV
    # (fidelity to the generated image). "dsv" = the WHOLE ink through DSV,
    # no tone fills — DSV's own stroke-elegant reduction, which simply drops
    # tone the way it did in the Aug 12 eval the artist responded to. Both
    # fall back to the v2 skeleton walk on any failure.
    polylines = None
    dsv_pure = False
    if engine in ("dsv_hybrid", "dsv"):
        try:
            from .dsv_hybrid import dsv_available, dsv_stroke_polylines
        except ImportError:
            from dsv_hybrid import dsv_available, dsv_stroke_polylines
        # Pure engine feeds DSV the RAW grayscale (no binarize — that layer
        # thickens and crushes; the Aug 12 eval verdict was earned on raw
        # gray). The hybrid's stroke layer is necessarily a binary mask.
        dsv_input = img if engine == "dsv" else strokes
        has_ink = binary.any() if engine == "dsv" else strokes.any()
        if dsv_available() and has_ink:
            try:
                polylines = dsv_stroke_polylines(dsv_input, thin=(engine == "dsv_hybrid"))
                dsv_pure = engine == "dsv"
                print(f"[v2] DSV ({engine}): {len(polylines)} strokes from Deep Sketch Vectorization")
            except Exception as e:
                print(f"[v2] DSV ({engine}) failed ({e}) — falling back to skeleton walk")
                polylines = None
        elif not dsv_available():
            print(f"[v2] {engine} requested but DSV_HOME not available — skeleton walk")

    if polylines is None:
        skel = skeletonize(strokes)
        if save_steps:
            cv2.imwrite(f"{base}_v2_step2_skeleton.png", skel.astype(np.uint8) * 255)
            cv2.imwrite(f"{base}_v2_step2_fills.png", fills.astype(np.uint8) * 255)

        paths, nodes, deg = skeleton_paths(skel)
        paths = prune_spurs(paths, nodes, deg, width_map, spur_factor)
        paths = merge_through_junctions(paths, merge_angle_deg)
        polylines = [simplify(p, simplify_epsilon, smooth_iterations)
                     for p in paths if len(p) >= min_path_px]
    if fills.any() and not dsv_pure:
        # Pure DSV IS the whole rendering — its reduction drops tone by
        # design; adding fills back would recreate the hybrid.
        if tone_fills:
            polylines += tone_fill_polylines(img, fills, w_half)
        else:
            polylines += hatch_fills(fills, w_half, hatch_angle_deg, hatch_spacing_factor)

    h, w = binary.shape
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
