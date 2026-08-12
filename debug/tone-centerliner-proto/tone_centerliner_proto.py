"""Prototype: tone-aware centerliner. Strokes -> skeleton centerlines (v2
machinery, unchanged). Ink regions -> hatching whose DENSITY carries the
gray value underneath and whose ANGLE follows each region's own axis.
Dark passages cross-hatch; light passages breathe. Region detection widened
beyond v2's width-cores to include fused scribble (the honeycomb source)."""
import sys

sys.path.insert(0, '/home/impostor/LOVE_YOURSELF_refactor_fixed_EXTREMELY-EPIC-BUILD---Copy')
import cv2
import numpy as np
from skimage.morphology import skeletonize

from bcnc.svg_centerliner_v2 import (binarize, skeleton_paths, prune_spurs,
                                     merge_through_junctions, simplify)


def serpentine_hatch(mask, spacing_px, angle_deg, outline=False):
    """v2's serpentine hatcher, parameterized: absolute spacing, optional outline."""
    polylines = []
    mask8 = mask.astype(np.uint8)
    if outline:
        contours, _ = cv2.findContours(mask8, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        for c in contours:
            if cv2.arcLength(c, True) < 4 * spacing_px:
                continue
            approx = cv2.approxPolyDP(c, 1.2, True).reshape(-1, 2).astype(np.float64)
            if len(approx) >= 3:
                polylines.append(np.vstack([approx, approx[:1]]))
    spacing = max(3, int(round(spacing_px)))
    h, w = mask.shape
    diag = int(np.ceil(np.hypot(h, w)))
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle_deg, 1.0)
    M[0, 2] += (diag - w) / 2.0
    M[1, 2] += (diag - h) / 2.0
    rot = cv2.warpAffine(mask8 * 255, M, (diag, diag), flags=cv2.INTER_NEAREST)
    Minv = cv2.invertAffineTransform(M)

    def unrotate(x, y):
        return (Minv[0, 0] * x + Minv[0, 1] * y + Minv[0, 2],
                Minv[1, 0] * x + Minv[1, 1] * y + Minv[1, 2])

    chains, open_chains = [], []
    for y in range(spacing // 2, diag, spacing):
        idx = np.flatnonzero(rot[y] > 127)
        runs = []
        if idx.size >= 2:
            splits = np.flatnonzero(np.diff(idx) > 1)
            runs = [(int(r[0]), int(r[-1])) for r in np.split(idx, splits + 1) if r.size >= max(2, spacing // 2)]
        next_open, used = [], set()
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
        if len(pts) >= 2:
            polylines.append(np.array([unrotate(x, y) for x, y in pts], dtype=np.float64))
    return polylines


def principal_angle(comp_mask):
    m = cv2.moments(comp_mask.astype(np.uint8), binaryImage=True)
    if m['mu20'] + m['mu02'] < 1e-3:
        return 45.0
    return float(np.degrees(0.5 * np.arctan2(2 * m['mu11'], m['mu20'] - m['mu02'])))


def tone_aware_centerline(src_path):
    """Round 3: v2's PROVEN classification (split_fills) untouched; only the
    RENDERING of a detected fill changes — tone-banded cumulative hatching
    instead of one uniform 45° screen."""
    from bcnc.svg_centerliner_v2 import split_fills

    img = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
    binary = binarize(img, 2.0, True, 180, 12)
    width_map = cv2.distanceTransform(binary.astype(np.uint8), cv2.DIST_L2, 3)
    strokes, region, w_half = split_fills(binary, width_map)

    skel = skeletonize(strokes)
    paths, nodes, deg = skeleton_paths(skel)
    paths = prune_spurs(paths, nodes, deg, width_map, 1.6)
    paths = merge_through_junctions(paths, 35.0)
    polylines = [simplify(p, 1.2, 1) for p in paths if len(p) >= 5]
    stats_out = {'w_half': w_half, 'strokes': len(polylines)}

    if not region.any():
        return polylines, stats_out

    # Round-2 fix 2: adaptive bands — quantiles of the gray UNDER the region,
    # so a light pencil-toned image bands as readily as a black mass.
    g = cv2.GaussianBlur(img, (0, 0), max(1.0, w_half))
    vals = g[region]
    q_dark, q_mid = np.percentile(vals, [40, 75])
    dark = region & (g <= q_dark)
    midplus = region & (g <= q_mid)  # dark+mid cumulative

    # Round-2 fix 3: CUMULATIVE layers with globally coherent angles — the
    # printmaker's build-up. Layer 1 covers the whole region sparsely; layer 2
    # adds over dark+mid; layer 3 adds over dark only. Rows stay collinear
    # across band boundaries, so tone reads as one system, not confetti.
    # Classical build: ONE hatch direction whose density carries the tone
    # (collinear layers deepen it), cross-hatch only where it's darkest.
    base_angle = principal_angle(region)
    layers = [
        ('light', region, 5.0 * w_half, base_angle),
        ('mid', midplus, 2.4 * w_half, base_angle),
        ('dark', dark, 3.0 * w_half, base_angle + 90.0),
    ]
    # One outline pass around each tonal mass — the pen edge a hand would give it.
    contours, _ = cv2.findContours(region.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for c in contours:
        if cv2.arcLength(c, True) < 12 * w_half:
            continue
        approx = cv2.approxPolyDP(c, 1.5, True).reshape(-1, 2).astype(np.float64)
        if len(approx) >= 3:
            polylines.append(np.vstack([approx, approx[:1]]))
    for name, mask, spacing, ang in layers:
        pls = serpentine_hatch(mask, spacing, ang, outline=False)
        polylines += pls
        stats_out[name] = len(pls)
    return polylines, stats_out


if __name__ == '__main__':
    import re
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from bcnc.svg_centerliner_v2 import raster_to_centerline_svg

    cases = {
        'foam-blob': '/home/impostor/ComfyUI/output/impostor-20260812_152527_00001_.png',
        'trash-bag': '/home/impostor/ComfyUI/output/impostor-20260812_172004_00001_.png',
        'mannequin': '/home/impostor/ComfyUI/output/impostor-20260812_171211_00001_.png',
    }
    fig, axes = plt.subplots(len(cases), 3, figsize=(21, 7 * len(cases)))
    for row, (name, src) in enumerate(cases.items()):
        img = cv2.imread(src, cv2.IMREAD_GRAYSCALE)
        axes[row][0].imshow(img, cmap='gray')
        axes[row][0].set_title(f'{name}: source')

        out = f'v2_{name}.svg'
        raster_to_centerline_svg(src, out)
        svg = open(out).read()
        for pts_attr in re.findall(r'points="([^"]+)"', svg):
            nums = [float(x) for x in re.findall(r'-?[\d.]+', pts_attr)]
            pts = [(nums[i], nums[i + 1]) for i in range(0, len(nums) - 1, 2)]
            axes[row][1].plot([p[0] for p in pts], [p[1] for p in pts], 'k-', lw=0.55)
        axes[row][1].set_title(f'{name}: current v2')
        axes[row][1].invert_yaxis()

        polylines, st = tone_aware_centerline(src)
        for pl in polylines:
            axes[row][2].plot(pl[:, 0], pl[:, 1], 'k-', lw=0.55)
        axes[row][2].set_title(f"{name}: tone-aware (strokes {st['strokes']}, dark {st.get('dark', 0)}, mid {st.get('mid', 0)}, light {st.get('light', 0)})")
        axes[row][2].invert_yaxis()
        for ax in axes[row]:
            ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig('tone_proto_compare.png', dpi=80)
    print('saved tone_proto_compare.png')
