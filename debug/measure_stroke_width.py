"""How much line-thickness signal is in a render, and does the centerline SVG carry it?

Samples the ink mask's distance transform along every vertex of an existing
_center_lined.svg: the half-width at a centerline vertex IS the local stroke
radius. The raw distance transform measures the ink BODY, which in cross-hatching
is the mesh rather than the stroke, so each vertex is clamped by the distance to
the nearest vertex of a DIFFERENT polyline. Reports the distribution in paper mm
(real warp window + ink-bounds normalization) so weight tiers can be judged
against a nib width.

Usage: python debug/measure_stroke_width.py <render.png> [centerlined.svg] [--nib 0.4]
"""

import json
import os
import re
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bcnc.svg_centerliner_v2 import binarize  # noqa: E402

WARP_JSON = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "grbl", "warp_calibration.json")


def parse_polylines(svg_path):
    txt = open(svg_path).read()
    polys = []
    for pts in re.findall(r'points="([^"]+)"', txt):
        p = [tuple(map(float, pair.split(","))) for pair in pts.split() if "," in pair]
        if len(p) > 1:
            polys.append(np.array(p, dtype=np.float64))
    for d in re.findall(r'd="([^"]+)"', txt):
        cur = []
        for cmd, x, y in re.findall(r"([ML])\s*([-\d.eE]+)[ ,]([-\d.eE]+)", d):
            if cmd == "M":
                if len(cur) > 1:
                    polys.append(np.array(cur, dtype=np.float64))
                cur = [(float(x), float(y))]
            else:
                cur.append((float(x), float(y)))
        if len(cur) > 1:
            polys.append(np.array(cur, dtype=np.float64))
    return polys


def mm_per_px(polys, ink_scale=1.0):
    """paper_window is (cx, cy, W, H, angle); the warp fits the INK BOUNDS into it."""
    cx, cy, win_w, win_h, ang = json.load(open(WARP_JSON))["paper_window"]
    allpts = np.vstack(polys)
    span_x = allpts[:, 0].max() - allpts[:, 0].min()
    span_y = allpts[:, 1].max() - allpts[:, 1].min()
    return min(win_w / max(span_x, 1e-6), win_h / max(span_y, 1e-6)) * ink_scale, (win_w, win_h), (span_x, span_y)


def neighbour_gap(polys, k=64):
    """Distance from each vertex to the nearest vertex of a different polyline.

    k must be large enough to escape the vertex's OWN polyline — DSV emits dense
    polylines, so a small k returns only self-neighbours and the clamp silently
    never binds (which is exactly what over-inked the first hatching preview).
    """
    from scipy.spatial import cKDTree

    owner = np.concatenate([np.full(len(p), i) for i, p in enumerate(polys)])
    allpts = np.vstack(polys)
    k = min(k, len(allpts))
    nbr_d, nbr_i = cKDTree(allpts).query(allpts, k=k)
    gap = np.full(len(allpts), np.inf)
    for j in range(1, k):
        other = owner[nbr_i[:, j]] != owner
        gap = np.where(other & (nbr_d[:, j] < gap), nbr_d[:, j], gap)
    return gap, owner


def vertex_widths(polys, dt, k=64):
    """Per-vertex stroke WIDTH in pixels, clamped so hatch meshes don't read as fat."""
    gap, _ = neighbour_gap(polys, k)
    raw, clamped, off = [], [], 0
    for p in polys:
        xs = np.clip(p[:, 0].round().astype(int), 0, dt.shape[1] - 1)
        ys = np.clip(p[:, 1].round().astype(int), 0, dt.shape[0] - 1)
        w_raw = 2.0 * dt[ys, xs]
        raw.append(w_raw)
        clamped.append(np.minimum(w_raw, gap[off : off + len(p)]))
        off += len(p)
    return clamped, raw, gap


def main():
    argv, args = sys.argv[1:], []
    i = 0
    while i < len(argv):
        if argv[i].startswith("--"):
            i += 2
        else:
            args.append(argv[i])
            i += 1
    nib = float(sys.argv[sys.argv.index("--nib") + 1]) if "--nib" in sys.argv else 0.4
    png = args[0]
    svg = args[1] if len(args) > 1 else png.rsplit(".", 1)[0] + "_center_lined.svg"

    img = cv2.imread(png, cv2.IMREAD_GRAYSCALE)
    binary = binarize(img, 2.0, True, 180, 12)
    dt = cv2.distanceTransform(binary.astype(np.uint8), cv2.DIST_L2, 3)

    polys = parse_polylines(svg)
    scale, (win_w, win_h), (span_x, span_y) = mm_per_px(polys)
    clamped, raw, gap = vertex_widths(polys, dt)

    seg_len, seg_w, seg_w_raw = [], [], []
    for p, w, wr in zip(polys, clamped, raw):
        seg_len.append(np.hypot(np.diff(p[:, 0]), np.diff(p[:, 1])))
        seg_w.append(0.5 * (w[:-1] + w[1:]))
        seg_w_raw.append(0.5 * (wr[:-1] + wr[1:]))
    seg_len = np.concatenate(seg_len)
    seg_w_mm = np.concatenate(seg_w) * scale
    seg_w_raw_mm = np.concatenate(seg_w_raw) * scale
    total_mm = seg_len.sum() * scale
    nverts = sum(len(p) for p in polys)

    print(f"render {img.shape[1]}x{img.shape[0]}  ink coverage {100*binary.mean():.1f}%")
    print(f"svg: {len(polys)} polylines, {nverts} vertices, ink span {span_x:.0f}x{span_y:.0f}px -> window {win_w:.0f}x{win_h:.0f}mm")
    print(f"scale {scale:.4f} mm/px   total pen travel {total_mm/1000:.2f} m   nib assumed {nib}mm")
    binds = np.concatenate([(w < wr - 1e-9) for w, wr in zip(clamped, raw)])
    print(f"neighbour clamp binds on {100*binds.mean():.1f}% of vertices (unreachable neighbours: {100*np.isinf(gap).mean():.1f}%)\n")

    qs = [5, 25, 50, 75, 90, 95, 99]
    reps = np.maximum(seg_len, 1e-9).astype(int) + 1
    print("stroke width, length-weighted percentiles (mm):")
    for label, arr in (("raw distance-transform", seg_w_raw_mm), ("neighbour-clamped     ", seg_w_mm)):
        pct = np.percentile(np.repeat(arr, reps), qs)
        print(f"  {label}: " + "  ".join(f"p{q}={v:.2f}" for q, v in zip(qs, pct)))

    print("\nlength by weight tier (passes needed at nib width):")
    edges = [0, 1.5 * nib, 2.5 * nib, 3.5 * nib, 5.5 * nib, 1e9]
    names = ["1 pass (<1.5 nib)", "2 passes", "3 passes", "4-5 passes", "6+ passes (mass)"]
    for lo, hi, name in zip(edges[:-1], edges[1:], names):
        m = (seg_w_mm >= lo) & (seg_w_mm < hi)
        L = seg_len[m].sum() * scale
        print(f"  {name:<20} {L/1000:6.2f} m  {100*L/total_mm:5.1f}%")

    extra = np.clip(np.round(seg_w_mm / nib), 1, 8) - 1
    print(f"\nextra pen travel if every stroke widened to its measured weight: " f"+{(extra*seg_len).sum()*scale/1000:.2f} m (+{100*(extra*seg_len).sum()/seg_len.sum():.0f}%)")


if __name__ == "__main__":
    main()
