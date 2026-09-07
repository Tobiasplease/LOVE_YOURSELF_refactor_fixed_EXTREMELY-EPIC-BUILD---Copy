"""How much line-thickness signal is in a render, and does the centerline SVG carry it?

Samples the ink mask's distance transform along every vertex of an existing
_center_lined.svg: half-width at a centerline vertex IS the local stroke radius.
Reports the distribution in paper mm (using the real warp window + ink-bounds
normalization) so weight tiers can be judged against a nib width.

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
    cx, cy, win_w, win_h, ang = json.load(open(WARP_JSON))["paper_window"]
    allpts = np.vstack(polys)
    span_x = allpts[:, 0].max() - allpts[:, 0].min()
    span_y = allpts[:, 1].max() - allpts[:, 1].min()
    return min(win_w / max(span_x, 1e-6), win_h / max(span_y, 1e-6)) * ink_scale, (win_w, win_h), (span_x, span_y)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    nib = 0.4
    if "--nib" in sys.argv:
        nib = float(sys.argv[sys.argv.index("--nib") + 1])
    png = args[0]
    svg = args[1] if len(args) > 1 else png.rsplit(".", 1)[0] + "_center_lined.svg"

    img = cv2.imread(png, cv2.IMREAD_GRAYSCALE)
    binary = binarize(img, 2.0, True, 180, 12)
    dt = cv2.distanceTransform(binary.astype(np.uint8), cv2.DIST_L2, 3)

    polys = parse_polylines(svg)
    scale, (win_w, win_h), (span_x, span_y) = mm_per_px(polys)

    # The distance transform measures the INK BODY, which in cross-hatching is
    # the mesh, not the stroke. Clamp each vertex's half-width by the distance
    # to the nearest vertex of a DIFFERENT polyline: two neighbouring hatch
    # lines can never each be wider than the gap between them.
    from scipy.spatial import cKDTree

    owner = np.concatenate([np.full(len(p), i) for i, p in enumerate(polys)])
    allpts = np.vstack(polys)
    tree = cKDTree(allpts)
    nbr_d, nbr_i = tree.query(allpts, k=12)
    neighbour_gap = np.full(len(allpts), np.inf)
    for k in range(1, nbr_d.shape[1]):
        other = owner[nbr_i[:, k]] != owner
        neighbour_gap = np.where(other & (nbr_d[:, k] < neighbour_gap), nbr_d[:, k], neighbour_gap)

    seg_len, seg_w, seg_w_raw = [], [], []
    off = 0
    for p in polys:
        xs = np.clip(p[:, 0].round().astype(int), 0, dt.shape[1] - 1)
        ys = np.clip(p[:, 1].round().astype(int), 0, dt.shape[0] - 1)
        w_raw = 2.0 * dt[ys, xs]
        w = np.minimum(w_raw, neighbour_gap[off : off + len(p)])
        off += len(p)
        d = np.hypot(np.diff(p[:, 0]), np.diff(p[:, 1]))
        seg_len.append(d)
        seg_w.append(0.5 * (w[:-1] + w[1:]))
        seg_w_raw.append(0.5 * (w_raw[:-1] + w_raw[1:]))
    seg_len = np.concatenate(seg_len)
    seg_w_px = np.concatenate(seg_w)
    seg_w_raw_px = np.concatenate(seg_w_raw)
    seg_w_mm = seg_w_px * scale
    total_mm = seg_len.sum() * scale

    print(f"render {img.shape[1]}x{img.shape[0]}  ink coverage {100*binary.mean():.1f}%")
    print(f"svg: {len(polys)} polylines, ink span {span_x:.0f}x{span_y:.0f}px -> window {win_w:.0f}x{win_h:.0f}mm")
    print(f"scale {scale:.4f} mm/px   total pen travel {total_mm/1000:.2f} m   nib assumed {nib}mm\n")

    qs = [5, 25, 50, 75, 90, 95, 99]
    reps = np.maximum(seg_len, 1e-9).astype(int) + 1
    print("stroke width, length-weighted percentiles (mm):")
    for label, arr in (("raw distance-transform", seg_w_raw_px * scale), ("neighbour-clamped   ", seg_w_mm)):
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
    print(f"\nextra pen travel if every stroke widened to its measured weight: +{(extra*seg_len).sum()*scale/1000:.2f} m " f"(+{100*(extra*seg_len).sum()/seg_len.sum():.0f}%)")

    fat = seg_w_mm > 2.5 * nib
    print(f"vertices inside fat ink: {100*fat.mean():.1f}% of segments, {100*seg_len[fat].sum()/seg_len.sum():.1f}% of length")


if __name__ == "__main__":
    main()
