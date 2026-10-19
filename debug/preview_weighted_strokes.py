"""Simulate width-aware multi-pass plotting against the current single-pass output.

Reads a render + its _center_lined.svg, estimates per-vertex stroke width
(measure_stroke_width.vertex_widths), then widens each stroke into n serpentine
passes at nib spacing — one pen-down per widened run, so weight costs travel but
not extra plunges. Rasterizes both at true nib width for a 3-panel comparison.

Usage: python debug/preview_weighted_strokes.py <render.png> [svg] [--nib 0.4]
       [--max-passes 4] [--gain 1.0] [--svg out.svg] [--out preview.png]
"""

import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from debug.measure_stroke_width import binarize, mm_per_px, parse_polylines, vertex_widths  # noqa: E402


def _smooth_int(n, k=9):
    """Median-filter the pass count so weight changes in runs, not per vertex."""
    if len(n) < k:
        return np.full(len(n), int(round(np.median(n))))
    pad = np.pad(n, (k // 2, k // 2), mode="edge")
    return np.array([int(round(np.median(pad[i : i + k]))) for i in range(len(n))])


def _normals(p):
    t = np.gradient(p, axis=0)
    L = np.hypot(t[:, 0], t[:, 1])
    L[L < 1e-9] = 1e-9
    return np.stack([-t[:, 1] / L, t[:, 0] / L], axis=1)


def widen(polys, widths, spacing_px, max_passes, gain=1.0):
    """Each run of constant pass-count becomes ONE serpentine polyline."""
    out, hist = [], []
    for p, w in zip(polys, widths):
        if len(p) < 2:
            continue
        n = _smooth_int(np.clip(np.round(gain * w / spacing_px), 1, max_passes).astype(int))
        nrm = _normals(p)
        start = 0
        for i in range(1, len(n) + 1):
            if i == len(n) or n[i] != n[start]:
                seg, seg_n, k = p[start : i + 1], nrm[start : i + 1], int(n[start])
                if len(seg) >= 2:
                    hist.append((k, np.hypot(*np.diff(seg, axis=0).T).sum()))
                    if k == 1:
                        out.append(seg)
                    else:
                        chain = []
                        for j in range(k):
                            lane = seg + seg_n * ((j - (k - 1) / 2.0) * spacing_px)
                            chain.append(lane if j % 2 == 0 else lane[::-1])
                        out.append(np.vstack(chain))
                start = i
    return out, hist


def raster(polys, shape, nib_px, scale_up=2):
    canvas = np.full((shape[0] * scale_up, shape[1] * scale_up), 255, np.uint8)
    t = max(1, int(round(nib_px * scale_up)))
    for p in polys:
        cv2.polylines(canvas, [(p * scale_up).round().astype(np.int32)], False, 0, t, cv2.LINE_AA)
    return cv2.resize(canvas, (shape[1], shape[0]), interpolation=cv2.INTER_AREA)


def _flag(name, cast, default):
    return cast(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default


def main():
    argv, args = sys.argv[1:], []
    i = 0
    while i < len(argv):
        if argv[i].startswith("--"):
            i += 2
        else:
            args.append(argv[i])
            i += 1
    nib = _flag("--nib", float, 0.4)
    max_passes = _flag("--max-passes", int, 4)
    gain = _flag("--gain", float, 1.0)
    png = args[0]
    svg = args[1] if len(args) > 1 else png.rsplit(".", 1)[0] + "_center_lined.svg"

    img = cv2.imread(png, cv2.IMREAD_GRAYSCALE)
    dt = cv2.distanceTransform(binarize(img, 2.0, True, 180, 12).astype(np.uint8), cv2.DIST_L2, 3)
    polys = parse_polylines(svg)
    scale, _, _ = mm_per_px(polys)
    nib_px, spacing_px = nib / scale, 0.85 * nib / scale

    widths, _, _ = vertex_widths(polys, dt)
    wide, hist = widen(polys, widths, spacing_px, max_passes, gain)

    def travel(ps):
        return sum(np.hypot(*np.diff(p, axis=0).T).sum() for p in ps if len(p) > 1) * scale / 1000.0

    t0, t1 = travel(polys), travel(wide)
    print(f"nib {nib}mm = {nib_px:.1f}px   spacing {spacing_px:.1f}px   max {max_passes} passes   gain {gain}")
    print(f"strokes: {len(polys)} -> {len(wide)} pen-downs (before vpype linemerge)")
    print(f"travel:  {t0:.2f} m -> {t1:.2f} m  (+{100*(t1-t0)/max(t0,1e-9):.0f}%)")
    print(f"draw time at 450mm/min: {1000*t0/450:.0f} min -> {1000*t1/450:.0f} min")
    tot = sum(L for _, L in hist) or 1.0
    for k in range(1, max_passes + 1):
        L = sum(L for kk, L in hist if kk == k)
        print(f"  {k} pass{'es' if k > 1 else '  '}: {100*L/tot:5.1f}% of centerline length")

    if "--svg" in sys.argv:
        import svgwrite

        out_svg = _flag("--svg", str, None)
        h, w = img.shape
        dwg = svgwrite.Drawing(out_svg, size=(f"{w}px", f"{h}px"))
        for p in wide:
            dwg.add(dwg.polyline(points=[(float(x), float(y)) for x, y in p], stroke="black", fill="none", stroke_width=1))
        dwg.save()
        print(f"wrote {out_svg}")

    panels = [img, raster(polys, img.shape, nib_px), raster(wide, img.shape, nib_px)]
    labels = ["render (what it wants)", f"current: one pass, {nib}mm nib", f"width-aware, max {max_passes} passes"]
    tall = []
    for pan, lab in zip(panels, labels):
        p = cv2.cvtColor(pan, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(p, (0, 0), (p.shape[1], 44), (255, 255, 255), -1)
        cv2.putText(p, lab, (14, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 0), 2, cv2.LINE_AA)
        tall.append(p)
    out = np.hstack([np.pad(t, ((6, 6), (6, 6), (0, 0)), constant_values=180) for t in tall])
    dest = _flag("--out", str, os.path.join(os.path.dirname(png), "weighted_stroke_preview.png"))
    cv2.imwrite(dest, out)
    print(f"preview -> {dest}")


if __name__ == "__main__":
    main()
