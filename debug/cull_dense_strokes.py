#!/usr/bin/env python3
"""Drop the strokes the nib cannot resolve, and show what the sheet would look like.

One pen of fixed width has three variables: whether a line is there, how far it
sits from its neighbours, and how many times you go over it. Thickness is not
one of them — so tone can only come from SPACING, and spacing only works above
the distance the nib can resolve. Below that, two lines are one wet mark.

Measured on the Sep 10 drawings at 0.18mm/px with a 0.4mm nib, 53% of the
workbench drawing's vertices sit closer together than 2x nib and 23% closer than
the nib itself. That is not a taste problem, it is arithmetic: those regions
cannot render as lines, so they grey out, the dark end saturates, and the tonal
range collapses into mud. The artist's word for the short ones is "messy dots".

So this culls on a physical criterion rather than a guess:

  - strokes shorter than --min-length land as dots, not lines
  - strokes packed below --min-gap x nib are thinned out, crowdedest first,
    until the local spacing clears what the nib can actually separate

What survives is a drawing whose darkest region sits AT the resolvable limit
instead of past it, which is where hierarchy comes from: the light areas stay
sparse while the dark ones stop saturating.

The preview rasterizes at TRUE NIB WIDTH — the 1px renders everyone looks at
flatter the problem, because they draw a 0.4mm nib as a hairline.

The pair to this is multi-pass weighting (debug/preview_weighted_strokes.py) for
the dark end: cull for the lights, weight for the darks, and the fixed nib gets
a range it otherwise has no way to express.

Usage:
  python debug/cull_dense_strokes.py <centerlined.svg> [--nib 0.4]
      [--min-gap 1.5] [--min-length 1.0] [--out culled.svg]
"""

import argparse
import os
import re
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from debug.measure_stroke_width import mm_per_px, neighbour_gap, parse_polylines  # noqa: E402


def stroke_lengths_mm(polys, mmpx):
    return np.array([np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1)) * mmpx for p in polys])


def gaps_by_stroke(polys, mmpx):
    """Median neighbour gap in mm, per stroke. Crowded strokes score low."""
    gap, owner = neighbour_gap(polys)
    gmm = gap * mmpx
    out = np.empty(len(polys))
    off = 0
    for i, p in enumerate(polys):
        seg = gmm[off : off + len(p)]
        seg = seg[np.isfinite(seg)]
        out[i] = np.median(seg) if len(seg) else np.inf
        off += len(p)
    return out, gmm


def cull(polys, mmpx, nib, min_gap_mult, min_len_mm, max_rounds=8):
    """Keep-mask over polys. Short strokes go first, then the crowdedest, in
    rounds — one pass cannot know what the spacing becomes once neighbours go."""
    keep = np.ones(len(polys), bool)
    lengths = stroke_lengths_mm(polys, mmpx)
    keep &= lengths >= min_len_mm
    print(f"  dots  : dropped {(~keep).sum()} strokes under {min_len_mm}mm")

    target = nib * min_gap_mult
    for rnd in range(max_rounds):
        idx = np.flatnonzero(keep)
        if len(idx) < 2:
            break
        live = [polys[i] for i in idx]
        gmed, gmm = gaps_by_stroke(live, mmpx)
        crowded = np.flatnonzero(gmed < target)
        if len(crowded) == 0:
            print(f"  round {rnd + 1}: spacing clear at {target:.2f}mm")
            break
        # Take the worst third each round: removing a stroke opens space for its
        # neighbours, so culling everything crowded at once guts the region.
        order = crowded[np.argsort(gmed[crowded])]
        drop = order[: max(1, len(order) // 3)]
        keep[idx[drop]] = False
        below = (gmm < target).mean()
        print(f"  round {rnd + 1}: {len(crowded)} crowded, dropped {len(drop)}, {below:.0%} of vertices still under {target:.2f}mm")
    return keep


def write_svg(src_svg, keep, out_path):
    """Filter <polyline> elements, leave the document otherwise untouched so the
    G-code step sees exactly the format it already handles."""
    txt = open(src_svg).read()
    parts = re.split(r"(<polyline\b[^>]*/>)", txt)
    out, i = [], 0
    for part in parts:
        if part.startswith("<polyline"):
            if i < len(keep) and keep[i]:
                out.append(part)
            i += 1
        else:
            out.append(part)
    open(out_path, "w").write("".join(out))
    return i


def preview(polys, keep, mmpx, nib, out_png, px_per_mm=8.0):
    """Both versions at true nib width — a 0.4mm nib is not a hairline."""
    pts = np.vstack(polys)
    ox, oy = pts[:, 0].min(), pts[:, 1].min()
    scale = mmpx * px_per_mm
    w = int((pts[:, 0].max() - ox) * scale) + 40
    h = int((pts[:, 1].max() - oy) * scale) + 40
    thick = max(1, int(round(nib * px_per_mm)))

    def draw(mask):
        img = np.full((h, w), 255, np.uint8)
        for p, k in zip(polys, mask):
            if not k:
                continue
            q = np.column_stack(((p[:, 0] - ox) * scale + 20, (p[:, 1] - oy) * scale + 20))
            cv2.polylines(img, [q.round().astype(np.int32)], False, 0, thick, cv2.LINE_AA)
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    def label(img, t):
        cv2.rectangle(img, (0, 0), (img.shape[1], 34), (0, 0, 0), -1)
        cv2.putText(img, t, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        return img

    before = label(draw(np.ones(len(polys), bool)), f"AS PLOTTED NOW - {len(polys)} strokes at {nib}mm nib")
    after = label(draw(keep), f"CULLED - {int(keep.sum())} strokes")
    cv2.imwrite(out_png, np.hstack([before, after]))
    return before.shape


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("svg")
    ap.add_argument("--nib", type=float, default=0.4, help="pen width in mm")
    ap.add_argument("--min-gap", type=float, default=1.5, help="x nib: spacing the drawing must clear")
    ap.add_argument("--min-length", type=float, default=1.0, help="mm: shorter strokes land as dots")
    ap.add_argument("--out", help="culled svg (default: alongside the source)")
    ap.add_argument("--preview", help="preview png")
    args = ap.parse_args()

    polys = parse_polylines(args.svg)
    mmpx, (ww, wh), _ = mm_per_px(polys)
    lengths = stroke_lengths_mm(polys, mmpx)
    _, gmm0 = gaps_by_stroke(polys, mmpx)
    print(f"{os.path.basename(args.svg)}: {len(polys)} strokes, {mmpx:.3f} mm/px, paper {ww:.0f}x{wh:.0f}mm, nib {args.nib}mm")
    print(
        f"  before: {(gmm0 < args.nib).mean():.1%} of vertices under nib, {(gmm0 < args.nib * 2).mean():.1%} under 2x nib, median stroke {np.median(lengths):.2f}mm\n"
    )

    keep = cull(polys, mmpx, args.nib, args.min_gap, args.min_length)

    live = [p for p, k in zip(polys, keep) if k]
    _, gmm1 = gaps_by_stroke(live, mmpx)
    ink0 = lengths.sum()
    ink1 = stroke_lengths_mm(live, mmpx).sum()
    print(f"\n  after : {int(keep.sum())}/{len(polys)} strokes kept ({keep.mean():.0%}), {ink1 / ink0:.0%} of the ink")
    print(f"          {(gmm1 < args.nib).mean():.1%} of vertices under nib, {(gmm1 < args.nib * 2).mean():.1%} under 2x nib")

    out_svg = args.out or args.svg.replace(".svg", "_culled.svg")
    n = write_svg(args.svg, keep, out_svg)
    print(f"\n  svg   : {out_svg} ({n} polylines seen)")

    out_png = args.preview or out_svg.replace(".svg", "_preview.png")
    shape = preview(polys, keep, mmpx, args.nib, out_png)
    print(f"  preview: {out_png} ({shape[1]}x{shape[0]} each, true {args.nib}mm nib)")


if __name__ == "__main__":
    main()
