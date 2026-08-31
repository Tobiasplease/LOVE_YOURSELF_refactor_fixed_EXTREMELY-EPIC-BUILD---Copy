"""A/B the centerline tracers: v1 (contour-of-skeleton) vs v2 (graph walk).

Renders original | v1 | v2 side by side and prints the numbers that matter:
paths, closed loops (v1's double-tracing), total ink length (= pen time).

    python debug/test_centerliner_v2.py /path/to/comfy_output.png [more.png ...]

Writes nothing outside /tmp. Does not touch the runtime pipeline.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import vpype
from PIL import Image

from bcnc.svg_centerliner import raster_to_centerline_svg as v1
from bcnc.svg_centerliner_v2 import raster_to_centerline_svg as v2


def stats(svg_path):
    doc = vpype.read_multilayer_svg(svg_path, quantization=0.5)
    n, total, closed, verts = 0, 0.0, 0, 0
    lines = []
    for lid in doc.layers:
        for line in doc.layers[lid]:
            n += 1
            verts += len(line)
            total += float(np.abs(np.diff(line)).sum())
            if len(line) > 3 and abs(line[0] - line[-1]) < 3:
                closed += 1
            lines.append(line)
    return dict(paths=n, closed=closed, ink=total, verts=verts, lines=lines)


def render(ax, lines, title):
    for line in lines:
        ax.plot(line.real, line.imag, "k-", lw=0.8)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.axis("off")
    ax.set_title(title, fontsize=9)


def main():
    pngs = sys.argv[1:]
    if not pngs:
        print(__doc__)
        return
    for png in pngs:
        base = os.path.basename(png).rsplit(".", 1)[0]
        svg1 = f"/tmp/{base}_v1.svg"
        svg2 = f"/tmp/{base}_v2.svg"
        v1(png, svg1, threshold_value=180, do_dilate=True, dilation_iterations=1)
        v2(png, svg2)
        s1, s2 = stats(svg1), stats(svg2)

        fig, axes = plt.subplots(1, 3, figsize=(18, 6.5))
        axes[0].imshow(Image.open(png).convert("L"), cmap="gray")
        axes[0].axis("off")
        axes[0].set_title("ComfyUI original", fontsize=9)
        render(axes[1], s1["lines"], f"v1: {s1['paths']} paths ({s1['closed']} closed=double-drawn), ink {s1['ink']:.0f}")
        render(axes[2], s2["lines"], f"v2: {s2['paths']} strokes ({s2['closed']} closed), ink {s2['ink']:.0f}")
        out = f"/tmp/{base}_ab.png"
        plt.tight_layout()
        plt.savefig(out, dpi=75, facecolor="white")
        plt.close()
        print(f"\n{base}:")
        print(f"  v1: {s1['paths']:4d} paths, {s1['closed']:3d} closed, ink {s1['ink']:9.0f}, {s1['verts']} vertices")
        print(f"  v2: {s2['paths']:4d} paths, {s2['closed']:3d} closed, ink {s2['ink']:9.0f}, {s2['verts']} vertices")
        print(f"  pen-travel saved: {100 * (1 - s2['ink'] / max(1, s1['ink'])):.0f}%   comparison: {out}")


if __name__ == "__main__":
    main()
