#!/usr/bin/env python3
"""Feed a centreline drawing back through generation. Does it come out more plottable?

The pipeline's loss is a representation mismatch: Flux renders tonal masses —
dense hatching, filled blacks — and the centreliner has to destroy them to get
single-width strokes. The drawing review named it without being asked ("all
wireframe, all loose sketch lines where there should be dense, heavy
cross-hatching"). The hypothesis under test is the artist's: condition the
generation on the CENTRELINE output instead of a depth map, because that is the
closest thing to what the pen actually reproduces, and see whether what comes
back is nearer to something a plotter can say.

What it does, on material already in ComfyUI/output — nothing is drawn, no paper
is involved:

  1. renders an existing _center_lined.svg back to a raster
  2. runs it through the live workflow with ControlNet switched from depth to
     canny and the depth preprocessor bypassed (clean line art IS an edge map)
  3. centrelines the result with the same bcnc pass the real pipeline uses
  4. measures two numbers

The two numbers:

  STROKE WIDTH — pass 1 vs pass 2, via debug/measure_stroke_width. This is the
  hypothesis stated quantitatively: if conditioning on centrelines really does
  bias toward plotter-native output, pass 2's ink should be thinner and more
  uniform, closer to a nib and further from tonal mass.

  PRESERVATION — the fraction of pass 2's strokes landing on pass 1's ink. This
  answers the OTHER question: whether an additive second pass is even possible.
  Adding detail to a real sheet means plotting only what is new, which needs the
  old lines to come back in the same place. Low preservation means the geometry
  drifted and a clean diff is off the table.

Flux needs the VRAM llama-server is holding, so --stop-llama frees it (the next
machine.py start brings it back; ~13s reload).

Usage:
  python debug/test_centerline_feedback.py --list
  python debug/test_centerline_feedback.py [--svg PATH] [--prompt TEXT] [--stop-llama]
"""

import argparse
import glob
import json
import os
import sys
import time
import urllib.request

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from debug.measure_stroke_width import binarize, parse_polylines, vertex_widths  # noqa: E402

COMFY = "http://localhost:8188"
COMFY_ROOT = os.path.expanduser("~/ComfyUI")
OUT_DIR = os.path.join(COMFY_ROOT, "output")
IN_DIR = os.path.join(COMFY_ROOT, "input")
TEMPLATE = "drawing/impostor-template-impostor-bot.json"
PREFIX = "clfeedback"


def render_svg(svg_path, out_png, thickness=1):
    """Centreline SVG back to a raster: black strokes on white, nib-thin."""
    polys = parse_polylines(svg_path)
    if not polys:
        raise SystemExit(f"no polylines in {svg_path}")
    allpts = np.vstack(polys)
    w = int(np.ceil(allpts[:, 0].max())) + 8
    h = int(np.ceil(allpts[:, 1].max())) + 8
    canvas = np.full((h, w), 255, np.uint8)
    for p in polys:
        cv2.polylines(canvas, [p.round().astype(np.int32)], False, 0, thickness, cv2.LINE_AA)
    cv2.imwrite(out_png, canvas)
    return canvas, len(polys)


def build_workflow(image_name, prompt, seed, size=None):
    """The live template, rewired: centreline in, depth preprocessor out."""
    wf = json.load(open(TEMPLATE))
    if size:
        # The template's latent is 1328x752 landscape. Generating at a different
        # aspect than the conditioning image stretches the ControlNet guidance,
        # which drifts the geometry and makes any preservation number meaningless.
        w, h = (max(256, (v // 16) * 16) for v in size)
        wf["5"]["inputs"].update({"width": w, "height": h})
    wf["607"]["inputs"]["image"] = image_name
    wf["711"]["inputs"]["image"] = ["607", 0]  # bypass DepthAnythingV2 (712)
    # Exact enum from the union pro 2 node — "canny" alone fails validation and
    # ComfyUI then silently ignores every output downstream of it.
    wf["713"]["inputs"]["type"] = "canny/lineart/anime_lineart/mlsd"
    wf["723"]["inputs"]["String"] = prompt
    wf["30"]["inputs"]["filename_prefix"] = PREFIX
    for nid, node in wf.items():
        if isinstance(node, dict) and "noise_seed" in node.get("inputs", {}):
            node["inputs"]["noise_seed"] = seed
    return wf


def queue_and_wait(wf, timeout=600):
    req = urllib.request.Request(f"{COMFY}/prompt", data=json.dumps({"prompt": wf}).encode(), headers={"Content-Type": "application/json"})
    pid = json.loads(urllib.request.urlopen(req, timeout=30).read())["prompt_id"]
    print(f"[comfy] queued {pid}")
    start = time.time()
    while time.time() - start < timeout:
        time.sleep(3)
        try:
            hist = json.loads(urllib.request.urlopen(f"{COMFY}/history/{pid}", timeout=15).read())
        except Exception:
            continue
        if pid in hist:
            # Node 30 is SaveImage. Take it specifically: the PreviewImage nodes
            # write temp files that get swept, and a validation failure leaves
            # only those behind while the run still reports success.
            out = hist[pid].get("outputs", {}).get("30", {})
            for img in out.get("images", []):
                return os.path.join(OUT_DIR, img.get("subfolder", ""), img["filename"])
            raise SystemExit("SaveImage produced nothing — check the tail of event_log/comfyui.log for a validation error")
        print(f"  … {time.time() - start:.0f}s")
    raise SystemExit(f"timed out after {timeout}s")


def widths_mm(svg_path, png_path):
    """Median/IQR stroke width in pixels of the raster it came from."""
    polys = parse_polylines(svg_path)
    img = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)
    ink = binarize(img)
    dt = cv2.distanceTransform((ink > 0).astype(np.uint8), cv2.DIST_L2, 5)
    clamped, _raw, _gap = vertex_widths(polys, dt)
    w = np.concatenate(clamped)
    w = w[np.isfinite(w) & (w > 0)]
    return polys, w


def preservation(old_svg, new_svg, shape, tol=3.0):
    """Fraction of new vertices landing within tol px of old ink."""
    old = np.zeros(shape, np.uint8)
    for p in parse_polylines(old_svg):
        cv2.polylines(old, [p.round().astype(np.int32)], False, 255, 1)
    dist = cv2.distanceTransform((old == 0).astype(np.uint8), cv2.DIST_L2, 5)
    hits = total = 0
    for p in parse_polylines(new_svg):
        xs = np.clip(p[:, 0].round().astype(int), 0, shape[1] - 1)
        ys = np.clip(p[:, 1].round().astype(int), 0, shape[0] - 1)
        hits += int((dist[ys, xs] <= tol).sum())
        total += len(p)
    return hits / max(1, total), total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--svg", help="a _center_lined.svg (default: most recent)")
    ap.add_argument("--prompt", default="impostor black and white sketch line art, thin single-weight pen strokes")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--stop-llama", action="store_true", help="free the VRAM Flux needs")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    svgs = sorted(glob.glob(os.path.join(OUT_DIR, "*_center_lined.svg")), key=os.path.getmtime, reverse=True)
    if args.list:
        for s in svgs[:15]:
            print(f"  {time.strftime('%m-%d %H:%M', time.localtime(os.path.getmtime(s)))}  {os.path.basename(s)}")
        return
    if not svgs:
        raise SystemExit(f"no _center_lined.svg in {OUT_DIR}")

    src_svg = args.svg or svgs[0]
    base = os.path.basename(src_svg).replace("_center_lined.svg", "")
    src_png = os.path.join(OUT_DIR, base + ".png")
    print(f"Source : {os.path.basename(src_svg)}")
    print(f"Render : {os.path.basename(src_png)} {'(found)' if os.path.exists(src_png) else '(MISSING — pass 1 widths skipped)'}\n")

    cond_name = f"{PREFIX}_cond_{base}.png"
    cond_path = os.path.join(IN_DIR, cond_name)
    canvas, n_old = render_svg(src_svg, cond_path)
    print(f"[1] Conditioning image: {canvas.shape[1]}x{canvas.shape[0]}, {n_old} strokes -> {cond_name}")

    if args.stop_llama:
        try:
            from utils.llama_server import stop_server

            stop_server()
            print("[vram] llama-server stopped")
            time.sleep(3)
        except Exception as e:
            print(f"[vram] could not stop llama-server: {e}")
    used = os.popen("nvidia-smi --query-gpu=memory.used --format=csv,noheader").read().strip()
    print(f"[vram] in use: {used}")

    print(f"\n[2] Generating at {canvas.shape[1]}x{canvas.shape[0]}, conditioned on the centreline (lineart, depth bypassed)…")
    out_png = queue_and_wait(build_workflow(cond_name, args.prompt, args.seed, size=(canvas.shape[1], canvas.shape[0])))
    print(f"    -> {os.path.basename(out_png)}")

    print("\n[3] Centrelining the result with the real bcnc pass…")
    from bcnc import raster_to_centerline_svg

    new_svg = out_png.replace(".png", "_center_lined.svg")
    raster_to_centerline_svg(out_png, new_svg)

    print("\n=== STROKE WIDTH (does it come out more plottable?) ===")
    if os.path.exists(src_png):
        _, w1 = widths_mm(src_svg, src_png)
        print(f"  pass 1: median {np.median(w1):.2f}px  IQR {np.percentile(w1, 25):.2f}-{np.percentile(w1, 75):.2f}  n={len(w1)}")
    _, w2 = widths_mm(new_svg, out_png)
    print(f"  pass 2: median {np.median(w2):.2f}px  IQR {np.percentile(w2, 25):.2f}-{np.percentile(w2, 75):.2f}  n={len(w2)}")
    print("  (thinner + tighter IQR = closer to what one nib can actually say)")

    frac, n_new = preservation(src_svg, new_svg, canvas.shape)
    print("\n=== PRESERVATION (is an additive second pass possible?) ===")
    print(f"  {frac:.0%} of pass 2's {n_new} vertices land within 3px of pass 1's ink")
    print(f"  strokes: {n_old} -> {len(parse_polylines(new_svg))}")
    print("  high = geometry held, a clean diff can isolate what is NEW")
    print("  low  = it redrew rather than added; additive plotting is off the table")

    print(f"\nLook at them side by side:\n  {src_png}\n  {out_png}")


if __name__ == "__main__":
    main()
