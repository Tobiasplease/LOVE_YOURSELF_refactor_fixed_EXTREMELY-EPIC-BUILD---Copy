"""DSV hybrid centerline engine (Aug 12 2026).

Deep Sketch Vectorization (SIGGRAPH 2024, MIT, vendored with its own venv +
weights at DSV_HOME) recovers scribble and hatching as REAL individual
strokes — the line-logic fluency skeleton-walking can't produce from fused
marks — but is blind to tone and worm-mazes solid areas. Its strengths are
exactly complementary to the tone-fill renderer, so the hybrid routes by
split_fills: ink MASSES -> tone_fill_polylines, the STROKE layer -> DSV.

Runs in the post-ComfyUI / pre-GRBL slot, before finish_drawing_generation
releases llama-server, so the GPU holds only ComfyUI's cache — which is
freed first (ComfyUI reloads its models every generation anyway, so this
costs nothing). Measured: ~24s on GPU even with 19GB held; ~7min CPU
fallback. Every failure falls through to the v2 skeleton walk — a drawing
must never die from this experiment.
"""
import os
import re
import shutil
import subprocess
import tempfile

import cv2
import numpy as np

DSV_HOME = os.getenv("DSV_HOME", "/home/impostor/Deep-Sketch-Vectorization")
DSV_PYTHON = os.path.join(DSV_HOME, ".venv", "bin", "python")
DSV_GPU_TIMEOUT_S = 180
DSV_CPU_TIMEOUT_S = 240  # downscaled-512 CPU pass measures ~2min; beyond this the skeleton walk is the better outcome


def dsv_available() -> bool:
    return os.path.exists(os.path.join(DSV_HOME, "predict_s1.py")) and os.path.exists(DSV_PYTHON)


def _parse_dsv_svg(svg_path: str):
    """DSV emits absolute M/L polyline paths. Returns (polys, canvas_long_edge).
    CAUTION (Aug 12, learned live): DSV's output canvas is 2x its PROCESSING
    resolution — for a 1024 input (internally resized to 512) that lands back
    on 1024 by coincidence, but for a pre-downscaled 512 input the output is
    ALSO 1024. Never assume; scale by the declared canvas."""
    txt = open(svg_path).read()
    m = re.search(r'viewBox="0 0 ([\d.]+) ([\d.]+)"', txt) or re.search(r'width="([\d.]+)".*?height="([\d.]+)"', txt)
    canvas_long = max(float(m.group(1)), float(m.group(2))) if m else None
    polys = []
    for d in re.findall(r'd="([^"]+)"', txt):
        cur = []
        for cmd, x, y in re.findall(r"([ML])\s*([-\d.eE]+),([-\d.eE]+)", d):
            if cmd == "M":
                if len(cur) > 1:
                    polys.append(np.array(cur, dtype=np.float64))
                cur = [(float(x), float(y))]
            else:
                cur.append((float(x), float(y)))
        if len(cur) > 1:
            polys.append(np.array(cur, dtype=np.float64))
    return polys, canvas_long


def _subprocess_env() -> dict:
    """ultralytics sets a process-global CUDA_VISIBLE_DEVICES="-1" to keep
    YOLO on CPU, which the DSV child would inherit — torch then sees no GPU
    and the cuda attempt fails instantly (first live run, Aug 12: every
    drawing fell to the 7-minute CPU path). Same cure llama_server uses:
    hand the child the visibility this process STARTED with."""
    env = dict(os.environ)
    try:
        from utils.llama_server import _PRISTINE_CUDA_VISIBLE_DEVICES as pristine

        if pristine is None:
            env.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            env["CUDA_VISIBLE_DEVICES"] = pristine
    except Exception:
        env.pop("CUDA_VISIBLE_DEVICES", None)
    return env


def _wait_for_vram(min_free_mb: int = 4500, timeout_s: float = 15.0) -> None:
    """ComfyUI's /free returns before VRAM actually drops; launching torch
    into a still-occupied card OOMs (the 22:47 race, Aug 12 — DSV needs
    ~3.5GB). Poll until enough is free or the timeout passes; the cuda
    attempt then either fits or fails fast into the cpu path."""
    import time

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                                 capture_output=True, timeout=5, text=True)
            if int(out.stdout.split()[0]) >= min_free_mb:
                return
        except Exception:
            return
        time.sleep(1.0)


def _free_comfyui() -> None:
    """Best-effort: drop ComfyUI's cached models so DSV gets the GPU. Safe in
    the drawing slot — ComfyUI reloads models at every generation regardless."""
    try:
        from utils.llama_server import _free_comfyui_vram

        _free_comfyui_vram()
    except Exception:
        pass


def dsv_stroke_polylines(stroke_input: np.ndarray, free_gpu: bool = True, thin: bool = True) -> list:
    """Run strokes through DSV; returns polylines in image pixels.
    Raises on failure so the caller can fall back to the skeleton walk.

    stroke_input: either a bool mask (the hybrid's stroke layer — binarized
    by necessity, split_fills needs binary) or a uint8 GRAYSCALE image (the
    pure engine — DSV's distance field works straight off the gray; feeding
    it the binarize() output re-introduces the thick lossy raster layer the
    artist flagged Aug 12, and the eval verdict was earned on raw grayscale).

    thin: line-thinning preprocess. True for the hybrid's stroke layer (thick
    marker outlines fragment without it). False for the pure engine — the
    un-thinned whole-image output is the stroke-elegant reduction the artist
    judged "by far the best result" (Aug 12)."""
    tmp = tempfile.mkdtemp(prefix="dsv_")
    try:
        png = os.path.join(tmp, "strokes.png")
        if stroke_input.dtype == np.bool_:
            cv2.imwrite(png, np.where(stroke_input, 0, 255).astype(np.uint8))
        else:
            cv2.imwrite(png, stroke_input)
        base_cmd = [DSV_PYTHON, "predict_s1.py", "-i", tmp, "-o", tmp, "--refine", "--rdp"]
        if thin:
            base_cmd.append("--thin")
        out_svg = os.path.join(tmp, "svg_full", "strokes_final.svg")
        env = _subprocess_env()
        if free_gpu:
            _free_comfyui()
        orig_long = max(stroke_input.shape[:2])
        # The artist's preference (Aug 12): wait longer for the REAL engine
        # rather than fall back — two cuda attempts, each behind a fresh
        # VRAM wait, before cpu is even considered. Every first-night
        # fallback traced to a fixable conflict, not to DSV itself.
        cuda_ok = False
        for attempt in (1, 2):
            _wait_for_vram()
            try:
                subprocess.run(base_cmd + ["-d", "cuda"], cwd=DSV_HOME, timeout=DSV_GPU_TIMEOUT_S,
                               check=True, capture_output=True, env=env)
                cuda_ok = True
                break
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                err_tail = ""
                if isinstance(e, subprocess.CalledProcessError) and e.stderr:
                    err_tail = e.stderr.decode(errors="ignore")[-400:]
                print(f"[dsv] cuda attempt {attempt} failed ({type(e).__name__}); stderr tail: {err_tail!r}")
                shutil.rmtree(os.path.join(tmp, "svg_full"), ignore_errors=True)
        if not cuda_ok:
            # CPU fallback. predict_s1 ignores its own resize on cpu (full-res
            # 1024 took ~7min); pre-downscale to 512 for speed.
            print("[dsv] both cuda attempts failed — cpu fallback (downscaled)")
            src = cv2.imread(png, cv2.IMREAD_GRAYSCALE)
            if max(src.shape) > 512:
                f = max(src.shape) / 512.0
                small = cv2.resize(src, (int(src.shape[1] / f), int(src.shape[0] / f)),
                                   interpolation=cv2.INTER_AREA)
                cv2.imwrite(png, small)
            subprocess.run(base_cmd + ["-d", "cpu"], cwd=DSV_HOME, timeout=DSV_CPU_TIMEOUT_S,
                           check=True, capture_output=True, env=env)
        polys, canvas_long = _parse_dsv_svg(out_svg)
        if not polys:
            raise RuntimeError("DSV produced no strokes")
        # Scale by DSV's own declared canvas — its output space is 2x its
        # processing resolution, so never assume it matches the input.
        if canvas_long and abs(canvas_long - orig_long) > 1:
            polys = [p * (orig_long / canvas_long) for p in polys]
        return polys
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
