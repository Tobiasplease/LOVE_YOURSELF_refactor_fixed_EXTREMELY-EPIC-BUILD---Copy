"""Sep 13: reproduce the YOLO fallback failure seen after the 11:22 relaunch.

The detection thread catches a CUDA OOM, sets force_cpu and calls
model.track(..., device="cpu") — but the pane showed the same OOM line 160+
times, i.e. the CPU retry itself keeps failing. Run this while the card is
full (llama-server at 22.5 GB) to see where the CPU path breaks and whether
resetting the cached predictor fixes it. Pass a frame path, else the newest
caption image is used.
"""
import glob
import os
import sys
import time

import traceback

import cv2
import torch
from ultralytics import YOLO

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import YOLO_MODEL_PATH  # noqa: E402

CPU_FIRST = "--cpu-first" in sys.argv
TRACE = "--trace" in sys.argv
args = [a for a in sys.argv[1:] if not a.startswith("--")]
frame_path = args[0] if args else sorted(glob.glob("event_log/**/*.jpg", recursive=True), key=os.path.getmtime)[-1]
frame = cv2.imread(frame_path)
print("frame", frame_path, frame.shape, "| model", YOLO_MODEL_PATH)
print("cuda available:", torch.cuda.is_available())
m = YOLO(YOLO_MODEL_PATH)


def attempt(label, **kw):
    t = time.time()
    try:
        r = m.track(frame, persist=True, tracker="config/bytetrack_custom.yaml", verbose=False, imgsz=512, **kw)[0]
        dev = next(m.predictor.model.model.parameters()).device if m.predictor is not None and m.predictor.model is not None else "?"
        print(f"{label}: OK {len(r.boxes)} boxes in {time.time() - t:.2f}s, predictor model on {dev}")
        return True
    except Exception as e:
        print(f"{label}: {type(e).__name__}: {str(e)[:160]!r}")
        if TRACE:
            tb = [f for f in traceback.extract_tb(e.__traceback__) if "site-packages" in f.filename or "debug/" in f.filename]
            for f in tb[-8:]:
                print(f"      {f.filename.split('site-packages/')[-1]}:{f.lineno} {f.name}: {f.line}")
        print(f"   predictor set: {m.predictor is not None}; predictor.model: {getattr(m.predictor, 'model', None) is not None}")
        return False


if "--recover-test" in sys.argv:
    # The design under test: a CPU model already running; a SEPARATE object tries
    # CUDA and fails; does the CPU model keep working afterwards?
    attempt("R1 cpu model works", device="cpu")
    cand = YOLO(YOLO_MODEL_PATH)
    print("   fresh load lands on:", next(cand.model.parameters()).device)
    try:
        cand.model.to("cuda")
        print("R2 candidate moved to cuda (room appeared?)")
    except Exception as e:
        print(f"R2 candidate cuda move failed as expected: {type(e).__name__}: {str(e)[:60]!r}")
    del cand
    try:
        torch.cuda.empty_cache()
        print("   empty_cache ok; free MiB now", torch.cuda.mem_get_info()[0] // 2**20)
    except Exception as e:
        print(f"   empty_cache/mem_get_info: {type(e).__name__}: {str(e)[:60]!r}")
    attempt("R3 cpu model still works after the failed cuda attempt", device="cpu")
    attempt("R4 and again", device="cpu")
    sys.exit(0)
if CPU_FIRST:
    attempt("0 device=cpu in a fresh process (no failed CUDA attempt before it)", device="cpu")
    attempt("0b device=cpu again", device="cpu")
    sys.exit(0)
attempt("1 default device (what the thread does first)")
attempt("2 device=cpu, predictor kept (what the thread does after OOM)")
attempt("3 device=cpu again")
m.predictor = None
attempt("4 device=cpu after predictor reset (proposed fix)")
attempt("5 device=cpu again, persist")
