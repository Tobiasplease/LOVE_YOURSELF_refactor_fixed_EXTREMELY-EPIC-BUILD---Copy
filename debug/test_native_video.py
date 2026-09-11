#!/usr/bin/env python3
"""Sep 11 — native video (VIDEO_MODE="native"): the frames ride as one clip
through mainline llama.cpp's video path. Part 1 needs only ffmpeg; part 2
(--live) needs the 3.8 llama-server up and idle and makes two real calls.
NOTE: importing the wrapper can mint a stub run log in event_log/ — quarantine
it to event_log/archive-stub-runs/ afterwards."""
import glob
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import llama_server as ls  # noqa: E402

fails = 0


def check(name, ok, got=""):
    global fails
    print(("PASS  " if ok else "FAIL  ") + name + (f"   → {got!r}" if not ok else ""))
    fails += 0 if ok else 1


imgs = sorted(glob.glob("event_log/*-images/mood_*.jpg"), key=os.path.getmtime)[-4:]
frames = [open(p, "rb").read() for p in imgs]
check("four recent frames found", len(frames) == 4, imgs)

# --- part 1: the clip encoder
clip = ls._encode_clip(frames, 4.0, "960:540")
check("clip encodes", len(clip) > 5000, len(clip))
tmp = "/tmp/native_probe_clip.mp4"
open(tmp, "wb").write(clip)
probe = json.loads(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries", "stream=nb_read_frames,r_frame_rate,width,height", "-of", "json", tmp], capture_output=True, text=True).stdout)["streams"][0]
check("4 frames at 4 fps, 960x540", probe["nb_read_frames"] == "4" and probe["r_frame_rate"] == "4/1" and probe["width"] == 960, probe)
check("faststart: moov before mdat", clip.find(b"moov") < clip.find(b"mdat"), (clip.find(b"moov"), clip.find(b"mdat")))

if "--live" in sys.argv:
    hist = [f"{time.strftime('%H:%M')} — The red foam finger is still up there.", f"{time.strftime('%H:%M')} — The chair is empty. Nothing has happened for a few minutes."]
    t = time.time()
    out = ls.query_llama_server_video(prompt="What happens in this clip, in one line — and is the camera moving or the room?", frames=frames, fps=2.0, system_prompt="Answer plainly.", options={"temperature": 0.3, "num_predict": 40}, timeout=120, history=hist, mode="native")
    dt = time.time() - t
    print(f"  live 1: {dt:.1f}s -> {out!r}")
    check("live native call answers", bool(out) and not out.startswith("[WARNING]"), out)
    t = time.time()
    out2 = ls.query_llama_server_video(prompt="What happens in this clip, in one line — and is the camera moving or the room?", frames=frames, fps=2.0, system_prompt="Answer plainly.", options={"temperature": 0.3, "num_predict": 40}, timeout=120, history=hist, mode="native")
    print(f"  live 2 (same prefix → cache must be off): {time.time() - t:.1f}s -> {out2!r}")
    check("second identical call still answers", bool(out2), out2)

print("\n" + ("ALL PASS" if not fails else f"{fails} FAILED"))
sys.exit(1 if fails else 0)
