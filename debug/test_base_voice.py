#!/usr/bin/env python3
"""
debug/test_base_voice.py
------------------------
Judge the NAKED base voice (config.BASE_VOICE_DETOX) on the live studio scene,
using the REAL prompt pipeline + REAL model, without the GUI/servo/CNC.

For each cycle it builds the actual detox prompt the caption loop would send:
  SYSTEM = get_monologue_system_prompt(mode)        # situation + genre + elicitation
  USER   = build_situational_line(agent, ...)       # time + gaze + sticky presence
and queries llama-server with a temporal super-frame (multi-image fallback) at
the real caption temperature (0.7). Prints prompt + caption so the register and
the studio-object perception can be read directly.

Prereq: llama-server up on :8080 (Qwen3.5-9B + mmproj).
Usage:
    python debug/test_base_voice.py                 # 6 cycles, relational+observational
    python debug/test_base_voice.py --cycles 10
    python debug/test_base_voice.py --mode observational
"""

import argparse
import base64
import os
import sys
import time
import types

import cv2
import numpy as np
import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

LLAMA_SERVER_URL = "http://localhost:8080"

from captioner.prompts import get_monologue_system_prompt, build_situational_line  # noqa: E402
from config import config  # noqa: E402

CAPTION_OPTS = {"temperature": 0.7, "top_p": 0.85, "repeat_penalty": 1.15}


def make_agent():
    """Minimal but faithful stand-in for the captioner: enough state for the
    detox situational line (session time + sticky presence belief)."""
    now = time.time()
    a = types.SimpleNamespace()
    a.true_session_start = now - 120          # "Been watching 2 minutes"
    a._presence_believed = True               # someone/something is in view
    a._presence_seen_now = True
    a._presence_since = now - 95
    a._presence_last_seen = now
    a._salience_hot = False
    return a


def capture(cap, seconds=8.0, fps=2.0):
    frames = []
    n = max(2, int(seconds * fps))
    for _ in range(n):
        ok, frame = cap.read()
        if not ok:
            break
        ts = time.time()
        _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        frames.append((ts, jpeg.tobytes()))
        time.sleep(1.0 / fps)
    return frames


def query(frames, system_prompt, user_prompt):
    """Super-frame if llama-video is available, else plain multi-image."""
    content = []
    payload_extra = {}
    try:
        from llama_video import Frame, Preprocessor, Settings
        from llama_video.client import LlamaServerClient
        lv = []
        for i, (ts, jpg) in enumerate(frames):
            img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
            if img is not None:
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                lv.append(Frame(data=rgb, index=i, timestamp=ts, width=img.shape[1], height=img.shape[0]))
        settings = Settings()
        vi = Preprocessor(settings.model).process(lv, fps=2.0)
        client = LlamaServerClient(settings.server)
        for sf in vi.super_frames:
            for url in client._super_frame_to_base64_pair(sf.data):
                content.append({"type": "image_url", "image_url": {"url": url}})
        payload_extra = {"mm_processor_kwargs": {
            "fps": vi.fps, "is_video": True, "grid_thw": list(vi.grid_thw),
            "temporal_positions": vi.temporal_positions}}
    except Exception as e:
        print(f"[harness] super-frame unavailable ({e}); using multi-image")
        content = []
        for _, jpg in frames:
            b64 = base64.b64encode(jpg).decode()
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})

    content.append({"type": "text", "text": user_prompt})
    payload = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ],
        "max_tokens": 80,
        "chat_template_kwargs": {"enable_thinking": False},
        **CAPTION_OPTS, **payload_extra,
    }
    r = requests.post(f"{LLAMA_SERVER_URL}/v1/chat/completions", json=payload, timeout=60)
    r.raise_for_status()
    return (r.json()["choices"][0]["message"].get("content") or "").strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycles", type=int, default=6)
    ap.add_argument("--mode", default="alternate",
                    help="relational | observational | introspective | alternate")
    ap.add_argument("--device", type=int, default=0)
    args = ap.parse_args()

    print(f"BASE_VOICE_DETOX = {getattr(config, 'BASE_VOICE_DETOX', False)}")
    try:
        requests.get(f"{LLAMA_SERVER_URL}/health", timeout=3)
    except Exception:
        print("llama-server not reachable on :8080 — start it first."); return

    cap = cv2.VideoCapture(args.device)
    if not cap.isOpened():
        print(f"camera {args.device} not available"); return

    agent = make_agent()
    modes = ["relational", "observational"]
    for i in range(args.cycles):
        mode = modes[i % len(modes)] if args.mode == "alternate" else args.mode
        # refresh presence "now" so the situational line stays current
        agent._presence_last_seen = time.time()
        frames = capture(cap, seconds=8.0, fps=2.0)
        if len(frames) < 2:
            print("no frames captured"); break
        system_prompt = get_monologue_system_prompt(mode, agent=agent)
        user_prompt = build_situational_line(agent, gaze_direction="ahead", gaze_state="aware")
        try:
            caption = query(frames, system_prompt, user_prompt)
        except Exception as e:
            caption = f"[query error: {e}]"
        print("\n" + "=" * 78)
        print(f"CYCLE {i+1}/{args.cycles}  mode={mode}  frames={len(frames)}")
        print("-" * 78)
        print(f"SYSTEM: {system_prompt}")
        print(f"USER:   {user_prompt}")
        print("-" * 78)
        print(f"CAPTION: {caption}")

    cap.release()
    print("\n" + "=" * 78 + "\nDone.")


if __name__ == "__main__":
    main()
