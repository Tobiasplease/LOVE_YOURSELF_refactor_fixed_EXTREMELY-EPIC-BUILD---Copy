#!/usr/bin/env python3
"""
debug/test_live_video.py
------------------------
Live webcam video analysis via llama-server with temporal super-frames.
Continuously buffers frames, sends them as video every N seconds, prints the model's response.

Usage:
    python debug/test_live_video.py                    # Default: 8s buffer, 2fps
    python debug/test_live_video.py --interval 5       # Faster cycle (every 5s)
    python debug/test_live_video.py --fps 4            # More frames per second
    python debug/test_live_video.py --single            # Single-image mode (comparison baseline)

Press Ctrl+C to stop.
"""

import argparse
import base64
import json
import sys
import time

import cv2
import numpy as np
import requests

sys.path.insert(0, ".")

LLAMA_SERVER_URL = "http://localhost:8080"

SYSTEM_PROMPT = (
    "You are watching a room through a camera. "
    "Say what you see and what changed since last time. React to it — wonder about it, get curious, get bored. "
    "Only mention things actually visible. Do not make up details. "
    "Short. A few words or a sentence or two. First person."
)


class FrameRing:
    """Lightweight ring buffer for JPEG frames with timestamps."""

    def __init__(self, target_fps: float = 2.0, max_seconds: float = 30.0):
        self.min_interval = 1.0 / target_fps
        self.max_frames = int(target_fps * max_seconds)
        self.frames = []  # (timestamp, jpeg_bytes, diff_score)
        self.last_push = 0.0
        self.prev_gray = None

    def push(self, frame: np.ndarray) -> None:
        now = time.time()
        if now - self.last_push < self.min_interval:
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (160, 120))
        diff = 0.0
        if self.prev_gray is not None:
            diff = float(cv2.absdiff(small, self.prev_gray).mean()) / 255.0
        self.prev_gray = small

        _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        self.frames.append((now, jpeg.tobytes(), diff))
        if len(self.frames) > self.max_frames:
            self.frames = self.frames[-self.max_frames:]
        self.last_push = now

    def get_recent(self, seconds: float, max_frames: int = 16):
        cutoff = time.time() - seconds
        recent = [(ts, jpg, d) for ts, jpg, d in self.frames if ts >= cutoff]
        if len(recent) > max_frames:
            first, last = recent[0], recent[-1]
            middle = sorted(recent[1:-1], key=lambda x: x[2], reverse=True)
            recent = sorted([first] + middle[:max_frames - 2] + [last], key=lambda x: x[0])
        return recent


def query_video_superframe(frames_data, thought_thread: str = "", fps: float = 2.0) -> str:
    """Mode A: Super-frame pipeline (Conv3D + M-RoPE temporal encoding)."""
    try:
        from llama_video import Frame, Preprocessor, Settings
    except ImportError:
        print("[WARN] llama-video not installed, falling back to single image")
        return query_single(frames_data[-1][1], thought_thread)

    lv_frames = []
    for i, (ts, jpg, _) in enumerate(frames_data):
        nparr = np.frombuffer(jpg, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is not None:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            lv_frames.append(Frame(data=img_rgb, index=i, timestamp=ts, width=img.shape[1], height=img.shape[0]))

    if len(lv_frames) < 2:
        return query_single(frames_data[-1][1], thought_thread)

    settings = Settings()
    preprocessor = Preprocessor(settings.model)
    video_input = preprocessor.process(lv_frames, fps=fps)

    from llama_video.client import LlamaServerClient
    client = LlamaServerClient(settings.server)
    content = []
    for sf in video_input.super_frames:
        for img_url in client._super_frame_to_base64_pair(sf.data):
            content.append({"type": "image_url", "image_url": {"url": img_url}})

    prompt = _build_prompt(thought_thread)
    content.append({"type": "text", "text": prompt})

    payload = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
        "max_tokens": 80,
        "temperature": 0.85,
        "chat_template_kwargs": {"enable_thinking": False},
        "mm_processor_kwargs": {
            "fps": video_input.fps,
            "is_video": True,
            "grid_thw": list(video_input.grid_thw),
            "temporal_positions": video_input.temporal_positions,
        },
    }

    resp = requests.post(f"{LLAMA_SERVER_URL}/v1/chat/completions", json=payload, timeout=45)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"].get("content") or ""


def query_video_multi_image(frames_data, thought_thread: str = "") -> str:
    """Mode B: Plain multi-image (no super-frames, model infers temporality from sequence + prompt)."""
    images = []
    for _, jpg, _ in frames_data:
        img_b64 = base64.b64encode(jpg).decode()
        images.append(f"data:image/jpeg;base64,{img_b64}")

    content = []
    for img_url in images:
        content.append({"type": "image_url", "image_url": {"url": img_url}})

    n = len(frames_data)
    span = frames_data[-1][0] - frames_data[0][0]
    prompt = _build_prompt(thought_thread, temporal_hint=f"These {n} frames span {span:.0f} seconds, earliest to latest.")
    content.append({"type": "text", "text": prompt})

    payload = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
        "max_tokens": 80,
        "temperature": 0.85,
        "chat_template_kwargs": {"enable_thinking": False},
    }

    resp = requests.post(f"{LLAMA_SERVER_URL}/v1/chat/completions", json=payload, timeout=45)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"].get("content") or ""


def _build_prompt(thought_thread: str = "", temporal_hint: str = "") -> str:
    """Build the user prompt with optional thread and temporal hint."""
    parts = []
    if temporal_hint:
        parts.append(temporal_hint)
    if thought_thread:
        parts.append(f"...{thought_thread}")
        parts.append("What do you notice now?")
    else:
        parts.append("What do you notice?")
    return "\n".join(parts)


def query_single(jpeg_bytes: bytes, thought_thread: str = "") -> str:
    """Send a single frame (baseline comparison)."""
    img_b64 = base64.b64encode(jpeg_bytes).decode()

    prompt = _build_prompt(thought_thread)
    content = [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
        {"type": "text", "text": prompt},
    ]

    payload = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
        "stream": False,
        "temperature": 0.85,
        "max_tokens": 80,
        "chat_template_kwargs": {"enable_thinking": False},
    }

    resp = requests.post(f"{LLAMA_SERVER_URL}/v1/chat/completions", json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"].get("content", "")


def main():
    parser = argparse.ArgumentParser(description="Live webcam video analysis")
    parser.add_argument("--interval", type=float, default=8.0, help="Seconds between analysis cycles")
    parser.add_argument("--fps", type=float, default=2.0, help="Frame buffer capture rate")
    parser.add_argument("--max-frames", type=int, default=6, help="Max frames per video query")
    parser.add_argument("--mode", choices=["single", "superframe", "multi"], default="multi",
                        help="single=one frame, superframe=Conv3D pipeline, multi=plain multi-image (default)")
    parser.add_argument("--single", action="store_true", help="Shorthand for --mode single")
    parser.add_argument("--motion-threshold", type=float, default=0.015, help="Min avg diff to trigger video mode")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    args = parser.parse_args()

    # Check server
    try:
        r = requests.get(f"{LLAMA_SERVER_URL}/health", timeout=2)
        if not r.ok:
            raise Exception("not ok")
    except Exception:
        print("llama-server not running at", LLAMA_SERVER_URL)
        sys.exit(1)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("Could not open camera")
        sys.exit(1)

    if args.single:
        args.mode = "single"
    ring = FrameRing(target_fps=args.fps)
    mode_labels = {
        "single": "SINGLE-IMAGE",
        "superframe": f"SUPER-FRAME ({args.fps}fps, max {args.max_frames}f)",
        "multi": f"MULTI-IMAGE ({args.fps}fps, max {args.max_frames}f)",
    }
    print(f"\nLive analysis — {mode_labels[args.mode]}, cycle every {args.interval}s, motion threshold {args.motion_threshold}")
    print("Press Ctrl+C to stop.\n")
    print("-" * 60)

    thought_thread = ""
    cycle = 0
    last_analysis = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            ring.push(frame)
            now = time.time()

            if now - last_analysis < args.interval:
                time.sleep(0.03)
                continue

            last_analysis = now
            cycle += 1

            try:
                avg_diff = 0.0
                recent = ring.get_recent(args.interval, args.max_frames)
                if not recent:
                    continue
                diffs = [d for _, _, d in recent]
                avg_diff = sum(diffs) / len(diffs) if diffs else 0

                use_video = (args.mode != "single"
                             and avg_diff >= args.motion_threshold
                             and len(recent) >= 2)

                if not use_video:
                    # Single frame — either by choice or nothing moved
                    start = time.time()
                    response = query_single(recent[-1][1], thought_thread)
                    elapsed = time.time() - start
                    n_frames = 1
                elif args.mode == "superframe":
                    start = time.time()
                    response = query_video_superframe(recent, thought_thread, fps=args.fps)
                    elapsed = time.time() - start
                    n_frames = len(recent)
                else:  # multi
                    start = time.time()
                    response = query_video_multi_image(recent, thought_thread)
                    elapsed = time.time() - start
                    n_frames = len(recent)

                response = response.strip()
                if response:
                    # Build thread — keep only last 2 short sentences
                    first_sentence = response.split(".")[0] + "." if "." in response else response
                    if len(first_sentence) > 80:
                        first_sentence = first_sentence[:80].rsplit(" ", 1)[0] + "..."
                    thread_parts = thought_thread.split(". ")[-1:] if thought_thread else []
                    thread_parts.append(first_sentence)
                    thought_thread = ". ".join(thread_parts)
                    # Cap total thread length
                    if len(thought_thread) > 160:
                        thought_thread = thought_thread[-160:].split(". ", 1)[-1]

                    diff_str = f" diff={avg_diff:.3f}" if not args.single else ""
                    info = f"[#{cycle} | {n_frames}f | {elapsed:.1f}s{diff_str}]"
                    print(f"{info} {response}")
                else:
                    print(f"[#{cycle}] (empty response)")

            except Exception as e:
                print(f"[#{cycle}] Error: {e}")

    except KeyboardInterrupt:
        print("\n\nStopped.")
    finally:
        cap.release()


if __name__ == "__main__":
    main()
