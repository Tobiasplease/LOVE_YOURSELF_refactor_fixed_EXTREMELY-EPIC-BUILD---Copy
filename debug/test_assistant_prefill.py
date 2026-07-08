#!/usr/bin/env python3
"""Probe llama-server assistant prefill: does a trailing assistant message get
CONTINUED (document mode) instead of answered?

Runs small requests against the live server. Safe to run alongside machine.py
(tiny max_tokens; requests just queue behind the caption loop).

Usage: python debug/test_assistant_prefill.py
"""
import base64
import io
import json
import sys

import requests

SERVER = "http://localhost:8080"

SYSTEM = (
    "You are a drawing machine bolted to a table in a workshop. You look around "
    "the room by turning your gaze. This is your inner voice, ongoing — plain, "
    "half-formed. A sentence or two."
)

PREFILL = (
    "The hum returns, a low vibration in the steel frame. The mannequin heads "
    "haven't moved, but the light across them has. I keep coming back to the "
    "scar on the floor — it holds the dust differently. "
)


def tiny_jpeg() -> str:
    """A 64x64 gray JPEG, base64-encoded (needs numpy+cv2 from the venv)."""
    import cv2
    import numpy as np

    img = np.full((64, 64, 3), 128, dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return base64.b64encode(buf.tobytes()).decode()


def call(name, messages, extra=None, max_tokens=40):
    payload = {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "cache_prompt": True,
    }
    if extra:
        payload.update(extra)
    r = requests.post(f"{SERVER}/v1/chat/completions", json=payload, timeout=120)
    print(f"\n=== {name} — HTTP {r.status_code}")
    if r.status_code != 200:
        print("  body:", r.text[:300])
        return None
    content = r.json()["choices"][0]["message"]["content"]
    print(f"  continuation: {content!r}")
    return content


def main():
    health = requests.get(f"{SERVER}/health", timeout=5).json()
    print("server health:", health)

    # 1. Text-only prefill, thinking disabled (expected to work)
    call(
        "1. text prefill + enable_thinking=false",
        [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": "Nothing new out there. The thought goes on."},
            {"role": "assistant", "content": PREFILL},
        ],
        extra={"chat_template_kwargs": {"enable_thinking": False}},
    )

    # 2. Same WITHOUT disabling thinking (documents the old failure, if any)
    call(
        "2. text prefill, thinking left at default",
        [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": "Nothing new out there. The thought goes on."},
            {"role": "assistant", "content": PREFILL},
        ],
    )

    # 3. Image in the user turn + trailing assistant prefill
    img = tiny_jpeg()
    call(
        "3. image + text prefill",
        [
            {"role": "system", "content": SYSTEM},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}},
                    {"type": "text", "text": "You glance up. Anything change?"},
                ],
            },
            {"role": "assistant", "content": PREFILL},
        ],
        extra={"chat_template_kwargs": {"enable_thinking": False}},
    )

    # 4. Two images (multi-frame, like the video fallback path) + prefill
    call(
        "4. two frames + text prefill",
        [
            {"role": "system", "content": SYSTEM},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}},
                    {"type": "text", "text": "You're seeing the last few seconds."},
                ],
            },
            {"role": "assistant", "content": PREFILL},
        ],
        extra={"chat_template_kwargs": {"enable_thinking": False}},
    )

    # 5. Control: same context WITHOUT prefill — how different is the output?
    call(
        "5. control, no prefill",
        [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": "Nothing new out there. The thought goes on."},
        ],
        extra={"chat_template_kwargs": {"enable_thinking": False}},
    )


if __name__ == "__main__":
    sys.exit(main())
