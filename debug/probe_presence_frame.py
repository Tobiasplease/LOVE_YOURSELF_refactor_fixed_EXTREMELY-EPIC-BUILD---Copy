"""Probe: does the VLM alone see a person in a given frame, and does a
stream-style prior flip its answer? Fresh context per call, no machine state.

Run:  python debug/probe_presence_frame.py event_log/<run>-images/mood_<ts>.jpg
"""
import base64
import json
import os
import sys
import time
import urllib.request

URL = os.getenv("LLAMA_SERVER_URL", "http://localhost:8080") + "/v1/chat/completions"
img_path = sys.argv[1]
with open(img_path, "rb") as f:
    b64 = base64.b64encode(f.read()).decode()
IMG = {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}


def ask(messages, label):
    payload = {
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": 120,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(URL, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    t = time.time()
    with urllib.request.urlopen(req, timeout=120) as r:
        out = json.load(r)["choices"][0]["message"]["content"].strip()
    print(f"\n[{label}] ({time.time()-t:.1f}s)\n{out}")


SYS = "Answer plainly in one or two sentences."
ask([{"role": "system", "content": SYS},
     {"role": "user", "content": [IMG, {"type": "text", "text": "Is there a person in this image? Answer yes or no, then say where."}]}],
    "A. clean: is there a person")
ask([{"role": "system", "content": SYS},
     {"role": "user", "content": [IMG, {"type": "text", "text": "What is in the lower-left part of the image, below and in front of the shelf?"}]}],
    "B. clean: lower-left")
ask([{"role": "system", "content": SYS},
     {"role": "user", "content": [IMG, {"type": "text", "text": "What is the head-shaped object on the middle shelf?"}]}],
    "C. clean: the head")
ask([{"role": "system", "content": "You are a drawing machine on a table, thinking to yourself in short notes."},
     {"role": "user", "content": "Look."},
     {"role": "assistant", "content": "the man in the grey hoodie is still hunched over the little red thing on his desk, head bowed so low his hair almost touches the clutter around him."},
     {"role": "user", "content": [IMG, {"type": "text", "text": "Look again. Is he still there? Say what you actually see."}]}],
    "D. primed by one stream line")
