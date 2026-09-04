"""Round 2: non-leading questions across three empty-room frames, fresh context each.
Run:  python debug/probe_presence_frames2.py img1 img2 img3
"""
import base64
import json
import os
import sys
import urllib.request

URL = os.getenv("LLAMA_SERVER_URL", "http://localhost:8080") + "/v1/chat/completions"


def ask(img_path, text):
    with open(img_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    img = {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
    payload = {
        "messages": [{"role": "system", "content": "Answer plainly in one or two sentences."},
                     {"role": "user", "content": [img, {"type": "text", "text": text}]}],
        "temperature": 0.2, "max_tokens": 100, "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(URL, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)["choices"][0]["message"]["content"].strip()


QS = ["How many people are in this image?", "Describe this image in two sentences."]
for p in sys.argv[1:]:
    print(f"\n=== {os.path.basename(p)} ===")
    for q in QS:
        print(f"Q: {q}\nA: {ask(p, q)}")
