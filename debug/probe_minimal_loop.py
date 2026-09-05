"""Minimal-loop control (Sep 5 eve): the early-system shape — a frame, a tiny
system line, and the machine's own previous answers as real chat turns.
No window, no status lines, no stamps, no ledgers. Does the plain loop wander
wider than the scaffolded one? Run: python debug/probe_minimal_loop.py [turns=6]
"""
import base64
import glob
import json
import os
import re
import sys
import urllib.request

URL = os.getenv("LLAMA_SERVER_URL", "http://localhost:8080") + "/v1/chat/completions"
TURNS = int(sys.argv[1]) if len(sys.argv) > 1 else 6
OPTS = {"temperature": 0.9, "top_p": 1.0, "min_p": 0.05, "presence_penalty": 0.6, "repeat_penalty": 1.05, "max_tokens": 70,
        "chat_template_kwargs": {"enable_thinking": False}}
ROOM = set("finger lamp desk pen paper wall chair curtain light wood cloth plastic hole shelf shelves dust wire wires laptop foam red white black grey gray stain ceiling floor room".split())


def chat(msgs):
    payload = {"model": "probe", "messages": msgs, "stream": False, **OPTS}
    req = urllib.request.Request(URL, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.load(r)["choices"][0]["message"]["content"].strip()


log = max(glob.glob("event_log/*-event-log.json"), key=os.path.getmtime)
run = os.path.basename(log).split("-")[0]
frame = sorted(glob.glob(f"event_log/{run}-images/*.jpg"))[-1]
IMG = {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(open(frame, "rb").read()).decode()}}
SYSTEM = "You are the mind of a small camera-and-pen machine on a desk in a studio. Think aloud, to yourself, a sentence or two at a time."
msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": [IMG, {"type": "text", "text": "What's on your mind?"}]}]
outs = []
for k in range(TURNS):
    t = chat(msgs)
    outs.append(t)
    msgs.append({"role": "assistant", "content": t})
    msgs.append({"role": "user", "content": [IMG, {"type": "text", "text": "And now?"}]})
words = [w for t in outs for w in re.findall(r"[a-z']+", t.lower())]
print(f"## minimal loop ({os.path.basename(frame)}): room%={round(100 * sum(w in ROOM for w in words) / max(1, len(words)))} words/turn={round(len(words) / TURNS, 1)}")
for i, t in enumerate(outs):
    print(f"  {i + 1}. {t.replace(chr(10), ' / ')[:230]}")
