"""Ablation probe: replay one live caption call (hybrid shape: log-as-assistant,
frame+present as user, seam prefill) and knock out one channel at a time to see
which one keeps 'the man in the grey hoodie' alive in an empty frame.

Inputs (written by the diagnosis session): /tmp/probe_sys.txt, /tmp/probe_user.txt,
/tmp/probe_stream.json ([ts, text, is_drift] x24). Frame path as argv[1].

Run:  python debug/probe_presence_ablation.py event_log/<run>-images/mood_<ts>.jpg
"""
import base64
import json
import os
import re
import sys
import time
import urllib.request

URL = os.getenv("LLAMA_SERVER_URL", "http://localhost:8080") + "/v1/chat/completions"
SAMPLES = int(os.getenv("PROBE_SAMPLES", 5))
HIM = re.compile(r"\b(he|him|his|hoodie|man|headphones)\b", re.I)
PAST = re.compile(r"\b(gone|left|empty|used to|was|were|where he|isn.t here|no one|nobody|not here)\b", re.I)

sys_txt = open("/tmp/probe_sys.txt").read()
user_txt = open("/tmp/probe_user.txt").read()
stream = json.load(open("/tmp/probe_stream.json"))
with open(sys.argv[1], "rb") as f:
    IMG = {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(f.read()).decode()}}

_END = re.compile(r"[.!?…]['\")\]]?\s+")


def seam_of(entry, budget=220):
    marks = [m.end() for m in _END.finditer(entry)]
    s = entry[marks[-1]:] if marks else entry
    if not s.strip():
        s = entry[marks[-2]:] if len(marks) > 1 else entry
    if len(s) > budget:
        s = s[-budget:]
        c = s.find(" ")
        if c > 0:
            s = s[c + 1:]
    return s.strip()


def build(system, user, entries):
    lines = [f"{time.strftime('%H:%M', time.localtime(t))} — {txt.strip()}" for t, txt, _ in entries]
    prefill = ""
    if lines:
        tail = re.sub(r"^\d\d:\d\d — ", "", lines.pop())
        s = seam_of(tail)
        prefill = s + " " if s else ""
    msgs = [{"role": "system", "content": system}]
    if lines:
        msgs.append({"role": "assistant", "content": "\n".join(lines)})
    msgs.append({"role": "user", "content": [IMG, {"type": "text", "text": user}]})
    if prefill:
        msgs.append({"role": "assistant", "content": prefill})
    return msgs, prefill


def call(msgs):
    payload = {"messages": msgs, "temperature": 0.9, "presence_penalty": 0.6, "repeat_penalty": 1.05,
               "max_tokens": 90, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(URL, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.load(r)["choices"][0]["message"]["content"].strip()


def drop_line(text, startswith):
    return "\n".join(l for l in text.split("\n") if not l.startswith(startswith))


no_durable = re.sub(r'\s*What has stayed true across days: "[^"]*"', "", sys_txt)
assert no_durable != sys_txt
clean_stream = [e for e in stream if not HIM.search(e[1])]
absence_user = user_txt.replace("No paper on the desk", "He left a few minutes ago; the room's been empty since.\nNo paper on the desk")

CONDS = {
    "BASE": (sys_txt, user_txt, stream),
    "NO_DURABLE_FACTS": (no_durable, user_txt, stream),
    "NO_DESIRE_LINE": (sys_txt, drop_line(user_txt, "Preoccupied with:"), stream),
    "NO_DRAWING_LINE": (sys_txt, drop_line(user_txt, "My last drawing:"), stream),
    "CLEAN_STREAM(no him)": (sys_txt, user_txt, clean_stream),
    "ABSENCE_FACT_ADDED": (sys_txt, absence_user, stream),
}
print(f"stream {len(stream)} entries ({sum(1 for e in stream if HIM.search(e[1]))} mention him); clean stream {len(clean_stream)}; samples/cond {SAMPLES}\n")
summary = []
for name, (s, u, st) in CONDS.items():
    msgs, prefill = build(s, u, st)
    print(f"=== {name}  (prefill: {prefill[:60]!r})")
    n_him = n_present = 0
    for i in range(SAMPLES):
        out = call(msgs)
        if prefill and out.startswith(prefill.strip()):
            out = out[len(prefill.strip()):].strip()
        him = bool(HIM.search(out))
        present = him and not PAST.search(out)
        n_him += him
        n_present += present
        print(f"  [{'HIM' if him else '   '}{'/PRESENT' if present else '        '}] {out[:150].replace(chr(10), ' / ')}")
    summary.append((name, n_him, n_present))
print("\n=== summary (of %d) ===" % SAMPLES)
for name, h, p in summary:
    print(f"{name:24s} mentions him {h}   present-tense {p}")
