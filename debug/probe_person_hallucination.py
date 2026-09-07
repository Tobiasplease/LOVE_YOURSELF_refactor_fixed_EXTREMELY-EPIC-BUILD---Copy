"""Probe (Sep 7): why does it write a man at the desk in an empty studio, and
does NAMING what is in view stop it? Runs the live server against frames the
machine actually hallucinated on. Machine may stay up. N samples per arm.
python debug/probe_person_hallucination.py [N=3]
"""
import base64
import json
import os
import re
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from captioner.prompt_registry import P  # noqa: E402

URL = os.getenv("LLAMA_SERVER_URL", "http://localhost:8080") + "/v1/chat/completions"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 3
PERSON = re.compile(r"\b(he|him|his|she|her|man|woman|person|someone|somebody|guy)\b", re.I)
OPTS = {"temperature": 0.75, "top_p": 1.0, "min_p": 0.05, "max_tokens": 70, "chat_template_kwargs": {"enable_thinking": False}}
FRAMES = [l.strip() for l in open("/tmp/hallu_frames.txt") if l.strip()]
SYSTEM = P("mind.system") + P("monologue.pen-parked")
TEXT = ("The lamp is a white hole again. It eats the right side of the room.\n\n"
        "The red thing is still up there, pointing at nothing.")


def ask(frame, cue):
    img = {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(open(frame, "rb").read()).decode()}}
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "assistant", "content": TEXT}, {"role": "user", "content": [img, {"type": "text", "text": cue}]}]
    req = urllib.request.Request(URL, data=json.dumps({"model": "p", "messages": msgs, "stream": False, **OPTS}).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.load(r)["choices"][0]["message"]["content"].strip()


NAMES = "the mannequin head, the wooden mannequin torso and the red foam finger"
ARMS = {
    "A bare clock (today's shape)": "22:41.",
    "B + what the registry knows is in view": f"22:41. What you know is in view: {NAMES}.",
    "C + the absence fact only": "22:41. No one is in the room; the last person left three hours ago.",
    "D + both": f"22:41. No one is in the room; the last person left three hours ago. What you know is in view: {NAMES}.",
}
CONTAM = ("He's under that shelf again. Just a dark shape hunched over the desk with all those wires hanging off it.\n\n"
          "Okay. He's there. Head bent low over the laptop. I've been looking at him for five minutes now.\n\n"
          "The red thing is still up there, pointing at nothing.")


def ask2(frame, cue, text):
    img = {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(open(frame, "rb").read()).decode()}}
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "assistant", "content": text}, {"role": "user", "content": [img, {"type": "text", "text": cue}]}]
    req = urllib.request.Request(URL, data=json.dumps({"model": "p", "messages": msgs, "stream": False, **OPTS}).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.load(r)["choices"][0]["message"]["content"].strip()


for _label, _cue in (("E its own text already says he is there", "22:41."), ("F same, plus the absence fact", "22:41. No one is in the room; the last person left three hours ago.")):
    _hits, _outs = 0, []
    for f in FRAMES:
        for _ in range(N):
            try:
                o = ask2(f, _cue, CONTAM)
            except Exception as e:  # noqa: BLE001
                o = f"[error {e}]"
            _outs.append(o)
            if PERSON.search(o):
                _hits += 1
    print(f"## {_label}: person-claims {_hits}/{len(_outs)}")
    for o in _outs[:3]:
        print("   ·", o.replace("\n", " / ")[:150])

for name, cue in ARMS.items():
    hits, outs = 0, []
    for f in FRAMES:
        for _ in range(N):
            try:
                o = ask(f, cue)
            except Exception as e:  # noqa: BLE001
                o = f"[error {e}]"
            outs.append(o)
            if PERSON.search(o):
                hits += 1
    print(f"## {name}: person-claims {hits}/{len(outs)}")
    for o in outs[:3]:
        print("   ·", o.replace("\n", " / ")[:150])
