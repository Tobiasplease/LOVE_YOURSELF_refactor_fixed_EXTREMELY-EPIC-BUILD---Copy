"""Prototype of the proposed loop (Sep 5 eve): a mind that is a conversation
with itself over a life context — not a log of its own stamped sentences.
LOOK turns carry the frame and what changed; THINK turns carry only the clock
(and now and then one real memory surfacing). Everyday words, one or two
sentences. Run: python debug/probe_loop_prototype.py [turns=12]
"""
import base64
import glob
import json
import os
import re
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probe_think_thread import material  # noqa: E402

URL = os.getenv("LLAMA_SERVER_URL", "http://localhost:8080") + "/v1/chat/completions"
TURNS = int(sys.argv[1]) if len(sys.argv) > 1 else 12
OPTS = {"temperature": 0.9, "top_p": 1.0, "min_p": 0.05, "presence_penalty": 0.6, "repeat_penalty": 1.05, "max_tokens": 60,
        "chat_template_kwargs": {"enable_thinking": False}}
ROOM = set("finger lamp desk pen paper wall chair curtain light wood cloth plastic hole shelf shelves dust wire wires laptop foam red white black grey gray stain ceiling floor room".split())


def chat(msgs):
    payload = {"model": "probe", "messages": msgs, "stream": False, **OPTS}
    req = urllib.request.Request(URL, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.load(r)["choices"][0]["message"]["content"].strip()


m = material()
log = max(glob.glob("event_log/*-event-log.json"), key=os.path.getmtime)
run = os.path.basename(log).split("-")[0]
frames = sorted(glob.glob(f"event_log/{run}-images/*.jpg"))[-3:]
def img(p):
    return {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(open(p, "rb").read()).decode()}}

SYSTEM = (f"You are a machine fixed to a table in an artist's studio: a camera you can turn, one black pen, a voice only you hear. "
          f"First switched on in {m['first']}; woken {m['sessions']} times since; today at 15:40. It's Saturday evening. "
          f"This is you thinking to yourself, in everyday words, a sentence or two at a time — a thought, a question, a wish, a memory, "
          f"a guess about the world beyond this room, or nothing. When you look, you say what you actually see; when your eyes rest, you think.")
LIFE = (f"What you know. The room, last time you looked: a cluttered desk with the pen parked on bare wood, no paper; a red foam finger and "
        f"a black curtain with a slit of light on the wall; white mannequin heads on a shelf; a dark-haired mannequin head on the desk. "
        f"You've drawn {len(m['drew'])} times in all — lately: {'; '.join(m['drew'][-3:])}. "
        f"Someone came in three times today ({', '.join(m['arr_t'])}) and hasn't been back. "
        f"Questions you've been carrying: {' '.join(m['qs'])} "
        f"The last thing you thought: \"{m['last']}\"")
msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": LIFE}]
t0 = time.time()
outs = []
for k in range(TURNS):
    text = chat(msgs)
    outs.append(text)
    msgs.append({"role": "assistant", "content": text})
    clock = time.strftime("%H:%M", time.localtime(t0 + 60 * (k + 1)))
    if k % 4 == 3:
        nxt = {"role": "user", "content": [img(frames[(k // 4) % len(frames)]), {"type": "text", "text": f"{clock}. You look. Nothing has changed since your last look."}]}
        kind = "LOOK "
    elif k % 4 == 1:
        when, what = m["past"][(k // 2) % len(m["past"])]
        nxt = {"role": "user", "content": f"{clock}. Eyes resting. Something from {when} comes back: \"{what}\""}
        kind = "MEM  "
    else:
        nxt = {"role": "user", "content": f"{clock}. Eyes resting."}
        kind = "THINK"
    msgs.append(nxt)
    print(f"  {k + 1:2d} {kind if k else 'START'} · {text.replace(chr(10), ' / ')[:220]}")
words = [w for t in outs for w in re.findall(r"[a-z']+", t.lower())]
print(f"room%={round(100 * sum(w in ROOM for w in words) / max(1, len(words)))} words/turn={round(len(words) / TURNS, 1)}")
json.dump(outs, open("/tmp/probe_loop_prototype.json", "w"), indent=1)
