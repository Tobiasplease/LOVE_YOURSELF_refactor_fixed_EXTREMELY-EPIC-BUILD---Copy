"""Context ablation (Sep 5 eve): replay the latest live caption call and knock
out one channel at a time — the self-knowledge/durable block, the status
lines, the seam prefill, the window, the frame — to see which channel makes
the voice bland, templated, number-bound, and which one conjures a person.

Reads everything from the newest event log (system prompt, user lines, the
last 24 kept captions as the window, the newest frame). No machine imports.

Run:  python debug/probe_context_ablation.py [samples=4]
"""
import base64
import collections
import glob
import json
import os
import re
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config.config as C  # noqa: E402  (plain constants, no side effects)

URL = os.getenv("LLAMA_SERVER_URL", "http://localhost:8080") + "/v1/chat/completions"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 4
WINDOW = 24

PERSON = re.compile(r"\b(he|him|his|she|her|man|woman|figure|someone|somebody|person|visitor|hunch\w*)\b", re.I)
TEMPLATE = re.compile(r"\b(it.s not|isn.t|it.s just|just a|is just|not a\b|no longer)\b", re.I)
NUMBER = re.compile(r"(?<![A-Za-z])\d+")
ROOM = set("finger lamp desk pen paper wall chair curtain light wood cloth plastic hole shelf shelves dust wire wires laptop foam red white black grey gray stain ceiling floor room".split())


def latest_call(log):
    rows = [json.loads(l) for l in open(log) if l.strip()]
    calls = [r for r in rows if r.get("type") == "llm_api_call" and r.get("prompt_type") == "caption" and r.get("success", True)]
    call = calls[-1]
    caps = [r for r in rows if r.get("type") == "caption" and r.get("caption") and r["timestamp"] <= call["timestamp"]]
    return call, caps[-WINDOW:]


def seam_of(entry, budget=220):
    end = re.compile(r"[.!?…]['\")\]]?\s+")
    marks = [m.end() for m in end.finditer(entry)]
    s = entry[marks[-1]:] if marks else entry
    if not s.strip():
        s = entry[marks[-2]:] if len(marks) > 1 else entry
    if len(s) > budget:
        s = s[-budget:]
        c = s.find(" ")
        if c > 0:
            s = s[c + 1:]
    return s.strip()


def options():
    o = {
        "temperature": getattr(C, "CAPTION_TEMP", 0.9),
        "top_p": getattr(C, "CAPTION_TOP_P", 1.0),
        "min_p": getattr(C, "CAPTION_MIN_P", 0.05),
        "presence_penalty": getattr(C, "CAPTION_PRESENCE_PENALTY", 0.6),
        "repeat_penalty": getattr(C, "CAPTION_REPEAT_PENALTY", 1.05),
        "max_tokens": 60,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    for k in ("dry_multiplier", "dry_base", "dry_allowed_length", "dry_penalty_last_n"):
        v = getattr(C, "CAPTION_" + k.upper(), None)
        if v is not None:
            o[k] = v
    return o


def call(system, history_lines, user_text, img, prefill):
    msgs = [{"role": "system", "content": system}]
    if history_lines:
        msgs.append({"role": "assistant", "content": "\n".join(history_lines)})
    content = ([img] if img else []) + [{"type": "text", "text": user_text or " "}]
    msgs.append({"role": "user", "content": content})
    if prefill:
        msgs.append({"role": "assistant", "content": prefill})
    payload = {"model": "probe", "messages": msgs, "stream": False, **options()}
    req = urllib.request.Request(URL, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.load(r)["choices"][0]["message"]["content"].strip()


def metrics(texts, known):
    words = [w for t in texts for w in re.findall(r"[a-z']+", t.lower())]
    content = [w for w in words if len(w) > 4]
    novel = [w for w in content if w not in known]
    return {
        "words": round(len(words) / max(1, len(texts)), 1),
        "person": sum(bool(PERSON.search(t)) for t in texts),
        "template": sum(len(TEMPLATE.findall(t)) for t in texts),
        "numbers": sum(len(NUMBER.findall(t)) for t in texts),
        "room%": round(100 * sum(w in ROOM for w in words) / max(1, len(words))),
        "novel%": round(100 * len(novel) / max(1, len(content))),
        "novel_words": sorted(set(novel))[:12],
    }


def main():
    log = max(glob.glob("event_log/*-event-log.json"), key=os.path.getmtime)
    call_rec, caps = latest_call(log)
    run = os.path.basename(log).split("-")[0]
    frame = sorted(glob.glob(f"event_log/{run}-images/*.jpg"))[-1]
    with open(frame, "rb") as f:
        img = {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(f.read()).decode()}}
    system = call_rec["system_prompt"]
    user = call_rec["prompt"]
    lines = [f"{time.strftime('%H:%M', time.localtime(c['timestamp']))} — {c['caption'].strip()}" for c in caps]
    tail = re.sub(r"^\d\d:\d\d — ", "", lines[-1])
    prefill = seam_of(tail)
    prefill = prefill + " " if prefill else ""
    hist = lines[:-1]
    cut = system.find(" What you've come to know about yourself")
    no_self = system[:cut] if cut > 0 else system
    bare = ("You are a machine fixed to a table in a studio, with a camera you can turn and one pen. "
            "This is your inner voice, said to yourself. It's Saturday evening.")
    known = set(re.findall(r"[a-z']+", (system + user + "\n".join(lines)).lower()))
    arms = {
        "full (as live)": (system, hist, user, img, prefill),
        "no self-block": (no_self, hist, user, img, prefill),
        "no status lines": (system, hist, "", img, prefill),
        "no prefill": (system, hist, user, img, ""),
        "no window": (system, [], user, img, ""),
        "bare system+status": (bare, hist, user, img, prefill),
        "bare, no status": (bare, hist, "", img, ""),
        "bare, no frame (think)": (bare, hist, "", None, ""),
    }
    print(f"log {run} | call {call_rec['iso_timestamp'][11:19]} | window {len(lines)} | frame {os.path.basename(frame)} | N={N}")
    print(f"prefill: {prefill!r}\n")
    out = {}
    for name, args in arms.items():
        texts = []
        for _ in range(N):
            try:
                texts.append(call(*args))
            except Exception as e:  # noqa: BLE001
                texts.append(f"[error {e}]")
        out[name] = texts
        m = metrics(texts, known)
        print(f"## {name}: {m}")
        for t in texts:
            print("   ·", t.replace("\n", " / ")[:170])
        print()
    json.dump(out, open("/tmp/probe_context_ablation.json", "w"), indent=1)


if __name__ == "__main__":
    main()
