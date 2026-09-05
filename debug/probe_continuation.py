"""Continuation probe (Sep 5 late): does thought n+1 build on thought n?
Measures the live mind-mode run (content-word overlap with the previous
thought vs. with older ones; verbatim 6-gram echoes), then replays the live
thread and samples the next thought under system-prompt variants.
Run: python debug/probe_continuation.py [samples=6]
"""
import json
import os
import re
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from captioner.mind import content_words  # noqa: E402
from captioner.prompt_registry import P  # noqa: E402

URL = os.getenv("LLAMA_SERVER_URL", "http://localhost:8080") + "/v1/chat/completions"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 6
OPTS = {"temperature": 0.9, "top_p": 1.0, "min_p": 0.05, "presence_penalty": 0.6, "repeat_penalty": 1.05, "max_tokens": 60,
        "dry_multiplier": 0.85, "dry_base": 1.75, "dry_allowed_length": 3, "dry_penalty_last_n": 384, "chat_template_kwargs": {"enable_thinking": False}}


def grams(t, n=6):
    w = re.findall(r"[a-z']+", t.lower())
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


def build_score(texts):
    """Per thought: overlap with the previous thought, overlap with thoughts 2..6 back, 6-gram echo of any earlier."""
    rows = []
    for i, t in enumerate(texts):
        cw = content_words(t)
        prev = content_words(texts[i - 1]) if i else set()
        older = set()
        for o in texts[max(0, i - 6):i - 1]:
            older |= content_words(o)
        echo = any(grams(t) & grams(o) for o in texts[max(0, i - 6):i])
        rows.append((len(cw & prev) / max(1, len(cw)), len(cw & older) / max(1, len(cw)), echo))
    return rows


def chat(msgs):
    payload = {"model": "probe", "messages": msgs, "stream": False, **OPTS}
    req = urllib.request.Request(URL, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.load(r)["choices"][0]["message"]["content"].strip()


d = json.load(open("event_log/mind_thread.json"))
thread = [e for e in d["thread"] if e.get("text")]
texts = [e["text"] for e in thread]
rows = build_score(texts)
print(f"LIVE thread: {len(texts)} thoughts | builds-on-previous (overlap≥0.15): {sum(r[0] >= 0.15 for r in rows)}/{len(rows)} | "
      f"mean overlap prev {sum(r[0] for r in rows)/len(rows):.2f} vs older {sum(r[1] for r in rows)/len(rows):.2f} | 6-gram echoes {sum(r[2] for r in rows)}")

turns = []
for i, e in enumerate(thread[-6:]):
    turns.append({"role": "user", "content": e.get("cue") or "…"})
    turns.append({"role": "assistant", "content": e["text"]})
cue = time.strftime("%H:%M", time.localtime()) + ". Eyes resting."
base = P("mind.system") + P("monologue.pen-parked")
REFLEX = (" The lines that arrive between your thoughts — the clock, what your eyes report — are your own senses, not someone talking to you. "
          "Nobody answers you. A question you ask is yours to answer, in the next thought or a later one; a thought you start is yours to carry. ")
variants = {"A current frame": base, "B + reflexive/self-answer": base + REFLEX}
last = texts[-1]
for name, system in variants.items():
    outs = []
    for _ in range(N):
        outs.append(chat([{"role": "system", "content": system}] + turns + [{"role": "user", "content": cue}]))
    sc = [(len(content_words(o) & content_words(last)) / max(1, len(content_words(o))), any(grams(o) & grams(t) for t in texts[-6:])) for o in outs]
    print(f"\n## {name}: builds-on-last {sum(s[0] >= 0.15 for s in sc)}/{N} (mean overlap {sum(s[0] for s in sc)/N:.2f}) | echoes {sum(s[1] for s in sc)}")
    for o in outs:
        print("   ·", o.replace("\n", " / ")[:170])
print("\nlast live thought was:", last[:160])
