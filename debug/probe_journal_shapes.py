"""Journal-continuity probe (Sep 6 morning). The artist: 'the collected output
should read as a continuous text… pages of what looks like an actual
journal.' Five prompt shapes over the same last 8 thoughts, N samples each:
  A  turns + premise cue (live shape)
  B  running text as ONE assistant message (no stamps) + clock-only cue
  C  running text + premise cue
  D  turns + paragraph premise (the last two thoughts quoted)
  E  running text as assistant PREFILL ending at a paragraph break (document continuation)
Measures: continues (connective/pronoun opener or ≥0.15 word overlap with the
last thought), clock narration ("It's HH:MM"), mean words. Run with the
machine OFF (shares the GPU). python debug/probe_journal_shapes.py [N=6]
"""
import json
import os
import re
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from captioner.mind import content_words, last_sentence  # noqa: E402
from captioner.prompt_registry import P  # noqa: E402

URL = os.getenv("LLAMA_SERVER_URL", "http://localhost:8080") + "/v1/chat/completions"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 6
OPTS = {"temperature": 0.9, "top_p": 1.0, "min_p": 0.05, "presence_penalty": 0.6, "repeat_penalty": 1.05, "max_tokens": 70,
        "dry_multiplier": 0.85, "dry_base": 1.75, "dry_allowed_length": 3, "dry_penalty_last_n": 384, "chat_template_kwargs": {"enable_thinking": False}}
CONT = re.compile(r"^\W*(and|but|or|so|then|still|maybe|it|its|it's|it’s|that|this|they|there|which|no,|yes,|not |because|if )", re.I)
CLOCK = re.compile(r"\b(it.s|at|now) \d\d?:\d\d\b|\b\d\d?:\d\d\b", re.I)


def chat(msgs, prefill=""):
    m = msgs + ([{"role": "assistant", "content": prefill}] if prefill else [])
    req = urllib.request.Request(URL, data=json.dumps({"model": "p", "messages": m, "stream": False, **OPTS}).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        out = json.load(r)["choices"][0]["message"]["content"]
    if prefill and out.startswith(prefill.rstrip()):
        out = out[len(prefill.rstrip()):]
    return out.strip()


d = json.load(open("event_log/mind_thread.json"))
th = [e for e in d["thread"] if e.get("text") and e.get("kind") in ("wake", "look", "think", "reflection")][-8:]
last = th[-1]["text"]
system = P("mind.system") + P("monologue.pen-parked")
clock = time.strftime("%H:%M", time.localtime())
turns = []
for e in th:
    turns.append({"role": "user", "content": e.get("cue") or "…"})
    turns.append({"role": "assistant", "content": e["text"]})
running = "\n\n".join(e["text"] for e in th)
prem = last_sentence(last)
shapes = {
    "A turns + premise (live)": lambda: chat([{"role": "system", "content": system}] + turns + [{"role": "user", "content": f"{clock}. Eyes resting." + P("mind.cue-premise").format(premise=prem)}]),
    "B running text + clock": lambda: chat([{"role": "system", "content": system}, {"role": "assistant", "content": running}, {"role": "user", "content": f"{clock}."}]),
    "C running text + premise": lambda: chat([{"role": "system", "content": system}, {"role": "assistant", "content": running}, {"role": "user", "content": f"{clock}. Eyes resting." + P("mind.cue-premise").format(premise=prem)}]),
    "D turns + paragraph premise": lambda: chat([{"role": "system", "content": system}] + turns + [{"role": "user", "content": f"{clock}. Eyes resting. You were on: \"{th[-2]['text']} {last}\" Go on from there."}]),
    "E running text as prefill": lambda: chat([{"role": "system", "content": system}, {"role": "user", "content": f"{clock}."}], prefill=running + "\n\n"),
}
print("last thought:", last[:150], "\n")
for name, fn in shapes.items():
    outs = []
    for _ in range(N):
        try:
            outs.append(fn())
        except Exception as e:  # noqa: BLE001
            outs.append(f"[error {e}]")
    cont = sum(1 for o in outs if CONT.match(o) or len(content_words(o) & content_words(last)) / max(1, len(content_words(o))) >= 0.15)
    clk = sum(bool(CLOCK.search(o)) for o in outs)
    words = sum(len(o.split()) for o in outs) / max(1, len(outs))
    empty = sum(1 for o in outs if len(o.split()) < 3)
    print(f"## {name}: continues {cont}/{N} | clock-narration {clk} | words {words:.0f} | empty {empty}")
    for o in outs:
        print("   ·", o.replace("\n", " / ")[:170])
    print()
