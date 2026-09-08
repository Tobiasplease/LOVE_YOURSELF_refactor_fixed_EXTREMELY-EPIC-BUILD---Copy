"""Probe (Sep 8): does a trailing space on the hybrid seam cause word-fragment
and double-space starts? Same call, prefill with vs without the trailing
space, N samples each. Machine may stay up. python debug/probe_prefill_space.py [N=8]"""
import json
import os
import re
import sys
import urllib.request

URL = os.getenv("LLAMA_SERVER_URL", "http://localhost:8080") + "/v1/chat/completions"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 8
OPTS = {"temperature": 0.9, "top_p": 1.0, "min_p": 0.05, "max_tokens": 24, "chat_template_kwargs": {"enable_thinking": False}}
log = max((os.path.join("event_log", f) for f in os.listdir("event_log") if f.endswith("-event-log.json")), key=os.path.getmtime)
rows = [json.loads(l) for l in open(log) if l.strip()]
calls = [r for r in rows if r.get("type") == "llm_api_call" and r.get("prompt_type") == "caption" and r.get("prefill_tail") and r.get("system_prompt")]
SEAMS = [
    "The red foam finger is still on the shelf, but it doesn't feel like a pulse anymore; it's just a bright,",
    "The room is still there, but it",
    "I've been staring at that space for an hour, trying to figure out if it's a mistake in my perception or a mistake in the room.",
    "So I don't move.",
]
c = calls[-1]


def ask(prefill):
    msgs = [{"role": "system", "content": c["system_prompt"]}, {"role": "user", "content": c["prompt"]}, {"role": "assistant", "content": prefill}]
    req = urllib.request.Request(URL, data=json.dumps({"model": "p", "messages": msgs, "stream": False, **OPTS}).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        out = json.load(r)["choices"][0]["message"]["content"]
    return out[len(prefill):] if out.startswith(prefill) else out


WORDS = set(w.strip().lower() for w in open("/usr/share/dict/words")) if os.path.exists("/usr/share/dict/words") else set()
for label, sp in (("WITH trailing space (today)", " "), ("WITHOUT trailing space", "")):
    frag = dbl = glue = 0
    ex = []
    for seam in SEAMS:
        for _ in range(N // len(SEAMS) or 1):
            o = ask(seam + sp)
            first = re.match(r"^\s*([A-Za-z']+)", o)
            w = first.group(1).lower() if first else ""
            if o.startswith("  "):
                dbl += 1
            if sp == "" and o and not o[0].isspace() and not seam.endswith((".", "!", "?")):
                glue += 1  # no leading space supplied: words would run together
            if w and WORDS and w not in WORDS and len(w) <= 6:
                frag += 1
            ex.append(repr(o[:34]))
    print(f"## {label}: fragment-starts {frag} | double-space {dbl} | glued (no space supplied) {glue} | of {len(ex)}")
    print("   ", ", ".join(ex[:6]))
