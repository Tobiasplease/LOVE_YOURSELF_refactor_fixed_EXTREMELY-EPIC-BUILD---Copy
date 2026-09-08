"""Probe (Sep 8): does a FRAGMENT seam bring back continuation and kill the
"it's not X, it's Y" tic?

Aug 20 added _trim_to_boundary: entries are cut to their last complete
sentence before storage, so the hybrid seam now hands the model a FINISHED
sentence and it must start a new one — and a fresh sentence start is where an
instruct model reaches for the antithesis. In the 3.6 runs the artist prefers,
entries were cut mid-clause by the token budget, so the seam was a fragment
and the next call continued inside a sentence.

Arms (same system prompt, same stream, live server, N samples each):
  A  seam = the last COMPLETE sentence (today)
  B  seam = the raw unfinished tail (as 3.6 had it)
  C  seam = the last sentence with its final clause cut at a comma
  D  no seam at all (world mode)
Scores: antithesis rate, opener variety, mid-clause continuation, words.
Run with the machine up. python debug/probe_seam_fragment.py [N=6]
"""
import json
import os
import re
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
URL = os.getenv("LLAMA_SERVER_URL", "http://localhost:8080") + "/v1/chat/completions"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 6
AN = re.compile(r"\b(it'?s not|isn'?t|not a\b|not the\b|no longer)\b[^.?!]{0,60}[,;]?\s*(it'?s|it is|just|but)\b", re.I)
LOWER_START = re.compile(r"^[a-z,;]")
OPTS = {"temperature": 0.9, "top_p": 1.0, "min_p": 0.05, "presence_penalty": 0.6, "repeat_penalty": 1.05,
        "dry_multiplier": 0.85, "dry_base": 1.75, "dry_allowed_length": 3, "dry_penalty_last_n": 384,
        "max_tokens": 60, "chat_template_kwargs": {"enable_thinking": False}}

log = max((os.path.join("event_log", f) for f in os.listdir("event_log") if f.endswith("-event-log.json")), key=os.path.getmtime)
rows = [json.loads(l) for l in open(log) if l.strip()]
calls = [r for r in rows if r.get("type") == "llm_api_call" and r.get("prompt_type") == "caption" and r.get("system_prompt")]
call = calls[-1]
caps = [r["caption"] for r in rows if r.get("type") == "caption" and r.get("caption")][-12:]
SYSTEM, USER = call["system_prompt"], call["prompt"]


def ask(lines, prefill):
    msgs = [{"role": "system", "content": SYSTEM}]
    if lines:
        msgs.append({"role": "assistant", "content": "\n".join(lines)})
    msgs.append({"role": "user", "content": USER})
    if prefill:
        msgs.append({"role": "assistant", "content": prefill})
    req = urllib.request.Request(URL, data=json.dumps({"model": "p", "messages": msgs, "stream": False, **OPTS}).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        out = json.load(r)["choices"][0]["message"]["content"]
    return out[len(prefill):].strip() if prefill and out.startswith(prefill) else out.strip()


last = caps[-1]
_END = re.compile(r"[.!?…]['\")\]]?\s+")
marks = [m.end() for m in _END.finditer(last + " ")]
complete_seam = last[marks[-2]:].strip() if len(marks) > 1 else last
raw_tail = re.sub(r"[.!?…]+\s*$", "", last).rsplit(" ", 4)[0] + " " + " ".join(re.sub(r"[.!?…]+\s*$", "", last).split()[-4:-2])
comma_cut = re.split(r",\s*", re.sub(r"[.!?…]+\s*$", "", last))[0] + "," if "," in last else re.sub(r"[.!?…]+\s*$", "", last).rsplit(" ", 3)[0]
ARMS = {
    "A complete-sentence seam (today)": (caps[:-1], complete_seam + " "),
    "B raw unfinished tail (3.6)": (caps[:-1], (re.sub(r"[.!?…]+\s*$", "", last).rsplit(" ", 3)[0] + " ")),
    "C cut at a comma": (caps[:-1], comma_cut + " "),
    "D no seam": (caps, ""),
}
print(f"log {os.path.basename(log)[:8]} | last entry: {last[:90]!r}\n")
for name, (lines, pre) in ARMS.items():
    outs = [ask(lines, pre) for _ in range(N)]
    an = sum(1 for o in outs if AN.search(o))
    low = sum(1 for o in outs if LOWER_START.match(o))
    openers = len({" ".join(o.split()[:2]).lower() for o in outs})
    print(f"## {name}\n   prefill={pre[-52:]!r}")
    print(f"   antithesis {an}/{N} | continues mid-clause {low}/{N} | distinct openers {openers}/{N} | words {sum(len(o.split()) for o in outs)//max(1,N)}")
    for o in outs[:3]:
        print("    ·", o.replace("\n", " / ")[:120])
    print()
