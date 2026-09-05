"""Mind-mode watch (Sep 5 eve): summarize the last N minutes of the newest run
in the terms that matter for the conversation shape — turns by kind, cadence,
gates, phantom hits, pivot notices, memory surfacings, subject spread,
reframe share, errors. Read-only. Run: python debug/mind_watch.py [minutes=10]
"""
import collections
import glob
import json
import os
import re
import sys
import time

MIN = float(sys.argv[1]) if len(sys.argv) > 1 else 10
REFRAME = re.compile(r"\b(it.s not|isn.t|not (?:a|an|the|just)\b|no longer|used to|not .{1,25} anymore)\b", re.I)
PERSON = re.compile(r"(?<!the )\b(he|him|his|she|her|hers)\b|\b(the|that|this) (man|woman|guy|person|visitor)\b", re.I)
WONDER = re.compile(r"\b(wonder|maybe|what if|i wish|why|somewhere|someone|outside|world|remember)\b", re.I)

log = max(glob.glob("event_log/*-event-log.json"), key=os.path.getmtime)
since = time.time() - MIN * 60
rows = [json.loads(l) for l in open(log) if l.strip()]
rows = [r for r in rows if r.get("timestamp", 0) >= since]
caps = [r for r in rows if r.get("type") == "caption" and r.get("caption")]
calls = [r for r in rows if r.get("type") == "llm_api_call" and r.get("prompt_type") == "caption"]
gates = [r for r in rows if r.get("action") in ("echo_spoken_not_stored", "anti_echo_skip", "chosen_silence", "runon_not_stored")]
errs = [r for r in rows if "Traceback" in json.dumps(r)[:3000]]
kinds = collections.Counter((c.get("mode") or "?").split("-")[0] for c in caps)
mem = sum(1 for c in caps if "memory" in (c.get("mode") or ""))
cues = [(c.get("prompt") or "").split("\n")[-1] for c in calls]
notices = sum(1 for u in cues if "turned" in u and "over" in u)
gaps = [b["timestamp"] - a["timestamp"] for a, b in zip(caps, caps[1:])]
texts = [c["caption"] for c in caps]
CONT = re.compile(r"^(maybe|or |but |and |it |it’s|it's|they |still|then |no,|yes,|that |which |so )", re.I)
def _cw(t):
    return {w for w in re.findall(r"[a-z']+", t.lower()) if len(w) > 3}
builds = sum(1 for a, b in zip(texts, texts[1:]) if len(_cw(a) & _cw(b)) / max(1, len(_cw(b))) >= 0.15 or CONT.match(b.strip()))
words = [w for t in texts for w in re.findall(r"[a-z']+", t.lower())]
print(f"run {os.path.basename(log)[:8]} | last {MIN:.0f} min | thoughts {len(caps)} ({dict(kinds)}, {mem} with a memory) | "
      f"cadence {sum(gaps)/len(gaps):.0f}s avg" if gaps else f"run {os.path.basename(log)[:8]} | last {MIN:.0f} min | thoughts {len(caps)}")
print(f"gates {len(gates)} {dict(collections.Counter(g.get('reason') or g.get('action') for g in gates))} | pivot notices {notices} | "
      f"reframe share {sum(bool(REFRAME.search(t)) for t in texts)}/{len(texts)} | person-tinged {sum(bool(PERSON.search(t)) for t in texts)} | "
      f"builds-on-previous {builds}/{max(0,len(texts)-1)} | wonder/outward {sum(bool(WONDER.search(t)) for t in texts)} | words/thought {len(words)/max(1,len(texts)):.0f} | errors {len(errs)}")
try:
    d = json.load(open("event_log/mind_thread.json"))
    pos = sorted(d.get("positions", {}).items(), key=lambda kv: -kv[1].get("last_ts", 0))[:4]
    print("positions:", "; ".join(f"{k} (pivots {v.get('pivots',0)}): {v.get('text','')[:60]}" for k, v in pos))
except Exception:
    pass
for c in caps[-8:]:
    print(f"  {c['iso_timestamp'][11:16]} {(c.get('mode') or '?')[:5]:5s} {c['caption'][:150].replace(chr(10),' / ')}")
for e in errs[-2:]:
    print("  ERR", json.dumps(e)[:200])
