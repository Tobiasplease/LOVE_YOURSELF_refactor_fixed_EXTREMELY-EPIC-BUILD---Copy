"""Night watch (Sep 5 2026): scan everything since the last check and report the
patterns that matter — presence (phantom gate, absence fact, belief edges,
relational-with-nobody), voice (refrains, template share, tone), memory (want/
belief/threads/questions changes, reflections), health (process, restarts,
errors, cadence, low-energy, clock). Appends markdown to docs/night-watch-<date>.md
and prints a compact summary. State in event_log/night_watch_state.json.

Run:  python debug/night_watch.py            (since last check)
      python debug/night_watch.py --since 2026-09-05T01:19   (explicit)
      python debug/night_watch.py --dry       (no state/report write)
"""

import collections
import glob
import json
import os
import re
import subprocess
import sys
import time
import urllib.request

REPO = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, REPO)
EV = os.path.join(REPO, "event_log")
STATE = os.path.join(EV, "night_watch_state.json")
REPORT = os.path.join(REPO, "docs", f"night-watch-{time.strftime('%b%d').lower()}.md")
PERSON = re.compile(r"\b(he|him|his|the man|the woman)\b", re.I)

dry = "--dry" in sys.argv
since = None
if "--since" in sys.argv:
    since = sys.argv[sys.argv.index("--since") + 1]
state = json.load(open(STATE)) if os.path.exists(STATE) else {}
if not since:
    since = state.get("last_check_iso") or time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(time.time() - 1800))
now_iso = time.strftime("%Y-%m-%dT%H:%M:%S")


def load_runs():
    runs = []
    for f in sorted(glob.glob(os.path.join(EV, "*-event-log.json")), key=os.path.getmtime)[-6:]:
        try:
            first = json.loads(open(f).readline())
        except Exception:
            continue
        start = first.get("start_time_iso", "")
        if os.path.getmtime(f) < time.mktime(time.strptime(since[:19], "%Y-%m-%dT%H:%M:%S")) - 60:
            continue
        runs.append((f, start))
    return runs


def words(s):
    return re.findall(r"[a-z']+", (s or "").lower())


out = {"since": since, "now": now_iso}
runs = load_runs()
entries = []
for f, start in runs:
    for l in open(f):
        try:
            d = json.loads(l)
        except Exception:
            continue
        if d.get("iso_timestamp", "") >= since:
            d["_run"] = os.path.basename(f)[:8]
            entries.append(d)
entries.sort(key=lambda d: d.get("timestamp", 0))
caps = [d for d in entries if d.get("type") == "caption" and "caption" in d]
calls = [d for d in entries if d.get("type") == "llm_api_call"]
acts = collections.Counter(d.get("action") for d in entries if d.get("action"))

# --- health
pid = subprocess.run(["pgrep", "-f", "python machin[e].py"], capture_output=True, text=True).stdout.split()
out["process"] = {"pids": pid, "runs_touched": [r[1][11:19] + " " + os.path.basename(r[0])[:8] for r in runs]}
# A real boot writes session_start; a debug script importing the captioner only mints run_metadata (clutter, not a restart).
_booted = {d["_run"] for d in entries if d.get("type") == "session_start"}
out["restarts"] = [d["iso_timestamp"][11:19] for d in entries if d.get("type") == "run_metadata" and d["_run"] in _booted]
out["empty_run_files"] = sum(1 for d in entries if d.get("type") == "run_metadata" and d["_run"] not in _booted)
out["errors"] = [(d["iso_timestamp"][11:19], (d.get("message") or "")[:100]) for d in entries if d.get("type") == "error"][-5:]
try:
    mode = json.load(open(os.path.join(EV, "runtime_mode.json")))
    out["low_energy"] = mode.get("low_energy")
except Exception:
    out["low_energy"] = None
out["captions"] = len(caps)
if len(caps) >= 2:
    span = caps[-1]["timestamp"] - caps[0]["timestamp"]
    out["cadence_s"] = round(span / max(1, len(caps) - 1), 1)
out["last_caption"] = caps[-1]["iso_timestamp"][11:19] if caps else None
try:
    req = urllib.request.Request("https://www.google.com", method="HEAD")
    with urllib.request.urlopen(req, timeout=5) as r:
        hdr = r.headers.get("Date")
    http_ts = time.mktime(time.strptime(hdr, "%a, %d %b %Y %H:%M:%S %Z")) - time.timezone
    out["clock_offset_s"] = round(time.time() - http_ts)
except Exception:
    out["clock_offset_s"] = None

# --- presence
gated = collections.Counter(d.get("reason") for d in entries if d.get("action") == "echo_spoken_not_stored")
out["gate_reasons"] = dict(gated)
him = [c for c in caps if PERSON.search(c["caption"])]
gated_prev = {(d.get("caption_preview") or "")[:40] for d in entries if d.get("action") == "echo_spoken_not_stored"}
# the gate logs the pre-trim text; match by time too (a gate event within 6 s of the caption)
_gate_ts = [
    d.get("timestamp", 0) for d in entries if d.get("action") == "echo_spoken_not_stored" or (d.get("action") == "drift_turn" and not d.get("stored"))
]


def _gated(c):
    return c["caption"][:40] in gated_prev or any(abs(c.get("timestamp", 0) - t) <= 6 for t in _gate_ts)


# belief timeline from the prompt edges: ON after an arrival line, OFF after a departure line
belief_on_spans = []
_on = None
for d in calls:
    for x in (d.get("prompt") or "").split("\n"):
        if re.match(r"^(He's come in|He's back|Someone's come in|People have come in)", x) and _on is None:
            _on = d.get("timestamp", 0)
        elif x.startswith("They've gone") and _on is not None:
            belief_on_spans.append((_on, d.get("timestamp", 0)))
            _on = None
if _on is not None:
    belief_on_spans.append((_on, float("inf")))


def believed_at(ts):
    return any(a - 5 <= ts <= b + 5 for a, b in belief_on_spans)


out["him_captions"] = len(him)
out["him_while_believed"] = sum(1 for c in him if believed_at(c.get("timestamp", 0)))
try:
    from utils.presence_text import is_phantom_presence as _phantom
except Exception:  # pragma: no cover
    _phantom = lambda t: True  # noqa: E731
out["him_kept"] = [
    (c["iso_timestamp"][11:19], c["caption"][:110])
    for c in him
    if not _gated(c) and not believed_at(c.get("timestamp", 0)) and _phantom(c["caption"])
][-5:]
edges = []
for d in calls:
    for x in (d.get("prompt") or "").split("\n"):
        if re.match(
            r"^(He's come in|He's back|Someone's come in|People have come in|They've gone|He left|Someone left|No one's been in the room)", x
        ):
            edges.append((d["iso_timestamp"][11:19], x[:60]))
out["presence_lines"] = collections.Counter(e[1].split(" ")[0] + " " + e[1].split(" ")[1] for e in edges)
out["belief_edges"] = [e for e in edges if not e[1].startswith(("He left", "Someone left", "No one's"))][-6:]
out["adjudications"] = [(d["iso_timestamp"][11:19], d["response"][:60]) for d in calls if d.get("prompt_type") == "presence_adjudication"][-4:]
rides = [d for d in entries if d.get("action") == "absence_standing" and not d.get("riding")]
out["absence_rides"] = {"count": len(rides), "calls": sum(int(d.get("calls") or 0) for d in rides)}
out["relational_captions"] = sum(1 for c in caps if c.get("mode") == "relational")
out["motion_events"] = sum(1 for d in calls if "Something is moving in the room" in (d.get("prompt") or ""))
try:
    pane = subprocess.run(["tmux", "capture-pane", "-p", "-t", "impostor-system", "-S", "-2000"], capture_output=True, text=True, timeout=10).stdout
    _tb_blocks = pane.split("Traceback (most recent call last)")[1:]
    _noise = ("ConnectionResetError", "BrokenPipeError", "ConnectionAbortedError")
    out["console"] = {
        "skeleton_rejects": pane.count("rejected by skeleton gate"),
        "tracebacks": sum(1 for b in _tb_blocks if not any(n in b[:1500] for n in _noise)),
        "feed_disconnects": sum(1 for b in _tb_blocks if any(n in b[:1500] for n in _noise)),
        "supervisor_restarts": pane.count("=== starting machine"),
    }
except Exception:
    out["console"] = {}

# --- voice
texts = [c["caption"] for c in caps]
if texts:
    ng = collections.Counter()
    for t in texts:
        seen = set()
        w = words(t)
        for i in range(len(w) - 3):
            g = " ".join(w[i : i + 4])
            if g not in seen:
                ng[g] += 1
                seen.add(g)
    out["top_4grams"] = [(g, n) for g, n in ng.most_common(6) if n >= 3]
    dup = 0
    sh = []
    for t in texts:
        w = words(t)
        s = {" ".join(w[i : i + 6]) for i in range(len(w) - 5)}
        if any(s & p for p in sh[-40:]):
            dup += 1
        sh.append(s)
    out["refrain_share_pct"] = round(100 * dup / len(texts))
    out["deflation_pct"] = round(100 * sum(1 for t in texts if re.search(r"\b(it's|its|it is) just\b", t.lower())) / len(texts))
    out["questions"] = sum(1 for t in texts if "?" in t)
    out["exclamations"] = sum(1 for t in texts if "!" in t)
    out["modes"] = dict(collections.Counter(c.get("mode") for c in caps))
    out["sample"] = [(c["iso_timestamp"][11:19], c["caption"][:120].replace("\n", " / ")) for c in caps[-3:]]

# --- memory
dist = [d for d in calls if d.get("prompt_type") == "reflection_distill"]
out["reflections"] = sum(1 for d in calls if d.get("prompt_type") == "reflection")
out["distills"] = [(d["iso_timestamp"][11:19], d["response"][:300].replace("\n", " / ")) for d in dist][-3:]
try:
    ident = json.load(open(os.path.join(EV, "machine_identity.json")))
    want = (ident.get("introspective_state") or ident).get("current_desire") or ident.get("desire")
    belief = (ident.get("introspective_state") or ident).get("current_belief") or ident.get("belief")
    out["want"] = want
    out["belief"] = belief
    out["want_changed"] = bool(state.get("want")) and state.get("want") != want
except Exception:
    pass
try:
    lore = json.load(open(os.path.join(EV, "lore_ledger.json")))
    out["lore"] = {
        "threads": len(lore.get("threads", [])),
        "questions": len(lore.get("questions", [])),
        "reveries": len(lore.get("reveries", [])),
        "name": lore.get("name"),
    }
    prev = state.get("lore") or {}
    out["lore_new"] = {k: out["lore"][k] - prev.get(k, out["lore"][k]) for k in ("threads", "questions", "reveries")}
    out["latest_thread"] = (lore.get("threads") or [{}])[-1].get("text", "")[:100]
    out["latest_question"] = (lore.get("questions") or [{}])[-1].get("text", "")[:100]
except Exception:
    pass
out["drift_stored"] = sum(1 for d in entries if d.get("action") == "drift_turn" and d.get("stored"))
out["memory_surface_lines"] = sum(
    1
    for d in calls
    if d.get("prompt_type") == "caption"
    for x in (d.get("prompt") or "").split("\n")
    if re.match(r"^(From |A thought you|A question you|You've noticed)", x)
)
out["draw_decisions"] = collections.Counter(d.get("reason") for d in entries if d.get("decision") == "trigger_decision")

# --- notable flags
flags = []
if not pid:
    flags.append("MACHINE DOWN — no python machine.py process")
if out["errors"]:
    flags.append(f"{len(out['errors'])} error entries")
if out.get("console", {}).get("tracebacks"):
    flags.append("Traceback on console")
if out["restarts"]:
    flags.append(f"restart(s) at {out['restarts']}")
if out["him_kept"]:
    flags.append(f"{len(out['him_kept'])} person captions KEPT while the belief was OFF — gate missed: {out['him_kept'][-1][1][:60]!r}")
rel_off = sum(1 for c in caps if c.get("mode") == "relational" and not believed_at(c.get("timestamp", 0)))
if rel_off:
    flags.append(f"{rel_off} relational captions while the belief was OFF")
if out["belief_edges"]:
    flags.append(f"belief edges: {out['belief_edges']}")
if out["absence_rides"]["calls"] > 0.3 * max(1, len(calls)):
    flags.append(f"absence fact riding heavily ({out['absence_rides']['calls']} calls)")
if out.get("want_changed"):
    flags.append(f"want changed → {out.get('want')}")
if out.get("lore_new", {}).get("threads"):
    flags.append(f"+{out['lore_new']['threads']} lore thread(s): {out.get('latest_thread')}")
if out.get("lore_new", {}).get("questions"):
    flags.append(f"+{out['lore_new']['questions']} question(s): {out.get('latest_question')}")
if out.get("clock_offset_s") is not None and abs(out["clock_offset_s"]) > 120:
    flags.append(f"CLOCK off by {out['clock_offset_s']}s vs HTTP")
if out.get("low_energy") is False:
    flags.append("LOW ENERGY IS OFF")
if len(caps) and out.get("cadence_s", 0) > 60:
    flags.append(f"slow cadence {out['cadence_s']}s")
if not caps and pid:
    flags.append("process alive but NO captions this interval")
out["flags"] = flags

# --- write
md = [f"\n## {now_iso[11:16]}  (since {since[11:16]})", ""]
md.append("**Flags:** " + ("; ".join(flags) if flags else "none"))
md.append(
    f"- process {out['process']['pids']} runs {out['process']['runs_touched']} empty-run-files {out.get('empty_run_files')} low_energy={out.get('low_energy')} clock_offset={out.get('clock_offset_s')}s"
)
md.append(
    f"- captions {out['captions']} cadence {out.get('cadence_s')}s last {out.get('last_caption')} modes {out.get('modes')} draw {dict(out['draw_decisions'])}"
)
md.append(
    f"- presence: him {out['him_captions']} (while believed {out.get('him_while_believed')}, kept-off {len(out['him_kept'])}) gate {out['gate_reasons']} absence rides {out['absence_rides']} relational {out['relational_captions']} motion {out['motion_events']} console {out.get('console')} lines {dict(out['presence_lines'])}"
)
md.append(
    f"- voice: refrain-share {out.get('refrain_share_pct')}% deflation {out.get('deflation_pct')}% questions {out.get('questions')} exclam {out.get('exclamations')} top4 {out.get('top_4grams')}"
)
md.append(
    f"- memory: reflections {out['reflections']} drift-stored {out['drift_stored']} surface-lines {out['memory_surface_lines']} lore {out.get('lore')} new {out.get('lore_new')}"
)
md.append(f"- want: {out.get('want')}\n- belief: {out.get('belief')}")
for t, r in out["distills"]:
    md.append(f"- distill {t}: {r}")
for t, c in out.get("sample", []):
    md.append(f"- {t} {c}")
text = "\n".join(md)
print(text)
if not dry:
    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    new = not os.path.exists(REPORT)
    with open(REPORT, "a") as fh:
        if new:
            fh.write(f"# Night watch — {time.strftime('%Y-%m-%d')}\n\nAppended every check by debug/night_watch.py (loop).\n")
        fh.write(text + "\n")
    state.update({"last_check_iso": now_iso, "want": out.get("want"), "lore": out.get("lore")})
    json.dump(state, open(STATE, "w"), indent=2)
