"""Prompt panel — a reading surface for the machine's mind.

Phase 1 (Aug 2): READ-ONLY, and it touches nothing. It reads the event log and
machine_identity.json, so it cannot perturb a run — which matters, because the
thing it exists to show is what the running system actually did.

    python prompt_panel/server.py          # then open http://localhost:8770

Design rule: show the prompt AS SENT. Every bug of the past week was invisible
in the template and obvious in the assembled call.
"""

import http.server
import json
import os
import socketserver
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

HERE = os.path.dirname(os.path.abspath(__file__))
PORT = int(os.getenv("PROMPT_PANEL_PORT", 8770))

try:
    from config.config import MOOD_SNAPSHOT_FOLDER
except Exception:
    MOOD_SNAPSHOT_FOLDER = os.path.join(os.path.dirname(HERE), "event_log")

# What each pass is, in one line. Ordered as the system experiences them:
# waking, thinking, remembering, drawing. Anything the log shows that isn't
# here appears as "undocumented" — that is the panel's own audit.
PASSES = [
    ("awakening", "Waking", "The first thought of a session — offline gap, lifetime, recall, reorientation."),
    ("caption", "Thinking", "The monologue. One per caption cycle."),
    ("caption_blind", "Thinking blind", "Fallback when the frame isn't on disk yet — same builder, no image."),
    ("memory", "Remembering", "Memory mode, roughly every 4 minutes."),
    ("stream_consolidation", "Folding the thread", "Compresses the oldest stream entries when the window gets long."),
    ("compression", "Memory diff", "Every 8 captions: room / self-note / event / mood."),
    ("concept_extraction", "Naming things", "Pulls solid objects out of the compression."),
    ("journal", "Diary", "Every 30 minutes and at shutdown."),
    ("reflection", "Reflecting", "Long-form thought every ~20 quiet minutes."),
    ("reflection_distill", "Becoming", "Turns a reflection into persona / belief / desire."),
    ("drawing_intent", "Wanting to draw", "Names the one image, in the machine's own words."),
    ("drawing_render", "Translating", "Turns the intent into a prompt an image model can use."),
    ("drawing_watch", "Watching itself draw", "Every 20s while the arm moves."),
    ("drawing_thematic_consolidation", "Drawing theme", "One consolidation at drawing start."),
    ("drawing_critique", "Critique", "After a drawing reaches paper."),
    ("drawing_summary", "Drawing summary", "Summarises the drawing during the flow."),
    ("artistic_arc", "The body of work", "Narrates the executed drawings so far."),
]
PASS_META = {k: (title, blurb) for k, title, blurb in PASSES}


def _load_events(path):
    txt = open(path, encoding="utf-8", errors="replace").read()
    try:
        d = json.loads(txt)
        return [e for e in d if isinstance(e, dict)]
    except json.JSONDecodeError:
        pass
    out = []
    for line in txt.splitlines():
        line = line.strip().rstrip(",")
        if line.startswith("{"):
            try:
                o = json.loads(line)
                if isinstance(o, dict):
                    out.append(o)
            except json.JSONDecodeError:
                pass
    return out


def _newest_run_with_calls():
    """The most recent run that actually talked to the model."""
    best = None
    try:
        names = [n for n in os.listdir(MOOD_SNAPSHOT_FOLDER) if n.endswith("-event-log.json")]
    except FileNotFoundError:
        return None, []
    for name in names:
        p = os.path.join(MOOD_SNAPSHOT_FOLDER, name)
        try:
            evs = _load_events(p)
        except Exception:
            continue
        if not evs:
            continue
        calls = [e for e in evs if e.get("type") == "llm_api_call"]
        if not calls:
            continue
        start = evs[0].get("start_time", 0) or evs[0].get("timestamp", 0)
        if best is None or start > best[0]:
            best = (start, name, evs)
    if not best:
        return None, []
    return best[1], best[2]


def _ago(ts):
    if not ts:
        return "—"
    s = max(0, int(time.time() - ts))
    if s < 60:
        return f"{s}s ago"
    if s < 3600:
        return f"{s // 60}m ago"
    if s < 86400:
        return f"{s // 3600}h ago"
    return f"{s // 86400}d ago"


def build_payload():
    run, evs = _newest_run_with_calls()
    calls = [e for e in evs if e.get("type") == "llm_api_call"]
    captions = [e for e in evs if e.get("type") == "caption" and e.get("caption")]

    by_type = {}
    for c in calls:
        by_type.setdefault(c.get("prompt_type") or "unlabelled", []).append(c)

    passes = []
    seen = set()
    for key, title, blurb in PASSES:
        group = by_type.get(key, [])
        seen.add(key)
        last = group[-1] if group else None
        passes.append(
            {
                "key": key,
                "title": title,
                "blurb": blurb,
                "count": len(group),
                "last_ago": _ago(last.get("timestamp")) if last else None,
                "documented": True,
                "call": _call_view(last) if last else None,
            }
        )
    # anything the log shows that the inventory doesn't know about
    for key, group in sorted(by_type.items()):
        if key in seen:
            continue
        last = group[-1]
        passes.append(
            {
                "key": key,
                "title": key,
                "blurb": "Not in the documented inventory — this pass talked to the model without being written down.",
                "count": len(group),
                "last_ago": _ago(last.get("timestamp")),
                "documented": False,
                "call": _call_view(last),
            }
        )

    gate_counts = {}
    for e in evs:
        reason = e.get("reason") or ""
        if e.get("action") in ("anti_echo_skip", "stream_erosion") or reason:
            k = reason or e.get("action")
            if k:
                gate_counts[k] = gate_counts.get(k, 0) + 1

    return {
        "run": run,
        "generated": time.strftime("%H:%M:%S"),
        "totals": {
            "calls": len(calls),
            "captions": len(captions),
            "pass_rate": round(100 * len(captions) / len(calls)) if calls else 0,
        },
        "passes": passes,
        "gates": sorted(gate_counts.items(), key=lambda kv: -kv[1]),
        "identity": _identity(),
    }


def _call_view(c):
    if not c:
        return None
    return {
        "when": (c.get("iso_timestamp") or "")[11:19],
        "model": c.get("model"),
        "stream_mode": c.get("stream_mode"),
        "history_len": c.get("history_len"),
        "frames": c.get("num_frames"),
        "system": c.get("system_prompt") or "",
        "user": c.get("prompt") or "",
        "prefill": c.get("prefill_tail") or "",
        "output": c.get("response") or "",
        "ok": bool(c.get("success", True)) and not str(c.get("response", "")).startswith("[WARNING]"),
    }


def _identity():
    p = os.path.join(MOOD_SNAPSHOT_FOLDER, "machine_identity.json")
    try:
        d = json.load(open(p, encoding="utf-8"))
    except Exception:
        return None
    return {
        "persona": d.get("core_facts", {}).get("self", ""),
        "desire": d.get("current_desire", ""),
        "belief": d.get("current_belief", ""),
        "desires": [h.get("desire", "") for h in d.get("desire_history", [])][-12:],
        "beliefs": [h.get("belief", h.get("text", "")) for h in d.get("belief_history", [])][-12:],
        "self_notes": [n.get("note", "") for n in d.get("self_notes", [])][-8:],
        "events": [e.get("event", "") for e in d.get("events", [])][-8:],
    }


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=HERE, **kw)

    def do_GET(self):
        if self.path.startswith("/api/data"):
            try:
                body = json.dumps(build_payload()).encode()
            except Exception as e:
                body = json.dumps({"error": str(e)}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path == "/":
            self.path = "/index.html"
        return super().do_GET()

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("127.0.0.1", PORT), Handler) as httpd:
        print(f"[panel] reading {MOOD_SNAPSHOT_FOLDER}")
        print(f"[panel] http://localhost:{PORT}")
        httpd.serve_forever()
