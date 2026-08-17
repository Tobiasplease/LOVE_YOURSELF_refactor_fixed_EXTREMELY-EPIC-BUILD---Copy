"""Prompt panel — the control surface for the system's authored prompt text.

Phase 2 (Aug 17): read/write. Reads the prompt registry (every hardcoded
fragment, the pass assembly manifests, the store loop map) and the event log
(calls as sent). Writes exactly one thing: config/prompt_overrides.json, via
the registry's validated set_override — an edit lands on the machine's next
cycle, git-tracked canonical text stays the baseline.

    python prompt_panel/server.py          # then open http://localhost:8770

Design rules kept from phase 1: show the prompt AS SENT next to the template;
any pass in the log that the registry doesn't know is flagged, not hidden.
"""

import http.server
import json
import os
import socketserver
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from captioner import prompt_registry as registry

HERE = os.path.dirname(os.path.abspath(__file__))
PORT = int(os.getenv("PROMPT_PANEL_PORT", 8770))

try:
    from config.config import MOOD_SNAPSHOT_FOLDER
except Exception:
    MOOD_SNAPSHOT_FOLDER = os.path.join(os.path.dirname(HERE), "event_log")


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
        "system_truncated": (c.get("full_system_prompt_length") or 0) > len(c.get("system_prompt") or ""),
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


def build_run_payload():
    """Live-run data: last call per pass, counts, gates, identity."""
    run, evs = _newest_run_with_calls()
    calls = [e for e in evs if e.get("type") == "llm_api_call"]
    captions = [e for e in evs if e.get("type") == "caption" and e.get("caption")]

    by_type = {}
    for c in calls:
        by_type.setdefault(c.get("prompt_type") or "unlabelled", []).append(c)

    activity = {}
    for key, group in by_type.items():
        last = group[-1]
        activity[key] = {
            "count": len(group),
            "last_ago": _ago(last.get("timestamp")),
            "call": _call_view(last),
            "known": key in registry.PASSES,
        }

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
        "activity": activity,
        "gates": sorted(gate_counts.items(), key=lambda kv: -kv[1]),
        "identity": _identity(),
    }


def build_registry_payload():
    ov = registry._load_overrides()
    fragments = []
    for fid, frag in registry.FRAGMENTS.items():
        fragments.append(
            {
                "id": fid,
                "title": frag.get("title", fid),
                "text": frag["text"],
                "note": frag.get("note", ""),
                "used_by": frag.get("used_by", []),
                "placeholders": frag.get("placeholders", []),
                "override": ov.get(fid),
            }
        )
    return {
        "fragments": fragments,
        "stores": registry.STORES,
        "passes": registry.PASSES,
        "overrides_path": os.path.relpath(registry.OVERRIDES_PATH, os.path.dirname(HERE)),
        "override_count": len(ov),
    }


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=HERE, **kw)

    def _json(self, obj, status=200):
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        try:
            if self.path.startswith("/api/run"):
                return self._json(build_run_payload())
            if self.path.startswith("/api/registry"):
                return self._json(build_registry_payload())
        except Exception as e:
            return self._json({"error": str(e)}, 500)
        if self.path == "/":
            self.path = "/index.html"
        return super().do_GET()

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
        except Exception:
            return self._json({"error": "bad request body"}, 400)
        try:
            if self.path == "/api/override":
                registry.set_override(body["id"], body["text"])
                return self._json({"ok": True, "override": registry._load_overrides().get(body["id"])})
            if self.path == "/api/revert":
                registry.clear_override(body["id"])
                return self._json({"ok": True})
        except (KeyError, ValueError) as e:
            return self._json({"error": str(e)}, 400)
        except Exception as e:
            return self._json({"error": str(e)}, 500)
        return self._json({"error": "unknown endpoint"}, 404)

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("127.0.0.1", PORT), Handler) as httpd:
        print(f"[panel] events: {MOOD_SNAPSHOT_FOLDER}")
        print(f"[panel] overrides: {registry.OVERRIDES_PATH}")
        print(f"[panel] http://localhost:{PORT}")
        httpd.serve_forever()
