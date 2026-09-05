"""Chosen glances (Sep 5 2026, agency round — the RC-car loop for a bolted-down
machine). The caption may end with two private lines, LOOK and EXPECT, in the
machine's own words. LOOK is resolved to a gaze target (a remembered object
from the spatial registry, a plain direction, or 'stay'), the gaze driver
executes it as a glance of kind "chosen", and the next world's turn states
the consequence — "You turned to look at the door." — plus, once the view
has settled, whether the expectation held, judged by the pose referee
(unchanged / changed / never looked there). The decision lines are stripped
before the mouth gate, never displayed, never stored: the stream keeps the
thought, the body keeps the act.
"""

import re
import threading
import time
from typing import Optional

_lock = threading.Lock()
_pending: Optional[dict] = None
_current: Optional[dict] = None

_STAY = re.compile(r"^\s*(stay|hold|here|nowhere|same|nothing|none|still|keep looking|don.t move)\b", re.I)
_DIRS = {
    "left": (-30.0, 0.0),
    "right": (30.0, 0.0),
    "up": (0.0, 20.0),
    "down": (0.0, -20.0),
    "ahead": None,
    "straight": None,
    "center": None,
    "centre": None,
    "forward": None,
}
_STOP = {
    "the",
    "a",
    "an",
    "at",
    "to",
    "of",
    "on",
    "in",
    "that",
    "this",
    "my",
    "its",
    "it",
    "and",
    "or",
    "look",
    "back",
    "again",
    "over",
    "toward",
    "towards",
}


def _content_words(text: str) -> set:
    return {w for w in re.findall(r"[a-z']+", (text or "").lower()) if w not in _STOP and len(w) > 2}


def resolve_target(look_text: str) -> Optional[dict]:
    """→ {"pan", "tilt", "label", "how"} | {"how": "stay"} | None (unresolved)."""
    text = (look_text or "").strip().strip("\"'.")
    if not text:
        return None
    if _STAY.match(text):
        return {"how": "stay"}
    words = _content_words(text)
    # 1. a remembered object: best content-word overlap with a registry term
    try:
        from perception.spatial_registry import spatial_registry

        best, best_n = None, 0
        for term, e in spatial_registry.get_entries().items():
            n = len(words & _content_words(term))
            if n > best_n:
                best, best_n = (term, e), n
        if best and best_n >= 1:
            term, e = best
            return {"pan": float(e["pan"]), "tilt": float(e["tilt"]), "label": term, "how": "registry"}
    except Exception:
        pass
    # 2. a plain direction, relative to where the head is now
    try:
        from config.config import PAN_MAX, PAN_MIN, TILT_MAX, TILT_MIN
        from vision.gaze import physics_state

        pan, tilt = float(physics_state.pan), float(physics_state.tilt)
        dpan = dtilt = 0.0
        hit = []
        for w in re.findall(r"[a-z]+", text.lower()):
            if w in _DIRS:
                hit.append(w)
                d = _DIRS[w]
                if d is None:
                    pan, tilt = (PAN_MIN + PAN_MAX) / 2.0, (TILT_MIN + TILT_MAX) / 2.0
                else:
                    dpan += d[0]
                    dtilt += d[1]
        if hit:
            return {
                "pan": max(PAN_MIN, min(PAN_MAX, pan + dpan)),
                "tilt": max(TILT_MIN, min(TILT_MAX, tilt + dtilt)),
                "label": " ".join(hit),
                "how": "direction",
            }
    except Exception:
        pass
    return None


def request(look: str, expect: str, target: dict, source: str = "decision") -> dict:
    global _pending, _current
    req = {
        "look": (look or "").strip()[:120],
        "expect": (expect or "").strip()[:160],
        "pan": target["pan"],
        "tilt": target["tilt"],
        "label": target.get("label") or (look or "").strip()[:60],
        "how": target.get("how"),
        "source": source,
        "requested": time.time(),
        "started": None,
        "checked": False,
    }
    with _lock:
        _pending = req
        _current = req
    return req


def pop() -> Optional[dict]:
    """One pending chosen glance for the gaze driver, or None."""
    global _pending
    with _lock:
        req, _pending = _pending, None
    return req


def mark_started(now: float = None) -> None:
    with _lock:
        if _current is not None:
            _current["started"] = now or time.time()


def current() -> Optional[dict]:
    with _lock:
        return _current


def pending() -> bool:
    with _lock:
        return _pending is not None or (_current is not None and _current.get("started") is None and time.time() - _current.get("requested", 0) < 60)
