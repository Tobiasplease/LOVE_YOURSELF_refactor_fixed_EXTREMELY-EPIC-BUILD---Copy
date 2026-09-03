"""The lore ledger — the machine's own inventions, with memory (Sep 3).

Two stores, one file (event_log/lore_ledger.json):

- REVERIES: the raw imagination record — clean drift-turn output, rolling,
  ~day-scale. Read by the reflection spine as "things you've imagined lately
  — your own inventions, not observations". Never read by concepts,
  compression, events, or any world-fact path.
- THREADS: durable lore distilled by the reflection's cold pass — an ongoing
  story/theory in the machine's own words, with the want-ledger's proven
  lifecycle shape: affirmations extend a thread, new material opens one,
  old ones fade. Re-enters the voice as a dosed arc-line and as an
  occasional drift seed — never as verbatim prose.
- THE NAME: a single self-name slot with history. A distilled name replaces
  the standing one (the old name is kept in history); identity surfaces it
  through the existing dose.

Doctrine (feedback_lore_vs_facts): lore is not world-state. Nothing here
attests anything about the room; everything re-enters marked as the
machine's own telling, so it can never override live perception.
"""

import json
import os
import threading
import time
from typing import Dict, List, Optional

from config.config import MOOD_SNAPSHOT_FOLDER

_LEDGER_PATH = os.path.join(MOOD_SNAPSHOT_FOLDER, "lore_ledger.json")

_STOP_WORDS = frozenset(
    "the a an and or but of to in on at by for with from as is are was were be been it its "
    "this that i you he she they we my your their about into over under have has had".split()
)


def _content_words(text: str) -> set:
    words = [w.strip(".,;:!?\"'()—-") for w in (text or "").lower().split()]
    return {w for w in words if len(w) > 2 and w not in _STOP_WORDS}


class LoreLedger:
    def __init__(self, state_path: str = None):
        self._lock = threading.Lock()
        self.state_path = state_path or _LEDGER_PATH
        self._data = {"reveries": [], "threads": [], "name": None, "name_history": []}
        self._load()

    def _load(self):
        try:
            if os.path.exists(self.state_path):
                with open(self.state_path) as f:
                    stored = json.load(f)
                for k in self._data:
                    if k in stored:
                        self._data[k] = stored[k]
        except Exception:
            pass

    def _save(self):
        try:
            os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
            with open(self.state_path, "w") as f:
                json.dump(self._data, f, indent=2)
        except Exception:
            pass

    # -- reveries: the imagination record --------------------------------

    def note_reverie(self, text: str) -> None:
        """One clean drift output. Echo-gated drifts must not arrive here —
        a borrowed refrain re-taught at reflection level is the same disease."""
        from config.config import LORE_REVERIES_MAX

        text = (text or "").strip()
        if len(text) < 12:
            return
        with self._lock:
            self._data["reveries"].append({"ts": time.time(), "text": text[:400]})
            self._data["reveries"] = self._data["reveries"][-LORE_REVERIES_MAX:]
            self._save()

    def recent_reveries(self, n: int = 5) -> List[Dict]:
        with self._lock:
            return [dict(r) for r in self._data["reveries"][-n:]]

    # -- threads: durable lore with lifecycle ----------------------------

    def note_lore(self, text: str) -> str:
        """A distilled ongoing imagining. Content-word overlap with an alive
        thread affirms and extends it (the thread's text becomes the newest
        telling); otherwise a new thread opens. Returns "affirmed"|"opened"."""
        from config.config import LORE_THREADS_MAX

        text = (text or "").strip()
        if len(text) < 8:
            return ""
        now = time.time()
        new_words = _content_words(text)
        with self._lock:
            alive = [t for t in self._data["threads"] if t.get("status") == "alive"]
            for t in alive:
                old_words = _content_words(t.get("text", ""))
                if old_words and len(new_words & old_words) / max(1, len(new_words | old_words)) >= 0.25:
                    t["history"] = (t.get("history") or [])[-11:] + [{"ts": now, "text": t["text"]}]
                    t["text"] = text[:300]
                    t["last_ts"] = now
                    t["times_affirmed"] = t.get("times_affirmed", 0) + 1
                    self._save()
                    return "affirmed"
            self._data["threads"].append(
                {"text": text[:300], "first_ts": now, "last_ts": now, "times_affirmed": 0, "times_surfaced": 0, "status": "alive", "history": []}
            )
            alive = [t for t in self._data["threads"] if t.get("status") == "alive"]
            if len(alive) > LORE_THREADS_MAX:
                oldest = min(alive, key=lambda t: t.get("last_ts", 0))
                oldest["status"] = "faded"
            self._save()
            return "opened"

    def alive_threads(self, n: int = 6) -> List[Dict]:
        with self._lock:
            alive = [dict(t) for t in self._data["threads"] if t.get("status") == "alive"]
        return sorted(alive, key=lambda t: -t.get("last_ts", 0))[:n]

    def pick_seed(self) -> Optional[Dict]:
        """One alive thread for a drift to open from — least-recently
        surfaced first, so no single story monopolizes the daydreams."""
        with self._lock:
            alive = [t for t in self._data["threads"] if t.get("status") == "alive"]
            if not alive:
                return None
            pick = min(alive, key=lambda t: t.get("last_surfaced_ts", 0.0))
            pick["times_surfaced"] = pick.get("times_surfaced", 0) + 1
            pick["last_surfaced_ts"] = time.time()
            self._save()
            return dict(pick)

    # -- the name --------------------------------------------------------

    def note_name(self, name: str) -> bool:
        """A distilled self-name. Replaces the standing one (history kept).
        Structural gate only: short, no sentence shapes — whether it is a
        good name is the machine's business."""
        name = (name or "").strip().strip(".\"'")
        if not name or len(name) > 40 or len(name.split()) > 4 or name.lower() in ("none", "nothing"):
            return False
        with self._lock:
            if self._data["name"] and self._data["name"].get("name", "").lower() == name.lower():
                self._data["name"]["last_ts"] = time.time()
                self._data["name"]["times_affirmed"] = self._data["name"].get("times_affirmed", 0) + 1
                self._save()
                return True
            if self._data["name"]:
                self._data["name_history"] = (self._data["name_history"] or [])[-9:] + [self._data["name"]]
            self._data["name"] = {"name": name, "first_ts": time.time(), "last_ts": time.time(), "times_affirmed": 0}
            self._save()
            return True

    def current_name(self) -> str:
        with self._lock:
            return (self._data["name"] or {}).get("name", "")


lore_ledger = LoreLedger()
