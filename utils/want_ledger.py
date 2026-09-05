"""
utils/want_ledger.py
--------------------
B3 (Aug 31): the want's lifecycle as a record — the resolution ledger.

Before this, a want had no history: distillation silently overwrote it, only
drawing could resolve it, and nothing remembered whether it was ever pursued,
refused, or answered. Wants were decorative. This ledger records the FACTS of
each want's life — when it formed, how often it was re-affirmed, how many
times its pursuit was refused, whether the machine acted on it, and what it
became (in the machine's own words, from the distiller's BECAME slot — never
ours, per the no-content-priors rule). The facts feed two surfaces: the
"Preoccupied with" line's arc tail, and the reflection prompt's standing-want
material — where an old refused want becomes something the machine can name
feelings about. Curdling is the machine's move; we only keep the receipts.
"""

import json
import os
import threading
import time
from typing import Dict, List, Optional

try:
    from config.config import MOOD_SNAPSHOT_FOLDER

    _LEDGER_PATH = os.path.join(MOOD_SNAPSHOT_FOLDER, "want_ledger.json")
except Exception:
    _LEDGER_PATH = None

MAX_ENTRIES = 50


class WantLedger:
    def __init__(self):
        self._lock = threading.Lock()
        self._entries: List[Dict] = []
        self._load()

    def _load(self):
        if not _LEDGER_PATH or not os.path.exists(_LEDGER_PATH):
            return
        try:
            with open(_LEDGER_PATH) as f:
                self._entries = json.load(f).get("wants", [])[-MAX_ENTRIES:]
        except Exception:
            self._entries = []

    def _save(self):
        if not _LEDGER_PATH:
            return
        try:
            os.makedirs(os.path.dirname(_LEDGER_PATH), exist_ok=True)
            with open(_LEDGER_PATH, "w") as f:
                json.dump({"wants": self._entries[-MAX_ENTRIES:]}, f, indent=2)
        except Exception:
            pass

    def _current(self) -> Optional[Dict]:
        if self._entries and self._entries[-1].get("ended_at") is None:
            return self._entries[-1]
        return None

    # ------------------------------------------------------------------
    # Lifecycle events (called from the distillation write path and the
    # drawing trigger/refusal path — the ledger never decides anything)
    # ------------------------------------------------------------------

    def note_want(self, text: str, affirmed: bool, became: str = "") -> None:
        """A distillation produced a want. affirmed=True means the write path
        judged it the same wish persisting (its clock keeps running); False
        closes the current want — outcome in the machine's own words if the
        distiller offered one, else 'superseded' — and opens the new one."""
        if not text or not text.strip():
            return
        now = time.time()
        with self._lock:
            cur = self._current()
            if affirmed and cur:
                cur["affirmed"] = int(cur.get("affirmed", 0)) + 1
                self._save()
                return
            if cur:
                cur["ended_at"] = now
                cur["outcome"] = (became or "").strip() or "abandoned"
                cur["kind"] = "became" if (became or "").strip() else "abandoned"  # Sep 5: replaced without resolution = abandoned, and counted
            self._entries.append(
                {
                    "text": text.strip()[:300],
                    "formed_at": now,
                    "affirmed": 0,
                    "refusals": 0,
                    "acted": False,
                    "ended_at": None,
                    "outcome": None,
                }
            )
            self._entries = self._entries[-MAX_ENTRIES:]
            self._save()

    def note_faded(self, became: str = "") -> None:
        """The want was cleared with no successor — it simply stopped."""
        with self._lock:
            cur = self._current()
            if cur:
                cur["ended_at"] = time.time()
                cur["outcome"] = (became or "").strip() or "faded"
                cur["kind"] = "drawn" if (became or "").lower().startswith(("drawn", "spent by drawing")) else "faded"
                self._save()

    def note_resolved(self, kind: str, words: str) -> None:
        """Sep 5 (agency round — artist: a want closes through whatever it is
        about, never routed to drawing). The distiller's RESOLVED slot, in the
        machine's own words: kind is 'understood' (thought through) or 'let go'."""
        with self._lock:
            cur = self._current()
            if cur:
                cur["ended_at"] = time.time()
                cur["outcome"] = (words or "").strip() or kind
                cur["kind"] = kind
                self._save()

    def note_met(self) -> None:
        """A real arrival while this want was about a person."""
        with self._lock:
            cur = self._current()
            if cur and not cur.get("met"):
                cur["met"] = True
                cur["met_at"] = time.time()
                self._save()

    def abandoned_count(self, n: int = 10) -> int:
        with self._lock:
            done = [e for e in self._entries if e.get("ended_at") is not None][-n:]
        return sum(1 for e in done if e.get("kind") == "abandoned")

    def note_refusal(self) -> None:
        """Pursuit was blocked (paper gate, hardware) while this want lived."""
        with self._lock:
            cur = self._current()
            if cur:
                cur["refusals"] = int(cur.get("refusals", 0)) + 1
                self._save()

    def note_acted(self) -> None:
        """The machine physically acted (a drawing executed) during this want."""
        with self._lock:
            cur = self._current()
            if cur and not cur.get("acted"):
                cur["acted"] = True
                cur["acted_at"] = time.time()
                self._save()

    # ------------------------------------------------------------------
    # Read surfaces (facts only — the mind does the meaning)
    # ------------------------------------------------------------------

    def current_facts(self) -> Optional[Dict]:
        """Age and pursuit counts for the live want, or None."""
        with self._lock:
            cur = self._current()
            if not cur:
                return None
            return {
                "text": cur["text"],
                "age_s": time.time() - cur.get("formed_at", time.time()),
                "affirmed": int(cur.get("affirmed", 0)),
                "refusals": int(cur.get("refusals", 0)),
                "acted": bool(cur.get("acted", False)),
                "met": bool(cur.get("met", False)),
            }

    def recently_resolved(self, n: int = 3) -> List[Dict]:
        """The last few wants that ended, oldest first — reflection material."""
        with self._lock:
            done = [e for e in self._entries if e.get("ended_at") is not None]
        return [
            {
                "text": e["text"],
                "outcome": e.get("outcome") or "abandoned",
                "kind": e.get("kind") or ("became" if e.get("outcome") not in (None, "superseded", "faded") else "abandoned"),
                "lived_s": (e.get("ended_at") or 0) - (e.get("formed_at") or 0),
                "refusals": int(e.get("refusals", 0)),
                "acted": bool(e.get("acted", False)),
            }
            for e in done[-n:]
        ]


want_ledger = WantLedger()
