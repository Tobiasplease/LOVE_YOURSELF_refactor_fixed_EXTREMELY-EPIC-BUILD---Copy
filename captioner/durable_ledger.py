"""Durable fact ledger (July 30) — the spine the seven memory stores never had.

Every store today is a ring buffer or a single slot: self_notes caps at 30,
events at 20, the persona is one sentence overwritten by every distill. A
self-given name had nowhere to LIVE — it had to re-win the persona slot every
20 minutes or drown (it survived the Nemo era by accident, and died every
other one). This ledger gives facts a permanence hierarchy:

  evolving  — noticed once; may fade (capped, least-recently-confirmed drops)
  stable    — confirmed >=3 times across >=2 distinct days; uncapped, earned
  permanent — promoted only by the (future) dream pass or the artist; never
              automatically. "I am called X" belongs here eventually.

The daytime writers (memory-diff self-facts, distill traits) call note_fact:
a rough-match against existing entries is a CONFIRMATION, not a duplicate —
re-noticing is how a fact earns permanence. Promotion happens on the spot for
evolving->stable; permanence is deliberately out of the fast loop's reach
(chinese-whispers containment: the fast loop can strengthen, never enshrine).

Model-agnostic plain JSON at event_log/durable_ledger.json — the thread must
survive model swaps. Read-back: render() feeds stable+permanent lines into
the system prompt, so what has stayed true rides every awakening.
"""

import json
import os
import threading
import time
from typing import List, Optional

_LEDGER_PATH = os.path.join("event_log", "durable_ledger.json")

_STOP = {"i", "to", "the", "a", "an", "and", "it", "my", "of", "for", "am", "is", "me"}


def _roughly_same(a: str, b: str) -> bool:
    wa = set(a.lower().rstrip(".").split()) - _STOP
    wb = set(b.lower().rstrip(".").split()) - _STOP
    if not wa or not wb:
        return False
    return len(wa & wb) / max(len(wa | wb), 1) >= 0.5


class DurableLedger:
    def __init__(self, path: str = _LEDGER_PATH):
        self._path = path
        self._lock = threading.Lock()
        self._facts: List[dict] = []
        self._load()

    def _load(self) -> None:
        try:
            with open(self._path) as f:
                data = json.load(f)
            self._facts = [e for e in data.get("facts", []) if e.get("fact")]
        except Exception:
            self._facts = []

    def _save(self) -> None:
        tmp = self._path + ".tmp"
        try:
            with open(tmp, "w") as f:
                json.dump({"facts": self._facts}, f, indent=1, ensure_ascii=False)
            os.replace(tmp, self._path)
        except Exception:
            pass

    def note_fact(self, fact: str, source: str = "memory_diff") -> Optional[str]:
        """Record or confirm a fact. Returns 'new' | 'confirmed' | 'promoted' | None."""
        fact = (fact or "").strip()
        if not fact or len(fact.split()) > 24:
            return None
        today = time.strftime("%Y-%m-%d")
        with self._lock:
            for e in self._facts:
                if _roughly_same(fact, e["fact"]):
                    e["last_confirmed"] = time.time()
                    e["confirmations"] = e.get("confirmations", 1) + 1
                    if e.get("cls") == "challenged":
                        e["reconfirm"] = e.get("reconfirm", 0) + 1
                        if e["reconfirm"] >= 2:
                            e["cls"] = "stable"
                            e.pop("challenged_ts", None)
                            print(f"[📌] challenged fact re-held — stable again: {e['fact']}")
                        self._save()
                        return "reconfirmed"
                    days = e.setdefault("days", [])
                    if today not in days:
                        days.append(today)
                    result = "confirmed"
                    if e.get("cls") == "evolving" and e["confirmations"] >= 3 and len(days) >= 2:
                        e["cls"] = "stable"
                        result = "promoted"
                        print(f"[📌] Fact held across days — now stable: {e['fact']}")
                    self._save()
                    return result
            self._facts.append(
                {
                    "fact": fact,
                    "cls": "evolving",
                    "established": time.time(),
                    "last_confirmed": time.time(),
                    "confirmations": 1,
                    "days": [today],
                    "source": source,
                }
            )
            evolving = [e for e in self._facts if e.get("cls") == "evolving"]
            if len(evolving) > 40:
                drop = min(evolving, key=lambda e: e.get("last_confirmed", 0))
                self._facts.remove(drop)
            self._save()
            return "new"

    def challenge(self, text: str) -> Optional[str]:
        """Sep 5 (time-and-loop round, persona baseline): the distill's NO LONGER
        TRUE slot quotes a held fact back. A rough match marks it CHALLENGED —
        it leaves the "stayed true" line and rides a "lately in doubt" line
        instead. Two fresh confirmations (in note_fact) restore it. This is the
        turn path a persona needs to develop rather than only deepen: for 41 of
        44 facts one night was pure confirmation of one idea."""
        text = (text or "").strip()
        if not text:
            return None
        with self._lock:
            for e in self._facts:
                if e.get("cls") in ("stable", "evolving") and _roughly_same(text, e["fact"]):
                    e["cls"] = "challenged"
                    e["challenged_ts"] = time.time()
                    e["reconfirm"] = 0
                    self._save()
                    print(f"[📌?] fact challenged: {e['fact']}")
                    return e["fact"]
        return None

    def render_challenged(self, max_chars: int = 240) -> str:
        with self._lock:
            keep = sorted((e for e in self._facts if e.get("cls") == "challenged"), key=lambda e: -e.get("challenged_ts", 0))
        out, total = [], 0
        for e in keep:
            line = e["fact"].rstrip(".")
            if total + len(line) > max_chars:
                break
            out.append(line)
            total += len(line)
        return " ".join(f"{l}." for l in out)

    def render_evolving_edge(self, max_chars: int = 240, min_confirmations: int = 2) -> str:
        """What is newly taking hold — evolving facts confirmed more than once."""
        with self._lock:
            keep = sorted(
                (e for e in self._facts if e.get("cls") == "evolving" and e.get("confirmations", 1) >= min_confirmations),
                key=lambda e: -e.get("last_confirmed", 0),
            )
        out, total = [], 0
        for e in keep:
            line = e["fact"].rstrip(".")
            if total + len(line) > max_chars:
                break
            out.append(line)
            total += len(line)
        return " ".join(f"{l}." for l in out)

    @staticmethod
    def days_words(n_days: int) -> str:
        if n_days >= 60:
            return "a couple of months"
        if n_days >= 28:
            return "over a month"
        if n_days >= 14:
            return "a couple of weeks"
        if n_days >= 7:
            return "over a week"
        if n_days >= 3:
            return "a few days"
        if n_days == 2:
            return "two days"
        return "a day"

    def held_spans(self) -> dict:
        """Audible time (Sep 5): how long the stable core has held, in words."""
        with self._lock:
            stable = [e for e in self._facts if e.get("cls") in ("permanent", "stable")]
        if not stable:
            return {}
        spans = []
        for e in stable:
            days = e.get("days") or []
            try:
                first = time.mktime(time.strptime(days[0], "%Y-%m-%d")) if days else e.get("established", time.time())
            except Exception:
                first = e.get("established", time.time())
            spans.append(max(1, int((time.time() - first) / 86400) + 1))
        return {"oldest": self.days_words(max(spans)), "newest": self.days_words(min(spans)), "count": len(stable)}

    def render(self, max_chars: int = 400) -> str:
        """Permanent then stable facts, most recently confirmed first — the
        lines that ride every awakening. Empty until something is earned."""
        with self._lock:
            keep = [e for e in self._facts if e.get("cls") in ("permanent", "stable")]
        keep.sort(key=lambda e: (e.get("cls") != "permanent", -e.get("last_confirmed", 0)))
        out = []
        total = 0
        for e in keep:
            line = e["fact"].rstrip(".")
            if total + len(line) > max_chars:
                break
            out.append(line)
            total += len(line)
        return " ".join(f"{l}." for l in out)

    def all_facts(self) -> List[dict]:
        """For the future dream pass — the sole authority for permanence."""
        with self._lock:
            return [dict(e) for e in self._facts]


_ledger: Optional[DurableLedger] = None
_ledger_lock = threading.Lock()


def get_durable_ledger() -> DurableLedger:
    global _ledger
    with _ledger_lock:
        if _ledger is None:
            _ledger = DurableLedger()
        return _ledger
