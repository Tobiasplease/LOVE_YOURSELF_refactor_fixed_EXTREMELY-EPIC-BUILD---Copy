# vocab_promotion.py

import json
import os
import threading
import time
from collections import Counter, deque
from datetime import datetime

from config.config import (
    MOOD_SNAPSHOT_FOLDER,
    OPEN_VOCAB_GHOST_AFTER,
    OPEN_VOCAB_MAX_TERMS,
    OPEN_VOCAB_PERSON_NOUNS,
    OPEN_VOCAB_PROMOTE_THRESHOLD,
    OPEN_VOCAB_PROMOTE_WINDOW,
    OPEN_VOCAB_PROMOTION_ENABLED,
    OPEN_VOCAB_REPROMOTE_COOLDOWN,
    OPEN_VOCAB_SELF_NOUNS,
    OPEN_VOCAB_STOP_HEAD_NOUNS,
    OPEN_VOCAB_STOP_TERMS,
    OPEN_VOCAB_VOCABULARY,
)

_ABSTRACT_SUFFIXES = ("ness", "tion", "sion", "ity", "ment", "ance", "ence", "hood", "ship")


class VocabularyPromoter:
    """Phase 2, the recursive part: noun phrases recurring in the monologue are
    promoted into the detector vocabulary, so what the machine says shapes what
    it can see. Coinages, abstractions and person-phrases never compile —
    mythology stays upstairs and connects via aliases later. Ghosts (promoted,
    never detected) are kept and logged: looking for what isn't there is data."""

    def __init__(self, state_path=None, log_events=True):
        self.state_path = state_path or os.path.join(MOOD_SNAPSHOT_FOLDER, "vocab_promotion.json")
        self.log_events = log_events
        self.lock = threading.Lock()
        self._detector = None
        self._window = deque(maxlen=OPEN_VOCAB_PROMOTE_WINDOW)  # one set of terms per accepted caption
        self._counts = Counter()
        self._observe_calls = 0
        self.promoted = []  # [{term, promoted_at, promoted_ts, mentions, hits, last_hit, ghost}]
        self.history = []  # [{event, term, time, ...}] — the readable log of what earned a name
        self._cooldowns = {}  # evicted term -> eviction ts; blocks immediate re-promotion churn
        self._load_state()

    def attach_detector(self, detector):
        with self.lock:
            self._detector = detector
            if self.promoted:
                detector.set_vocabulary(self._merged_vocabulary())
                print(f"[VocabPromo] Restored {len(self.promoted)} promoted terms into detector vocabulary")

    def observe_caption(self, caption):
        if not OPEN_VOCAB_PROMOTION_ENABLED or not caption or not caption.strip():
            return
        terms = self._extract_candidates(caption)
        with self.lock:
            if len(self._window) == self._window.maxlen:
                for t in self._window[0]:
                    self._counts[t] -= 1
            self._window.append(terms)
            for t in terms:
                self._counts[t] += 1
            now = time.time()
            ripe = [
                t
                for t in terms
                if self._counts[t] >= OPEN_VOCAB_PROMOTE_THRESHOLD
                and not self._in_vocabulary(t)
                and now - self._cooldowns.get(t, 0) > OPEN_VOCAB_REPROMOTE_COOLDOWN
            ]
            for t in ripe:
                self._promote(t)
            self._observe_calls += 1
            if self._observe_calls % 25 == 0:
                self._update_ghosts()

    def get_promotion_history(self):
        with self.lock:
            return list(self.history)

    def _extract_candidates(self, caption):
        from utils.pattern_recognition import nlp  # module-level singleton, already loaded by the mood engine

        terms = set()
        for chunk in nlp(caption).noun_chunks:
            if any(t.pos_ == "PROPN" for t in chunk):
                continue  # coinages and names stay upstairs
            toks = [t for t in chunk if t.pos_ in ("NOUN", "ADJ")]
            if not toks or toks[-1].pos_ != "NOUN" or len(toks) > 4:
                continue
            head = toks[-1].lemma_.lower()
            if head in OPEN_VOCAB_PERSON_NOUNS or head in OPEN_VOCAB_STOP_HEAD_NOUNS or head in OPEN_VOCAB_SELF_NOUNS or head.endswith(_ABSTRACT_SUFFIXES):
                continue
            term = " ".join(t.lemma_.lower() for t in toks)
            if len(term) < 3 or not all(w.isalpha() or "-" in w for w in term.split()):
                continue
            if len(toks) == 1 and head in OPEN_VOCAB_STOP_TERMS:
                continue  # bare abstractions/body parts; they survive inside phrases
            terms.add(term)
        return terms

    def _in_vocabulary(self, term):
        if term in OPEN_VOCAB_VOCABULARY or any(p["term"] == term for p in self.promoted):
            return True
        # Subsumption: "monitor" vs "computer monitor", "foam finger" vs "red
        # foam finger" — near-duplicates waste slots and split the contest.
        padded = f" {term} "
        for existing in self._merged_vocabulary():
            if padded in f" {existing} " or f" {existing} " in padded:
                return True
        return False

    def _merged_vocabulary(self):
        return list(OPEN_VOCAB_VOCABULARY) + [p["term"] for p in self.promoted]

    def promote_term(self, term, origin="audit", note=""):
        """External promotion bypassing the recurrence threshold — the label
        audit's path (the rooster pattern, automated). Same filters as organic
        promotion do NOT apply: the caller vouches for the term."""
        with self.lock:
            if self._in_vocabulary(term):
                return False
            self._promote(term, origin=origin, note=note)
            return True

    def _promote(self, term, origin="monologue", note=""):
        now = time.time()
        entry = {
            "term": term,
            "promoted_at": datetime.fromtimestamp(now).isoformat(),
            "promoted_ts": now,
            "mentions": self._counts[term],
            "hits": 0,
            "last_hit": None,
            "ghost": False,
            "origin": origin,
        }
        self.promoted.append(entry)
        extra = {"mentions": entry["mentions"], "origin": origin}
        if note:
            extra["note"] = note
        self._record("promote", term, **extra)
        while len(OPEN_VOCAB_VOCABULARY) + len(self.promoted) > OPEN_VOCAB_MAX_TERMS:
            self._evict()
        if self._detector:
            self._detector.set_vocabulary(self._merged_vocabulary())
        self._save_state()

    def _evict(self):
        candidates = [p for p in self.promoted[:-1]] or self.promoted
        victim = sorted(candidates, key=lambda p: (not p["ghost"], p["hits"], p["promoted_ts"]))[0]
        self.promoted.remove(victim)
        self._cooldowns[victim["term"]] = time.time()
        self._record("evict", victim["term"], hits=victim["hits"], was_ghost=victim["ghost"])

    def _update_ghosts(self):
        if not self._detector:
            return
        hit_counts = self._detector.get_term_hit_counts()
        now = time.time()
        for p in self.promoted:
            stats = hit_counts.get(p["term"])
            if stats:
                p["hits"] = stats["count"]
                p["last_hit"] = stats["last"]
            if not p["ghost"] and p["hits"] == 0 and now - p["promoted_ts"] > OPEN_VOCAB_GHOST_AFTER:
                p["ghost"] = True
                self._record("ghost", p["term"], promoted_at=p["promoted_at"])
        self._save_state()

    def _record(self, event, term, **extra):
        stamp = datetime.now().isoformat()
        self.history.append({"event": event, "term": term, "time": stamp, **extra})
        print(f"[VocabPromo] {event.upper()}: {term} {extra if extra else ''}")
        if self.log_events:
            try:
                from event_logging.event_logger import log_json_entry
                from event_logging.log_type import LogType

                log_json_entry(LogType.VOCAB_PROMOTION, {"event": event, "term": term, **extra})
            except Exception as e:
                print(f"[VocabPromo] Event log failed: {e}")

    def _load_state(self):
        try:
            if os.path.exists(self.state_path):
                with open(self.state_path) as f:
                    data = json.load(f)
                self.promoted = data.get("promoted", [])
                self.history = data.get("history", [])
        except Exception as e:
            print(f"[VocabPromo] Could not load state: {e}")

    def _save_state(self):
        try:
            os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
            with open(self.state_path, "w") as f:
                json.dump({"promoted": self.promoted, "history": self.history}, f, indent=2)
        except Exception as e:
            print(f"[VocabPromo] Could not save state: {e}")


vocab_promoter = VocabularyPromoter()
