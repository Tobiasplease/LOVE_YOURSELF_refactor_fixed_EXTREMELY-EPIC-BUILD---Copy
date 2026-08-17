# spatial_registry.py

import json
import os
import random
import threading
import time
from datetime import datetime

from config.config import (
    MOOD_SNAPSHOT_FOLDER,
    PAN_MAX,
    PAN_MIN,
    SPATIAL_MENTION_BOOST_S,
    SPATIAL_REGISTRY_EMA,
    SPATIAL_REGISTRY_ENABLED,
    SPATIAL_REGISTRY_HFOV,
    SPATIAL_REGISTRY_MAX_AGE,
    SPATIAL_REGISTRY_VFOV,
    TILT_MAX,
    TILT_MIN,
)


class SpatialRegistry:
    """The world map: object term -> where it is, in servo angles. Fed by the
    open-vocab detector (Phase 3); anchors are EMA-smoothed across sightings,
    decay when unseen, and persist across restarts. This is what turns idle
    gaze from a random walk into a lookup — and makes absence an event: going
    to where a thing was and not detecting it is a real observation."""

    def __init__(self, state_path=None):
        self.state_path = state_path or os.path.join(MOOD_SNAPSHOT_FOLDER, "spatial_registry.json")
        self.lock = threading.Lock()
        self.entries = {}  # term -> {pan, tilt, conf, hits, first_seen, last_seen, mention_ts?, misses?}
        self._absence_events = []  # {term, ts} — verified-gone moments, drained by the caption loop
        self._last_save = 0.0
        self._load()

    def update_from_detections(self, detections, frame_shape):
        """Fold one detector pass into the map. Only settled detections move
        anchors — a mid-move capture's geometry is untrustworthy."""
        if not SPATIAL_REGISTRY_ENABLED or not detections:
            return
        h, w = frame_shape[0], frame_shape[1]
        now = time.time()
        with self.lock:
            for det in detections:
                if not det.get("settled", True) or det.get("pan") is None:
                    continue
                x1, y1, x2, y2 = det["box"]
                cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                # Same convention as machine.py's person_angle (~60 deg HFOV);
                # image y grows downward, servo tilt grows upward.
                pan = det["pan"] + ((cx / w) - 0.5) * SPATIAL_REGISTRY_HFOV
                tilt = det["tilt"] + (0.5 - (cy / h)) * SPATIAL_REGISTRY_VFOV
                e = self.entries.get(det["term"])
                if e is None:
                    self.entries[det["term"]] = {
                        "pan": pan,
                        "tilt": tilt,
                        "conf": det["conf"],
                        "hits": 1,
                        "first_seen": now,
                        "last_seen": now,
                    }
                else:
                    a = SPATIAL_REGISTRY_EMA
                    e["pan"] += (pan - e["pan"]) * a
                    e["tilt"] += (tilt - e["tilt"]) * a
                    e["conf"] += (det["conf"] - e["conf"]) * a
                    e["hits"] += 1
                    e["last_seen"] = now
                    e["misses"] = 0  # re-seen anywhere = the memory holds
            stale = [t for t, e in self.entries.items() if now - e["last_seen"] > SPATIAL_REGISTRY_MAX_AGE]
            for t in stale:
                del self.entries[t]  # forgetting as garbage collection
        if now - self._last_save > 60.0:
            self._last_save = now
            self._save()

    def get_entries(self):
        with self.lock:
            return {t: dict(e) for t, e in self.entries.items()}

    def get_anchor(self, term):
        with self.lock:
            e = self.entries.get(term)
            return (e["pan"], e["tilt"]) if e else None

    def mark_audit(self, term, verdict, detail=""):
        """Label-audit bookkeeping on the entry (persisted with the map)."""
        with self.lock:
            e = self.entries.get(term)
            if e is not None:
                e["last_audit"] = time.time()
                e["audit_verdict"] = verdict
                if detail:
                    e["audit_detail"] = detail
        self._save()

    def pick_glance_target(self, explore_weight=0.25):
        """A place worth looking: usually the known object gone longest
        unchecked (staleness-weighted, recent discoveries boosted, and a term
        the monologue just spoke of pulls hardest — thought leads gaze),
        sometimes an under-visited stretch of the room (explore). Returns
        {pan, tilt, term|None, kind} or None if the map is empty."""
        now = time.time()
        with self.lock:
            entries = [(t, dict(e)) for t, e in self.entries.items()]
        if not entries or random.random() < explore_weight:
            return self._explore_target(entries)
        weights = []
        for t, e in entries:
            staleness = min(now - e["last_seen"], 3600.0) + 30.0
            novelty = 3.0 if now - e["first_seen"] < 600.0 else 1.0
            mentioned = 4.0 if now - e.get("mention_ts", 0.0) < SPATIAL_MENTION_BOOST_S else 1.0
            weights.append(staleness * novelty * mentioned)
        term, e = random.choices(entries, weights=weights)[0]
        return {"pan": self._clamp_pan(e["pan"]), "tilt": self._clamp_tilt(e["tilt"]), "term": term, "kind": "revisit"}

    def note_mentions(self, text):
        """The monologue spoke — terms it named get a temporary pull on the
        next glances (the mind's eye leads the physical one)."""
        if not text:
            return
        low = text.lower()
        now = time.time()
        with self.lock:
            for t, e in self.entries.items():
                if t.lower() in low:
                    e["mention_ts"] = now

    def note_glance_result(self, term, found):
        """Discernment: the gaze went to where a thing should be and the
        detector reported. Seen -> the memory holds. Not seen -> confidence
        decays; the 2nd consecutive miss becomes an absence EVENT for the
        mind ("it isn't where it was"); the 4th forgets the entry — the map
        stores where things ARE, not where they once were. Returns
        "seen" | "missed" | "gone" | None."""
        now = time.time()
        with self.lock:
            e = self.entries.get(term)
            if e is None:
                return None
            if found:
                e["misses"] = 0
                return "seen"
            e["misses"] = e.get("misses", 0) + 1
            e["conf"] *= 0.7
            if e["misses"] == 2:
                self._absence_events.append({"term": term, "ts": now})
                self._absence_events = self._absence_events[-6:]
            if e["misses"] >= 4:
                del self.entries[term]
                return "gone"
            return "missed"

    def pop_absence_event(self):
        """One verified absence for the caption loop, oldest first, or None."""
        with self.lock:
            return self._absence_events.pop(0) if self._absence_events else None

    def _explore_target(self, entries):
        n_buckets = 6
        span = (PAN_MAX - PAN_MIN) / n_buckets
        coverage = [0.0] * n_buckets
        now = time.time()
        for _, e in entries:
            b = int(min(max((e["pan"] - PAN_MIN) / span, 0), n_buckets - 1))
            coverage[b] += 1.0 / (1.0 + (now - e["last_seen"]) / 600.0)
        emptiest = min(range(n_buckets), key=lambda b: coverage[b])
        pan = PAN_MIN + span * (emptiest + random.random())
        tilt = random.uniform(TILT_MIN + 10, TILT_MAX - 20)
        return {"pan": self._clamp_pan(pan), "tilt": self._clamp_tilt(tilt), "term": None, "kind": "explore"}

    @staticmethod
    def _clamp_pan(v):
        return max(PAN_MIN, min(PAN_MAX, v))

    @staticmethod
    def _clamp_tilt(v):
        return max(TILT_MIN, min(TILT_MAX, v))

    def _load(self):
        try:
            if os.path.exists(self.state_path):
                with open(self.state_path) as f:
                    self.entries = json.load(f).get("entries", {})
        except Exception as e:
            print(f"[SpatialReg] Could not load state: {e}")

    def _save(self):
        try:
            os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
            with open(self.state_path, "w") as f:
                json.dump({"entries": self.entries, "saved_at": datetime.now().isoformat()}, f, indent=2)
        except Exception as e:
            print(f"[SpatialReg] Could not save state: {e}")


spatial_registry = SpatialRegistry()
