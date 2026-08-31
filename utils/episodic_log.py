"""
utils/episodic_log.py
---------------------
Episodic event log — captures significant moments with timestamps so the
identity line can speak in terms of what's happened during the session,
not just current state.

Distinct from semantic memory (which tracks WHAT exists) and the flowing
thread (which is just the last 30 seconds). This tracks WHAT HAPPENED.

Usage:
    from utils.episodic_log import episodic_log
    episodic_log.record("person_arrived", "someone arrived")
    events = episodic_log.get_recent_events(window_seconds=3600)
"""

import json
import os
import threading
import time
from typing import Dict, List, Optional

try:
    from config.config import MOOD_SNAPSHOT_FOLDER

    _LOG_PATH = os.path.join(MOOD_SNAPSHOT_FOLDER, "episodic_events.json")
except Exception:
    _LOG_PATH = None

MAX_EVENTS = 500  # Rolling buffer
DEFAULT_WINDOW_SECONDS = 3600  # 1 hour


class EpisodicLog:
    def __init__(self):
        self._lock = threading.Lock()
        self._events: List[Dict] = []
        self._load()

    def _load(self):
        if not _LOG_PATH or not os.path.exists(_LOG_PATH):
            return
        try:
            with open(_LOG_PATH, "r") as f:
                data = json.load(f)
                self._events = data.get("events", [])[-MAX_EVENTS:]
        except Exception:
            self._events = []

    def _save(self):
        if not _LOG_PATH:
            return
        try:
            os.makedirs(os.path.dirname(_LOG_PATH), exist_ok=True)
            with open(_LOG_PATH, "w") as f:
                json.dump({"events": self._events[-MAX_EVENTS:]}, f, indent=2)
        except Exception:
            pass

    def record(self, event_type: str, description: str, metadata: Optional[Dict] = None) -> None:
        """Record a significant event. Silently ignored if duplicate within 5 seconds."""
        if not description or not description.strip():
            return
        event = {
            "timestamp": time.time(),
            "type": event_type,
            "description": description.strip()[:200],
        }
        if metadata:
            event["metadata"] = {k: str(v)[:100] for k, v in metadata.items()}

        with self._lock:
            # De-dupe rapid repeats
            if self._events:
                last = self._events[-1]
                if last.get("type") == event_type and last.get("description") == event["description"] and time.time() - last.get("timestamp", 0) < 5:
                    return
            self._events.append(event)
            if len(self._events) > MAX_EVENTS:
                self._events = self._events[-MAX_EVENTS:]
            self._save()

    def get_recent_events(self, window_seconds: int = DEFAULT_WINDOW_SECONDS, types: Optional[List[str]] = None) -> List[Dict]:
        """Return events from the last `window_seconds`, optionally filtered by type(s)."""
        cutoff = time.time() - window_seconds
        with self._lock:
            recent = [e for e in self._events if e.get("timestamp", 0) >= cutoff]
        if types is not None:
            type_set = set(types)
            recent = [e for e in recent if e.get("type") in type_set]
        return recent

    def get_last_event(self, event_type: str) -> Optional[Dict]:
        """Return the most recent event of a given type, or None."""
        with self._lock:
            for e in reversed(self._events):
                if e.get("type") == event_type:
                    return e
        return None

    def get_pairs_in_window(self, start_type: str, end_type: str, window_seconds: int = DEFAULT_WINDOW_SECONDS) -> List[Dict]:
        """For event pairs like person_arrived/person_left, return matched pairs.

        Returns a list of {start: <event>, end: <event or None>, duration_seconds: float}.
        Unmatched starts (still ongoing) get end=None and duration=time-since-start.
        """
        events = self.get_recent_events(window_seconds, types=[start_type, end_type])
        events.sort(key=lambda e: e["timestamp"])

        pairs = []
        i = 0
        now = time.time()
        while i < len(events):
            if events[i].get("type") == start_type:
                start = events[i]
                end = None
                # Look forward for matching end
                for j in range(i + 1, len(events)):
                    if events[j].get("type") == end_type:
                        end = events[j]
                        break
                    if events[j].get("type") == start_type:
                        # Another start before end — orphaned
                        break
                duration = (end["timestamp"] if end else now) - start["timestamp"]
                pairs.append({"start": start, "end": end, "duration_seconds": duration})
                i = events.index(end) + 1 if end else i + 1
            else:
                i += 1
        return pairs

    @staticmethod
    def format_ago(seconds: float) -> str:
        """Format a duration as a natural-language relative time."""
        if seconds < 60:
            return "just now"
        elif seconds < 120:
            return "a minute ago"
        elif seconds < 600:
            return f"{int(seconds / 60)} minutes ago"
        elif seconds < 3600:
            return f"about {int(seconds / 60)} minutes ago"
        elif seconds < 5400:
            return "about an hour ago"
        else:
            hours = round(seconds / 3600, 1)
            if hours == int(hours):
                hours = int(hours)
            return f"about {hours} hours ago"


# Singleton
_instance: Optional[EpisodicLog] = None
_init_lock = threading.Lock()


def get_episodic_log() -> EpisodicLog:
    global _instance
    if _instance is None:
        with _init_lock:
            if _instance is None:
                _instance = EpisodicLog()
    return _instance


# Convenience for direct use
episodic_log = get_episodic_log()
