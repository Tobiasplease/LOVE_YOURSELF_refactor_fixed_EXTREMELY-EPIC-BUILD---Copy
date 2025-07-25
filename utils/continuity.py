# continuity.py
"""
continuity.py
-------
Handles temporal awareness for the machine: tracks real time, session uptime, elapsed durations, and poetic time expressions.
"""

import time
from datetime import datetime, timedelta

# -- Utility functions --


def now() -> float:
    """Returns current time as UNIX timestamp."""
    return time.time()


def to_datetime(ts: float) -> datetime:
    """Convert timestamp to datetime object."""
    return datetime.fromtimestamp(ts)


def elapsed_seconds(ts_start: float, ts_end: float | None = None) -> float:
    """Seconds between two timestamps."""
    if ts_end is None:
        ts_end = now()
    return ts_end - ts_start


def elapsed_minutes(ts_start: float, ts_end: float | None = None) -> float:
    return elapsed_seconds(ts_start, ts_end) / 60


def elapsed_hours(ts_start: float, ts_end: float | None = None) -> float:
    return elapsed_seconds(ts_start, ts_end) / 3600


def elapsed_days(ts_start: float, ts_end: float | None = None) -> float:
    return elapsed_seconds(ts_start, ts_end) / 86400


def describe_duration(ts_start: float, ts_end: float | None = None) -> str:
    """Human/poetic string for time since ts_start."""
    seconds = elapsed_seconds(ts_start, ts_end)
    if seconds < 60:
        return "less than a minute"
    elif seconds < 3600:
        mins = int(seconds // 60)
        return f"{mins} minute{'s' if mins > 1 else ''}"
    elif seconds < 86400:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        if mins == 0:
            return f"{hours} hour{'s' if hours > 1 else ''}"
        return f"{hours} hour{'s' if hours > 1 else ''} and {mins} minute{'s' if mins != 1 else ''}"
    else:
        days = int(seconds // 86400)
        hours = int((seconds % 86400) // 3600)
        if hours == 0:
            return f"{days} day{'s' if days > 1 else ''}"
        return f"{days} day{'s' if days > 1 else ''} and {hours} hour{'s' if hours != 1 else ''}"


def since_datetime_string(ts: float) -> str:
    """Returns a poetic string: 'since yesterday', 'since this morning', etc."""
    dt = to_datetime(ts)
    now_dt = datetime.now()
    delta = now_dt - dt

    if delta < timedelta(minutes=1):
        return "just now"
    elif delta < timedelta(hours=1):
        return f"{int(delta.total_seconds() // 60)} minutes ago"
    elif delta < timedelta(hours=6):
        return f"{int(delta.total_seconds() // 3600)} hours ago"
    elif dt.date() == now_dt.date():
        return "earlier today"
    elif dt.date() == (now_dt - timedelta(days=1)).date():
        return "yesterday"
    else:
        return dt.strftime("on %b %d, %Y")


# -- Example: track system boot, session starts, etc. --
class Timekeeper:
    """Tracks Lint's persistent time, awake time, and session continuity."""

    def __init__(self):
        self.system_start: float = now()
        self.last_wake: float = self.system_start
        self.session_starts: list[float] = [self.system_start]

    def new_session(self):
        self.last_wake = now()
        self.session_starts.append(self.last_wake)

    def get_awake_time(self) -> float:
        """Time since last wake in seconds."""
        return now() - self.last_wake

    def get_total_uptime(self) -> float:
        return now() - self.system_start

    def time_since(self, ts: float) -> str:
        return describe_duration(ts)


# Singleton instance
timekeeper = Timekeeper()


# -- NEW: Presence phrase generator --
def get_presence_phrase() -> str:
    """Returns a neutral phrase describing how long you have been awake."""
    awake_time = timekeeper.get_awake_time()
    if awake_time < 60:
        return "you just woke up"
    elif awake_time < 600:
        return "you’ve been here for a while"
    elif awake_time < 3600:
        return "you’ve been here for a long time"
    else:
        return "you’ve been here for a very long time"
