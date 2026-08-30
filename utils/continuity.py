# continuity.py
"""
continuity.py
-------
Handles temporal awareness for the machine: tracks real time, session duration,
and provides human-readable duration descriptions for AI prompts.
"""

import time
from datetime import datetime


class TimeKeeper:
    """Tracks session time and provides duration helpers."""

    def __init__(self):
        self.start_time = time.time()
        self.last_reset = self.start_time

    def get_awake_time(self):
        """Returns time awake in seconds."""
        return time.time() - self.start_time

    def reset_awake_time(self):
        """Resets the awake timer."""
        self.start_time = time.time()

    def get_session_time(self):
        """Returns current session time in seconds."""
        return time.time() - self.last_reset

    def new_session(self):
        """Starts a new session timer."""
        self.last_reset = time.time()


# Global timekeeper instance
timekeeper = TimeKeeper()


def now():
    """Return current timestamp."""
    return time.time()


def describe_duration(start_time):
    """
    Convert a duration into a human-readable description.

    Args:
        start_time: Unix timestamp of when the duration started

    Returns:
        Human-readable duration string
    """
    elapsed = time.time() - start_time

    if elapsed < 60:
        return f"{int(elapsed)} seconds"
    elif elapsed < 3600:
        minutes = int(elapsed / 60)
        return f"{minutes} minute{'s' if minutes != 1 else ''}"
    elif elapsed < 86400:
        hours = int(elapsed / 3600)
        minutes = int((elapsed % 3600) / 60)
        if minutes == 0:
            return f"{hours} hour{'s' if hours != 1 else ''}"
        return f"{hours} hour{'s' if hours != 1 else ''} and {minutes} minute{'s' if minutes != 1 else ''}"
    else:
        days = int(elapsed / 86400)
        hours = int((elapsed % 86400) / 3600)
        if hours == 0:
            return f"{days} day{'s' if days != 1 else ''}"
        return f"{days} day{'s' if days != 1 else ''} and {hours} hour{'s' if hours != 1 else ''}"


def describe_time_gap(timestamp):
    """
    Describe how long ago something happened.

    Args:
        timestamp: Unix timestamp of the event

    Returns:
        Human-readable description of time gap
    """
    gap = time.time() - timestamp

    if gap < 60:
        return "just now"
    elif gap < 3600:
        minutes = int(gap / 60)
        return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
    elif gap < 86400:
        hours = int(gap / 3600)
        return f"{hours} hour{'s' if hours != 1 else ''} ago"
    elif gap < 604800:  # 1 week
        days = int(gap / 86400)
        return f"{days} day{'s' if days != 1 else ''} ago"
    else:
        weeks = int(gap / 604800)
        return f"{weeks} week{'s' if weeks != 1 else ''} ago"


def get_current_time_description():
    """Get a human-readable description of current time."""
    dt = datetime.now()

    hour = dt.hour
    if 5 <= hour < 12:
        time_of_day = "morning"
    elif 12 <= hour < 17:
        time_of_day = "afternoon"
    elif 17 <= hour < 21:
        time_of_day = "evening"
    else:
        time_of_day = "night"

    return f"{time_of_day} ({dt.strftime('%I:%M %p')})"


