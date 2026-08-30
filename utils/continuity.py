# continuity.py
"""
continuity.py
-------
Handles temporal awareness for the machine: tracks real time, session duration,
and provides human-readable duration descriptions for AI prompts.
"""

import time
from datetime import datetime


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


