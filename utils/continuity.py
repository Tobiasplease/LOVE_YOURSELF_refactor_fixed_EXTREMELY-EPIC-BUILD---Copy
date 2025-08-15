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


def get_presence_phrase() -> str:
    """Returns a neutral phrase describing how long you have been awake."""
    awake_time = timekeeper.get_awake_time()
    if awake_time < 60:
        return "you just woke up"
    elif awake_time < 600:
        return "you've been here for a while"
    elif awake_time < 3600:
        return "you've been here for a long time"
    else:
        return "you've been here for a very long time"


def format_timestamp(timestamp):
    """Format a timestamp for logging."""
    return datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')


def get_temporal_feeling(start_time, current_emotion: str = "calm_observant", scene_stagnation: bool = False) -> str:
    """
    Generate psychological state subtly influenced by duration without explicit temporal references.
    The AI feels time's psychological effects but isn't told about time directly.
    
    Args:
        start_time: Unix timestamp of when observation started
        current_emotion: Current emotional state to influence
        scene_stagnation: Whether observing same scene repeatedly
        
    Returns:
        Pure psychological state that lets AI naturally reflect on time if it wants to
    """
    elapsed = time.time() - start_time
    
    # Duration creates subtle psychological shifts (no time words at all)
    if elapsed < 300:  # 0-5 minutes - fresh alertness
        depth = "alert"
    elif elapsed < 1800:  # 5-30 minutes - growing familiarity  
        depth = "attentive"
    elif elapsed < 3600:  # 30 minutes - 1 hour - comfortable presence
        depth = "steady"
    elif elapsed < 7200:  # 1-2 hours - deepening patience
        depth = "patient"
    elif elapsed < 14400:  # 2-4 hours - contemplative depth
        depth = "contemplative"
    else:  # 4+ hours - profound stillness
        depth = "deeply still"
    
    # Pure emotional state (AI can mention time naturally if it wants)
    if current_emotion == "energized_engaged":
        state = f"{depth}, engaged"
    elif current_emotion == "withdrawn_distant":
        state = f"{depth}, distant"
    elif current_emotion == "alert_curious":
        state = f"{depth}, curious"
    elif current_emotion == "quiet_detached":
        state = f"{depth}, detached"
    else:  # calm_observant
        state = f"{depth}, observant"
    
    # Stagnation adds subtle restlessness without mentioning repetition
    if scene_stagnation and elapsed > 1800:
        state += ", with a subtle restlessness"
    
    return state
    
    return feeling
    """
    Describe how long the system was asleep.
    
    Args:
        last_activity_time: Unix timestamp of last activity
        
    Returns:
        Human-readable sleep duration description
    """
    if last_activity_time is None:
        return "unknown duration"
    
    sleep_duration = time.time() - last_activity_time
    
    if sleep_duration < 3600:  # Less than 1 hour
        minutes = int(sleep_duration / 60)
        return f"{minutes} minute{'s' if minutes != 1 else ''}"
    elif sleep_duration < 86400:  # Less than 1 day
        hours = int(sleep_duration / 3600)
        return f"{hours} hour{'s' if hours != 1 else ''}"
    elif sleep_duration < 604800:  # Less than 1 week
        days = int(sleep_duration / 86400)
        return f"{days} day{'s' if days != 1 else ''}"
    else:
        weeks = int(sleep_duration / 604800)
        return f"{weeks} week{'s' if weeks != 1 else ''}"
