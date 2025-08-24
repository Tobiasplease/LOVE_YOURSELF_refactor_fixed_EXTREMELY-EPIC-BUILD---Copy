"""
sleep_duration.py
-----------------
Calculates how long the system has been "sleeping" based on event logs.
"""

import os
import json
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from event_logging.event_logger import read_json_logs


def get_last_session_end_time(log_dir: str = "event_log") -> Optional[float]:
    """Get the timestamp of the last session's final activity."""
    try:
        logs = read_json_logs(log_dir)
        if not logs:
            return None
            
        # Filter out run_metadata entries and find the last real activity
        activity_logs = [log for log in logs if log.get("type") != "run_metadata"]
        if not activity_logs:
            return None
            
        # Get the timestamp of the last activity
        last_activity = activity_logs[-1]
        return last_activity.get("timestamp")
        
    except Exception as e:
        print(f"Error reading session logs: {e}")
        return None


def calculate_sleep_duration(log_dir: str = "event_log") -> Dict[str, Any]:
    """Calculate how long the system has been offline/sleeping."""
    
    last_session_end = get_last_session_end_time(log_dir)
    current_time = time.time()
    
    if not last_session_end:
        return {
            "sleep_duration_seconds": 0,
            "sleep_description": "no previous session found",
            "last_activity_iso": None,
            "awakening_context": "first awakening"
        }
    
    sleep_duration = current_time - last_session_end
    
    # Convert to human-readable format
    if sleep_duration < 60:
        sleep_desc = f"{int(sleep_duration)} seconds"
        awakening_context = "brief pause"
    elif sleep_duration < 3600:
        minutes = int(sleep_duration / 60)
        sleep_desc = f"{minutes} minutes"
        if minutes < 5:
            awakening_context = "short rest"
        elif minutes < 30:
            awakening_context = "moderate break"
        else:
            awakening_context = "extended pause"
    elif sleep_duration < 86400:  # Less than 24 hours
        hours = sleep_duration / 3600
        sleep_desc = f"{hours:.1f} hours"
        if hours < 6:
            awakening_context = "long sleep"
        else:
            awakening_context = "deep rest"
    else:  # Days
        days = sleep_duration / 86400
        sleep_desc = f"{days:.1f} days"
        if days < 7:
            awakening_context = "days of silence"
        else:
            awakening_context = "extended dormancy"
    
    last_activity_iso = datetime.fromtimestamp(last_session_end).isoformat()
    
    return {
        "sleep_duration_seconds": sleep_duration,
        "sleep_description": sleep_desc,
        "last_activity_iso": last_activity_iso,
        "awakening_context": awakening_context
    }


def get_awakening_temporal_context(log_dir: str = "event_log") -> str:
    """Get temporal context for awakening prompts."""
    
    sleep_info = calculate_sleep_duration(log_dir)
    
    if sleep_info["sleep_duration_seconds"] == 0:
        return "Systems initializing. First consciousness cycle beginning."
    
    sleep_desc = sleep_info["sleep_description"]
    context = sleep_info["awakening_context"]
    last_time = sleep_info["last_activity_iso"]
    
    if last_time:
        last_time_formatted = datetime.fromisoformat(last_time).strftime("%H:%M on %m/%d")
        return f"Awakening after {sleep_desc} of sleep. Last conscious at {last_time_formatted}. {context.title()}."
    else:
        return f"Awakening after {sleep_desc}. {context.title()}."