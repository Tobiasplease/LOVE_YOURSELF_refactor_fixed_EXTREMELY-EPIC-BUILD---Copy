import json
import os
import time
from typing import Any, Dict, Optional, List
from datetime import datetime


def log_json_entry(log_type: str, data: Dict[str, Any], log_dir: str) -> str:
    """
    Log a JSON entry with timestamp to a file.
    
    Args:
        log_type: Type of log entry (e.g., 'mood', 'evaluation', 'drawing_prompt', 'internal_note')
        data: Dictionary containing the data to log
        log_dir: Directory where log files are stored
    
    Returns:
        Path to the created log file
    """
    timestamp = int(time.time())
    iso_timestamp = datetime.fromtimestamp(timestamp).isoformat()
    
    # Create the log entry
    entry = {
        "timestamp": timestamp,
        "iso_timestamp": iso_timestamp,
        "type": log_type,
        **data
    }
    
    # Determine filename using the format: log-<timestamp>-<logtype>.json
    filename = f"log-{timestamp}-{log_type}.json"
    
    filepath = os.path.join(log_dir, filename)
    
    # Ensure directory exists
    os.makedirs(log_dir, exist_ok=True)
    
    # Write JSON to file
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(entry, f, indent=2, ensure_ascii=False)
    
    return filepath


def read_json_logs(log_dir: str, log_type: str = None) -> List[Dict[str, Any]]:
    """
    Read and parse JSON log files from a directory.
    
    Args:
        log_dir: Directory containing log files
        log_type: Optional filter by log type
        
    Returns:
        List of parsed log entries, sorted by timestamp
    """
    if not os.path.exists(log_dir):
        return []
    
    logs = []
    for filename in os.listdir(log_dir):
        if not filename.endswith('.json'):
            continue
        
        # Handle both old and new filename formats
        # New format: log-<timestamp>-<logtype>.json
        # Old format: <logtype>_<timestamp>.json or <prefix>_<timestamp>.json
        filepath = os.path.join(log_dir, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                entry = json.load(f)
                
            # Filter by log type if specified
            if log_type and entry.get('type') != log_type:
                continue
                
            logs.append(entry)
        except (json.JSONDecodeError, IOError) as e:
            print(f"[⚠️] Error reading log file {filepath}: {e}")
            continue
    
    # Sort by timestamp
    logs.sort(key=lambda x: x.get('timestamp', 0))
    return logs


def get_latest_log_entry(log_dir: str, log_type: str) -> Optional[Dict[str, Any]]:
    """
    Get the most recent log entry of a specific type.
    
    Args:
        log_dir: Directory containing log files
        log_type: Type of log entry to find
        
    Returns:
        Most recent log entry or None if not found
    """
    logs = read_json_logs(log_dir, log_type)
    return logs[-1] if logs else None


def append_to_log_file(log_dir: str, filename: str, entry: Dict[str, Any]) -> None:
    """
    Append a JSON entry to a log file (for aggregated logs).
    
    Args:
        log_dir: Directory containing log files
        filename: Name of the log file
        entry: Dictionary to append
    """
    filepath = os.path.join(log_dir, filename)
    os.makedirs(log_dir, exist_ok=True)
    
    # Read existing entries
    entries = []
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                entries = json.load(f)
        except (json.JSONDecodeError, IOError):
            entries = []
    
    # Append new entry
    entries.append(entry)
    
    # Write back to file
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)


def read_evaluations(log_dir: str, limit: int = None) -> List[Dict[str, Any]]:
    """Read self-evaluation logs."""
    logs = read_json_logs(log_dir, "self_evaluation")
    if limit:
        logs = logs[-limit:]
    return logs


def read_drawing_prompts(log_dir: str, limit: int = None) -> List[Dict[str, Any]]:
    """Read drawing prompt logs."""
    logs = read_json_logs(log_dir, "drawing_prompt")
    if limit:
        logs = logs[-limit:]
    return logs


def read_internal_notes(log_dir: str, limit: int = None) -> List[Dict[str, Any]]:
    """Read internal note logs."""
    logs = read_json_logs(log_dir, "internal_note")
    if limit:
        logs = logs[-limit:]
    return logs