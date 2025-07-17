import json
import os
import time
import uuid
from typing import Any, Dict, Optional, List
from datetime import datetime


# Global run ID - generated once per application run
_current_run_id: Optional[str] = None


def get_current_run_id() -> str:
    """Get or generate the current run ID."""
    global _current_run_id
    if _current_run_id is None:
        _current_run_id = str(uuid.uuid4())[:8]  # Use first 8 chars for readability
    return _current_run_id


def set_run_id(run_id: str) -> None:
    """Set a custom run ID."""
    global _current_run_id
    _current_run_id = run_id


def log_json_entry(log_type: str, data: Dict[str, Any], log_dir: str, run_id: Optional[str] = None) -> str:
    """
    Log a JSON entry with timestamp to a run-specific event log file.

    Args:
        log_type: Type of log entry (e.g., 'mood', 'evaluation', 'drawing_prompt', 'internal_note')
        data: Dictionary containing the data to log
        log_dir: Directory where log files are stored
        run_id: Optional run ID. If not provided, uses the current global run ID.

    Returns:
        Path to the event log file
    """
    if run_id is None:
        run_id = get_current_run_id()

    timestamp = int(time.time())
    iso_timestamp = datetime.fromtimestamp(timestamp).isoformat()

    # Create the log entry
    entry = {"timestamp": timestamp, "iso_timestamp": iso_timestamp, "type": log_type, "run_id": run_id, **data}

    # Use run-based event log filename
    filename = f"{run_id}-event-log.json"
    filepath = os.path.join(log_dir, filename)

    # Ensure directory exists
    os.makedirs(log_dir, exist_ok=True)

    # Append to the event log file
    append_to_log_file(log_dir, filename, entry)

    return filepath


def read_json_logs(log_dir: str, log_type: Optional[str] = None) -> List[Dict[str, Any]]:
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
        if not filename.endswith(".json"):
            continue

        filepath = os.path.join(log_dir, filename)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Handle different log file formats
            if filename.endswith("-event-log.json") or filename.startswith("event_log_"):
                # Event log format: array of entries (new and old format)
                if isinstance(data, list):
                    for entry in data:
                        if isinstance(entry, dict):
                            # Filter by log type if specified
                            if log_type and entry.get("type") != log_type:
                                continue
                            logs.append(entry)
                else:
                    # Single entry format
                    if isinstance(data, dict):
                        if log_type and data.get("type") != log_type:
                            continue
                        logs.append(data)
            else:
                # Old format: single entry per file
                # New format: log-<timestamp>-<logtype>.json
                # Old format: <logtype>_<timestamp>.json or <prefix>_<timestamp>.json
                if isinstance(data, dict):
                    # Filter by log type if specified
                    if log_type and data.get("type") != log_type:
                        continue
                    logs.append(data)
                elif isinstance(data, list):
                    # Handle array format
                    for entry in data:
                        if isinstance(entry, dict):
                            if log_type and entry.get("type") != log_type:
                                continue
                            logs.append(entry)

        except (json.JSONDecodeError, IOError) as e:
            print(f"[⚠️] Error reading log file {filepath}: {e}")
            continue

    # Sort by timestamp
    logs.sort(key=lambda x: x.get("timestamp", 0))
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
            with open(filepath, "r", encoding="utf-8") as f:
                entries = json.load(f)
        except (json.JSONDecodeError, IOError):
            entries = []

    # Append new entry
    entries.append(entry)

    # Write back to file
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)


def read_evaluations(log_dir: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Read self-evaluation logs."""
    logs = read_json_logs(log_dir, "self_evaluation")
    if limit:
        logs = logs[-limit:]
    return logs


def read_drawing_prompts(log_dir: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Read drawing prompt logs."""
    logs = read_json_logs(log_dir, "drawing_prompt")
    if limit:
        logs = logs[-limit:]
    return logs


def read_internal_notes(log_dir: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Read internal note logs."""
    logs = read_json_logs(log_dir, "internal_note")
    if limit:
        logs = logs[-limit:]
    return logs


def log_llava_api_call(
    prompt: str, 
    model: str = "llava", 
    image_path: Optional[str] = None, 
    response: Optional[str] = None, 
    success: bool = True, 
    error_message: Optional[str] = None,
    timeout: Optional[int] = None,
    log_dir: str = "mood_snapshots"
) -> str:
    """
    Log LLaVA API call details for monitoring and debugging.
    
    Args:
        prompt: The prompt sent to LLaVA
        model: The model name (default: "llava")
        image_path: Path to input image if any
        response: The response from LLaVA
        success: Whether the API call was successful
        error_message: Error message if call failed
        timeout: Request timeout used
        log_dir: Directory to store the log
        
    Returns:
        Path to the log file
    """
    # Truncate very long prompts and responses for readability
    truncated_prompt = prompt[:500] + "..." if len(prompt) > 500 else prompt
    truncated_response = response[:1000] + "..." if response and len(response) > 1000 else response
    
    data = {
        "prompt": truncated_prompt,
        "full_prompt_length": len(prompt),
        "model": model,
        "image_path": image_path if image_path and os.path.exists(image_path) else None,
        "has_image": image_path is not None and os.path.exists(image_path) if image_path else False,
        "response": truncated_response,
        "full_response_length": len(response) if response else 0,
        "success": success,
        "error_message": error_message,
        "timeout": timeout,
        "api_endpoint": "http://localhost:11434/api/generate"
    }
    
    return log_json_entry("llava_api_call", data, log_dir)


def read_llava_api_calls(log_dir: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Read LLaVA API call logs."""
    logs = read_json_logs(log_dir, "llava_api_call")
    if limit:
        logs = logs[-limit:]
    return logs
