import json
import os
import time
import uuid
from typing import Any, Dict, Optional, List
from datetime import datetime
import importlib.util

from config.config import OLLAMA_MODEL


# Global run ID - generated once per application run
_current_run_id: Optional[str] = None
_config_metadata: Optional[Dict[str, Any]] = None
_start_time: Optional[float] = None


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


def set_start_time(start_time: float) -> None:
    """Set the start time for elapsed time calculations."""
    global _start_time
    _start_time = start_time


def get_elapsed_time() -> str:
    """Get elapsed time since start as formatted string (HH:MM:SS)."""
    global _start_time
    if _start_time is None:
        return "00:00:00"

    elapsed = time.time() - _start_time
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = int(elapsed % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def event_print(message: str, event_type: Optional[str] = None, data: Optional[Dict[str, Any]] = None, log_dir: str = "mood_snapshots") -> None:
    """
    Unified print and log function with elapsed time.

    Args:
        message: The message to print and optionally log
        event_type: If provided, also log to JSON with this event type
        data: Additional data to include in JSON log
        log_dir: Directory for JSON logs
    """
    elapsed = get_elapsed_time()
    formatted_message = f"[{elapsed}] {message}"
    print(formatted_message)

    if event_type:
        log_data = {"message": message, "elapsed_time": elapsed}
        if data:
            log_data.update(data)
        log_json_entry(event_type, log_data, log_dir)


def load_config_metadata() -> Dict[str, Any]:
    """Load configuration values from config.py as metadata."""
    global _config_metadata
    if _config_metadata is not None:
        return _config_metadata

    # Try to load config.py
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "config.py")
    if not os.path.exists(config_path):
        _config_metadata = {}
        return _config_metadata

    try:
        spec = importlib.util.spec_from_file_location("config", config_path)
        config = importlib.util.module_from_spec(spec)  # type: ignore
        spec.loader.exec_module(config)  # type: ignore

        # Extract all uppercase variables (config constants)
        config_vars = {name: getattr(config, name) for name in dir(config) if not name.startswith("_") and name.isupper()}

        _config_metadata = config_vars
        return _config_metadata
    except Exception as e:
        print(f"[⚠️] Error loading config metadata: {e}")
        _config_metadata = {}
        return _config_metadata


def create_run_metadata(run_id: str) -> Dict[str, Any]:
    """Create run metadata including config values."""
    config_metadata = load_config_metadata()

    return {
        "run_id": run_id,
        "start_time": int(time.time()),
        "start_time_iso": datetime.fromtimestamp(int(time.time())).isoformat(),
        "config": config_metadata,
    }


def update_all_run_log(log_dir: str, entry: Dict[str, Any]) -> None:
    """Update the aggregated all-run-log.json file with a log entry."""
    all_run_log_path = os.path.join(log_dir, "all-run-log.json")

    # Load existing entries
    all_entries = []
    if os.path.exists(all_run_log_path):
        try:
            with open(all_run_log_path, "r", encoding="utf-8") as f:
                all_entries = json.load(f)
        except (json.JSONDecodeError, IOError):
            all_entries = []

    # Add new log entry
    all_entries.append(entry)

    # Write back to file
    os.makedirs(log_dir, exist_ok=True)
    with open(all_run_log_path, "w", encoding="utf-8") as f:
        json.dump(all_entries, f, indent=2, ensure_ascii=False)


def log_json_entry(
    log_type: str, data: Dict[str, Any], log_dir: str, run_id: Optional[str] = None, auto_print: bool = False, print_message: Optional[str] = None
) -> str:
    """
    Log a JSON entry with timestamp to a run-specific event log file.

    Args:
        log_type: Type of log entry (e.g., 'mood', 'evaluation', 'drawing_prompt', 'internal_note')
        data: Dictionary containing the data to log
        log_dir: Directory where log files are stored
        run_id: Optional run ID. If not provided, uses the current global run ID.
        auto_print: If True, also print the message with elapsed time
        print_message: Custom message to print. If None and auto_print=True, uses data.get('message')

    Returns:
        Path to the event log file
    """
    if run_id is None:
        run_id = get_current_run_id()

    timestamp = int(time.time())
    iso_timestamp = datetime.fromtimestamp(timestamp).isoformat()
    elapsed_time = get_elapsed_time()

    # Create the log entry
    entry = {"timestamp": timestamp, "iso_timestamp": iso_timestamp, "type": log_type, "run_id": run_id, "elapsed_time": elapsed_time, **data}

    # Use run-based event log filename
    filename = f"{run_id}-event-log.json"
    filepath = os.path.join(log_dir, filename)

    # Ensure directory exists
    os.makedirs(log_dir, exist_ok=True)

    # Check if this is a new run log file and create metadata if needed
    if not os.path.exists(filepath):
        # Create run metadata with config values
        run_metadata = create_run_metadata(run_id)

        # Create the run log file with metadata as first entry
        metadata_entry = {"timestamp": timestamp, "iso_timestamp": iso_timestamp, "type": "run_metadata", "run_id": run_id, **run_metadata}

        # Write metadata entry first to individual run log
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump([metadata_entry], f, indent=2, ensure_ascii=False)

        # Also add metadata to all-run-log.json
        update_all_run_log(log_dir, metadata_entry)

    # Append to the individual run event log file
    append_to_log_file(log_dir, filename, entry)

    # Also append to all-run-log.json
    update_all_run_log(log_dir, entry)

    # Auto-print if requested
    if auto_print:
        message = print_message or data.get("message", f"{log_type} event")
        elapsed = get_elapsed_time()
        print(f"[{elapsed}] {message}")

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


def log_ollama_api_call(
    prompt: str,
    model: str = OLLAMA_MODEL,
    image_path: Optional[str] = None,
    response: Optional[str] = None,
    success: bool = True,
    error_message: Optional[str] = None,
    timeout: Optional[int] = None,
    log_dir: str = "mood_snapshots",
) -> str:
    """
    Legacy function for backward compatibility.
    Redirects to the new ollama module's log_ollama_call function.
    """
    from utils.ollama import log_ollama_call

    return log_ollama_call(
        prompt=prompt,
        model=model,
        image_path=image_path,
        response=response,
        success=success,
        error_message=error_message,
        timeout=timeout,
        log_dir=log_dir,
    )
