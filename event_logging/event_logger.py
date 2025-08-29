import json
import os
import shutil
import time
import uuid
from typing import Any, Dict, Optional, List, Union
from datetime import datetime
import importlib.util

from config.config import LOG_TYPES_TO_PRINT, MOOD_SNAPSHOT_FOLDER
from event_logging.log_type import LogType


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
        print(f"[WARNING] Error loading config metadata: {e}")
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
    log_type: Union[LogType, str],
    data: Dict[str, Any],
    log_dir: str = MOOD_SNAPSHOT_FOLDER,
    run_id: Optional[str] = None,
    print_message: Optional[str] = None,
) -> str:
    """
    Log a JSON entry with timestamp to a run-specific event log file.

    Args:
        log_type: Type of log entry (LogType enum or string for backward compatibility)
        data: Dictionary containing the data to log
        log_dir: Directory where log files are stored
        run_id: Optional run ID. If not provided, uses the current global run ID.
        print_message: Custom message to print to terminal if in print whitelist.

    Returns:
        Path to the event log file
    """
    if run_id is None:
        run_id = get_current_run_id()

    # Convert enum to string value if needed
    log_type_str = log_type.value if isinstance(log_type, LogType) else log_type

    timestamp = int(time.time())
    iso_timestamp = datetime.fromtimestamp(timestamp).isoformat()
    elapsed_time = get_elapsed_time()

    # Create the log entry
    entry = {"timestamp": timestamp, "iso_timestamp": iso_timestamp, "type": log_type_str, "run_id": run_id, "elapsed_time": elapsed_time, **data}

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
        metadata_entry = {
            "timestamp": timestamp,
            "iso_timestamp": iso_timestamp,
            "type": LogType.RUN_METADATA.value,
            "run_id": run_id,
            **run_metadata,
        }

        # Write metadata entry first to individual run log
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump([metadata_entry], f, indent=2, ensure_ascii=False)

        update_all_run_log(log_dir, metadata_entry)

    append_to_log_file(log_dir, filename, entry)

    update_all_run_log(log_dir, entry)

    if log_type_str.lower() in LOG_TYPES_TO_PRINT or ("all" in LOG_TYPES_TO_PRINT and log_type_str.lower() != "debug"):
        if print_message:
            elapsed = get_elapsed_time()
            print(f"[{elapsed}] {print_message}")

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
            print(f"[WARNING] Error reading log file {filepath}: {e}")
            continue

    # Sort by timestamp
    logs.sort(key=lambda x: x.get("timestamp", 0))
    return logs


# def get_latest_log_entry(log_dir: str, log_type: str) -> Optional[Dict[str, Any]]:
#     """
#     Get the most recent log entry of a specific type.

#     Args:
#         log_dir: Directory containing log files
#         log_type: Type of log entry to find

#     Returns:
#         Most recent log entry or None if not found
#     """
#     logs = read_json_logs(log_dir, log_type)
#     return logs[-1] if logs else None


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

    entries = []
    if os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                entries = json.load(f)
        except UnicodeDecodeError as e:
            print(f"[WARNING] Corrupted log file {filepath}, attempting recovery: {e}")
            entries = _recover_corrupted_log_file(filepath)
        except (json.JSONDecodeError, IOError) as e:
            print(f"[WARNING] Invalid JSON in {filepath}, starting fresh: {e}")
            entries = []

    entries.append(entry)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)


def _recover_corrupted_log_file(filepath: str) -> List[Dict[str, Any]]:
    """
    Attempt to recover entries from a corrupted log file.

    Args:
        filepath: Path to the corrupted file

    Returns:
        List of recovered entries (may be empty if recovery fails)
    """
    backup_path = filepath + ".corrupted_backup"
    entries = []

    try:
        shutil.copy2(filepath, backup_path)
        print(f"[RECOVERY] Backed up corrupted file to {backup_path}")
    except Exception as e:
        print(f"[WARNING] Could not backup corrupted file: {e}")

    try:
        with open(filepath, "rb") as f:
            raw_data = f.read()

        try:
            decoded = raw_data.decode("utf-8", errors="replace")
            entries = json.loads(decoded)
            print(f"[RECOVERY] Successfully recovered {len(entries)} entries using error replacement")
        except json.JSONDecodeError:
            try:
                decoded = raw_data.decode("latin1")
                entries = json.loads(decoded)
                print(f"[RECOVERY] Successfully recovered {len(entries)} entries using latin1 encoding")
            except (UnicodeDecodeError, json.JSONDecodeError):
                try:
                    lines = raw_data.decode("utf-8", errors="ignore").split("\n")
                    for line in lines:
                        line = line.strip()
                        if line and line.startswith("{") and line.endswith("}"):
                            try:
                                entry = json.loads(line)
                                entries.append(entry)
                            except json.JSONDecodeError:
                                continue
                    if entries:
                        print(f"[RECOVERY] Line-by-line recovery found {len(entries)} entries")
                except Exception:
                    pass

    except Exception as e:
        print(f"[ERROR] Could not recover corrupted file: {e}")

    if not entries:
        print(f"[WARNING] Recovery failed, starting with empty log for {filepath}")

    return entries
