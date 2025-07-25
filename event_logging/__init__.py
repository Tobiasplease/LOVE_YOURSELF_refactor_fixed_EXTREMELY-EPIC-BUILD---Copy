"""
Event logging utilities for the LOVE_YOURSELF project.
"""

from .event_logger import (
    LogType,
    log_json_entry,
    read_json_logs,
    get_latest_log_entry,
    append_to_log_file,
    # read_drawing_prompts,
    # read_internal_notes,
    log_ollama_api_call,
)

__all__ = [
    "LogType",
    "log_json_entry",
    "read_json_logs",
    "get_latest_log_entry",
    "append_to_log_file",
    # "read_drawing_prompts",
    # "read_internal_notes",
    "log_ollama_api_call",
]
