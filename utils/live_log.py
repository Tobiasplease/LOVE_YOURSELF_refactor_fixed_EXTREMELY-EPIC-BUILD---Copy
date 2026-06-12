"""
utils/live_log.py
-----------------
Tiny helper for appending events to event_log/live_captions.txt
which is consumed by debug/caption_monitor.py.

All entries are written as plain text — no tags, no labels.
The monitor displays them as a continuous stream.
"""

import os

try:
    from config.config import MOOD_SNAPSHOT_FOLDER
    _LIVE_LOG_PATH = os.path.join(MOOD_SNAPSHOT_FOLDER, "live_captions.txt")
except Exception:
    _LIVE_LOG_PATH = None


def _write(line: str) -> None:
    """Append one line to the live log. Silent on failure — never blocks the main loop."""
    if not _LIVE_LOG_PATH:
        return
    try:
        with open(_LIVE_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line.replace("\n", " ").strip() + "\n")
    except Exception:
        pass


def log_caption(caption: str) -> None:
    """Log a regular caption."""
    if caption and caption.strip():
        _write(caption)


def log_drawing_intent(prompt: str) -> None:
    """Log when a drawing prompt has been generated and queued."""
    # Strip the standard ComfyUI preamble for readability
    p = prompt.strip()
    for prefix in ("Black ink line drawing on white paper. ",
                   "Black ink line drawing on white paper.",
                   "black ink line drawing on white paper. "):
        if p.lower().startswith(prefix.lower()):
            p = p[len(prefix):]
            break
    if len(p) > 220:
        p = p[:220].rsplit(" ", 1)[0] + "..."
    _write(f"Drawing: {p}")


def log_drawing_failed(reason: str) -> None:
    """Log a drawing attempt that failed (no paper, ComfyUI error, etc.)."""
    if "paper" in reason.lower():
        _write("Wanted to draw, but there's no paper.")
    else:
        _write(f"Tried to draw, but couldn't ({reason}).")


def log_drawing_complete(description: str) -> None:
    """Log when a drawing finishes physically executing."""
    if description and description.strip():
        _write(f"Finished drawing: {description}")


def log_reflection(subject: str, reflection: str) -> None:
    """Log when the reflection loop produces a new long-form thought."""
    if reflection and reflection.strip():
        _write(f"Reflected on {subject}: {reflection}")
