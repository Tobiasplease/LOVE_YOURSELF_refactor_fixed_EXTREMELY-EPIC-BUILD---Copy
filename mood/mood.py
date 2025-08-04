# mood/mood.py
from __future__ import annotations

import os
import time
import json
import numpy as np  # type: ignore
from typing import List, Optional

from config.config import MOOD_SNAPSHOT_FOLDER
from event_logging.event_logger import log_json_entry, read_json_logs, LogType


# ---------------------------------------------------------------------------#
# Pure MoodEngine - analyzes captions without generating them               #
# ---------------------------------------------------------------------------#
class MoodEngine:
    def __init__(self) -> None:
        self.current_mood = 0.5
        self.last_caption = ""
        self.last_person_detected = False
        self.memory = []
        
        # Track novelty and boredom for external systems (like hand control)
        self.novelty = 0.0  # Current novelty level (0.0 to 1.0)
        self.boredom = 0.0  # Current boredom level (0.0 to 1.0)

    # -------------------------------------------------------------- main hook
    def analyze_mood(self, caption: str, image_path: Optional[str] = None, temporal_context: Optional[dict] = None) -> float:
        """Analyze mood from a provided caption without generating new captions."""
        saw_person = "person" in caption.lower() or "individual" in caption.lower()

        novelty = self.calculate_novelty(caption)
        
        # Store novelty as instance attribute for external systems
        self.novelty = novelty
        
        # Calculate boredom based on low novelty over time
        # If novelty is consistently low, boredom increases
        if novelty < 0.3:
            self.boredom = min(1.0, self.boredom + 0.05)
        else:
            self.boredom = max(0.0, self.boredom - 0.02)
        
        # Apply temporal context modifiers if available
        temporal_modifier = 0.0
        if temporal_context:
            consciousness_state = temporal_context.get('consciousness_state', '')
            if 'deeply present' in consciousness_state:
                temporal_modifier += 0.1  # Sustained attention boosts mood
            elif 'freshly awakened' in consciousness_state:
                temporal_modifier += 0.05  # New awareness is slightly positive
        
        mood_change = self.compute_mood_change(novelty, saw_person) + temporal_modifier
        self.current_mood = np.clip(self.current_mood + mood_change, 0.0, 1.0)

        log_mood(caption, self.current_mood, mood_change, image_path=image_path)
        self.last_caption = caption
        self.last_person_detected = saw_person
        return self.current_mood

    # --------------------------------------------------------------- helpers
    def get_current_mood(self):
        return self.current_mood

    def calculate_novelty(self, caption):
        if not self.last_caption:
            return 1.0
        
        # More sophisticated novelty detection
        current_caption = caption.strip().lower()
        last_caption = self.last_caption.strip().lower()
        
        # Exact match = no novelty
        if current_caption == last_caption:
            return 0.0
        
        # Very similar captions (minor word changes) = low novelty
        current_words = set(current_caption.split())
        last_words = set(last_caption.split())
        
        if len(current_words) > 0 and len(last_words) > 0:
            overlap = len(current_words.intersection(last_words))
            total_unique = len(current_words.union(last_words))
            similarity = overlap / total_unique if total_unique > 0 else 0
            
            # If captions are >80% similar, consider low novelty
            if similarity > 0.8:
                return 0.2
            elif similarity > 0.6:
                return 0.5
        
        # Significantly different caption = full novelty
        return 1.0

    def compute_mood_change(self, novelty, saw_person):

        # HMMM
        # The natural decay (-0.02) is too weak compared to novelty increases (+0.05), so
        # any small variation in scene descriptions keeps the mood elevated.

        # Solutions:

        # 1. Increase decay rate: Make the no-novelty penalty stronger (e.g., -0.03 to
        # -0.04)
        # 2. Add time-based decay: Gradually reduce mood over time regardless of scene
        # changes
        # 3. Make novelty detection more strict: Only count significant caption changes as
        # novelty
        # 4. Cap positive changes: Reduce the novelty bonus when mood is already high

        change = 0.0
        if novelty:
            change += 0.02
        else:
            change -= 0.07

        if saw_person and not self.last_person_detected:
            change += 0.07
        elif not saw_person and self.last_person_detected:
            change -= 0.05
        return change


def log_mood(caption, mood, mood_change, image_path: Optional[str] = None):
    """
    Log mood data in JSON format with timestamp, caption, mood value, and image path.
    """
    data = {
        "caption": caption,
        "mood": mood,
        "mood_change": mood_change,
        "image_path": image_path if image_path and os.path.exists(image_path) else None,
    }

    if mood > 0.7:
        emoji = "😊"
    elif mood > 0.5:
        emoji = "🙂"
    elif mood > 0.3:
        emoji = "😐"
    elif mood > 0.1:
        emoji = "😔"
    else:
        emoji = "😞"

    log_json_entry(LogType.MOOD, data, MOOD_SNAPSHOT_FOLDER, auto_print=False)


def read_mood_logs(limit: Optional[int] = None) -> List[dict]:
    """
    Read mood logs from JSON files, with backward compatibility for old text format.

    Args:
        limit: Maximum number of entries to return (most recent first)

    Returns:
        List of mood log entries
    """
    # Read new JSON format logs
    json_logs = read_json_logs(MOOD_SNAPSHOT_FOLDER, "mood")

    # Convert to consistent format
    mood_logs = []
    for log in json_logs:
        mood_logs.append(
            {
                "timestamp": log.get("timestamp"),
                "iso_timestamp": log.get("iso_timestamp"),
                "caption": log.get("caption", ""),
                "mood": log.get("mood", 0.0),
                "image_path": log.get("image_path"),
                "type": "mood",
            }
        )

    # Handle backward compatibility for old text format if needed
    if os.path.exists(MOOD_SNAPSHOT_FOLDER):
        for filename in os.listdir(MOOD_SNAPSHOT_FOLDER):
            if filename.endswith(".txt") and filename.startswith("mood_"):
                filepath = os.path.join(MOOD_SNAPSHOT_FOLDER, filename)
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        content = f.read().strip()

                    # Parse old format: "Caption: ...\nMood: ..."
                    lines = content.split("\n")
                    caption = ""
                    mood = 0.0

                    for line in lines:
                        if line.startswith("Caption: "):
                            caption = line[9:]  # Remove "Caption: "
                        elif line.startswith("Mood: "):
                            mood = float(line[6:])  # Remove "Mood: "

                    # Extract timestamp from filename
                    timestamp_str = filename.replace("mood_", "").replace(".txt", "")
                    timestamp = int(timestamp_str)

                    # Add to logs if not already present in JSON format
                    if not any(log["timestamp"] == timestamp for log in mood_logs):
                        mood_logs.append(
                            {
                                "timestamp": timestamp,
                                "iso_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(timestamp)),
                                "caption": caption,
                                "mood": mood,
                                "image_path": filepath.replace(".txt", ".jpg"),
                                "type": "mood",
                            }
                        )

                except (ValueError, IOError) as e:
                    print(f"[⚠️] Error reading old mood log {filepath}: {e}")

    # Also handle old JSON format with different naming convention
    if os.path.exists(MOOD_SNAPSHOT_FOLDER):
        for filename in os.listdir(MOOD_SNAPSHOT_FOLDER):
            # Handle old format: mood_<timestamp>.json, evaluation_<timestamp>.json, etc.
            # Skip event log files as they're handled by read_json_logs
            if filename.endswith(".json") and not filename.startswith("log-") and not filename.endswith("-event-log.json"):
                filepath = os.path.join(MOOD_SNAPSHOT_FOLDER, filename)
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    # Handle both single entry and array formats
                    entries_to_process = []
                    if isinstance(data, list):
                        entries_to_process = data
                    elif isinstance(data, dict):
                        entries_to_process = [data]

                    for entry in entries_to_process:
                        if isinstance(entry, dict):
                            # Only add mood logs and avoid duplicates
                            if entry.get("type") == "mood":
                                timestamp = entry.get("timestamp", 0)
                                if not any(log["timestamp"] == timestamp for log in mood_logs):
                                    mood_logs.append(
                                        {
                                            "timestamp": timestamp,
                                            "iso_timestamp": entry.get("iso_timestamp", ""),
                                            "caption": entry.get("caption", ""),
                                            "mood": entry.get("mood", 0.0),
                                            "image_path": entry.get("image_path"),
                                            "type": "mood",
                                        }
                                    )

                except (json.JSONDecodeError, IOError) as e:
                    print(f"[⚠️] Error reading old JSON mood log {filepath}: {e}")

    # Sort by timestamp (newest first) and apply limit
    mood_logs.sort(key=lambda x: x["timestamp"], reverse=True)

    if limit:
        mood_logs = mood_logs[:limit]

    return mood_logs


def get_latest_mood() -> Optional[dict]:
    """
    Get the most recent mood log entry.

    Returns:
        Most recent mood log entry or None if not found
    """
    logs = read_mood_logs(limit=1)
    return logs[0] if logs else None
