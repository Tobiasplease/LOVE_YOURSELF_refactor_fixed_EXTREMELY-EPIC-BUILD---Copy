# mood/mood.py
from __future__ import annotations

import os
import base64
import time
import requests
import re
import json
import cv2  # type: ignore
import numpy as np  # type: ignore
from typing import List, Union, Optional

from config.config import MOOD_SNAPSHOT_FOLDER, LLAVA_TIMEOUT_EVAL, LLAVA_TIMEOUT_SUMMARY
from json_logging.json_logger import log_json_entry, read_json_logs, get_latest_log_entry
from json_logging.run_manager import get_run_image_path


# ---------------------------------------------------------------------------#
# LLaVA scalar‑mood helper (needed by Captioner)                              #
# ---------------------------------------------------------------------------#
def estimate_mood_llava(
    caption: str, url: str = "http://localhost:11434/api/generate", timeout: int = LLAVA_TIMEOUT_EVAL
) -> float:
    """
    Ask the local LLaVA server for a scalar mood value in [‑1, +1].
    Falls back to 0.0 if the request fails.
    """
    prompt = (
        f"This is my most recent internal thought: '{caption}'\n"
        "As this voice inside me, how would I describe the mood I’m in? "
        "Reply with a number between -1 and +1. Just the number."
    )
    try:
        r = requests.post(
            url,
            json={"model": "llava", "prompt": prompt, "images": [], "stream": False},
            timeout=timeout,
        )
        r.raise_for_status()
        response_text = r.json().get("response", "0.0").strip()
        match = re.search(r"[-+]?\d*\.\d+|\d+", response_text)
        if match:
            return max(-1.0, min(1.0, float(match.group(0))))
        else:
            raise ValueError("No valid mood score found")
    except Exception as e:
        print(f"[⚠️] Mood estimation failed: {e}")
        return 0.0


# ---------------------------------------------------------------------------#
# Snapshot‑based MoodEngine (your original code, updated with timeout)       #
# ---------------------------------------------------------------------------#
class MoodEngine:
    def __init__(self) -> None:
        self.current_mood = 0.5
        self.last_caption = ""
        self.last_person_detected = False
        self.memory = []

    # -------------------------------------------------------------- main hook
    def update_feeling_brain(self, frame, image_path: Optional[str] = None):
        caption = self.generate_caption(frame)
        saw_person = "person" in caption.lower() or "individual" in caption.lower()

        novelty = self.calculate_novelty(caption)
        mood_change = self.compute_mood_change(novelty, saw_person)
        self.current_mood = np.clip(self.current_mood + mood_change, 0.0, 1.0)

        # note = generate_internal_note(
        #     caption,
        #     self.last_caption,
        #     self.current_mood,
        #     self.current_mood - mood_change,
        #     saw_person,
        #     self.last_person_detected,
        # )

        log_mood(caption, self.current_mood, image_path=image_path)
        self.last_caption = caption
        self.last_person_detected = saw_person
        return caption

    # --------------------------------------------------------------- helpers
    def get_current_mood(self):
        return self.current_mood

    def calculate_novelty(self, caption):
        if not self.last_caption:
            return 1.0
        return 0.0 if caption.strip() == self.last_caption.strip() else 1.0

    def compute_mood_change(self, novelty, saw_person):
        change = 0.0
        if novelty:
            change += 0.05
        else:
            change -= 0.02

        if saw_person and not self.last_person_detected:
            change += 0.07
        elif not saw_person and self.last_person_detected:
            change -= 0.05
        return change

    # -------------------------------------------------------- LLaVA caption
    def generate_caption(self, frame, timeout: int = LLAVA_TIMEOUT_SUMMARY):
        _, img_encoded = cv2.imencode(".jpg", frame)
        img_base64 = base64.b64encode(img_encoded).decode("utf-8")  # type: ignore

        payload = {
            "model": "llava",
            "prompt": "Describe the scene",
            "images": [img_base64],
            "stream": False,
        }

        try:
            response = requests.post("http://localhost:11434/api/generate", json=payload, timeout=timeout)
            if response.status_code == 200:
                return response.json().get("response", "")
            return f"Error: LLaVA API returned status {response.status_code}"
        except Exception as e:
            return f"Error: {e}"


# ---------------------------------------------------------------------------#
# Stateless helpers (unchanged)                                              #
# ---------------------------------------------------------------------------#
def generate_internal_note(caption, last_caption, mood, last_mood, saw_person, last_person_detected):
    """
    Generate and log an internal note about changes in mood and observations.
    """
    changes = []
    if caption != last_caption:
        changes.append("new observation")
    if saw_person and not last_person_detected:
        changes.append("person appeared")
    elif not saw_person and last_person_detected:
        changes.append("person disappeared")
    if mood > last_mood:
        changes.append("mood improved")
    elif mood < last_mood:
        changes.append("mood declined")
    
    note_text = f"{', '.join(changes)}, {caption.strip()}"
    
    # Log internal note in JSON format
    data = {
        "note": note_text,
        "changes": changes,
        "caption": caption,
        "last_caption": last_caption,
        "mood": mood,
        "last_mood": last_mood,
        "saw_person": saw_person,
        "last_person_detected": last_person_detected
    }
    
    log_json_entry("internal_note", data, MOOD_SNAPSHOT_FOLDER)
    
    return note_text


def log_mood(caption, mood, image_path: Optional[str] = None):
    """
    Log mood data in JSON format with timestamp, caption, mood value, and image path.
    """
    data = {
        "caption": caption,
        "mood": mood,
        "image_path": image_path if image_path and os.path.exists(image_path) else None
    }
    
    log_json_entry("mood", data, MOOD_SNAPSHOT_FOLDER)


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
        mood_logs.append({
            "timestamp": log.get("timestamp"),
            "iso_timestamp": log.get("iso_timestamp"),
            "caption": log.get("caption", ""),
            "mood": log.get("mood", 0.0),
            "image_path": log.get("image_path"),
            "type": "mood"
        })
    
    # Handle backward compatibility for old text format if needed
    if os.path.exists(MOOD_SNAPSHOT_FOLDER):
        for filename in os.listdir(MOOD_SNAPSHOT_FOLDER):
            if filename.endswith('.txt') and filename.startswith('mood_'):
                filepath = os.path.join(MOOD_SNAPSHOT_FOLDER, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                    
                    # Parse old format: "Caption: ...\nMood: ..."
                    lines = content.split('\n')
                    caption = ""
                    mood = 0.0
                    
                    for line in lines:
                        if line.startswith("Caption: "):
                            caption = line[9:]  # Remove "Caption: "
                        elif line.startswith("Mood: "):
                            mood = float(line[6:])  # Remove "Mood: "
                    
                    # Extract timestamp from filename
                    timestamp_str = filename.replace('mood_', '').replace('.txt', '')
                    timestamp = int(timestamp_str)
                    
                    # Add to logs if not already present in JSON format
                    if not any(log["timestamp"] == timestamp for log in mood_logs):
                        mood_logs.append({
                            "timestamp": timestamp,
                            "iso_timestamp": time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime(timestamp)),
                            "caption": caption,
                            "mood": mood,
                            "image_path": filepath.replace('.txt', '.jpg'),
                            "type": "mood"
                        })
                        
                except (ValueError, IOError) as e:
                    print(f"[⚠️] Error reading old mood log {filepath}: {e}")
    
    # Also handle old JSON format with different naming convention
    if os.path.exists(MOOD_SNAPSHOT_FOLDER):
        for filename in os.listdir(MOOD_SNAPSHOT_FOLDER):
            # Handle old format: mood_<timestamp>.json, evaluation_<timestamp>.json, etc.
            if filename.endswith('.json') and not filename.startswith('log-'):
                filepath = os.path.join(MOOD_SNAPSHOT_FOLDER, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        entry = json.load(f)
                    
                    # Only add mood logs and avoid duplicates
                    if entry.get('type') == 'mood':
                        timestamp = entry.get('timestamp', 0)
                        if not any(log["timestamp"] == timestamp for log in mood_logs):
                            mood_logs.append({
                                "timestamp": timestamp,
                                "iso_timestamp": entry.get('iso_timestamp', ''),
                                "caption": entry.get('caption', ''),
                                "mood": entry.get('mood', 0.0),
                                "image_path": entry.get('image_path'),
                                "type": "mood"
                            })
                            
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
