# mood/mood.py
from __future__ import annotations

import os
import base64
import time
import requests
import re
import cv2                 # type: ignore
import numpy as np          # type: ignore

from config.config import MOOD_SNAPSHOT_FOLDER, INTERNAL_VOICE_LOG


# ---------------------------------------------------------------------------#
# LLaVA scalar‑mood helper (needed by Captioner)                              #
# ---------------------------------------------------------------------------#
def estimate_mood_llava(caption: str,
                        url: str = "http://localhost:11434/api/generate",
                        timeout: int = 30) -> float:
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
            json={"model": "llava",
                  "prompt": prompt,
                  "images": [],
                  "stream": False},
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
# Snapshot‑based MoodEngine (your original code, untouched)                  #
# ---------------------------------------------------------------------------#
class MoodEngine:
    def __init__(self) -> None:
        self.current_mood = 0.5
        self.last_caption = ""
        self.last_person_detected = False
        self.memory = []

    # -------------------------------------------------------------- main hook
    def update_feeling_brain(self, frame, image_path: str | None = None):
        caption = self.generate_caption(frame)
        saw_person = "person" in caption.lower() or "individual" in caption.lower()

        novelty     = self.calculate_novelty(caption)
        mood_change = self.compute_mood_change(novelty, saw_person)
        self.current_mood = np.clip(self.current_mood + mood_change, 0.0, 1.0)

        note = generate_internal_note(
            caption,
            self.last_caption,
            self.current_mood,
            self.current_mood - mood_change,
            saw_person,
            self.last_person_detected,
        )

        log_mood(caption, self.current_mood, image_path=image_path)
        self.last_caption          = caption
        self.last_person_detected  = saw_person
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
    def generate_caption(self, frame):
        _, img_encoded = cv2.imencode(".jpg", frame)
        img_base64     = base64.b64encode(img_encoded).decode("utf-8")

        payload = {
            "model":  "llava",
            "prompt": "Describe the scene",
            "images": [img_base64],
            "stream": False,
        }

        try:
            response = requests.post("http://localhost:11434/api/generate", json=payload)
            if response.status_code == 200:
                return response.json().get("response", "")
            return f"Error: LLaVA API returned status {response.status_code}"
        except Exception as e:
            return f"Error: {e}"

# ---------------------------------------------------------------------------#
# Stateless helpers (unchanged)                                              #
# ---------------------------------------------------------------------------#
def generate_internal_note(caption, last_caption, mood, last_mood,
                           saw_person, last_person_detected):
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
    return f"{', '.join(changes)}, {caption.strip()}"


def log_mood(caption, mood, image_path: str | None = None):
    timestamp = int(time.time())
    if image_path and os.path.exists(image_path):
        filename = image_path.replace(".jpg", ".txt")
    else:
        filename = os.path.join(MOOD_SNAPSHOT_FOLDER, f"mood_{timestamp}.txt")
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"Caption: {caption}\nMood: {mood}\n")
