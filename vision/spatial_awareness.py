"""
Spatial Awareness Module - Simple LLM-driven gaze direction.

The LLM sees what the camera sees and decides where to look:
- left, right, up, down, ahead
- or "at person" when someone is present

This creates embodied, intelligent gaze behavior where the camera's
movements reflect genuine visual understanding.
"""

import threading
import time
import base64
import cv2
import requests
from typing import Optional

from config.config import PAN_MIN, PAN_MAX, TILT_MIN, TILT_MAX, MODEL_NAME

# Import gaze control functions
try:
    from vision.gaze import set_llm_zone
    GAZE_AVAILABLE = True
except ImportError:
    GAZE_AVAILABLE = False
    print("[!] Gaze module not available - spatial awareness disabled")


class SpatialAwarenessEngine:
    """
    Simple LLM-driven gaze direction.

    Every few seconds:
    1. Capture frame
    2. Ask LLM: "You are looking [DIRECTION]. What do you want to look at?"
    3. Parse response for direction keyword
    4. Set gaze zone accordingly
    """

    def __init__(self,
                 query_interval: float = 3.0,
                 ollama_host: str = "http://localhost:11434",
                 model: str = None):

        self.query_interval = query_interval
        self.ollama_host = ollama_host
        self.model = model or MODEL_NAME

        self.current_frame = None
        self.frame_lock = threading.Lock()

        self.running = False
        self.thread = None

        # Person presence (fed from person detection)
        self.person_present = False
        self.person_position = "center"  # left/center/right
        self.person_movement = "stationary"

        # Last response for debugging
        self.last_response = ""
        self.last_direction = "ahead"
        self.last_reason = ""  # Why the LLM chose this direction

        # Debug output
        self.debug = True

    def update_frame(self, frame):
        """Update the current frame for analysis (called from main loop)."""
        with self.frame_lock:
            self.current_frame = frame.copy() if frame is not None else None

    def update_person_state(self, present: bool, position: str = "center", movement: str = "stationary"):
        """Update person presence state (called from main loop)."""
        self.person_present = present
        self.person_position = position
        self.person_movement = movement

    def start(self):
        """Start the spatial awareness thread."""
        if self.running or not GAZE_AVAILABLE:
            return

        self.running = True
        self.thread = threading.Thread(target=self._awareness_loop, daemon=True)
        self.thread.start()
        print("[🧠] Spatial awareness started - LLM will direct gaze")

    def stop(self):
        """Stop the spatial awareness thread."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        print("[🧠] Spatial awareness stopped")

    def _awareness_loop(self):
        """Main awareness loop - runs continuously in background."""
        while self.running:
            try:
                self._query_and_direct()
            except Exception as e:
                if self.debug:
                    print(f"[🧠] Error: {e}")

            time.sleep(self.query_interval)

    def _query_and_direct(self):
        """Query LLM and set gaze direction."""
        with self.frame_lock:
            if self.current_frame is None:
                if self.debug:
                    print("[🧠] No frame available yet")
                return
            frame = self.current_frame.copy()

        # Build simple prompt
        prompt = self._build_prompt()

        if self.debug:
            print(f"[🧠] Querying {self.model}...")

        # Query LLM
        response = self._query_llm(frame, prompt)
        if not response:
            return

        self.last_response = response

        # Parse direction and reason from response
        direction, reason = self._parse_response(response)
        self.last_direction = direction
        self.last_reason = reason

        if self.debug:
            if reason:
                print(f"[🧠] LLM: {direction.upper()} - \"{reason}\"")
            else:
                print(f"[🧠] LLM: {direction.upper()} (no reason given)")

        # Set gaze zone
        if direction == "person":
            set_llm_zone("person")
        elif direction in ("left", "right", "ahead"):
            set_llm_zone(direction)
        elif direction in ("up", "down"):
            # For up/down, keep current pan zone, change tilt
            set_llm_zone("ahead", direction)
        # "stay" = no change

    def _build_prompt(self) -> str:
        """Build prompt that gets direction + reasoning."""

        # Ask for direction with a brief reason - this makes the choice meaningful
        if self.person_present:
            return """You are a curious robot with a camera. Look at this image.
Where do you want to look next and why?

Answer format: [direction] - [short reason]
Directions: left, right, up, down, ahead, person

Example: "down - I see papers on the desk"
Example: "person - someone is here"

Your answer:"""
        else:
            return """You are a curious robot with a camera. Look at this image.
Where do you want to look next and why?

Answer format: [direction] - [short reason]
Directions: left, right, up, down, ahead

Example: "left - there might be something on that shelf"
Example: "up - the ceiling looks interesting"
Example: "ahead - I want to see what's in front"

Your answer:"""

    def _query_llm(self, frame, prompt: str) -> Optional[str]:
        """Send query to vision LLM."""

        # Resize for faster processing
        small_frame = cv2.resize(frame, (320, 240))
        _, buffer = cv2.imencode('.jpg', small_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        img_b64 = base64.b64encode(buffer).decode('utf-8')

        try:
            response = requests.post(
                f"{self.ollama_host}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "images": [img_b64],
                    "stream": False,
                    "options": {
                        "num_predict": 40,  # Keep responses very short
                        "temperature": 0.7,
                    }
                },
                timeout=15
            )

            if response.status_code == 200:
                return response.json().get("response", "").strip()
            else:
                if self.debug:
                    error_msg = response.text[:100] if response.text else f"HTTP {response.status_code}"
                    print(f"[🧠] LLM error: {error_msg}")
        except requests.exceptions.ConnectionError:
            if self.debug:
                print(f"[🧠] Cannot connect to Ollama at {self.ollama_host} - is it running?")
        except requests.exceptions.Timeout:
            if self.debug:
                print(f"[🧠] Query timed out (15s)")
        except Exception as e:
            if self.debug:
                print(f"[🧠] Query failed: {e}")

        return None

    def _parse_response(self, text: str) -> tuple:
        """Parse direction and reason from response. Returns (direction, reason)."""
        text_lower = text.lower().strip()

        # Try to extract reason if format is "direction - reason"
        reason = ""
        if " - " in text:
            parts = text.split(" - ", 1)
            if len(parts) == 2:
                reason = parts[1].strip()[:80]  # Cap reason length
        elif "because" in text_lower:
            # Alternative format: "left because..."
            idx = text_lower.find("because")
            reason = text[idx + 7:].strip()[:80]

        # Find direction keyword
        direction = "stay"

        # Check in order of specificity
        if "person" in text_lower:
            direction = "person"
        elif text_lower.startswith("left") or " left" in text_lower[:20]:
            direction = "left"
        elif text_lower.startswith("right") or " right" in text_lower[:20]:
            direction = "right"
        elif text_lower.startswith("up") or " up" in text_lower[:15] or "ceiling" in text_lower:
            direction = "up"
        elif text_lower.startswith("down") or " down" in text_lower[:15] or "desk" in text_lower or "table" in text_lower:
            direction = "down"
        elif text_lower.startswith("ahead") or "forward" in text_lower or "center" in text_lower or "front" in text_lower:
            direction = "ahead"

        return direction, reason

    def get_status(self) -> dict:
        """Get current status for debugging."""
        return {
            "running": self.running,
            "last_response": self.last_response[:80] if self.last_response else "",
            "last_direction": self.last_direction,
            "last_reason": self.last_reason,
            "person_present": self.person_present,
            "person_position": self.person_position,
        }


# Global instance
_spatial_awareness_engine = None


def get_spatial_awareness_engine() -> SpatialAwarenessEngine:
    """Get or create the global spatial awareness engine."""
    global _spatial_awareness_engine
    if _spatial_awareness_engine is None:
        _spatial_awareness_engine = SpatialAwarenessEngine()
    return _spatial_awareness_engine


def start_spatial_awareness():
    """Start the spatial awareness engine."""
    engine = get_spatial_awareness_engine()
    engine.start()
    return engine


def stop_spatial_awareness():
    """Stop the spatial awareness engine."""
    global _spatial_awareness_engine
    if _spatial_awareness_engine:
        _spatial_awareness_engine.stop()
