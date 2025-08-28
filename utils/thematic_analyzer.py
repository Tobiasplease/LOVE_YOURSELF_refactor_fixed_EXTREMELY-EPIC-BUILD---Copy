"""
Periodic thematic analysis system - extracts brief emotional themes every 3 captions.
Replaces the bloated per-caption concept extraction with meaningful periodic insights.
"""

import time
import requests
from typing import List, Optional, Tuple
from collections import deque
from event_logging.event_logger import log_json_entry
from event_logging.log_type import LogType
from config.config import TINYLLAMA_TOP_P

OLLAMA_BASE_URL = "http://localhost:11434"


class ThematicAnalyzer:
    """Analyzes accumulated captions for emotional themes periodically."""

    def __init__(self, analysis_interval: int = 3):
        self.caption_buffer = deque(maxlen=analysis_interval)
        self.analysis_interval = analysis_interval
        self.caption_count = 0
        self.current_themes: List[str] = []
        self.theme_history: List[Tuple[float, List[str]]] = []

        # Rotating thematic prompts for variety
        self.theme_prompts = [
            "What 3 emotional themes (max 3 words each)?",
            "What 3 focal points (max 3 words each)?",
            "What 3 mood patterns (max 3 words each)?",
            "What 3 key interests (max 3 words each)?",
        ]
        self.prompt_index = 0

    def add_caption(self, caption: str) -> Optional[List[str]]:
        """
        Add caption to buffer. Returns themes if it's time to analyze, None otherwise.
        """
        self.caption_buffer.append(caption)
        self.caption_count += 1

        # Only analyze every N captions
        if self.caption_count % self.analysis_interval == 0:
            return self._analyze_themes()
        return None

    def _analyze_themes(self) -> List[str]:
        """Extract brief themes from accumulated captions using TinyLlama."""
        if not self.caption_buffer:
            return []

        # Combine recent captions
        combined_text = " ... ".join(self.caption_buffer)

        # Rotate through different thematic questions
        current_prompt = self.theme_prompts[self.prompt_index % len(self.theme_prompts)]
        self.prompt_index += 1

        # Build TinyLlama prompt for brief themes
        prompt = f"""Analyze these observations:
{combined_text[:300]}

{current_prompt}

Return ONLY 3 themes, each 1-3 words maximum.
Format: theme1, theme2, theme3

Example: creative energy, quiet contemplation, digital focus"""

        try:
            themes = self._query_tinyllama_themes(prompt)

            # Update state
            self.current_themes = themes
            self.theme_history.append((time.time(), themes))

            # Log thematic analysis
            self._log_themes(themes)

            return themes

        except Exception as e:
            print(f"[WARNING] Thematic analysis failed: {e}")
            # Fallback to simple heuristic themes
            return self._heuristic_themes()

    def _query_tinyllama_themes(self, prompt: str) -> List[str]:
        """Query TinyLlama for brief thematic insights."""
        payload = {
            "model": "tinyllama:latest",
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.4,  # Balanced creativity for meaningful themes
                "top_p": TINYLLAMA_TOP_P,
                "num_predict": 20,  # Allow full theme phrases
                "stop": ["\n\n", "Example:"],
            },
        }

        response = requests.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload, timeout=60.0)  # Generous timeout for meaningful analysis
        response.raise_for_status()

        result = response.json()
        raw_response = result.get("response", "").strip()

        # Parse comma-separated themes
        themes = []
        for theme in raw_response.split(","):
            cleaned = theme.strip().lower()
            # Ensure max 3 words
            words = cleaned.split()[:3]
            if words:
                themes.append(" ".join(words))

        # Return max 3 themes
        return themes[:3] if themes else ["observing quietly", "neutral mood", "present moment"]

    def _heuristic_themes(self) -> List[str]:
        """Fallback heuristic themes when TinyLlama unavailable."""
        # Simple word frequency analysis
        if not self.caption_buffer:
            return ["quiet observation", "steady presence", "calm awareness"]

        combined = " ".join(self.caption_buffer).lower()

        # Look for emotional indicators
        themes = []
        if "person" in combined or "face" in combined:
            themes.append("human presence")
        if "laptop" in combined or "screen" in combined:
            themes.append("digital focus")
        if "calm" in combined or "quiet" in combined:
            themes.append("peaceful mood")
        elif "move" in combined or "change" in combined:
            themes.append("active energy")

        # Pad with defaults if needed
        defaults = ["steady observation", "neutral space", "present moment"]
        while len(themes) < 3:
            themes.append(defaults[len(themes)])

        return themes[:3]

    def _log_themes(self, themes: List[str]):
        """Log thematic insights for tracking."""
        log_entry = {
            "timestamp": time.time(),
            "themes": themes,
            "caption_count": self.caption_count,
            "analysis_number": self.caption_count // self.analysis_interval,
        }

        message = f"[🔍] {' | '.join(themes)}"
        log_json_entry(LogType.MOTIF, log_entry, print_message=message)

    def get_current_themes(self) -> List[str]:
        """Get the most recent themes."""
        return self.current_themes

    def get_theme_evolution(self, minutes_back: int = 5) -> List[Tuple[float, List[str]]]:
        """Get theme evolution over time."""
        cutoff_time = time.time() - (minutes_back * 60)
        return [(ts, themes) for ts, themes in self.theme_history if ts > cutoff_time]


# Global analyzer instance
_thematic_analyzer = None


def get_thematic_analyzer(interval: int = 3) -> ThematicAnalyzer:
    """Get the global thematic analyzer instance."""
    global _thematic_analyzer
    if _thematic_analyzer is None:
        _thematic_analyzer = ThematicAnalyzer(interval)
    return _thematic_analyzer
