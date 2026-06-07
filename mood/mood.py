# mood/mood.py
from __future__ import annotations

import os
import time
from typing import List, Optional, Tuple

import numpy as np  # type: ignore

from config.config import MOOD_SNAPSHOT_FOLDER
from event_logging.event_logger import log_json_entry
from event_logging.log_type import LogType
from utils.inference import query_model
from utils.pattern_recognition import PatternRecognitionEngine


# ---------------------------------------------------------------------------#
# Pure MoodEngine - analyzes captions without generating them               #
# ---------------------------------------------------------------------------#
class MoodEngine:
    def __init__(self) -> None:
        self.current_mood = 0.5  # Backward compatibility scalar
        self.mood_vector = (0.0, 0.0, 0.0)  # Initial: truly neutral valence, arousal, clarity
        self.previous_mood_vector = (0.0, 0.0, 0.0)  # For emotional momentum tracking
        self.emotional_momentum = 0.2  # How much previous mood influences new mood (0.0-1.0) - lower for more responsive emotional evolution
        self.last_caption = ""
        self.last_person_detected = False
        self.memory = []
        self.session_start = time.time()  # Track session duration for decay

        # GPT-5's suggestion: Long-term mood bias for tone evolution over weeks
        self.long_bias = getattr(self, "long_bias", (0.0, 0.0, 0.0))  # val, aro, clr drift

        # Initialize unified pattern recognition engine
        self.pattern_engine = PatternRecognitionEngine()

    # -------------------------------------------------------------- main hook
    def analyze_mood(
        self,
        caption: str,
        saw_person: bool = False,
        image_path: str | None = None,
        memory_context: Optional[any] = None,  # type: ignore
        temporal_feeling: Optional[str] = None,
    ) -> float:
        """Analyze mood using 3D valence/arousal/clarity system via Ollama."""
        saw_person = "person" in caption.lower() or "individual" in caption.lower()

        # Simple mood analysis (3D analysis moved to context compression)
        scalar_mood = self.analyze_caption_sentiment(caption)

        # Apply unified pattern analysis (motifs + novelty)
        pattern_data = self.pattern_engine.analyze_caption(caption)
        novelty = pattern_data["novelty"]
        self._last_novelty = novelty  # Store for external access
        traditional_change = self.compute_mood_change(novelty, saw_person)

        # Combine simple analysis with traditional factors
        self.current_mood = np.clip(scalar_mood + traditional_change, 0.0, 1.0)

        # CRITICAL: Update 3D mood vector from sentiment analysis (was missing!)
        # Map sentiment to valence, use caption content for arousal/clarity
        valence = scalar_mood  # Direct mapping from sentiment

        # Calculate arousal from action/energy words in caption
        energy_words = ["energetic", "excited", "dynamic", "movement", "active", "engaged", "focus", "intense", "alert", "vibrant"]
        calm_words = ["quiet", "still", "peaceful", "calm", "sitting", "resting", "dim", "soft", "tired", "withdrawn", "slouching", "heavy"]
        arousal_score = 0.0
        for word in energy_words:
            if word in caption.lower():
                arousal_score += 0.15
        for word in calm_words:
            if word in caption.lower():
                arousal_score -= 0.1
        arousal = np.clip(0.3 + arousal_score, 0.0, 1.0)  # Base arousal + content adjustment

        clarity = np.clip((len(caption.split()) - 10) / 20, -1.0, 1.0)  # Caption length suggests clarity

        # Apply emotional momentum to smooth transitions
        prev_valence, prev_arousal, prev_clarity = self.mood_vector
        momentum = self.emotional_momentum
        self.mood_vector = (
            (1 - momentum) * valence + momentum * prev_valence,
            (1 - momentum) * arousal + momentum * prev_arousal,
            (1 - momentum) * clarity + momentum * prev_clarity
        )

        log_mood(caption, self.current_mood, traditional_change, image_path=image_path)
        self.last_caption = caption
        self.last_person_detected = saw_person
        return self.current_mood

    # --------------------------------------------------------------- helpers
    def get_current_mood(self):
        return self.current_mood

    def get_emotion_for_hand_controller(self) -> str:
        """Map 3D mood vector to hand controller emotion states with enhanced natural variation."""
        valence, arousal, clarity = self.mood_vector

        # Light natural variation (reduced to not overwhelm sentiment analysis)
        time_factor = (time.time() - self.session_start) / 1800.0  # 30-minute cycles (slower)
        valence_variation = 0.05 * np.sin(time_factor)  # ±0.05 oscillation (much smaller)
        arousal_variation = 0.08 * np.cos(time_factor * 1.3)  # Different frequency
        clarity_variation = 0.03 * np.sin(time_factor * 0.7)  # Slower clarity drift

        adjusted_valence = valence + valence_variation
        adjusted_arousal = arousal + arousal_variation
        adjusted_clarity = clarity + clarity_variation

        # Sentiment-responsive mapping - order matters!
        # 1. High energy positive states first
        if adjusted_valence > 0.05 and adjusted_arousal > 0.6:
            return "energized_engaged"  # High positive energy + high arousal

        # 2. Negative states
        elif adjusted_valence < -0.05:  # Lower threshold for negative detection
            if adjusted_arousal < 0.4:  # Higher threshold for low arousal
                return "withdrawn_distant"  # Low energy + negative = withdrawn
            else:
                return "alert_curious"  # Negative but high arousal = anxious curiosity

        # 3. Low arousal states (calm/quiet)
        elif adjusted_arousal < 0.25:
            if adjusted_valence > 0.02:
                return "calm_observant"  # Positive but calm
            else:
                return "quiet_detached"  # Low arousal + neutral/slightly negative

        # 4. High arousal with curiosity indicators
        elif adjusted_arousal > 0.4 and adjusted_clarity > 0.05:
            return "alert_curious"  # High arousal + decent clarity = curious

        # 5. Medium positive states
        elif adjusted_valence > 0.02:
            return "calm_observant"  # Positive but moderate arousal

        # 6. Default for everything else
        else:
            return "alert_curious"  # Default curious state

    def get_pattern_data(self) -> dict:
        """Get current pattern recognition data for external access."""
        return {
            "recent_motifs": list(self.pattern_engine.current_motifs),
            "motif_summary": self.pattern_engine.get_motif_summary(),
            "novelty_score": getattr(self, "_last_novelty", 0.0),
        }

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

        # Base novelty effect - much smaller and proportional
        change += novelty * 0.03  # Scale down novelty impact significantly

        # Natural mood decay - balanced with sentiment strength
        change -= 0.03  # Reduced to let sentiment analysis have more impact

        # Person presence effects - reduced
        if saw_person and not self.last_person_detected:
            change += 0.04  # Reduced from 0.07
        elif not saw_person and self.last_person_detected:
            change -= 0.03  # Reduced from 0.05

        # Add time-based decay for long sessions
        session_duration = time.time() - self.session_start
        if session_duration > 1800:  # After 30 minutes
            change -= 0.01  # Additional decay for long sessions
        if session_duration > 3600:  # After 1 hour
            change -= 0.02  # Even more decay

        return change

    def analyze_caption_sentiment(self, caption: str) -> float:
        """Analyze sentiment from caption content using keyword matching."""
        positive_words = [
            "happy",
            "joy",
            "smile",
            "laugh",
            "bright",
            "warm",
            "comfortable",
            "peaceful",
            "content",
            "energetic",
            "engaged",
            "focused",
            "curious",
            "confident",
            "relaxed",
            "pleasant",
            "cozy",
            "vibrant",
            "cheerful",
        ]

        negative_words = [
            "sad",
            "tired",
            "dark",
            "cold",
            "worried",
            "anxious",
            "stressed",
            "distant",
            "withdrawn",
            "confused",
            "lonely",
            "bored",
            "frustrated",
            "empty",
            "dull",
            "lifeless",
            "gloomy",
            "melancholy",
            "troubled",
        ]

        neutral_words = ["thoughtful", "contemplative", "observant", "quiet", "still", "reflective", "pensive", "calm", "serene", "composed"]

        caption_lower = caption.lower()
        sentiment_score = 0.0

        # Count positive indicators
        for word in positive_words:
            if word in caption_lower:
                sentiment_score += 0.02

        # Count negative indicators
        for word in negative_words:
            if word in caption_lower:
                sentiment_score -= 0.03

        # Neutral words have slight positive effect (better than nothing)
        for word in neutral_words:
            if word in caption_lower:
                sentiment_score += 0.005

        return np.clip(sentiment_score, -0.3, 0.3)  # Increased range for better emotion detection

    # analyze_3d_mood_with_ollama removed - now uses natural language sentiment from context compression

def log_mood(caption, mood, mood_change, image_path: Optional[str] = None):
    """
    Log mood data in JSON format with timestamp, caption, mood value, and image path.
    Only print to console when mood change is meaningful (>0.05).
    """
    data = {
        "caption": caption,
        "mood": mood,
        "mood_change": mood_change,
        "image_path": image_path if image_path and os.path.exists(image_path) else None,
    }

    # Only print mood updates for meaningful changes (>0.05) to reduce noise
    print_message = None
    if abs(mood_change) > 0.05:
        change_indicator = "↗" if mood_change > 0 else "↘"
        print_message = f"[😊] Mood {change_indicator} {mood:.2f} (Δ{mood_change:+.2f})"

    log_json_entry(LogType.MOOD, data, MOOD_SNAPSHOT_FOLDER, print_message=print_message)


