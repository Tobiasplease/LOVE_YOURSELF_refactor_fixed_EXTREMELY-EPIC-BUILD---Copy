# mood/mood.py
from __future__ import annotations

import os
import time
from typing import Optional

import numpy as np  # type: ignore

from config.config import MOOD_SNAPSHOT_FOLDER
from event_logging.event_logger import log_json_entry
from event_logging.log_type import LogType


def mood_to_feeling(valence: float, arousal: float) -> str:
    """Translate the abstract valence/arousal mood vector into a plain, literal
    feeling the LLM can grasp — an emotion word plus its intensity, e.g.
    "a little bored", "very calm", "really anxious". Deterministic (no LLM, no
    poetry), but DEGREED: slightly happy vs very happy vs really happy. Think of
    naming a feeling in the most direct, unambiguous terms — emotion + how strong.
    This is the felt-state's job: make the numeric mood legible to the model.
    """
    # Spaces: valence is signed (-1..1); arousal is the ENGINE's 0..1 space
    # (base ~0.35). The old thresholds treated arousal as signed, so "low"
    # was unreachable and anything above 0.2 read as "high" — every feeling
    # came out excited/anxious/restless.
    arousal_dev = arousal - 0.35
    strength = max(abs(valence), abs(arousal_dev) * 1.5)
    if strength < 0.12:
        return "calm"  # basically neutral — no strong feeling either way

    pos, neg = valence > 0.15, valence < -0.15
    high, low = arousal_dev > 0.15, arousal_dev < -0.15
    if pos and high:
        word = "excited"
    elif pos and low:
        word = "content"
    elif pos:
        word = "happy"
    elif neg and high:
        word = "anxious"
    elif neg and low:
        word = "down"
    elif neg:
        word = "uneasy"
    elif high:
        word = "restless"
    elif low:
        word = "calm"
    else:
        word = "okay"

    if strength < 0.30:
        adv = "a little "
    elif strength < 0.55:
        adv = ""
    elif strength < 0.78:
        adv = "very "
    else:
        adv = "really "

    # "a little calm/okay/content" reads oddly — keep low-key words plain.
    if adv == "a little " and word in ("calm", "okay", "content"):
        adv = ""
    return (adv + word).strip()


# ---------------------------------------------------------------------------#
# Pure MoodEngine - analyzes captions without generating them               #
# ---------------------------------------------------------------------------#
class MoodEngine:
    def __init__(self) -> None:
        self.current_mood = 0.5  # Backward compatibility scalar
        self.mood_vector = (0.0, 0.0, 0.0)  # Initial: truly neutral valence, arousal, clarity
        self.emotional_momentum = 0.2  # How much previous mood influences new mood (0.0-1.0) - lower for more responsive emotional evolution
        self.last_caption = ""
        self.last_person_detected = False
        self.session_start = time.time()  # Oscillator phase in get_emotion_for_hand_controller

    # -------------------------------------------------------------- main hook
    def analyze_mood(
        self,
        caption: str,
        saw_person: bool = False,
        image_path: str | None = None,
    ) -> float:
        """Analyze mood: LLM mood read as the core signal + real-event nudges.

        REINTEGRATED July 10. The old core was a keyword lexicon matching
        emotion adjectives ("happy", "gloomy", "cozy") — dead against the
        post-teardown voice, which never names emotions, so valence flatlined
        at ~0 for weeks and every downstream consumer starved (felt-state
        "calm" forever, hand controller stuck on defaults, drawing step-2
        inventing drama to contradict "balanced"). The core signal is now the
        compression thread's mood read (context_compression._mood_read): the
        model reading the undertone of the machine's own recent thoughts.
        Real events (company, novelty) still nudge on top — state signals,
        not text, loop-safe. Momentum smooths per caption as before.
        """
        saw_person = saw_person or "person" in caption.lower() or "individual" in caption.lower()

        # Core affect from the mood read (fresh within 15 min); neutral until
        # the first read of the session lands.
        read_valence, read_arousal = 0.0, 0.35
        try:
            from captioner.context_compression import context_compressor
            read = context_compressor.get_last_mood_read()
            if read:
                read_valence = read.get("valence", 0.0)
                read_arousal = read.get("arousal", 0.35)
        except Exception:
            pass

        valence = np.clip(read_valence + (0.08 if saw_person else 0.0), -1.0, 1.0)
        arousal = np.clip(read_arousal + (0.12 if saw_person else 0.0), 0.0, 1.0)
        clarity = np.clip((len(caption.split()) - 10) / 20, -1.0, 1.0)  # Caption length suggests clarity

        # Apply emotional momentum to smooth transitions
        prev_valence, prev_arousal, prev_clarity = self.mood_vector
        momentum = self.emotional_momentum
        self.mood_vector = (
            (1 - momentum) * valence + momentum * prev_valence,
            (1 - momentum) * arousal + momentum * prev_arousal,
            (1 - momentum) * clarity + momentum * prev_clarity
        )

        # Legacy scalar (0..1) now derives from blended valence — the keyword
        # sentiment + decay arithmetic it used to carry is retired.
        previous_scalar = self.current_mood
        self.current_mood = float(np.clip(0.5 + 0.5 * self.mood_vector[0], 0.0, 1.0))

        log_mood(caption, self.current_mood, self.current_mood - previous_scalar, image_path=image_path)
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

    # Keyword sentiment (analyze_caption_sentiment) + compute_mood_change
    # retired July 10 — the lexicon matched emotion adjectives the voice never
    # uses; the core signal is now the LLM mood read (context_compression).

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


