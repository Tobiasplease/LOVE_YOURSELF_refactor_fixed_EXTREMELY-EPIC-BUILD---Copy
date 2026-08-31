# mood/mood.py
from __future__ import annotations

import time

import numpy as np  # type: ignore


# ---------------------------------------------------------------------------#
# Pure MoodEngine - analyzes captions without generating them               #
# ---------------------------------------------------------------------------#
class MoodEngine:
    def __init__(self) -> None:
        self.current_mood = 0.5  # Backward compatibility scalar
        self.mood_vector = (0.0, 0.0, 0.0)  # Initial: truly neutral valence, arousal, clarity
        self.emotional_momentum = 0.2  # How much previous mood influences new mood (0.0-1.0) - lower for more responsive emotional evolution
        self.session_start = time.time()  # Oscillator phase in get_emotion_for_hand_controller

    # -------------------------------------------------------------- main hook
    def analyze_mood(
        self,
        caption: str,
        saw_person: bool = False,
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
        A real event (company) still nudges on top — a state signal, not
        text, loop-safe. Momentum smooths per caption as before.
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
            (1 - momentum) * clarity + momentum * prev_clarity,
        )

        # Legacy scalar (0..1) now derives from blended valence — the keyword
        # sentiment + decay arithmetic it used to carry is retired.
        previous_scalar = self.current_mood
        self.current_mood = float(np.clip(0.5 + 0.5 * self.mood_vector[0], 0.0, 1.0))

        change = self.current_mood - previous_scalar
        if abs(change) > 0.05:
            print(f"[😊] Mood {'↗' if change > 0 else '↘'} {self.current_mood:.2f} (Δ{change:+.2f})")
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
