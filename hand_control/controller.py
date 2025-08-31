#!/usr/bin/env python3
"""
Hand Controller
===============

Direct hand controller integration without bridge complexity.
Maps mood states to emotional hand movements.

Emotional States:
- energized_engaged: High energy, quick movements
- alert_curious: Moderate energy, exploratory movements
- calm_observant: Gentle, contemplative movements
- quiet_detached: Slow, minimal movements
- withdrawn_distant: Very minimal, distant movements
"""

import json
import os
import time
from typing import Any, Dict, Optional

from mood.mood import MoodEngine


class HandController:
    """Direct hand controller without bridge complexity."""

    def __init__(self):
        """Initialize the hand controller."""
        self.current_emotion_state = None
        self.last_update_time = 0
        self.update_interval = 2.0  # Minimum interval between state changes

        # Emotional state mapping based on mood values
        self.emotion_thresholds = {
            "withdrawn_distant": 0.0,  # 0.0 - 0.2
            "quiet_detached": 0.2,  # 0.2 - 0.4
            "calm_observant": 0.4,  # 0.4 - 0.6
            "alert_curious": 0.6,  # 0.6 - 0.8
            "energized_engaged": 0.8,  # 0.8 - 1.0
        }

        print("🤲 Hand Controller initialized (direct integration)")

    def map_mood_to_emotion(self, mood: float, boredom: float = 0.5, novelty: float = 0.5) -> str:
        """
        Map numerical mood to emotional state.

        Args:
            mood: 0.0-1.0 mood value
            boredom: 0.0-1.0 boredom factor (affects withdrawal)
            novelty: 0.0-1.0 novelty factor (affects curiosity)

        Returns:
            Emotional state string
        """
        # Factor in boredom and novelty for more nuanced emotional mapping
        effective_mood = mood

        # High boredom pushes toward withdrawn states
        if boredom > 0.7:
            effective_mood = max(0.0, effective_mood - (boredom - 0.7) * 0.5)

        # High novelty can boost alertness
        if novelty > 0.7:
            effective_mood = min(1.0, effective_mood + (novelty - 0.7) * 0.3)

        # Map to emotional states
        if effective_mood <= 0.2:
            return "withdrawn_distant"
        elif effective_mood <= 0.4:
            return "quiet_detached"
        elif effective_mood <= 0.6:
            return "calm_observant"
        elif effective_mood <= 0.8:
            return "alert_curious"
        else:
            return "energized_engaged"

    def update(
        self,
        mood_engine: MoodEngine,
        force_update: bool = False,
        reactivity_data: Optional[Dict[str, float]] = None,
        temporal_data: Optional[Dict[str, Any]] = None,
    ):
        """
        Update hand controller with current emotional state.

        Args:
            mood_engine: Active mood engine instance
            force_update: Force update even if state hasn't changed
            reactivity_data: Real-time camera reactivity metrics
            temporal_data: Temporal awareness data (boredom, session time)
        """
        current_time = time.time()

        # Rate limiting
        if not force_update and (current_time - self.last_update_time) < self.update_interval:
            return

        # Get current mood
        current_mood = mood_engine.get_current_mood()

        # Extract temporal factors
        boredom = temporal_data.get("boredom", 0.5) if temporal_data else 0.5
        novelty = temporal_data.get("novelty", 0.5) if temporal_data else 0.5

        # Map to emotional state with temporal factors
        emotion_state = self.map_mood_to_emotion(current_mood, boredom, novelty)

        # Check if state actually changed
        if not force_update and emotion_state == self.current_emotion_state:
            return

        # Handle pause state from reactivity
        is_paused = reactivity_data.get("is_paused", False) if reactivity_data else False

        if is_paused:
            self._handle_pause_state(reactivity_data)
        else:
            self._activate_emotion_state(emotion_state, current_mood, temporal_data)

        self.current_emotion_state = emotion_state
        self.last_update_time = current_time

    def _activate_emotion_state(self, emotion_state: str, mood_value: float, temporal_data: Optional[Dict[str, Any]] = None):
        """Activate a specific emotional state."""
        # Clean, minimal emotional state change notification
        if emotion_state != self.current_emotion_state:
            session_time = temporal_data.get("session_duration", 0) if temporal_data else 0
            boredom = temporal_data.get("boredom", 0) if temporal_data else 0

            # Include temporal context in the display - commented out to avoid duplicates
            # if session_time > 3600:  # Over 1 hour
            #     hours = session_time / 3600
            #     print(f"🎭 {emotion_state} (session: {hours:.1f}h, boredom: {boredom:.2f})")
            # else:
            #     print(f"🎭 {emotion_state}")

        # Here you would implement actual hand movement control
        # For now, just log the state change
        self._log_state_change(emotion_state, mood_value, temporal_data)

    def _handle_pause_state(self, reactivity_data: Dict[str, float]):
        """Handle camera activity pause state."""
        pause_remaining = reactivity_data.get("pause_remaining", 3.0)
        print(f"🔄 Hand movement paused ({pause_remaining:.1f}s remaining)")

        # Here you would pause hand movements
        # For now, just log the pause

    def _log_state_change(self, emotion_state: str, mood_value: float, temporal_data: Optional[Dict[str, Any]] = None):
        """Log state change for debugging/monitoring."""
        log_data = {"timestamp": time.time(), "emotion_state": emotion_state, "mood_value": mood_value, "source": "direct_hand_controller"}

        if temporal_data:
            log_data["temporal"] = temporal_data

        # Could optionally write to event_log/ for consistency with the rest of the system

    def get_status(self) -> Dict[str, Any]:
        """Get current hand controller status."""
        return {
            "current_emotion_state": self.current_emotion_state,
            "last_update_time": self.last_update_time,
            "update_interval": self.update_interval,
            "integration_type": "direct",
        }
