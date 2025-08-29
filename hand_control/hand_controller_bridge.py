#!/usr/bin/env python3
"""
Hand Controller Bridge
======================

Bridges machine.py's mood engine output to th        try:
            if self.communication_method == 'file':
                return self._send_via_file(emotion_state, mood_value, reactivity_data)
            elif self.communication_method == 'tcp':
                return self._send_via_tcp(emotion_state, mood_value, reactivity_data)ndalone hand controller's emotional states.
Maps numerical mood values to discrete emotional states and triggers appropriate Markov generation.

Integration approach:
1. Read mood from machine.py's mood engine
2. Map to hand controller's emotional states: energized_engaged, alert_curious, calm_observant, quiet_detached, withdrawn_distant
3. Send state change commands to hand controller (via file, TCP, or direct API)
4. Hand controller switches to corresponding emotion and cycles through datasets

Author: Bridge System for Emotional Hand Control
"""
import time
import json

# import os
# import socket
from typing import Optional, Dict, Any
from mood.mood import MoodEngine


class HandControllerBridge:
    """Bridge between machine.py mood and hand controller emotional states."""

    def __init__(self, communication_method: str = "file"):
        """
        Initialize the bridge.

        Args:
            communication_method: "file", "tcp", or "direct"
                - "file": Write state to shared JSON file
                - "tcp": Send via TCP socket to hand controller
                - "direct": Import and call hand controller directly (future)
        """
        self.communication_method = communication_method
        self.last_emotion_state = None
        self.last_update_time = 0
        self.update_interval = 2.0  # Don't spam updates - 2 second minimum interval

        # Emotional state mapping based on mood values
        self.emotion_thresholds = {
            "withdrawn_distant": 0.0,  # 0.0 - 0.2
            "quiet_detached": 0.2,  # 0.2 - 0.4
            "calm_observant": 0.4,  # 0.4 - 0.6
            "alert_curious": 0.6,  # 0.6 - 0.8
            "energized_engaged": 0.8,  # 0.8 - 1.0
        }

        # File-based communication setup
        if self.communication_method == "file":
            # Write directly to the HandControlStandalone directory
            self.state_file = r"C:\Users\tobia\Downloads\HandControlStandalone\hand_controller_state.json"

        # TCP communication setup (future)
        elif self.communication_method == "tcp":
            self.tcp_host = "localhost"
            self.tcp_port = 9999

        print(f"Hand Controller Bridge initialized (method: {communication_method})")

    def map_mood_to_emotion(self, mood: float, novelty: float = 0.5, boredom: float = 0.5) -> str:
        """
        Map numerical mood value to discrete emotional state.

        Args:
            mood: 0.0-1.0 mood value from machine.py
            novelty: 0.0-1.0 novelty factor (future use)
            boredom: 0.0-1.0 boredom factor (future use)

        Returns:
            Emotional state string matching hand controller's states
        """
        # Simple mood-based mapping (can be enhanced with novelty/boredom later)
        if mood <= 0.2:
            return "withdrawn_distant"
        elif mood <= 0.4:
            return "quiet_detached"
        elif mood <= 0.6:
            return "calm_observant"
        elif mood <= 0.8:
            return "alert_curious"
        else:
            return "energized_engaged"

    def update_hand_controller(self, mood_engine: MoodEngine, force_update: bool = False, reactivity_data: Optional[Dict[str, float]] = None):
        """
        Update hand controller with current emotional state from mood engine.

        Args:
            mood_engine: Active mood engine instance from machine.py
            force_update: Force update even if state hasn't changed
            reactivity_data: Real-time camera reactivity metrics for organic behavior
        """
        current_time = time.time()

        # Rate limiting - don't spam updates
        if not force_update and (current_time - self.last_update_time) < self.update_interval:
            return

        # Get current mood
        current_mood = mood_engine.get_current_mood()

        # Map to emotional state
        emotion_state = self.map_mood_to_emotion(current_mood)

        # Check if state actually changed
        if not force_update and emotion_state == self.last_emotion_state:
            return

        # Send state update
        success = self._send_state_update(emotion_state, current_mood, reactivity_data)

        if success:
            # Only print when emotional state actually changes (not every mood update)
            if emotion_state != self.last_emotion_state:
                print(f"🎭 {emotion_state}")  # Clean, minimal emotional state change
            self.last_emotion_state = emotion_state
            self.last_update_time = current_time

    def _send_state_update(self, emotion_state: str, mood_value: float, reactivity_data: Optional[Dict[str, float]] = None) -> bool:
        """Send state update via chosen communication method."""
        try:
            if self.communication_method == "file":
                return self._send_via_file(emotion_state, mood_value)
            elif self.communication_method == "tcp":
                return self._send_via_tcp(emotion_state, mood_value)
            else:
                print(f"ERROR Unknown communication method: {self.communication_method}")
                return False
        except Exception as e:
            print(f"ERROR Failed to send state update: {e}")
            return False

    def _send_via_file(self, emotion_state: str, mood_value: float, reactivity_data: Optional[Dict[str, float]] = None) -> bool:
        """Send state via shared JSON file."""
        state_data = {"emotion_state": emotion_state, "mood_value": mood_value, "timestamp": time.time(), "source": "machine_py_bridge"}

        # Add camera reactivity data if available
        if reactivity_data:
            state_data["reactivity"] = reactivity_data
            # Add explicit pause command for hand controller
            if reactivity_data.get("is_paused", False):
                state_data["pause_command"] = {
                    "action": "pause",
                    "duration": reactivity_data.get("pause_remaining", 3.0),
                    "reason": "camera_activity",
                }
                print(f"🔄 Sending pause command to hand controller: {reactivity_data.get('pause_remaining', 3.0):.1f}s remaining")
            else:
                state_data["pause_command"] = {"action": "resume", "reason": "camera_activity"}

        try:
            with open(self.state_file, "w") as f:
                json.dump(state_data, f, indent=2)
            return True
        except Exception as e:
            print(f"ERROR Failed to write bridge file: {e}")
            return False

    def _send_via_tcp(self, emotion_state: str, mood_value: float, reactivity_data: Optional[Dict[str, float]] = None) -> bool:
        """Send state via TCP socket (future implementation)."""
        # TODO: Implement TCP communication with reactivity data
        print(f"🌐 TCP communication not yet implemented")
        return False

    def get_status(self) -> Dict[str, Any]:
        """Get current bridge status."""
        return {
            "communication_method": self.communication_method,
            "last_emotion_state": self.last_emotion_state,
            "last_update_time": self.last_update_time,
            "update_interval": self.update_interval,
        }


# Simple test function
def test_bridge():
    """Test the bridge with simulated mood values."""
    bridge = HandControllerBridge("file")

    # Create mock mood engine
    class MockMoodEngine:
        def __init__(self, mood):
            self.current_mood = mood

        def get_current_mood(self):
            return self.current_mood

    # Test different mood values
    test_moods = [0.1, 0.3, 0.5, 0.7, 0.9]

    for mood in test_moods:
        mock_engine = MockMoodEngine(mood)
        bridge.update_hand_controller(mock_engine, force_update=True)  # type: ignore
        time.sleep(0.5)

    print("SUCCESS Bridge test complete")


if __name__ == "__main__":
    test_bridge()
