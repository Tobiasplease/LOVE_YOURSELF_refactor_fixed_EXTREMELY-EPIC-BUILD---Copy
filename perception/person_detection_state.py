"""
Person Detection State Management
Unifies face detection and YOLO person detection into a coherent person presence system
that can influence breathing, hand control, and consciousness.
"""

import threading
import time
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

@dataclass
class PersonDetectionEvent:
    """Represents a person detection event with confidence and source."""
    confidence: float
    source: str  # "face" or "yolo"
    timestamp: float
    bbox: Optional[Tuple[int, int, int, int]] = None

class PersonDetectionState:
    """Unified person detection state that combines multiple detection sources."""

    def __init__(self):
        self._lock = threading.Lock()

        # Current state
        self.is_person_present = False
        self.person_confidence = 0.0
        self.last_detection_time = 0.0
        self.last_departure_time = 0.0

        # Detection sources
        self.face_detection: Optional[PersonDetectionEvent] = None
        self.yolo_detection: Optional[PersonDetectionEvent] = None

        # Behavioral state tracking
        self.person_presence_duration = 0.0  # How long person has been present
        self.person_absence_duration = 0.0   # How long person has been absent
        self.detection_stability = 0.0       # Stability of detection (0.0-1.0)

        # Configuration
        self.confidence_threshold = 0.5
        self.stability_window = 3.0  # seconds to stabilize detection
        self.departure_delay = 2.0   # seconds to confirm person departure

        # Event tracking for behavior
        self.recent_arrivals = []    # List of recent arrival timestamps
        self.recent_departures = []  # List of recent departure timestamps

    def update_face_detection(self, confidence: float, bbox: Optional[Tuple[int, int, int, int]] = None):
        """Update face detection state."""
        with self._lock:
            if confidence > 0.0:
                self.face_detection = PersonDetectionEvent(
                    confidence=confidence,
                    source="face",
                    timestamp=time.time(),
                    bbox=bbox
                )
            else:
                self.face_detection = None
            self._update_person_state()

    def update_yolo_detection(self, person_detected: bool, confidence: float = 0.8):
        """Update YOLO person detection state."""
        with self._lock:
            if person_detected:
                self.yolo_detection = PersonDetectionEvent(
                    confidence=confidence,
                    source="yolo",
                    timestamp=time.time()
                )
            else:
                self.yolo_detection = None
            self._update_person_state()

    def _update_person_state(self):
        """Internal method to update overall person presence state."""
        now = time.time()

        # Determine current detection confidence
        current_confidence = 0.0
        best_detection = None

        # Check face detection (higher priority, more immediate)
        if self.face_detection and (now - self.face_detection.timestamp) < 3.0:
            current_confidence = max(current_confidence, self.face_detection.confidence)
            best_detection = self.face_detection

        # Check YOLO detection (lower priority, broader context)
        if self.yolo_detection and (now - self.yolo_detection.timestamp) < 8.0:
            yolo_confidence = self.yolo_detection.confidence * 0.7  # Weight YOLO lower
            if yolo_confidence > current_confidence:
                current_confidence = yolo_confidence
                best_detection = self.yolo_detection

        # Update confidence and stability
        self.person_confidence = current_confidence

        # Determine person presence with hysteresis
        was_present = self.is_person_present
        person_detected = current_confidence > self.confidence_threshold

        if person_detected and not was_present:
            # Person arrival
            self.is_person_present = True
            self.last_detection_time = now
            self.recent_arrivals.append(now)
            self.person_absence_duration = 0.0

            # Clean old arrivals
            self.recent_arrivals = [t for t in self.recent_arrivals if now - t < 300]  # 5 minutes

        elif not person_detected and was_present:
            # Check departure delay
            if now - self.last_detection_time > self.departure_delay:
                # Person departure confirmed
                self.is_person_present = False
                self.last_departure_time = now
                self.recent_departures.append(now)
                self.person_presence_duration = 0.0

                # Clean old departures
                self.recent_departures = [t for t in self.recent_departures if now - t < 300]  # 5 minutes

        # Update duration tracking
        if self.is_person_present:
            self.person_presence_duration = now - self.last_detection_time
        else:
            self.person_absence_duration = now - self.last_departure_time if self.last_departure_time > 0 else float('inf')

        # Update detection stability
        if best_detection:
            detection_age = now - best_detection.timestamp
            self.detection_stability = max(0.0, 1.0 - (detection_age / self.stability_window))
        else:
            self.detection_stability = 0.0

    def get_person_state(self) -> Dict:
        """Get current person detection state for other systems."""
        with self._lock:
            return {
                "is_present": self.is_person_present,
                "confidence": self.person_confidence,
                "presence_duration": self.person_presence_duration,
                "absence_duration": self.person_absence_duration,
                "stability": self.detection_stability,
                "last_detection": self.last_detection_time,
                "last_departure": self.last_departure_time,
                "recent_arrivals": len(self.recent_arrivals),
                "recent_departures": len(self.recent_departures),
            }

    def get_breathing_modifiers(self, emotion_state: str) -> Tuple[float, float]:
        """Get breathing speed and pause modifiers based on person presence."""
        state = self.get_person_state()

        if not state["is_present"]:
            return 1.0, 1.0  # No modifiers when alone

        # Base response to person presence
        breath_modifier = 1.1  # Slightly slower when observed
        pause_modifier = 1.2   # Longer pauses when person present

        # Adjust based on how long person has been present
        presence_duration = state["presence_duration"]
        if presence_duration < 5.0:
            # Initial reaction - more pronounced
            breath_modifier *= 1.15
            pause_modifier *= 1.4
        elif presence_duration > 30.0:
            # Comfortable with presence
            breath_modifier *= 0.95
            pause_modifier *= 0.9

        # Emotion-specific responses
        if emotion_state == "alert_curious":
            breath_modifier *= 0.85  # More alert breathing
            pause_modifier *= 0.7    # Shorter pauses when curious
        elif emotion_state == "withdrawn_distant":
            breath_modifier *= 1.3   # Much slower when withdrawn
            pause_modifier *= 1.8    # Long pauses when shy
        elif emotion_state == "energized_engaged":
            breath_modifier *= 0.9   # Slight excitement
            pause_modifier *= 0.8    # Quick responses when social

        return breath_modifier, pause_modifier

    def should_trigger_hand_freeze(self) -> bool:
        """Determine if hand control should freeze due to person detection."""
        state = self.get_person_state()

        # Trigger freeze on fresh person arrival
        if state["is_present"] and state["presence_duration"] < 2.0:
            return True

        # Trigger freeze if person detection is very sudden and confident
        if (state["is_present"] and
            state["confidence"] > 0.8 and
            state["stability"] > 0.7 and
            state["recent_arrivals"] > 0):
            return True

        return False

    def get_consciousness_context(self) -> str:
        """Get person presence context for consciousness prompts."""
        state = self.get_person_state()

        if not state["is_present"]:
            if state["absence_duration"] < 10.0:
                return "Person just left - lingering awareness of their presence"
            elif state["recent_departures"] > 0:
                return "Solitude after recent social presence"
            else:
                return "Alone in this space"

        presence_duration = state["presence_duration"]
        if presence_duration < 5.0:
            return f"Person just arrived - {presence_duration:.1f}s of fresh social awareness"
        elif presence_duration < 30.0:
            return f"Person present for {presence_duration:.1f}s - developing social interaction"
        else:
            minutes = presence_duration / 60.0
            return f"Sustained social presence for {minutes:.1f} minutes"


# Global instance
_person_detection_state = None

def get_person_detection_state() -> PersonDetectionState:
    """Get the global person detection state instance."""
    global _person_detection_state
    if _person_detection_state is None:
        _person_detection_state = PersonDetectionState()
    return _person_detection_state