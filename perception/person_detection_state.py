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

        # Event tracking for behavior
        self.recent_arrivals = []    # List of recent arrival timestamps
        self.recent_departures = []  # List of recent departure timestamps

        # Spatial memory - where was the person last seen
        self.last_seen_servo_pan = None
        self.last_seen_servo_tilt = None
        self.current_servo_pan = None
        self.current_servo_tilt = None

        # Tri-state presence: "absent" | "visible" | "remembered"
        self.person_state = "absent"

        # Extended timeouts for gaze-aware logic
        # 180s (was 60): YOLO misses a still, distant person for stretches at
        # idle cadence — the 60s hard timeout churned phantom departures
        # ("come and gone 14 times" while one person sat at the desk).
        # Genuine exits are caught much faster by the sweep path above.
        self.remembered_timeout = 180.0  # Remember presence when not looking at last location
        self.looking_away_timeout = 8.0  # Normal timeout when looking at last known location

        # Sweep-based departure detection
        # Must scan a significant portion of the range before concluding person is gone
        self.scan_zones_visited = set()  # Which zones have been scanned since last detection
        self.scan_zone_size = 30  # Degrees per zone
        self.min_zones_for_departure = 4  # Must visit at least 4 zones (120°) to confirm departure

        # Detection smoothing - simple time-based grace period
        # Since false positives are rare but losing detection is constant, use time-based hold
        self._last_raw_detection_time = 0.0  # Last time YOLO actually detected someone
        self._detection_grace_period = 1.0  # Hold detection for 1 second after losing it
        self._smoothed_detection_active = False  # Current smoothed detection state
        self._last_valid_bbox: Optional[Tuple[int, int, int, int]] = None  # Last valid bbox for grace period

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
                # Reset sweep tracking when face detected
                self.scan_zones_visited.clear()
            # Don't clear face_detection - let it expire via timestamp
            self._update_person_state()

    def update_yolo_detection(self, person_detected: bool, confidence: float = 0.8, bbox: Optional[Tuple[int, int, int, int]] = None):
        """Update YOLO person detection state with time-based grace period to prevent flickering.

        Uses a simple 1-second hold: once detected, stays detected for 1 second even if
        raw detection drops out. This handles the case where the camera/YOLO frequently
        loses detection momentarily but false positives are rare.

        The bbox is stored when available and persists through the grace period for tracking.
        """
        with self._lock:
            now = time.time()

            if person_detected:
                # Raw detection is positive - update timestamp and mark as active
                self._last_raw_detection_time = now
                self._smoothed_detection_active = True
                # Store bbox when available
                if bbox is not None:
                    self._last_valid_bbox = bbox
            else:
                # Raw detection is negative - check grace period
                time_since_last = now - self._last_raw_detection_time
                if time_since_last < self._detection_grace_period:
                    # Within grace period - keep detection active (bbox persists)
                    self._smoothed_detection_active = True
                else:
                    # Past grace period - detection is truly lost
                    self._smoothed_detection_active = False
                    self._last_valid_bbox = None  # Clear bbox when truly lost

            # Update YOLO detection event based on smoothed state
            if self._smoothed_detection_active:
                self.yolo_detection = PersonDetectionEvent(
                    confidence=confidence,
                    source="yolo",
                    timestamp=now,
                    bbox=self._last_valid_bbox  # Use persisted bbox
                )
                # Reset sweep tracking when person detected
                self.scan_zones_visited.clear()
            # Don't clear yolo_detection when not detected - let it expire via timestamp
            self._update_person_state()

    def update_servo_position(self, pan: float, tilt: float):
        """Update current gaze position - called from machine.py main loop."""
        with self._lock:
            self.current_servo_pan = pan
            self.current_servo_tilt = tilt

            # Track which zones have been visited (for sweep-based departure)
            # Only track when person is not actively detected
            if not self._is_actively_detecting():
                zone = int(pan // self.scan_zone_size)
                self.scan_zones_visited.add(zone)

    def _is_actively_detecting(self) -> bool:
        """Check if we're actively detecting a person right now."""
        now = time.time()
        if self.yolo_detection and (now - self.yolo_detection.timestamp) < 2.0:
            return True
        if self.face_detection and (now - self.face_detection.timestamp) < 2.0:
            return True
        return False

    def _has_completed_sweep(self) -> bool:
        """Check if camera has scanned enough zones to confirm departure."""
        return len(self.scan_zones_visited) >= self.min_zones_for_departure

    def is_looking_at_last_known_location(self, tolerance: float = 25.0) -> bool:
        """Check if camera is pointing near where person was last seen."""
        if self.last_seen_servo_pan is None or self.current_servo_pan is None:
            return False
        if self.last_seen_servo_tilt is None or self.current_servo_tilt is None:
            return False
        pan_diff = abs(self.current_servo_pan - self.last_seen_servo_pan)
        tilt_diff = abs(self.current_servo_tilt - self.last_seen_servo_tilt)
        return pan_diff < tolerance and tilt_diff < tolerance

    def get_person_direction(self) -> Optional[str]:
        """Get relative direction of last known person location."""
        if self.last_seen_servo_pan is None or self.current_servo_pan is None:
            return None
        pan_diff = self.last_seen_servo_pan - self.current_servo_pan
        if pan_diff > 15:
            return "to my right"
        elif pan_diff < -15:
            return "to my left"
        return "ahead"

    def _update_person_state(self):
        """Internal method to update overall person presence state with spatial awareness."""
        now = time.time()

        # Determine current detection confidence
        current_confidence = 0.0
        best_detection = None

        # YOLO detection is now PRIMARY (more reliable body detection)
        if self.yolo_detection and (now - self.yolo_detection.timestamp) < 10.0:
            current_confidence = self.yolo_detection.confidence
            best_detection = self.yolo_detection

        # Face detection is SECONDARY (boosts confidence if also detected)
        if self.face_detection and (now - self.face_detection.timestamp) < 3.0:
            face_conf = self.face_detection.confidence * 0.8
            if face_conf > current_confidence:
                current_confidence = face_conf
                best_detection = self.face_detection

        # Update confidence and stability
        self.person_confidence = current_confidence

        # Determine person presence with hysteresis
        was_present = self.is_person_present
        person_detected = current_confidence > self.confidence_threshold

        if person_detected:
            # Person actively detected
            self.person_state = "visible"
            self.scan_zones_visited.clear()  # Reset sweep tracking
            if not was_present:
                # Fresh arrival
                self.is_person_present = True
                self.last_detection_time = now
                self.recent_arrivals.append(now)
                self.person_absence_duration = 0.0
                self.recent_arrivals = [t for t in self.recent_arrivals if now - t < 300]

            # Store spatial memory - where did we see them
            if self.current_servo_pan is not None:
                self.last_seen_servo_pan = self.current_servo_pan
                self.last_seen_servo_tilt = self.current_servo_tilt

        elif was_present:
            # Not detecting but was present - SWEEP-BASED departure logic
            # Must scan multiple zones before concluding person is truly gone
            time_since = now - self.last_detection_time
            looking_at_last = self.is_looking_at_last_known_location()
            sweep_complete = self._has_completed_sweep()

            # Only mark absent if BOTH conditions met:
            # 1. Looking at last known location (they're not where we last saw them)
            # 2. Have completed a sweep (we've looked around enough to be sure)
            if looking_at_last and time_since > self.looking_away_timeout and sweep_complete:
                # Looked where they were, did a sweep, didn't find them
                self.person_state = "absent"
                self.is_person_present = False
                self.last_departure_time = now
                self.recent_departures.append(now)
                self.person_presence_duration = 0.0
                self.recent_departures = [t for t in self.recent_departures if now - t < 300]
                print(f"[👤] Person marked absent after sweep ({len(self.scan_zones_visited)} zones scanned)")
                try:
                    from utils.episodic_log import episodic_log
                    episodic_log.record("person_left", "confirmed gone after sweep")
                except Exception:
                    pass
            elif time_since > self.remembered_timeout:
                # Hard timeout - even without sweep, don't remember forever
                self.person_state = "absent"
                self.is_person_present = False
                self.last_departure_time = now
                self.recent_departures.append(now)
                self.person_presence_duration = 0.0
                self.recent_departures = [t for t in self.recent_departures if now - t < 300]
                print(f"[👤] Person marked absent after timeout ({int(self.remembered_timeout)}s)")
                try:
                    from utils.episodic_log import episodic_log
                    episodic_log.record("person_left", "gone after timeout")
                except Exception:
                    pass
            else:
                # Stay in "remembered" state - person might still be here
                self.person_state = "remembered"
                self.is_person_present = True

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
                "person_state": self.person_state,
                "direction": self.get_person_direction(),
                "confidence": self.person_confidence,
                "presence_duration": self.person_presence_duration,
                "absence_duration": self.person_absence_duration,
                "stability": self.detection_stability,
                "last_detection": self.last_detection_time,
                "last_departure": self.last_departure_time,
                "recent_arrivals": len(self.recent_arrivals),
                "recent_departures": len(self.recent_departures),
            }

    def get_smoothed_bbox(self) -> Optional[Tuple[int, int, int, int]]:
        """Get the bbox that persists through the grace period for tracking."""
        with self._lock:
            if self._smoothed_detection_active:
                return self._last_valid_bbox
            return None

    def get_last_raw_detection_time(self) -> float:
        """Get the timestamp of the most recent positive detection (not arrival time)."""
        with self._lock:
            return self._last_raw_detection_time

    def get_consciousness_context(self) -> str:
        """Get person presence context for consciousness prompts with spatial awareness."""
        state = self.get_person_state()
        person_state = state.get("person_state", "absent")

        if person_state == "visible":
            duration = state["presence_duration"]
            if duration < 5.0:
                return "Someone just appeared"
            return f"Someone here for {duration:.0f}s"

        elif person_state == "remembered":
            direction = state.get("direction", "nearby")
            return f"Someone is {direction}, not looking"

        else:  # absent
            if state["absence_duration"] < 10.0:
                return "They just left"
            elif state["recent_departures"] > 0:
                return "Solitude after presence"
            return "Alone"


# Global instance
_person_detection_state = None

def get_person_detection_state() -> PersonDetectionState:
    """Get the global person detection state instance."""
    global _person_detection_state
    if _person_detection_state is None:
        _person_detection_state = PersonDetectionState()
    return _person_detection_state