# detection_memory.py

import threading
from typing import Dict, List, Optional, Tuple


class DetectionMemory:
    _lock = threading.Lock()
    _labels = []
    _timestamp = None
    _image = None
    _person_bbox: Optional[Tuple[int, int, int, int]] = None
    _person_confidence: float = 0.0
    _person_count: int = 0
    _best_track_id: Optional[int] = None
    _person_tracks: Dict[int, Tuple[Tuple[int, int, int, int], float]] = {}  # {track_id: (bbox, conf)}

    @classmethod
    def update(cls, labels, timestamp=None, image=None, person_bbox=None, person_confidence=0.0,
               person_count=0, best_track_id=None, person_tracks=None):
        with cls._lock:
            cls._labels = labels
            cls._timestamp = timestamp
            cls._image = image
            cls._person_bbox = person_bbox
            cls._person_confidence = person_confidence
            cls._person_count = person_count
            cls._best_track_id = best_track_id
            cls._person_tracks = person_tracks or {}

    @classmethod
    def get_labels(cls):
        with cls._lock:
            return cls._labels.copy()

    @classmethod
    def get_person_bbox(cls) -> Optional[Tuple[int, int, int, int]]:
        with cls._lock:
            return cls._person_bbox

    @classmethod
    def get_person_confidence(cls) -> float:
        with cls._lock:
            return cls._person_confidence

    @classmethod
    def get_person_count(cls) -> int:
        with cls._lock:
            return cls._person_count

    @classmethod
    def get_best_track_id(cls) -> Optional[int]:
        """Get the ByteTrack ID of the highest-confidence person detection."""
        with cls._lock:
            return cls._best_track_id

    @classmethod
    def get_person_tracks(cls) -> Dict[int, Tuple[Tuple[int, int, int, int], float]]:
        """Get all tracked persons: {track_id: (bbox, confidence)}."""
        with cls._lock:
            return cls._person_tracks.copy()
