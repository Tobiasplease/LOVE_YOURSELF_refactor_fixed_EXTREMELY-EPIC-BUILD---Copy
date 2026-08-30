# detection_memory.py

import threading
from typing import Dict, List, Optional, Tuple


class DetectionMemory:
    _lock = threading.Lock()
    _labels = []
    _image = None
    _person_bbox: Optional[Tuple[int, int, int, int]] = None
    _person_confidence: float = 0.0
    _person_count: int = 0
    _best_track_id: Optional[int] = None

    @classmethod
    def update(cls, labels, image=None, person_bbox=None, person_confidence=0.0,
               person_count=0, best_track_id=None):
        with cls._lock:
            cls._labels = labels
            cls._image = image
            cls._person_bbox = person_bbox
            cls._person_confidence = person_confidence
            cls._person_count = person_count
            cls._best_track_id = best_track_id

    @classmethod
    def get_labels(cls):
        with cls._lock:
            return cls._labels.copy()

    @classmethod
    def get_person_bbox(cls) -> Optional[Tuple[int, int, int, int]]:
        with cls._lock:
            return cls._person_bbox

    @classmethod
    def get_person_crop(cls):
        """Atomic (frame, bbox) snapshot cropped to the best person, or None."""
        with cls._lock:
            if cls._image is None or cls._person_bbox is None:
                return None
            x1, y1, x2, y2 = cls._person_bbox
            crop = cls._image[max(0, y1) : y2, max(0, x1) : x2]
            return crop.copy() if crop.size else None

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

