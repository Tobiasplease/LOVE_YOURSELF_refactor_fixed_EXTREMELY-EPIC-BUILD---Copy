# detection_memory.py

import threading
from typing import List, Optional, Tuple


class DetectionMemory:
    _lock = threading.Lock()
    _labels = []
    _timestamp = None
    _image = None
    _person_bbox: Optional[Tuple[int, int, int, int]] = None
    _person_confidence: float = 0.0

    @classmethod
    def update(cls, labels, timestamp=None, image=None, person_bbox=None, person_confidence=0.0):
        with cls._lock:
            cls._labels = labels
            cls._timestamp = timestamp
            cls._image = image
            cls._person_bbox = person_bbox
            cls._person_confidence = person_confidence

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
