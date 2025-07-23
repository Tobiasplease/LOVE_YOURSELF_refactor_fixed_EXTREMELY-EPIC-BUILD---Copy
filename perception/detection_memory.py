
# detection_memory.py

import threading

class DetectionMemory:
    _lock = threading.Lock()
    _labels = []
    _timestamp = None
    _image = None

    @classmethod
    def update(cls, labels, timestamp=None, image=None):
        with cls._lock:
            cls._labels = labels
            cls._timestamp = timestamp
            cls._image = image

    @classmethod
    def get_labels(cls):
        with cls._lock:
            return cls._labels.copy()

    @classmethod
    def get_image(cls):
        with cls._lock:
            return cls._image