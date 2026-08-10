# presence_identity.py

import threading
import time
from collections import deque

from config.config import (
    PRESENCE_REID_ENABLED,
    PRESENCE_REID_GALLERY_SIZE,
    PRESENCE_REID_MAX_AGE,
    PRESENCE_REID_SAMPLE_INTERVAL,
    PRESENCE_REID_THRESHOLD,
)
from perception.detection_memory import DetectionMemory


class PresenceIdentity:
    """Session-scoped person re-identification: CLIP image embeddings of person
    crops in a rolling gallery. Answers one question — is the person on camera
    the same one seen recently? — so a lapsed presence belief resuming does not
    fire a false arrival. No names, no persistent biometrics; the gallery dies
    with the process and entries expire after PRESENCE_REID_MAX_AGE."""

    def __init__(self):
        self.lock = threading.Lock()
        self._model = None
        self._preprocess = None
        self._gallery = deque(maxlen=PRESENCE_REID_GALLERY_SIZE)  # (ts, unit embedding)
        self._last_sample = 0.0

    def note_sighting(self):
        """Add the current person crop to the gallery. Rate-limited; call freely."""
        if not PRESENCE_REID_ENABLED:
            return
        now = time.time()
        with self.lock:
            if now - self._last_sample < PRESENCE_REID_SAMPLE_INTERVAL:
                return
            self._last_sample = now
        emb = self._embed_current_person()
        if emb is not None:
            with self.lock:
                self._gallery.append((now, emb))

    def matches_recent(self):
        """True = same person as a recent sighting, False = looks like someone
        else, None = can't tell (no crop / empty gallery / model unavailable)."""
        if not PRESENCE_REID_ENABLED:
            return None
        emb = self._embed_current_person()
        if emb is None:
            return None
        now = time.time()
        with self.lock:
            while self._gallery and now - self._gallery[0][0] > PRESENCE_REID_MAX_AGE:
                self._gallery.popleft()
            gallery = list(self._gallery)
        if not gallery:
            return None
        best = max(float(emb @ g) for _, g in gallery)
        matched = best >= PRESENCE_REID_THRESHOLD
        print(f"[PresenceID] Re-ID check: best similarity {best:.3f} -> {'same person' if matched else 'unfamiliar'}")
        return matched

    def _embed_current_person(self):
        crop = DetectionMemory.get_person_crop()
        if crop is None or crop.shape[0] < 40 or crop.shape[1] < 20:
            return None  # too small to say anything about
        try:
            return self.embed_crop(crop)
        except Exception as e:
            print(f"[PresenceID] Embedding failed: {e}")
            return None

    def embed_crop(self, bgr_crop):
        """BGR person crop -> unit-norm CLIP image embedding (numpy). Public for
        the debug threshold-tuning script."""
        import cv2
        import torch
        from PIL import Image

        if self._model is None:
            import clip

            self._model, self._preprocess = clip.load("ViT-B/32", device="cpu")
            self._model.eval()
            print("[PresenceID] CLIP image encoder loaded (CPU)")
        rgb = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)
        tensor = self._preprocess(Image.fromarray(rgb)).unsqueeze(0)
        with torch.no_grad():
            emb = self._model.encode_image(tensor)[0]
        emb = emb / emb.norm()
        return emb.numpy()


presence_identity = PresenceIdentity()
