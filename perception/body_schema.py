# body_schema.py

import json
import os
import threading
import time

import numpy as np

from config.config import (
    BODY_GALLERY_SIZE,
    BODY_HARVEST_INTERVAL,
    BODY_POSE_TOLERANCE,
    BODY_REGION_OVERLAP,
    BODY_SCHEMA_ENABLED,
    BODY_SELF_STRICT,
    BODY_SELF_THRESHOLD,
    MOOD_SNAPSHOT_FOLDER,
)


class BodySchema:
    """Visual self-recognition, two-factor: PLACE and APPEARANCE. The arms
    enter the frame from a fixed mount, so at a given camera pan/tilt the body
    can only occupy certain image regions (the reach envelope); and they look
    roughly like their reference crops. Either cue alone fails here — the
    studio is full of sculpted limbs that out-score the arm's own next pose in
    CLIP space — but a detection inside the envelope AND loosely matching the
    gallery is self. References are harvested by proprioception: while the CNC
    executes and the gaze is parked on the paper, a faceless "person" at the
    workspace is the machine's own arm. Persistent; the body is stable."""

    def __init__(self, state_path=None):
        self.state_path = state_path or os.path.join(MOOD_SNAPSHOT_FOLDER, "body_schema.json")
        self.lock = threading.Lock()
        self._refs = []  # {ts, emb, pan, tilt, box} box normalized (x1,y1,x2,y2)/(w,h)
        self._last_harvest = 0.0
        self._last_self_seen = 0.0
        self._load()

    def gallery_size(self):
        with self.lock:
            return len(self._refs)

    def maybe_harvest(self):
        """Call freely from the caption cycle; proprioceptive guards inside."""
        if not BODY_SCHEMA_ENABLED:
            return
        now = time.time()
        with self.lock:
            if now - self._last_harvest < BODY_HARVEST_INTERVAL:
                return
        from utils.state_manager import state_manager

        if getattr(state_manager, "current_drawing_phase", None) != "executing":
            return
        try:
            import vision.gaze as gaze

            if not getattr(gaze, "drawing_sequence_active", False):
                return
            pan, tilt = gaze.physics_state.pan, gaze.physics_state.tilt
        except Exception:
            return
        from perception.detection_memory import DetectionMemory

        bbox = DetectionMemory.get_person_bbox()
        crop = DetectionMemory.get_person_crop()
        if bbox is None or crop is None:
            return
        emb = self._embed(crop)
        if emb is None:
            return
        frame_shape = self._current_frame_shape()
        if frame_shape is None:
            return
        norm_box = self._normalize_box(bbox, frame_shape)
        with self.lock:
            self._last_harvest = now
            if any(float(emb @ r["emb"]) > 0.97 and self._overlap(norm_box, r["box"]) > 0.8 for r in self._refs):
                return  # near-duplicate view
            self._refs.append({"ts": now, "emb": emb, "pan": pan, "tilt": tilt, "box": norm_box})
            if len(self._refs) > BODY_GALLERY_SIZE:
                self._refs.pop(0)
            n = len(self._refs)
        print(f"[BodySchema] Harvested own-arm reference during drawing ({n} refs)")
        self._save()

    def is_self(self, bgr_crop, norm_box=None, pan=None, tilt=None):
        """(is_self, similarity). Place+appearance when pose/box are given;
        appearance-only falls back to the strict bar."""
        if not BODY_SCHEMA_ENABLED:
            return None, 0.0
        with self.lock:
            refs = list(self._refs)
        if not refs:
            return None, 0.0
        emb = self._embed(bgr_crop)
        if emb is None:
            return None, 0.0
        in_envelope = False
        if norm_box is not None and pan is not None:
            for r in refs:
                if abs(r["pan"] - pan) < BODY_POSE_TOLERANCE and abs(r["tilt"] - tilt) < BODY_POSE_TOLERANCE:
                    if self._overlap(norm_box, r["box"]) > BODY_REGION_OVERLAP:
                        in_envelope = True
                        break
        best = max(float(emb @ r["emb"]) for r in refs)
        verdict = best >= (BODY_SELF_THRESHOLD if in_envelope else BODY_SELF_STRICT)
        if verdict:
            with self.lock:
                self._last_self_seen = time.time()
        return verdict, best

    def is_self_current_person(self):
        """Is the person the tracker sees the machine's own arm? Cached 5s."""
        now = time.time()
        cached = getattr(self, "_person_verdict", None)
        if cached and now - cached[0] < 5.0:
            return cached[1], cached[2]
        from perception.detection_memory import DetectionMemory

        bbox = DetectionMemory.get_person_bbox()
        crop = DetectionMemory.get_person_crop()
        frame_shape = self._current_frame_shape()
        if bbox is None or crop is None or frame_shape is None:
            return None, 0.0
        try:
            import vision.gaze as gaze

            pan, tilt = gaze.physics_state.pan, gaze.physics_state.tilt
        except Exception:
            pan = tilt = None
        verdict, sim = self.is_self(crop, self._normalize_box(bbox, frame_shape), pan, tilt)
        self._person_verdict = (now, verdict, sim)
        return verdict, sim

    def cached_person_verdict(self, max_age=30.0):
        """Last computed person-is-self verdict WITHOUT computing — safe for
        the main camera loop (an embed would hitch the servo physics)."""
        cached = getattr(self, "_person_verdict", None)
        if cached and time.time() - cached[0] < max_age:
            return cached[1], cached[2]
        return None, 0.0

    def recently_self_visible(self, within=30.0):
        with self.lock:
            return time.time() - self._last_self_seen < within

    def add_reference(self, bgr_crop, norm_box, pan, tilt):
        """Manual enrollment (debug/seeding). Returns True if added."""
        emb = self._embed(bgr_crop)
        if emb is None:
            return False
        with self.lock:
            self._refs.append({"ts": time.time(), "emb": emb, "pan": pan, "tilt": tilt, "box": list(norm_box)})
            if len(self._refs) > BODY_GALLERY_SIZE:
                self._refs.pop(0)
        self._save()
        return True

    @staticmethod
    def _normalize_box(box, frame_shape):
        h, w = frame_shape[0], frame_shape[1]
        return [box[0] / w, box[1] / h, box[2] / w, box[3] / h]

    @staticmethod
    def _overlap(a, b):
        """Intersection over the smaller box, on normalized boxes."""
        ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
        iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
        area_a = max(1e-6, (a[2] - a[0]) * (a[3] - a[1]))
        area_b = max(1e-6, (b[2] - b[0]) * (b[3] - b[1]))
        return (ix * iy) / min(area_a, area_b)

    @staticmethod
    def _current_frame_shape():
        try:
            from utils.state_manager import state_manager

            frame = state_manager.get_shared_frame(max_age=5.0)
            return frame.shape if frame is not None else None
        except Exception:
            return None

    def _embed(self, bgr_crop):
        if bgr_crop is None or bgr_crop.shape[0] < 30 or bgr_crop.shape[1] < 20:
            return None
        try:
            from perception.presence_identity import presence_identity

            return presence_identity.embed_crop(bgr_crop)
        except Exception as e:
            print(f"[BodySchema] Embedding failed: {e}")
            return None

    def _load(self):
        try:
            if os.path.exists(self.state_path):
                with open(self.state_path) as f:
                    data = json.load(f)
                self._refs = [
                    {"ts": r["ts"], "emb": np.array(r["emb"], dtype=np.float32), "pan": r["pan"], "tilt": r["tilt"], "box": r["box"]}
                    for r in data.get("refs", [])
                ]
                print(f"[BodySchema] Loaded {len(self._refs)} own-body references")
        except Exception as e:
            print(f"[BodySchema] Could not load state: {e}")

    def _save(self):
        try:
            os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
            with self.lock:
                data = {
                    "refs": [
                        {"ts": r["ts"], "emb": [round(float(v), 5) for v in r["emb"]], "pan": r["pan"], "tilt": r["tilt"], "box": r["box"]}
                        for r in self._refs
                    ]
                }
            with open(self.state_path, "w") as f:
                json.dump(data, f)
        except Exception as e:
            print(f"[BodySchema] Could not save state: {e}")


body_schema = BodySchema()
