# effigy_memory.py

import json
import os
import threading
import time

from config.config import (
    EFFIGY_ENABLED,
    EFFIGY_MATCH_IOU,
    EFFIGY_STILL_S,
    EFFIGY_TTL_S,
    MOOD_SNAPSHOT_FOLDER,
)


class EffigyMemory:
    """Still, faceless, person-shaped things that are NOT people — the
    sweater doll, the legless floor robot the tracker keeps calling a child.

    The discriminator is time: a real person cannot hold a pixel-identical
    pose for EFFIGY_STILL_S. A person-labelled, faceless box that stays put
    that long enrolls as an effigy; from then on a person-hit at that place
    is vetoed (no presence, no arrival, no gaze tracking). A face appearing
    at the place evicts it instantly — face evidence always wins. Effigies
    expire after EFFIGY_TTL_S unseen (furniture gets rearranged); persisted
    so the robot doesn't get to be a person again after every restart."""

    def __init__(self, state_path=None):
        self.state_path = state_path or os.path.join(MOOD_SNAPSHOT_FOLDER, "effigy_memory.json")
        self.lock = threading.Lock()
        self._effigies = []  # {box, enrolled, last_seen}
        self._cand_box = None
        self._cand_since = 0.0
        self._load()

    def observe(self, norm_box, face_present, now=None):
        """One person-detection observation. Returns True if this box is a
        known effigy (caller should veto the person state)."""
        if not EFFIGY_ENABLED or norm_box is None:
            return False
        now = now or time.time()

        with self.lock:
            self._effigies = [e for e in self._effigies if now - e["last_seen"] < EFFIGY_TTL_S]
            hit = next((e for e in self._effigies if self._iou(norm_box, e["box"]) >= EFFIGY_MATCH_IOU), None)
            if hit is not None:
                if face_present:
                    # a face at the effigy's place = a real person stands there
                    self._effigies.remove(hit)
                    print("[Effigy] Face at effigy place — evicted, person restored")
                    self._save_locked()
                    return False
                hit["last_seen"] = now
                return True

            # candidate tracking: person-shaped, faceless, unmoving
            if face_present:
                self._cand_box = None
                return False
            if self._cand_box is not None and self._iou(norm_box, self._cand_box) >= EFFIGY_MATCH_IOU:
                if now - self._cand_since >= EFFIGY_STILL_S:
                    self._effigies.append({"box": list(norm_box), "enrolled": now, "last_seen": now})
                    self._cand_box = None
                    print(
                        f"[Effigy] Enrolled: faceless person-shape held still {EFFIGY_STILL_S / 60:.0f} min — not a person ({len(self._effigies)} known)"
                    )
                    self._save_locked()
                    return True
            else:
                self._cand_box = list(norm_box)
                self._cand_since = now
        return False

    def count(self):
        with self.lock:
            return len(self._effigies)

    @staticmethod
    def _iou(a, b):
        ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
        iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
        inter = ix * iy
        area_a = max(1e-6, (a[2] - a[0]) * (a[3] - a[1]))
        area_b = max(1e-6, (b[2] - b[0]) * (b[3] - b[1]))
        return inter / (area_a + area_b - inter)

    def _load(self):
        try:
            if os.path.exists(self.state_path):
                with open(self.state_path) as f:
                    self._effigies = json.load(f).get("effigies", [])
        except Exception as e:
            print(f"[Effigy] Could not load state: {e}")

    def _save_locked(self):
        try:
            with open(self.state_path, "w") as f:
                json.dump({"effigies": self._effigies}, f, indent=2)
        except Exception as e:
            print(f"[Effigy] Could not save state: {e}")


effigy_memory = EffigyMemory()
