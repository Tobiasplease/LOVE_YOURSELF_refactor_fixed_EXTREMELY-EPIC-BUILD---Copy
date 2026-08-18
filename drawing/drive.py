"""The drawing drive — a continuous energy replacing every timer in the
trigger (docs/drawing-drive-plan.md, Aug 18 2026).

Charges from the emotional system (arousal, continuously) and from a standing
drawing-directed want; discharges fully when a drawing physically completes.
A flat, wantless machine takes days to reach threshold; an agitated one can
refill during the very execution of the previous drawing. Rhythm is a symptom
of inner state, not a schedule.

Monotonic time only — the RTC-skew lesson. Offline hours are not credited:
no experience, no charging.
"""

import json
import os
import time

from config.config import MOOD_SNAPSHOT_FOLDER

THRESHOLD = 1.0
CAP = 1.2
BASE_PER_H = float(os.getenv("DRIVE_BASE_PER_H", 0.03))
AROUSAL_PER_H = float(os.getenv("DRIVE_AROUSAL_PER_H", 0.55))
WANT_PER_H = float(os.getenv("DRIVE_WANT_PER_H", 0.9))
BOOT_LEVEL = float(os.getenv("DRIVE_BOOT_LEVEL", 0.9))

_STATE_PATH = os.path.join(MOOD_SNAPSHOT_FOLDER, "drawing_drive.json")


class DrawingDrive:
    def __init__(self, mono=time.monotonic):
        self._mono = mono
        self._last_mono = mono()
        self.get_arousal = None  # injected by machine.py: mood_engine.mood_vector[1]
        self.level = self._load()

    def _load(self) -> float:
        try:
            with open(_STATE_PATH) as f:
                stored = float(json.load(f).get("level", 0.0))
            # Wake at least at the boot seed (testing era: first drawing comes
            # quickly), or higher if it went to sleep charged.
            return min(CAP, max(stored, BOOT_LEVEL))
        except Exception:
            return BOOT_LEVEL

    def _save(self) -> None:
        try:
            with open(_STATE_PATH, "w") as f:
                json.dump({"level": round(self.level, 4), "saved_at": time.time()}, f)
        except Exception:
            pass

    def tick(self, want_active: bool) -> float:
        """Advance the level by monotonic elapsed time. Called on every
        trigger evaluation; cheap enough for any cadence."""
        now = self._mono()
        dt_h = max(0.0, now - self._last_mono) / 3600.0
        self._last_mono = now
        arousal = 0.0
        try:
            if callable(self.get_arousal):
                arousal = max(0.0, min(1.0, float(self.get_arousal())))
        except Exception:
            pass
        rate = BASE_PER_H + AROUSAL_PER_H * arousal + (WANT_PER_H if want_active else 0.0)
        self.level = min(CAP, self.level + rate * dt_h)
        self._save()
        return self.level

    def full(self) -> bool:
        return self.level >= THRESHOLD

    def spend(self) -> None:
        """A drawing physically completed — the act satisfies fully."""
        self.level = 0.0
        self._last_mono = self._mono()
        self._save()
