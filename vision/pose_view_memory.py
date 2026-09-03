"""Per-pose view memory — the camera-vs-world referee (Sep 3, queue #2).

The July 26 view-replacement check held ONE previous frame: any gaze turn
discarded the comparison, so "the world changed while you were looking away"
was invisible — the machine could only catch a change it happened to be
staring at. This keeps a small grayscale reference per servo pose (6° cells),
so the static background at every habitual gaze direction — desk, walls,
shelf, floor — becomes remembered world, not just the last frame.

Honesty rules, all conservative:
- compare only when the current pose sits within WORLD_POSE_COMPARE_DEG of
  the stored reference (the regime the July 26 check proved: breathing sway
  ~1° measures 0.05-0.1 at 64px, a scene replacement ~0.4);
- references older than WORLD_POSE_REF_MAX_AGE_S re-baseline silently —
  lighting drifts, and a change the code can't attest must not mint an event;
- a confirmed-unchanged comparison rolls the reference forward, so slow
  drift never accumulates into a false change;
- callers must skip saccade/ego-motion frames (blur is not evidence).

Callers: captioner._assess_scene (the only one). Not persisted — references
are perceptual; a restart starts from fresh baselines.
"""

import numpy as np


class PoseViewMemory:
    def __init__(self) -> None:
        self._refs = {}  # (pan_cell, tilt_cell) -> {gray, pan, tilt, ts}

    def observe(self, gray, pan, tilt, now) -> dict:
        """Compare a settled 64px grayscale frame against this pose's reference.

        Returns {"status": ...} where status is one of:
            no_pose      — pan/tilt unknown, nothing learned
            baselined    — first look this way, reference stored
            rebaselined  — reference too old to attest change, replaced
            off_center   — a reference exists but the pose is too far off for
                           an honest comparison; the established one is kept
            unchanged    — compared, world still (score, away_s attached)
            changed      — compared, world different (score, away_s attached)
        """
        if pan is None or tilt is None:
            return {"status": "no_pose"}
        from config.config import (
            WORLD_POSE_CELL_DEG,
            WORLD_POSE_COMPARE_DEG,
            WORLD_POSE_MAX_REFS,
            WORLD_POSE_REF_MAX_AGE_S,
            WORLD_VIEW_DIFF_THRESHOLD,
        )

        key = (int(round(pan / WORLD_POSE_CELL_DEG)), int(round(tilt / WORLD_POSE_CELL_DEG)))
        entry = {"gray": gray, "pan": float(pan), "tilt": float(tilt), "ts": float(now)}
        ref = self._refs.get(key)

        if ref is None:
            if len(self._refs) >= WORLD_POSE_MAX_REFS:
                oldest = min(self._refs, key=lambda k: self._refs[k]["ts"])
                del self._refs[oldest]
            self._refs[key] = entry
            return {"status": "baselined"}

        away_s = now - ref["ts"]
        if away_s > WORLD_POSE_REF_MAX_AGE_S:
            self._refs[key] = entry
            return {"status": "rebaselined", "away_s": away_s}

        if abs(pan - ref["pan"]) + abs(tilt - ref["tilt"]) > WORLD_POSE_COMPARE_DEG:
            return {"status": "off_center"}

        score = float(np.mean(np.abs(gray.astype(np.int16) - ref["gray"].astype(np.int16))) / 255.0)
        self._refs[key] = entry
        status = "changed" if score > WORLD_VIEW_DIFF_THRESHOLD else "unchanged"
        return {"status": status, "score": score, "away_s": away_s}
