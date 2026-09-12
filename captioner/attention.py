"""Room attention (Sep 12 2026) — one value that says how much of the page the
room gets, earned by what the eyes find rather than by a clock.

Artist: "Someone that is bored and sick of a space doesn't still dart around
the room… lower energy and lower novelty should contribute to how and where it
looks." And on the picture: "The picture in the frame is not the problem. Its
size is" — at ~1024 tokens the room outweighs the mind by mass; at a quarter
of that it is the room seen out of the corner of the eye.

    value ∈ [0, 1]. Novelty pulls it up — the pose-view referee's "changed"
    (+0.6) or a first look this way (+0.3), scene motion, an arrival, eye
    contact or a close face (→ 1.0). Sameness lets it fall: exponential decay
    toward ATTENTION_FLOOR with time constant ATTENTION_DECAY_TAU_S, so with
    nothing new the room is peripheral after ten to fifteen minutes.

Consumers: the caption picture and the clip (utils/image_tokens
.tokens_for_attention), the gaze (vision/gaze.set_attention: zone expiry,
glance lottery, explore weight, wander range) and the LOOK ask cadence
(prompts.build_decision_ask). Nothing here tells the model anything; the
composition of the page is the state.
"""

from __future__ import annotations

import math
import time

from config import config as _c


def _cfg(name, default):
    return getattr(_c, name, default)


class RoomAttention:
    def __init__(self, now: float = None) -> None:
        self.value = 1.0  # a fresh look
        self.ts = now or time.time()
        self.last_reason = "boot"

    def _decay_to(self, now: float) -> None:
        floor = float(_cfg("ATTENTION_FLOOR", 0.15))
        tau = float(_cfg("ATTENTION_DECAY_TAU_S", 600))
        dt = max(0.0, now - self.ts)
        if dt > 0 and tau > 0:
            self.value = floor + (self.value - floor) * math.exp(-dt / tau)
        self.ts = now

    def bump(self, amount: float, reason: str, now: float = None) -> None:
        now = now or time.time()
        self._decay_to(now)
        self.value = min(1.0, self.value + amount)
        self.last_reason = reason

    def snap(self, reason: str, now: float = None) -> None:
        now = now or time.time()
        self._decay_to(now)
        self.value = 1.0
        self.last_reason = reason

    def update(self, scene: dict, now: float = None, view_verdict: str = None, new_view: bool = False, unseen_share: float = 1.0) -> float:
        """One caption cycle: apply what the eyes found, then decay. `new_view` is
        a 20° cell not looked at this session (the captioner keeps the set) — the
        referee's 6° "baselined" fired on nearly every wander and pinned attention
        at 1.0 for the whole first run (Sep 12 11:37)."""
        now = now or time.time()
        if not bool(_cfg("ATTENTION_ENABLED", True)):
            self.value = 1.0
            self.ts = now
            return self.value
        if scene.get("eye_contact") or scene.get("face_close") or (scene.get("salience_hot") and scene.get("presence_believed")):
            self.snap("someone", now)  # a person: full attention, no argument
        elif (scene.get("salience_hot") or scene.get("scene_motion")) and int(scene.get("ego_count", 0) or 0) >= 2:
            # Sep 12 13:10: the head's own glance leaks into the motion signal
            # (each earlier re-pump followed a "Glance (explore)" or a glance
            # check). While the camera itself was moving, motion is not the
            # room's — no credit; a person is caught above regardless.
            self._decay_to(now)
            self.last_reason = "own motion"
        elif scene.get("salience_hot") or scene.get("scene_motion"):
            # Sep 12 12:20: motion or salience with NOBODY believed present was
            # snapping attention from 0.45 to 1.0 in an empty room (twice each
            # in half an hour — the lamp, the flow's own noise). Now it adds,
            # it does not reset: real events keep adding, noise fades.
            self.bump(float(_cfg("ATTENTION_BUMP_MOTION", 0.4)), "motion" if scene.get("scene_motion") else "salience", now)
        elif scene.get("presence_believed") and self.value < float(_cfg("ATTENTION_PRESENT_FLOOR", 0.8)):
            self._decay_to(now)
            self.value = float(_cfg("ATTENTION_PRESENT_FLOOR", 0.8))
            self.last_reason = "presence"
        elif scene.get("view_changed") or view_verdict == "changed":
            self.bump(float(_cfg("ATTENTION_BUMP_CHANGED", 0.6)), "view changed", now)
        elif new_view:
            # Novelty depletes: a first look somewhere is worth less the more of
            # the room has already been seen (unseen_share = cells not yet looked
            # at / reachable cells). Without this the curious gaze refuelled
            # itself — each explore found a new cell, each cell paid +0.3, and
            # attention never settled below 0.5 (Sep 12, first hour).
            self.bump(float(_cfg("ATTENTION_BUMP_NEW_VIEW", 0.3)) * max(0.0, min(1.0, unseen_share)), "new view", now)
        else:
            self._decay_to(now)
            self.last_reason = "same"
        return self.value

    def state(self) -> dict:
        return {"attention": round(self.value, 3), "reason": self.last_reason}


def decide_every(base_n: int, attention: float) -> int:
    """The LOOK ask cadence on the dial: every base_n quiet captions when
    attention is full, rarer as it falls (at the floor about four times rarer).
    A depleted machine is asked less often where to look, and its answer holds."""
    a = min(1.0, max(0.25, float(attention)))
    return max(int(base_n), int(round(base_n / a)))


def curious(attention: float) -> bool:
    return float(attention) >= float(_cfg("ATTENTION_CURIOUS", 0.5))
