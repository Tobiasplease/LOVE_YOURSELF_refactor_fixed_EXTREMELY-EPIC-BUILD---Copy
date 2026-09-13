"""Tedium as pressure, with discharge (Sep 13 2026).

Artist: "I still fundamentally have a problem with persistence producing
hesitation, because the continuous flow of time and interplay with boredom is a
constant shift within the persistence. Tedium is material and repetition is in
and of itself an event. But we are missing something in the architecture to
truly convey this framework."

What was missing, named on Sep 13: repetition was never counted where the
machine could feel it, and tedium had no discharge. The stretch line
(prompts.get_unchanged_line) fixed the first half — the machine now hears how
many times it has named the same thing. This is the second half: the counting
accumulates into a pressure, and the pressure has somewhere to go.

    value in [0, 1]. It RISES per thought call only while nothing is happening
    — the stretch is running — and rises faster when the call repeated the
    phrase it keeps naming. It does not rise at all while someone is here or
    while attention is up: novelty and tedium are exclusive, and a machine
    looking at something new is not bored by definition. It DISCHARGES when
    the pressure found a way out — a drift taken, a reflection, a rare event, a
    drawing wanted, a run of chosen silence — never to nothing, always by a
    fraction of what had built up. Otherwise it decays slowly on its own.

Consumers, each an offer and never a command (the interval is fixed and
stillness stays the model's choice): the drift comes due sooner
(captioner._drift_due), the decision ask names the pressure and puts the exits
in view (prompts.build_decision_ask), and a reflection may come early
(reflection._should_reflect). Nothing here writes text; what the machine does
with the pressure is its own.
"""

from __future__ import annotations

import math
import time

from config import config as _c


def _cfg(name, default):
    return getattr(_c, name, default)


class Tedium:
    def __init__(self, now: float = None) -> None:
        self.value = 0.0
        self.ts = now or time.time()
        self.last_reason = "boot"
        self.last_discharge_ts = 0.0
        self._silence_run = 0

    def _decay_to(self, now: float) -> None:
        floor = float(_cfg("TEDIUM_FLOOR", 0.0))
        tau = float(_cfg("TEDIUM_DECAY_TAU_S", 1800))
        dt = max(0.0, now - self.ts)
        if dt > 0 and tau > 0:
            self.value = floor + (self.value - floor) * math.exp(-dt / tau)
        self.ts = now

    def update(self, now: float = None, *, unchanged: bool = False, repeated: bool = False, engaged: bool = False, silent: bool = False) -> float:
        """One thought call's worth. `unchanged` — the stretch is running;
        `repeated` — this call named the same thing again; `engaged` — someone
        is here or attention is up; `silent` — the machine chose to say
        nothing."""
        now = now or time.time()
        self._decay_to(now)
        if silent:
            self._silence_run += 1
            run = int(_cfg("TEDIUM_SILENCE_RUN", 2))
            if self._silence_run >= run:
                # Staying quiet on purpose IS a way of spending it (the Sep 12
                # night chose silence 597 times and nothing registered).
                self.discharge(float(_cfg("TEDIUM_DISCHARGE_SILENCE", 0.15)), "kept quiet", now)
                return self.value
        else:
            self._silence_run = 0
        if engaged:
            self.last_reason = "engaged"
            return self.value
        if not unchanged:
            self.last_reason = "something happening"
            return self.value
        rise = float(_cfg("TEDIUM_RISE_BASE", 0.02))
        if repeated:
            rise += float(_cfg("TEDIUM_RISE_REPEAT", 0.06))
        self.value = max(0.0, min(1.0, self.value + rise))
        self.last_reason = "said it again" if repeated else "nothing happening"
        return self.value

    def discharge(self, fraction: float, reason: str, now: float = None) -> float:
        """The pressure found a way out. A fraction of what had built up, never
        all of it unless asked: a machine that empties completely every time
        has no history of its own tedium."""
        now = now or time.time()
        self._decay_to(now)
        f = max(0.0, min(1.0, float(fraction)))
        self.value = max(0.0, self.value * (1.0 - f))
        self.last_reason = reason
        self.last_discharge_ts = now
        return self.value

    def state(self) -> dict:
        return {"tedium": round(self.value, 3), "reason": self.last_reason}


def pressing(value: float) -> bool:
    """High enough that the exits belong in view."""
    return float(value or 0.0) >= float(_cfg("TEDIUM_ASK_AT", 0.6))


def unbearable(value: float) -> bool:
    """High enough that a reflection need not wait its full interval."""
    return float(value or 0.0) >= float(_cfg("TEDIUM_REFLECT_AT", 0.8))
