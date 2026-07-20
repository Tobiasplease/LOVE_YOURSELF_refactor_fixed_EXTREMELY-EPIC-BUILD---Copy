"""Joint markov engine for whole-body choreography — channel-agnostic.

Any set of motor channels (gantry x/y, arm servos, fingers, lung…) recorded
as ONE joint state: a chain trained on the recording can only produce
configurations and transitions that were actually performed, so collision
safety and mutual relation between limbs come from the choreography itself.

Channels split into three actuation families:
  ease channels  — servo-like: the generator interpolates them at substep
                   rate for smooth motion (elbow, shoulder, fingers, lung)
  plan channels  — planner-like: sent once per transition with timing; the
                   controller does its own easing (GRBL x/y)
  step channels  — state-like: emitted ONLY when the value changes (pen
                   M3 S). Never interpolated (a half-lowered pen drags) and
                   never streamed (each M3 barriers the GRBL writer queue —
                   20Hz would destroy motion pipelining).

This module is UI-free; motor_panel/session.py owns tracks/transport and
motor_panel/panel.py provides the surfaces. Run  python motor_panel/panel.py
"""

import ast
import json
import os
import random
import threading
import time
from typing import Callable, Dict, List, Optional

DEFAULT_BINS = {
    "elbow": 1.0,
    "shoulder": 1.0,
    "x": 2.0,
    "y": 2.0,
    "finger0": 4.0,
    "finger1": 4.0,
    "finger2": 4.0,
    "finger3": 4.0,
    "lung": 2.0,
    "pen": 2.0,
}
PLAN_CHANNELS = ("x", "y")
STEP_CHANNELS = ("pen",)  # everything else eases
SAMPLE_RATE = 20.0  # Hz
TAKES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "movement_recordings", "arms")


def _key(state: Dict[str, float], channels: List[str], bins: Dict[str, float]) -> str:
    return str(tuple(int(round(state[c] / bins[c])) for c in channels))


def _unkey(key: str, channels: List[str], bins: Dict[str, float]) -> Dict[str, float]:
    vals = ast.literal_eval(key)
    return {c: v * bins[c] for c, v in zip(channels, vals)}


class Recorder:
    """Samples a joint motor state at SAMPLE_RATE while the user performs."""

    def __init__(self, get_state: Callable[[], Dict[str, float]]):
        self.get_state = get_state
        self.samples: List[dict] = []
        self._running = False
        self._thread = None

    def start(self):
        self.samples = []
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        t0 = time.time()
        last = t0
        while self._running:
            now = time.time()
            s = dict(self.get_state())
            s["t"] = now - t0
            s["dt"] = now - last
            last = now
            self.samples.append(s)
            time.sleep(max(0.0, 1.0 / SAMPLE_RATE - (time.time() - now)))

    def stop(self) -> List[dict]:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        return self.samples


def train(samples: List[dict], channels: List[str], bins: Optional[Dict[str, float]] = None) -> dict:
    """First + second-order transition chain over discretized joint states."""
    bins = bins or {c: DEFAULT_BINS.get(c, 1.0) for c in channels}
    keys = [_key(s, channels, bins) for s in samples]
    dts = [max(0.02, s.get("dt", 1.0 / SAMPLE_RATE)) for s in samples]

    first: Dict[str, Dict[str, dict]] = {}
    second: Dict[str, Dict[str, dict]] = {}
    for i in range(1, len(keys)):
        a, b, dt = keys[i - 1], keys[i], dts[i]
        if a == b:
            continue
        slot = first.setdefault(a, {}).setdefault(b, {"count": 0, "dts": []})
        slot["count"] += 1
        slot["dts"].append(dt)
        if i >= 2:
            so = f"{keys[i - 2]}|{a}"
            slot2 = second.setdefault(so, {}).setdefault(b, {"count": 0, "dts": []})
            slot2["count"] += 1
            slot2["dts"].append(dt)

    def finalize(table):
        out = {}
        for src, nxts in table.items():
            total = sum(n["count"] for n in nxts.values())
            out[src] = {dst: {"prob": n["count"] / total, "avg_dt": sum(n["dts"]) / len(n["dts"]), "count": n["count"]} for dst, n in nxts.items()}
        return out

    return {
        "servo_transitions": finalize(first),
        "servo_second_order": finalize(second),
        "second_order_enabled": True,
        "discretization": bins,
        "channels": channels,
        "total_samples": len(samples),
        "unique_states": len(set(keys)),
    }


def _weighted(nxts: Dict[str, dict]) -> str:
    r = random.random()
    acc = 0.0
    for dst, info in nxts.items():
        acc += info["prob"]
        if r <= acc:
            return dst
    return dst  # rounding fallthrough


class Generator:
    """Walks the joint chain: ease channels interpolate at SUBSTEP rate via
    send_ease(dict); plan channels go out once per transition via
    send_plan(dict, dt_seconds)."""

    SUBSTEP = 0.05

    def __init__(
        self,
        chain: dict,
        send_ease: Callable[[Dict[str, float]], None],
        send_plan: Callable[[Dict[str, float], float], None],
        speed: float = 1.0,
        on_state: Optional[Callable[[str], None]] = None,
        send_step: Optional[Callable[[Dict[str, float]], None]] = None,
    ):
        self.chain = chain
        self.channels = chain["channels"]
        self.bins = chain["discretization"]
        self.ease_channels = [c for c in self.channels if c not in PLAN_CHANNELS and c not in STEP_CHANNELS]
        self.plan_channels = [c for c in self.channels if c in PLAN_CHANNELS]
        self.step_channels = [c for c in self.channels if c in STEP_CHANNELS]
        self.send_ease = send_ease
        self.send_plan = send_plan
        self.send_step = send_step
        self.speed = speed
        self.on_state = on_state
        self._running = False
        self._thread = None

    def start(self, initial: Dict[str, float]):
        self._running = True
        self._thread = threading.Thread(target=self._loop, args=(dict(initial),), daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)

    def _loop(self, current: Dict[str, float]):
        first = self.chain["servo_transitions"]
        second = self.chain.get("servo_second_order", {})
        prev_key = None
        cur_key = _key(current, self.channels, self.bins)
        if cur_key not in first:  # ease to a known state to enter the chain
            cur_key = random.choice(list(first.keys()))
            self._transition(current, self._state(cur_key), 2.0)
            current = self._state(cur_key)

        while self._running:
            nxts = second.get(f"{prev_key}|{cur_key}") if prev_key else None
            nxts = nxts or first.get(cur_key)
            if not nxts:  # dead end — re-enter the chain gently
                prev_key, cur_key = None, random.choice(list(first.keys()))
                self._transition(current, self._state(cur_key), 2.0)
                current = self._state(cur_key)
                continue
            nxt = _weighted(nxts)
            dt = max(0.1, nxts[nxt]["avg_dt"] / max(0.1, self.speed))
            target = self._state(nxt)
            self._transition(current, target, dt)
            current = target
            prev_key, cur_key = cur_key, nxt
            if self.on_state:
                self.on_state(cur_key)

    def _state(self, key: str) -> Dict[str, float]:
        return _unkey(key, self.channels, self.bins)

    def _transition(self, frm: Dict[str, float], to: Dict[str, float], dt: float):
        if self.step_channels and self.send_step:
            changed = {c: to[c] for c in self.step_channels if to[c] != frm.get(c)}
            if changed:
                self.send_step(changed)
        if self.plan_channels:
            self.send_plan({c: to[c] for c in self.plan_channels}, dt)
        steps = max(1, int(dt / self.SUBSTEP))
        for i in range(1, steps + 1):
            if not self._running:
                return
            f = i / steps
            if self.ease_channels:
                self.send_ease({c: frm[c] + (to[c] - frm[c]) * f for c in self.ease_channels})
            time.sleep(self.SUBSTEP)


class Player:
    """Replays recorded samples on a channel subset — the overdub/loop deck."""

    PLAN_INTERVAL = 0.2  # GRBL planner wants coarser targets than servos

    def __init__(
        self,
        samples: List[dict],
        channels: List[str],
        send_ease: Callable[[Dict[str, float]], None],
        send_plan: Callable[[Dict[str, float], float], None],
        loop: bool = False,
        speed: float = 1.0,
        on_done: Optional[Callable[[], None]] = None,
        send_step: Optional[Callable[[Dict[str, float]], None]] = None,
    ):
        self.samples = samples
        self.ease_channels = [c for c in channels if c not in PLAN_CHANNELS and c not in STEP_CHANNELS]
        self.plan_channels = [c for c in channels if c in PLAN_CHANNELS]
        self.step_channels = [c for c in channels if c in STEP_CHANNELS]
        self.send_ease = send_ease
        self.send_plan = send_plan
        self.send_step = send_step
        self._last_step: Dict[str, float] = {}
        self.loop = loop
        self.speed = max(0.1, speed)  # 2.0 = twice recorded tempo
        self.on_done = on_done
        self._running = False
        self._thread = None

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._loop_fn, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)

    def _loop_fn(self):
        while self._running:
            t0 = time.time()
            last_plan = 0.0
            for s in self.samples:
                if not self._running:
                    break
                wait = s["t"] / self.speed - (time.time() - t0)
                if wait > 0:
                    time.sleep(wait)
                if self.ease_channels:
                    self.send_ease({c: s[c] for c in self.ease_channels if c in s})
                if self.step_channels and self.send_step:
                    changed = {c: s[c] for c in self.step_channels if c in s and s[c] != self._last_step.get(c)}
                    if changed:
                        self.send_step(changed)
                        self._last_step.update(changed)
                if self.plan_channels and time.time() - last_plan >= self.PLAN_INTERVAL:
                    self.send_plan({c: s[c] for c in self.plan_channels if c in s}, self.PLAN_INTERVAL)
                    last_plan = time.time()
            if not self.loop:
                break
        self._running = False
        if self.on_done:
            self.on_done()


if __name__ == "__main__":
    print(__doc__)
    print("This is a library — run the panel instead:  python motor_panel/panel.py")
