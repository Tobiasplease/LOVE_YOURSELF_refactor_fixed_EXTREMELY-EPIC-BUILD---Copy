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
import math
import os
import random
import sys
import threading
import time
from typing import Callable, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from config.config import KINETIC_STATE_BIN_SCALE
except Exception:  # panel/tests can run without the full config
    KINETIC_STATE_BIN_SCALE = 8.0

DEFAULT_BINS = {
    "elbow": 1.0,
    "shoulder": 1.0,
    "wrist": 1.0,
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
    """First + second-order transition chain over discretized joint states.

    IDENTITY IS COARSE, POSE IS FINE (July 31). The state key exists only to
    decide which moments count as "the same place"; the pose the body is
    actually sent is the MEAN of the real samples that landed there. Before
    this split, identity WAS the pose: every channel keyed at 1 degree meant
    two moments had to agree on all seven joints at once to merge, so a
    600-sample take trained ~568 states with exactly one successor each —
    branching factor 1.00. The "chain" was a linked list of the recording,
    and every modifier that reweights a CHOICE (the gaze bias) had nothing
    to choose between. Coarsening alone would have quantized the poses to
    the bin grid (blocky movement); storing the true pose keeps the walk
    smooth while the identity is loose enough to actually fork."""
    bins = bins or {c: DEFAULT_BINS.get(c, 1.0) * KINETIC_STATE_BIN_SCALE for c in channels}
    keys = [_key(s, channels, bins) for s in samples]
    dts = [max(0.02, s.get("dt", 1.0 / SAMPLE_RATE)) for s in samples]

    sums: Dict[str, Dict[str, float]] = {}
    counts: Dict[str, int] = {}
    for k, s in zip(keys, samples):
        acc = sums.setdefault(k, {c: 0.0 for c in channels})
        for c in channels:
            acc[c] += float(s[c])
        counts[k] = counts.get(k, 0) + 1
    poses = {k: {c: acc[c] / counts[k] for c in channels} for k, acc in sums.items()}

    # A transition's dt is the time SPENT in the state before leaving it, not
    # one sample tick. With coarse identity the body dwells many samples in
    # one state; charging the move a single tick would replay the take many
    # times too fast (8x bins: ~14 degrees of travel billed at 0.05s). Summing
    # the dwell keeps total duration — and therefore tempo — identical to the
    # performance, with fewer waypoints along the way.
    first: Dict[str, Dict[str, dict]] = {}
    second: Dict[str, Dict[str, dict]] = {}
    prev_prev, prev_key, dwell = None, keys[0], 0.0
    for i in range(1, len(keys)):
        dwell += dts[i]
        b = keys[i]
        if b == prev_key:
            continue
        slot = first.setdefault(prev_key, {}).setdefault(b, {"count": 0, "dts": []})
        slot["count"] += 1
        slot["dts"].append(dwell)
        if prev_prev is not None:
            so = f"{prev_prev}|{prev_key}"
            slot2 = second.setdefault(so, {}).setdefault(b, {"count": 0, "dts": []})
            slot2["count"] += 1
            slot2["dts"].append(dwell)
        prev_prev, prev_key, dwell = prev_key, b, 0.0

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
        "state_poses": poses,  # key -> the real averaged pose (identity is coarse, pose is fine)
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
    send_plan(dict, dt_seconds).

    Entry is SEAMLESS: from wherever the body is, the generator eases over
    `enter_ease` seconds into the NEAREST demonstrated state (bin-normalized
    distance), not a random one — this is what makes temperament switches
    read as one continuous body changing its mind. freeze() holds the body
    mid-motion for startle/hesitation beats."""

    SUBSTEP = 0.05

    def __init__(
        self,
        chain: dict,
        send_ease: Callable[[Dict[str, float]], None],
        send_plan: Callable[[Dict[str, float], float], None],
        speed: float = 1.0,
        on_state: Optional[Callable[[str], None]] = None,
        send_step: Optional[Callable[[Dict[str, float]], None]] = None,
        enter_ease: float = 2.0,
        bias: Optional[Callable[[], Dict[str, float]]] = None,
        bias_strength: float = 1.5,
        tempo_strength: float = 0.0,
        temperature=1.0,  # float or callable — read live, so mood can move it mid-walk
        repetition=1.0,
        min_p: float = 0.0,
        repetition_window: int = 24,
    ):
        self.chain = chain
        self.channels = chain["channels"]
        self.bins = chain["discretization"]
        self.poses = chain.get("state_poses", {})  # real poses; empty for chains trained before the split
        self.bias = bias  # -> {channel: -1..1 direction preference} (gaze etc.)
        self.bias_strength = bias_strength
        self.tempo_strength = tempo_strength  # aligned transitions eager, opposed reluctant
        self.temperature = temperature
        self.repetition = repetition
        self.min_p = min_p
        self.repetition_window = repetition_window
        self._recent: List[str] = []
        self.ease_channels = [c for c in self.channels if c not in PLAN_CHANNELS and c not in STEP_CHANNELS]
        self.plan_channels = [c for c in self.channels if c in PLAN_CHANNELS]
        self.step_channels = [c for c in self.channels if c in STEP_CHANNELS]
        self.send_ease = send_ease
        self.send_plan = send_plan
        self.send_step = send_step
        self.speed = speed
        self.on_state = on_state
        self.enter_ease = enter_ease
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

    def _nearest_key(self, current: Dict[str, float], first: Dict[str, dict]) -> str:
        """The demonstrated state closest to where the body actually is,
        distance in bin units so channels with coarse bins don't dominate."""
        best, best_d = None, float("inf")
        for key in first:
            state = self._state(key)  # real pose, so "nearest" means nearest in the body, not in the grid
            d = sum(((state[c] - current.get(c, state[c])) / self.bins[c]) ** 2 for c in self.channels)
            if d < best_d:
                best, best_d = key, d
        return best

    def _loop(self, current: Dict[str, float]):
        first = self.chain["servo_transitions"]
        second = self.chain.get("servo_second_order", {})
        if not first:  # a constant take trains ZERO transitions (a==b skipped) — nothing to walk
            return
        if any(c not in current for c in self.channels):
            # the live body can't report every channel (gantry x/y is
            # commanded, not sensed) — adopt the nearest demonstrated
            # values for the gaps instead of dying on the first key
            near = self._state(self._nearest_key(current, first))
            current = {**near, **{c: current[c] for c in self.channels if c in current}}
        cur_key = _key(current, self.channels, self.bins)
        if cur_key not in first:  # ease to the nearest known state to enter the chain
            cur_key = self._nearest_key(current, first)
            self._transition(current, self._state(cur_key), self.enter_ease)
            current = self._state(cur_key)

        # Pipeline: the walk chooses (and DISPATCHES gantry/pen for) one
        # transition beyond the one currently easing, so the GRBL planner
        # always holds a queued segment and blends junctions instead of
        # decelerating to a stop at every markov state — the choppy gait.
        # Servo ease stays on the wall clock; the gantry leads by at most
        # one transition. Pen/step commands ride the dispatch, keeping their
        # stream order relative to the segments they belong to.
        walk_prev, walk_cur = None, cur_key
        queue: List[Tuple[str, float, Dict[str, float]]] = []
        while self._running:
            while len(queue) < 2 and self._running:
                nxts = self._candidates(first, second, walk_prev, walk_cur)
                if not nxts:
                    break
                walk_state = self._state(walk_cur)
                nxt = self._pick(nxts, walk_state)
                dt = max(0.1, nxts[nxt]["avg_dt"] / max(0.1, self.speed))
                target = self._state(nxt)
                if self.bias is not None and self.tempo_strength:
                    b = self.bias()
                    if b:  # eager toward the look, reluctant away — tempo, not position
                        dt = max(0.06, dt * math.exp(-self.tempo_strength * self._alignment(walk_state, target, b)))
                self._dispatch(walk_state, target, dt)
                queue.append((nxt, dt, target))
                walk_prev, walk_cur = walk_cur, nxt
            if not queue:  # dead end with nothing in flight — re-enter gently
                walk_prev = None
                walk_cur = random.choice(list(first.keys()))
                self._transition(current, self._state(walk_cur), 2.0)
                current = self._state(walk_cur)
                continue
            nxt, dt, target = queue.pop(0)
            self._ease(current, target, dt)
            current = target
            if self.on_state:
                self.on_state(nxt)

    def _state(self, key: str) -> Dict[str, float]:
        """The REAL pose recorded at this state, not the bin centre."""
        pose = self.poses.get(key)
        return dict(pose) if pose else _unkey(key, self.channels, self.bins)

    def _alignment(self, frm: Dict[str, float], to: Dict[str, float], b: Dict[str, float]) -> float:
        """How well a transition's direction agrees with the bias vector,
        per channel in bin units, clamped so one big step can't dominate."""
        a = 0.0
        for c, pref in b.items():
            if c in to and c in frm:
                step = (to[c] - frm[c]) / self.bins.get(c, 1.0)
                a += pref * max(-2.0, min(2.0, step)) / 2.0
        return a

    def _pick(self, nxts: Dict[str, dict], cur_state: Dict[str, float]) -> str:
        """Transition choice — the body's sampler, same vocabulary as the
        model's (July 31).

        Base weight is the demonstrated probability, then in order:
          BIAS        exp(strength * direction alignment) — the gaze current;
                      the body tends toward where it looks without being
                      pushed (only demonstrated states stay reachable).
          TEMPERATURE p ** (1/temp): <1 replays the strongest path faithfully,
                      >1 flattens toward the roads less travelled.
          REPETITION  recently visited states are divided down, so the walk
                      stops falling into the same loop — the movement
                      counterpart of the text-side repetition penalty.
          MIN_P       drop candidates far below the strongest one, so
                      wandering never becomes incoherent.
        With one candidate every knob is a no-op, which is exactly what the
        chain used to be everywhere (branching 1.00) — see train()."""
        b = self.bias() if self.bias else None
        weights = {}
        for dst, info in nxts.items():
            w = info["prob"]
            if b and any(abs(v) > 1e-3 for v in b.values()):
                w *= math.exp(self.bias_strength * self._alignment(cur_state, self._state(dst), b))
            temp = self.temperature() if callable(self.temperature) else self.temperature
            if temp and abs(temp - 1.0) > 1e-3:
                w = w ** (1.0 / max(0.05, temp))
            pen = self.repetition() if callable(self.repetition) else self.repetition
            if pen and pen > 1.0 and dst in self._recent:
                w /= pen
            weights[dst] = w
        if len(weights) > 1:
            floor = self.min_p * max(weights.values())
            kept = {d: w for d, w in weights.items() if w >= floor}
            if kept:
                weights = kept
        total = sum(weights.values())
        if total <= 0:
            return _weighted(nxts)
        r = random.random() * total
        acc = 0.0
        for dst, w in weights.items():
            acc += w
            if r <= acc:
                return self._remember(dst)
        return self._remember(dst)  # rounding fallthrough

    def _candidates(self, first: dict, second: dict, walk_prev: Optional[str], walk_cur: str) -> Optional[Dict[str, dict]]:
        """Which transition table to choose from — and whether to break the
        groove. Second-order (where we came from, where we are) is what keeps
        movement coherent instead of jittering, but on hand-performed takes
        almost every second-order context has exactly ONE exit: the walk is
        momentum-locked and no amount of temperature can loosen it. So when
        the context forces a single exit while the state ITSELF offers real
        choices, temperature above 1 decides how often to drop to first order
        and take a different road. Heat is what makes the body break its own
        habit — at temp 1.0 nothing changes."""
        nxts = second.get(f"{walk_prev}|{walk_cur}") if walk_prev else None
        alts = first.get(walk_cur)
        if nxts and len(nxts) == 1 and alts and len(alts) > 1:
            temp = self.temperature() if callable(self.temperature) else self.temperature
            dropout = max(0.0, min(0.9, (float(temp or 1.0) - 1.0) * 0.5))
            if dropout and random.random() < dropout:
                return alts
        return nxts or alts

    def _remember(self, key: str) -> str:
        self._recent.append(key)
        if len(self._recent) > self.repetition_window:
            self._recent.pop(0)
        return key

    def _dispatch(self, frm: Dict[str, float], to: Dict[str, float], dt: float):
        """Plan + step leave at WALK time (ahead of the ease), in stream
        order — a pen change stays interleaved with exactly the segments it
        was demonstrated between."""
        if self.step_channels and self.send_step:
            changed = {c: to[c] for c in self.step_channels if to[c] != frm.get(c)}
            if changed:
                self.send_step(changed)
        if self.plan_channels:
            self.send_plan({c: to[c] for c in self.plan_channels}, dt)

    def _ease(self, frm: Dict[str, float], to: Dict[str, float], dt: float):
        steps = max(1, int(dt / self.SUBSTEP))
        for i in range(1, steps + 1):
            if not self._running:
                return
            if self.ease_channels:
                f = i / steps
                self.send_ease({c: frm[c] + (to[c] - frm[c]) * f for c in self.ease_channels})
            time.sleep(self.SUBSTEP)

    def _transition(self, frm: Dict[str, float], to: Dict[str, float], dt: float):
        """Entry / dead-end re-entry: dispatch and ease together."""
        self._dispatch(frm, to, dt)
        self._ease(frm, to, dt)


class Player:
    """Replays recorded samples on a channel subset — the overdub/loop deck.

    Plan (gantry) targets are streamed LOOKAHEAD seconds ahead of the
    playback clock: a target that arrives exactly on time leaves the GRBL
    planner with zero lookahead, so it decelerates to a full stop at every
    waypoint — the choppy stop-start gait. A small lead keeps 1-2 segments
    queued and junctions blend; tempo stays honest because it's carried by
    the per-segment feeds, not by arrival time. (Cost: pen/step commands
    ride the wall clock, so their position in the ink can lead by up to
    LOOKAHEAD — acceptable for choreography; a pen-marker path queue is the
    proper fix if it ever matters.)"""

    PLAN_INTERVAL = 0.2  # GRBL planner wants coarser targets than servos
    LOOKAHEAD = 0.35  # seconds of gantry commands kept ahead of the clock

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
        self._stopped = False
        self._lifecycle = threading.Lock()  # stop() can race start() (homing released mid-assembly)

    def start(self):
        with self._lifecycle:
            if self._stopped:
                return  # stopped before it ever began — stay stopped, no zombie playback
            self._running = True
            self._thread = threading.Thread(target=self._loop_fn, daemon=True)
            self._thread.start()

    def stop(self):
        with self._lifecycle:
            self._stopped = True
            self._running = False
            t = self._thread
        if t is not None and t.ident is not None:
            t.join(timeout=2)

    def _plan_points(self) -> List[Tuple[float, float, Dict[str, float]]]:
        """Take decimated to PLAN_INTERVAL spacing: (t, dt_from_prev, target)."""
        pts = []
        last_t = None
        for s in self.samples:
            if not all(c in s for c in self.plan_channels):
                continue
            if last_t is None or s["t"] - last_t >= self.PLAN_INTERVAL:
                pts.append((s["t"], self.PLAN_INTERVAL if last_t is None else s["t"] - last_t, {c: s[c] for c in self.plan_channels}))
                last_t = s["t"]
        return pts

    def _loop_fn(self):
        plan_pts = self._plan_points() if self.plan_channels else []
        while self._running:
            t0 = time.time()
            pj = 0
            for s in self.samples:
                if not self._running:
                    break
                wait = s["t"] / self.speed - (time.time() - t0)
                if wait > 0:
                    time.sleep(wait)
                elapsed = time.time() - t0
                while pj < len(plan_pts) and plan_pts[pj][0] / self.speed <= elapsed + self.LOOKAHEAD:
                    t, dt, target = plan_pts[pj]
                    self.send_plan(dict(target), dt / self.speed)
                    pj += 1
                if self.ease_channels:
                    self.send_ease({c: s[c] for c in self.ease_channels if c in s})
                if self.step_channels and self.send_step:
                    changed = {c: s[c] for c in self.step_channels if c in s and s[c] != self._last_step.get(c)}
                    if changed:
                        self.send_step(changed)
                        self._last_step.update(changed)
            if not self.loop:
                break
        self._running = False
        if self.on_done:
            self.on_done()


if __name__ == "__main__":
    print(__doc__)
    print("This is a library — run the panel instead:  python motor_panel/panel.py")
