"""Multitrack session + transport for whole-body choreography.

A session is a looper: fixed loop length, one track per body subsystem
(right arm = grbl x/y, left arm = elbow/shoulder, later hand and lung).
Recording a pass captures the ARMED tracks while every other track with a
take plays back — layers stack against the same loop, like a looper pedal.
Training merges all takes on a common time grid into one joint chain, so
generation preserves the relations you performed between layers.

UI-free; panel.py provides surfaces and channel routing.
"""

import json
import os
import threading
import time
from typing import Callable, Dict, List, Optional

from motor_panel import arms_markov as engine

SESSIONS_DIR = engine.TAKES_DIR  # movement_recordings/arms/


class Track:
    def __init__(self, name: str, channels: List[str], group: str = "A"):
        self.name = name
        self.channels = channels
        self.group = group  # "A"/"B": trained jointly with groupmates; "solo": own chain
        self.samples: Optional[List[dict]] = None  # on the loop timeline
        self.armed = False
        self.mute = False

    @property
    def has_take(self) -> bool:
        return bool(self.samples)


GROUPS = ["A", "B", "solo"]


class Session:
    """One emotional temperament: tracks grouped into joint chains (limbs
    that learned to move in relation) plus solo chains (independent parts),
    all generated simultaneously. Name sessions after emotion states and the
    runtime picks the whole bundle per mood."""

    def __init__(self, name: str = "session", loop_len: float = 30.0):
        self.name = name
        self.loop_len = loop_len
        self.tracks: List[Track] = [
            Track("right arm (grbl)", ["x", "y"], group="A"),
            Track("pen (right hand)", ["pen"], group="solo"),  # group with the arm to learn WHERE it draws
            # wrist is its own LANE (overdub it against an arm take — layers
            # within the limb) but linked by default: training rejoins them
            # into one chain, so generation keeps wrist-in-relation-to-arm
            Track("left arm", ["elbow", "shoulder"], group="A"),
            Track("wrist", ["wrist"], group="A"),
            Track("hand (fingers)", ["finger0", "finger1", "finger2", "finger3"], group="solo"),
            Track("lung", ["lung"], group="solo"),
        ]

    def chain_groups(self) -> Dict[str, List[Track]]:
        """Active tracks bucketed by training destiny."""
        groups: Dict[str, List[Track]] = {}
        for t in self.tracks:
            if t.has_take and not t.mute:
                key = t.name if t.group == "solo" else f"group {t.group}"
                groups.setdefault(key, []).append(t)
        return groups

    def _joint_samples(self, tracks: List[Track]) -> List[dict]:
        """Resample the given takes onto a common SAMPLE_RATE grid."""
        n = int(self.loop_len * engine.SAMPLE_RATE)
        grid = [i / engine.SAMPLE_RATE for i in range(n)]
        joint = [{"t": t, "dt": 1.0 / engine.SAMPLE_RATE} for t in grid]
        for track in tracks:
            idx = 0
            for j, t in enumerate(grid):
                while idx + 1 < len(track.samples) and track.samples[idx + 1]["t"] <= t:
                    idx += 1
                for c in track.channels:
                    joint[j][c] = track.samples[idx].get(c, 0.0)
        return joint

    def train_all(self) -> Dict[str, dict]:
        """One chain per group + one per solo track."""
        out = {}
        for key, tracks in self.chain_groups().items():
            channels = [c for t in tracks for c in t.channels]
            out[key] = engine.train(self._joint_samples(tracks), channels)
        return out

    def save(self, export: bool = False) -> str:
        """Project saves (the default) live in projects/ — the working area,
        free to iterate, never scanned by the runtime. export=True publishes
        into the runtime library root, where the kinetic bus picks bundles
        by their state-name prefix."""
        base = SESSIONS_DIR if export else os.path.join(SESSIONS_DIR, "projects")
        os.makedirs(base, exist_ok=True)
        safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in self.name) or "session"
        path = os.path.join(base, f"session_{safe}.json")
        with open(path, "w") as f:
            json.dump(
                {
                    "format_version": "4.1_session_groups",
                    "name": self.name,
                    "loop_len": self.loop_len,
                    "tracks": [{"name": t.name, "channels": t.channels, "group": t.group, "samples": t.samples} for t in self.tracks],
                },
                f,
            )
        return path

    @classmethod
    def load(cls, filename: str) -> "Session":
        with open(os.path.join(SESSIONS_DIR, filename)) as f:
            d = json.load(f)
        s = cls(d["name"], d["loop_len"])
        by_name = {t.name: t for t in s.tracks}
        for td in d["tracks"]:
            t = by_name.get(td["name"]) or Track(td["name"], td["channels"])
            t.samples = td["samples"]
            t.group = td.get("group", t.group)
            if td["name"] not in by_name:
                s.tracks.append(t)
        return s

    @staticmethod
    def list_saved() -> List[str]:
        """Projects first (the working set, shown as projects/...), then the
        runtime library. Load() takes either form."""
        out = []
        proj = os.path.join(SESSIONS_DIR, "projects")
        if os.path.isdir(proj):
            out += [f"projects/{f}" for f in sorted(os.listdir(proj)) if f.startswith("session_") and f.endswith(".json")]
        if os.path.isdir(SESSIONS_DIR):
            out += sorted(f for f in os.listdir(SESSIONS_DIR) if f.startswith("session_") and f.endswith(".json"))
        return out


LEGACY_HAND_DIR = os.path.join(os.path.dirname(SESSIONS_DIR))  # movement_recordings/


def list_legacy_hand_datasets() -> List[str]:
    if not os.path.isdir(LEGACY_HAND_DIR):
        return []
    return sorted(f for f in os.listdir(LEGACY_HAND_DIR) if f.endswith(".json") and os.path.isfile(os.path.join(LEGACY_HAND_DIR, f)))


def import_legacy_hand_take(filename: str, loop_len: float) -> List[dict]:
    """Convert a hand_control movement_recordings dataset (servo_movements
    with finger_positions) into a hand-track take on the loop timeline,
    tiled to fill the loop. The original file is never modified."""
    with open(os.path.join(LEGACY_HAND_DIR, filename)) as f:
        d = json.load(f)
    src = [
        {"t": m.get("relative_time", 0.0), "f": m.get("finger_positions") or m.get("servo_positions")}
        for m in d.get("servo_movements", [])
        if m.get("finger_positions") or m.get("servo_positions")
    ]
    if len(src) < 4:
        raise ValueError(f"{filename}: no finger movement data")
    duration = max(0.5, src[-1]["t"])
    n = int(loop_len * engine.SAMPLE_RATE)
    out, idx = [], 0
    for i in range(n):
        t = i / engine.SAMPLE_RATE
        ts = t % duration  # tile the source across the loop
        if ts < src[idx]["t"]:
            idx = 0
        while idx + 1 < len(src) and src[idx + 1]["t"] <= ts:
            idx += 1
        fp = src[idx]["f"]
        out.append(
            {
                "t": t,
                "dt": 1.0 / engine.SAMPLE_RATE,
                "finger0": float(fp[0]),
                "finger1": float(fp[1]),
                "finger2": float(fp[2]),
                "finger3": float(fp[3]),
            }
        )
    return out


class Transport:
    """One state machine over the session: idle / recording / playing / generating.

    Routing callbacks (from the panel):
      get_state()            -> dict of every live channel value
      send_ease(dict)        -> servo-like channels
      send_plan(dict, dt)    -> planner-like channels (grbl)
      send_step(dict)        -> state-like channels, on change only (pen)
    """

    def __init__(
        self,
        session: Session,
        get_state: Callable[[], Dict[str, float]],
        send_ease: Callable[[Dict[str, float]], None],
        send_plan: Callable[[Dict[str, float], float], None],
        on_status: Callable[[str], None],
        send_step: Optional[Callable[[Dict[str, float]], None]] = None,
    ):
        self.session = session
        self.get_state = get_state
        self.send_ease = send_ease
        self.send_plan = send_plan
        self.send_step = send_step
        self.on_status = on_status
        self.state = "idle"
        self._players: List[engine.Player] = []
        self._recorder: Optional[engine.Recorder] = None
        self._generators: List[engine.Generator] = []
        self._rec_timer: Optional[threading.Timer] = None
        self._t0: Optional[float] = None  # transport start, for the playhead
        self._speed = 1.0

    def loop_pos(self) -> Optional[float]:
        """Seconds into the loop right now, or None when there's no playhead
        (idle / generating — generation freewheels, it has no timeline)."""
        if self._t0 is None or self.state not in ("recording", "playing"):
            return None
        t = (time.time() - self._t0) * self._speed
        return t % self.session.loop_len if self.state == "playing" else min(t, self.session.loop_len)

    # --- record one loop pass (layered) --------------------------------------
    def record(self):
        self.stop()
        armed = [t for t in self.session.tracks if t.armed]
        if not armed:
            self.on_status("enable ● rec on a track first")
            return
        for t in self.session.tracks:
            if t.has_take and not t.armed and not t.mute:
                p = engine.Player(t.samples, t.channels, self.send_ease, self.send_plan, loop=False, send_step=self.send_step)
                self._players.append(p)
                p.start()
        self._recorder = engine.Recorder(self.get_state)
        self._recorder.start()
        self.state = "recording"
        self._t0 = time.time()
        self._speed = 1.0
        self.on_status(f"● recording {'+'.join(t.name for t in armed)} — {self.session.loop_len:.0f}s pass (Stop keeps the partial)")
        self._rec_timer = threading.Timer(self.session.loop_len, self._finish_record)
        self._rec_timer.daemon = True
        self._rec_timer.start()

    def _commit_take(self, samples: List[dict]):
        """Hand the captured pass to every rec-enabled track. Called from the
        full-pass timer AND from stop() — a performance is never discarded
        silently (the pre-July-26 behavior: Stop mid-pass threw the take
        away, which read as 'recording is a facade')."""
        for t in self.session.tracks:
            if t.armed:
                t.samples = [{**{c: s[c] for c in t.channels if c in s}, "t": s["t"], "dt": s["dt"]} for s in samples]
                t.armed = False

    def _finish_record(self):
        if self.state != "recording":
            return
        samples = self._recorder.stop()
        self._recorder = None
        self._stop_players()
        self._commit_take(samples)
        self.state = "idle"
        self._t0 = None
        self.on_status(f"pass captured ({len(samples)} samples)")

    # --- playback -------------------------------------------------------------
    def play(self, speed: float = 1.0):
        self.stop()
        started = 0
        for t in self.session.tracks:
            if t.has_take and not t.mute:
                p = engine.Player(t.samples, t.channels, self.send_ease, self.send_plan, loop=True, speed=speed, send_step=self.send_step)
                self._players.append(p)
                p.start()
                started += 1
        if started:
            self.state = "playing"
            self._t0 = time.time()
            self._speed = max(0.1, speed)
            self.on_status(f"looping {started} track(s) at {speed:.2g}x")
        else:
            self.on_status("nothing to play — record a take first")

    # --- generation -----------------------------------------------------------
    def generate(self, speed: float = 1.0):
        """One Generator per chain: grouped tracks improvise in relation,
        solo tracks improvise independently — all simultaneously."""
        self.stop()
        chains = self.session.train_all()
        if not chains:
            self.on_status("no takes to train on")
            return
        state = self.get_state()
        for key, chain in chains.items():
            gen = engine.Generator(chain, self.send_ease, self.send_plan, speed=speed, send_step=self.send_step)
            gen.start(state)
            self._generators.append(gen)
        self.state = "generating"
        self.on_status("generating " + ", ".join(f"{k} ({c['unique_states']} states)" for k, c in chains.items()))

    def stop(self):
        if self._rec_timer:
            self._rec_timer.cancel()
            self._rec_timer = None
        msg = "stopped"
        if self._recorder:
            samples = self._recorder.stop()
            self._recorder = None
            if self.state == "recording" and samples:  # Stop ends the pass, keeps the take
                self._commit_take(samples)
                msg = f"stopped — partial take kept ({samples[-1]['t']:.1f}s)"
        for gen in self._generators:
            gen.stop()
        self._generators = []
        self._stop_players()
        self._t0 = None
        if self.state != "idle":
            self.state = "idle"
            self.on_status(msg)

    def _stop_players(self):
        for p in self._players:
            p.stop()
        self._players = []
