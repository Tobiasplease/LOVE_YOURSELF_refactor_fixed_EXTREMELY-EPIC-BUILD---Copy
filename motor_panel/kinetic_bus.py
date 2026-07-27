"""Runtime kinetic bus — recorded temperament instead of blind wandering.

This is roadmap item 5 of the motor consolidation (docs/motor-panel-handover.md
§8): the practice room's markov generators lifted into the running build,
behind the mood system. The panel and the runtime are now two modes of the
same stack — same devices.py actuator layer, same arms_markov engine, same
session files.

v1 SCOPE: the bus owns the LEFTHAND device only (fingers, elbow, shoulder,
wrist). It replaces organic_left_arm's blind wander and the old hand
interface's generation loop. The gantry stays with the drawing pipeline +
grbl/idle_movements until port arbitration is designed; gaze and lung stay
with their own systems. Sessions may contain gantry/pen takes — the bus
simply ignores tracks whose channels it doesn't own (a "link" between left
arm and right arm degrades to left-arm-only at runtime, by design).

How the body chooses what to be:
  - Sessions are named per state in the panel: "{emotion}_*" for the five
    mood states, "drawing_*" for the machine-is-drawing state.
  - The bus picks a bundle: drawing state overrides emotion (the left hand
    watching the right hand draw is its own temperament); several bundles
    per state rotate on a dwell timer for variety.
  - SEAMLESS transitions: switching bundles never snaps. New generators are
    seeded with the body's live positions and ease into the NEAREST
    demonstrated state of the new chain over KINETIC_CROSSFADE_S (Generator
    enter_nearest). One continuous body, changing its mind.

Modifiers (context leaking into recorded movement):
  - The gaze current (KINETIC_GAZE_*): one direction vector, three
    coordinated effects — a smoothed bounded LEAN every applicable channel
    drifts along (felt immediately, together), TEMPO eagerness (aligned
    transitions quick, opposed reluctant), and the transition CHOICE bias.
    Directional logic throughout: poses only ever lean by a bounded,
    settling amount; the walk never leaves demonstrated states.
  - Startle: flinch -> hold -> slow release. Servos NUDGE partway toward
    a startle pose (the per-channel median of a take assigned under the
    "startle" state — record yourself holding the flinch; built-in deltas
    as fallback), freeze in that tension for HOLD_S, then blend back into
    the running dataset via the normal slow crossfade. Cooldown against
    detector flicker; gantry/pen never flinch.

Wiring (machine.py, behind KINETIC_BUS_ENABLED, default False):
    bus = KineticBus()
    bus.enable()                      # opens /dev/arduino_lefthand
    bus.set_emotion(emotion)          # alongside change_to_emotion(...)
    bus.shutdown()                    # in graceful/emergency cleanup
Context (drawing state, gaze, person presence) is polled by the bus itself
from the runtime singletons; providers are injectable for tests and for the
panel's future "practice room" mode.
"""

import os
import random
import sys
import threading
import time
from typing import Callable, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import (
    KINETIC_CROSSFADE_S,
    KINETIC_GAZE_CHOICE_K,
    KINETIC_GAZE_LEAN,
    KINETIC_GAZE_LEAN_TAU,
    KINETIC_GAZE_STRENGTH,
    KINETIC_GAZE_TEMPO_K,
    KINETIC_ROTATE_S,
    KINETIC_STARTLE_COOLDOWN_S,
    KINETIC_STARTLE_DELTAS,
    KINETIC_STARTLE_ENABLED,
    KINETIC_STARTLE_HOLD_S,
    KINETIC_STARTLE_NUDGE,
)
from motor_panel import arms_markov as engine
from motor_panel.session import SESSIONS_DIR, Session

EMOTIONS = ["energized_engaged", "alert_curious", "calm_observant", "quiet_detached", "withdrawn_distant"]
DRAWING_STATE = "drawing"
STARTLE_STATE = "startle"  # recorded startle gestures — interrupts, never mood-picked
STATES = [DRAWING_STATE, STARTLE_STATE] + EMOTIONS
OWNED_CHANNELS = {"finger0", "finger1", "finger2", "finger3", "elbow", "shoulder", "wrist"}


class TemperamentLibrary:
    """Session files bucketed by the state their name declares.

    "session_energized_engaged_a.json" -> energized_engaged bundle;
    "session_drawing_slow.json" -> the drawing-state bundle. Chains are
    trained lazily and cached per (file, mtime), restricted to the channels
    the bus owns. The projects/ subfolder is the panel's working area —
    deliberately never scanned; only Export ▸ runtime publishes here."""

    def __init__(self, sessions_dir: str = SESSIONS_DIR, owned: set = OWNED_CHANNELS):
        self.sessions_dir = sessions_dir
        self.owned = owned
        self._chain_cache: Dict[str, Tuple[float, Dict[str, dict]]] = {}

    def scan(self) -> Dict[str, List[str]]:
        """state -> [session filenames], rescanned each call (files are
        cheap to list; the user drops new recordings in while running)."""
        buckets: Dict[str, List[str]] = {}
        if not os.path.isdir(self.sessions_dir):
            return buckets
        for fn in sorted(os.listdir(self.sessions_dir)):
            if not (fn.startswith("session_") and fn.endswith(".json")):
                continue
            stem = fn[len("session_") : -len(".json")]
            for state in STATES:
                if stem == state or stem.startswith(state + "_"):
                    buckets.setdefault(state, []).append(fn)
                    break
        return buckets

    def bundle_for(self, emotion: str, drawing: bool) -> Optional[str]:
        """Pick a session filename for the current state. Drawing overrides
        emotion; a missing bucket falls back to any emotion bundle rather
        than stillness (an unfamiliar mood shouldn't paralyze the body).
        Startle datasets are interrupts — never mood-picked, never fallback."""
        buckets = self.scan()
        if drawing and buckets.get(DRAWING_STATE):
            return random.choice(buckets[DRAWING_STATE])
        if buckets.get(emotion):
            return random.choice(buckets[emotion])
        pool = [fn for state, fns in buckets.items() if state not in (DRAWING_STATE, STARTLE_STATE) for fn in fns]
        return random.choice(pool) if pool else None

    def startle_pose(self, filename: str) -> Dict[str, float]:
        """The flinch pose: per-channel MEDIAN of a startle take (record
        yourself HOLDING the flinch and the median IS the pose). Gantry and
        pen are excluded — a flinch lives in the servos."""
        session = Session.load(filename)
        pose: Dict[str, float] = {}
        for t in session.tracks:
            if not t.has_take:
                continue
            for c in t.channels:
                if c in self.owned and c not in ("x", "y", "pen"):
                    vals = sorted(s[c] for s in t.samples if c in s)
                    if vals:
                        pose[c] = float(vals[len(vals) // 2])
        return pose

    def retire(self, filename: str) -> str:
        """Un-publish a runtime bundle: move it back to projects/ (the take
        survives, the bus stops seeing it on its next scan)."""
        src = os.path.join(self.sessions_dir, filename)
        proj = os.path.join(self.sessions_dir, "projects")
        os.makedirs(proj, exist_ok=True)
        dst = os.path.join(proj, filename)
        i = 1
        while os.path.exists(dst):
            dst = os.path.join(proj, filename[: -len(".json")] + f"_{i}.json")
            i += 1
        os.rename(src, dst)
        self._chain_cache.pop(filename, None)
        return dst

    def chains(self, filename: str) -> Dict[str, dict]:
        """Trained chains for a session, owned channels only, mtime-cached."""
        path = os.path.join(self.sessions_dir, filename)
        mtime = os.path.getmtime(path)
        cached = self._chain_cache.get(filename)
        if cached and cached[0] == mtime:
            return cached[1]
        session = Session.load(filename)
        out: Dict[str, dict] = {}
        for key, tracks in session.chain_groups().items():
            tracks = [t for t in tracks if set(t.channels) <= self.owned]
            if not tracks:
                continue
            channels = [c for t in tracks for c in t.channels]
            chain = engine.train(session._joint_samples(tracks), channels)
            if chain["servo_transitions"]:  # constant takes (e.g. an unmoved pen) train zero transitions — nothing to play
                out[key] = chain
        self._chain_cache[filename] = (mtime, out)
        return out


def _default_drawing_provider() -> bool:
    try:
        from utils.state_manager import state_manager

        return bool(state_manager.is_executing_cnc)
    except Exception:
        return False


def _default_gaze_provider() -> Tuple[float, float]:
    """Normalized gaze deflection: (-1 left .. +1 right, -1 down .. +1 up)."""
    try:
        from config.config import PAN_MAX, PAN_MIN, TILT_MAX, TILT_MIN
        from vision.gaze import get_gaze_state

        s = get_gaze_state()
        gx = (s["pan"] - (PAN_MIN + PAN_MAX) / 2) / max(1, (PAN_MAX - PAN_MIN) / 2)
        gy = (s["tilt"] - (TILT_MIN + TILT_MAX) / 2) / max(1, (TILT_MAX - TILT_MIN) / 2)
        return max(-1.0, min(1.0, gx)), max(-1.0, min(1.0, gy))
    except Exception:
        return 0.0, 0.0


def _default_person_provider() -> str:
    try:
        from perception.person_detection_state import get_person_detection_state

        return get_person_detection_state().get_person_state()["person_state"]
    except Exception:
        return "absent"


class KineticBus:
    """Keeps one temperament's generators alive, morphs between temperaments
    as context changes. Two hosting modes, same behavior:

    RUNTIME (machine.py): no callbacks given — the bus builds and owns the
    lefthand device itself.
    PRACTICE ROOM (the panel's temperament lab): send_ease/send_plan/
    send_step/get_state injected, `owned` widened to the full body — the
    SAME bus drives the panel's routing, so what you audition in the lab is
    literally the runtime code path."""

    SUPERVISOR_TICK = 0.2  # startle needs sub-second arrival detection

    def __init__(
        self,
        device=None,
        library: Optional[TemperamentLibrary] = None,
        get_emotion: Optional[Callable[[], str]] = None,
        is_drawing: Callable[[], bool] = _default_drawing_provider,
        get_gaze: Callable[[], Tuple[float, float]] = _default_gaze_provider,
        get_person: Callable[[], str] = _default_person_provider,
        on_log: Callable[[str], None] = lambda m: print(f"[kinetic] {m}"),
        send_ease: Optional[Callable[[Dict[str, float]], None]] = None,
        send_plan: Optional[Callable[[Dict[str, float], float], None]] = None,
        send_step: Optional[Callable[[Dict[str, float]], None]] = None,
        get_state: Optional[Callable[[], Dict[str, float]]] = None,
        owned: Optional[set] = None,
    ):
        self.device = device  # built lazily in enable() so tests can inject
        self.owned = owned or OWNED_CHANNELS
        self.library = library or TemperamentLibrary(owned=self.owned)
        self._emotion = "calm_observant"
        self.get_emotion = get_emotion  # optional pull; set_emotion() is the push path
        self.is_drawing = is_drawing
        self.get_gaze = get_gaze
        self.get_person = get_person
        self.log = on_log
        self._ext_ease = send_ease
        self._ext_plan = send_plan
        self._ext_step = send_step
        self._ext_state = get_state
        self.gaze_strength = KINETIC_GAZE_STRENGTH  # master gaze influence; the runtime tab slider
        self._startle_until = 0.0  # while set, a recorded startle owns the body
        self._gens: List[engine.Generator] = []
        self._offsets: Dict[str, float] = {}
        self._active_bundle: Optional[str] = None
        self._active_state: Optional[str] = None
        self._bundle_since = 0.0
        self._last_person = "absent"
        self._last_startle = 0.0
        self._running = False
        self._thread: Optional[threading.Thread] = None

    # --- context pushes -------------------------------------------------------
    def set_emotion(self, emotion: str):
        self._emotion = emotion

    def status(self) -> dict:
        rotate_in = None
        if self._running and self._active_bundle is not None:
            rotate_in = max(0.0, KINETIC_ROTATE_S - (time.time() - self._bundle_since))
        return {
            "running": self._running,
            "state": self._active_state,
            "bundle": self._active_bundle,
            "chains": len(self._gens),
            "emotion": self._emotion,
            "rotate_in": rotate_in,
        }

    # --- lifecycle ------------------------------------------------------------
    def enable(self):
        if self._ext_ease is None:  # runtime mode: the bus owns the device
            if self.device is None:
                from motor_panel.devices import build_devices

                self.device = build_devices()[1]  # lefthand
            msg = self.device.connect()
            self.log(f"lefthand: {msg}")
        self._running = True
        self._bundle_since = 0.0  # re-enable picks a bundle immediately
        self._thread = threading.Thread(target=self._supervise, daemon=True)
        self._thread.start()

    def shutdown(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)
        self._stop_gens()
        self._active_bundle = self._active_state = None
        self._startle_until = 0.0
        self._offsets = {}
        if self.device is not None:
            try:
                self.device.all_neutral()
                time.sleep(0.3)  # let the writer drain the neutral pose
                self.device.disconnect()
            except Exception:
                pass

    # --- actuation ------------------------------------------------------------
    def _send_ease(self, d: Dict[str, float]):
        d = {c: v + self._offsets.get(c, 0.0) for c, v in d.items()}
        if self._ext_ease is not None:
            self._ext_ease(d)  # panel routing clamps per channel
        else:
            for c, v in d.items():
                self.device.set_channel(c, v)  # device clamps

    def _send_plan(self, d: Dict[str, float], dt: float):
        if self._ext_plan is not None:
            d = {c: v + self._offsets.get(c, 0.0) for c, v in d.items()}  # the lean current reaches the gantry too
            self._ext_plan(d, dt)
        # runtime v1 has no gantry — plan channels never train there (owned)

    def _send_step(self, d: Dict[str, float]):
        if self._ext_step is not None:
            self._ext_step(d)

    def _live_state(self) -> Dict[str, float]:
        if self._ext_state is not None:
            return dict(self._ext_state())
        return {c: float(ch.value) for c, ch in self.device.channels.items()}

    def _stop_gens(self):
        for g in self._gens:
            g.stop()
        self._gens = []

    # --- the supervisor -------------------------------------------------------
    def _supervise(self):
        last_slow = 0.0
        while self._running:
            now = time.time()
            self._watch_person(now)
            self._update_lean()  # the lean current settles/releases at tick rate
            if now - last_slow >= 2.0:  # bundle choice at a calmer rate
                last_slow = now
                self._update_bundle(now)
            time.sleep(self.SUPERVISOR_TICK)

    def _desired_state(self) -> str:
        if self.get_emotion is not None:
            try:
                self._emotion = self.get_emotion() or self._emotion
            except Exception:
                pass
        return DRAWING_STATE if self.is_drawing() else self._emotion

    def _update_bundle(self, now: float):
        if now < self._startle_until:
            return  # a recorded startle owns the body until it has played through
        desired = self._desired_state()
        dwell_over = now - self._bundle_since > KINETIC_ROTATE_S
        if desired == self._active_state and not dwell_over:
            return
        bundle = self.library.bundle_for(self._emotion, desired == DRAWING_STATE)
        if bundle is None:
            if self._active_bundle is not None or int(now) % 60 == 0:
                self.log("no session bundles found — body idle (record some in the panel)")
            self._stop_gens()
            self._active_bundle, self._active_state = None, desired
            self._bundle_since = now
            return
        if bundle == self._active_bundle and desired == self._active_state:
            self._bundle_since = now  # same bundle re-picked at dwell — keep it
            return
        self._switch(bundle, desired)
        self._bundle_since = now

    def _switch(self, bundle: str, state: str):
        """The seamless morph: stop the old temperament wherever it stands,
        seed the new generators with the body's LIVE positions — they ease
        into the nearest demonstrated state over the crossfade."""
        try:
            chains = self.library.chains(bundle)
        except Exception as e:
            self.log(f"bundle {bundle} failed to train: {e}")
            return
        if not chains:
            self.log(f"bundle {bundle} has no takes on owned channels")
            return
        self._start_chains(chains, KINETIC_CROSSFADE_S)
        self._active_bundle, self._active_state = bundle, state
        self.log(f"temperament -> {bundle} ({state}, {len(chains)} chain(s))")

    def _start_chains(self, chains: Dict[str, dict], enter_ease: float):
        self._stop_gens()
        seed = self._live_state()
        for key, chain in chains.items():
            gen = engine.Generator(
                chain,
                send_ease=self._send_ease,
                send_plan=self._send_plan,
                send_step=self._send_step,
                enter_ease=enter_ease,
                bias=self._gaze_bias,
                bias_strength=KINETIC_GAZE_CHOICE_K,
                tempo_strength=KINETIC_GAZE_TEMPO_K,
            )
            gen.start(seed)
            self._gens.append(gen)

    # --- modifiers: the gaze current -------------------------------------------
    # One direction vector, three coordinated effects (config block
    # KINETIC_GAZE_*): a smoothed bounded LEAN on every applicable channel
    # (felt immediately, together), TEMPO eagerness on aligned transitions,
    # and the CHOICE bias — all scaled by gaze_strength.
    def _gaze_vector(self) -> Tuple[float, float]:
        gx, gy = self.get_gaze()
        return gx * self.gaze_strength, gy * self.gaze_strength

    def _gaze_bias(self) -> Dict[str, float]:
        """Direction preference per channel, read LIVE by every generator at
        each transition choice (and for tempo eagerness)."""
        gx, gy = self._gaze_vector()
        if abs(gx) < 1e-3 and abs(gy) < 1e-3:
            return {}
        return {c: (gx if axis == "x" else gy) for c, (axis, _deg) in KINETIC_GAZE_LEAN.items()}

    def _update_lean(self):
        """The felt component: every applicable channel drifts toward the
        gaze — bounded degrees, settling over LEAN_TAU and releasing the
        same way. A current the whole body moves inside, never a snap."""
        gx, gy = self._gaze_vector()
        f = min(1.0, self.SUPERVISOR_TICK / KINETIC_GAZE_LEAN_TAU)
        for c, (axis, deg) in KINETIC_GAZE_LEAN.items():
            target = (gx if axis == "x" else gy) * deg
            cur = self._offsets.get(c, 0.0)
            self._offsets[c] = cur + (target - cur) * f

    def _watch_person(self, now: float):
        state = self.get_person()
        arrived = state == "visible" and self._last_person != "visible"
        self._last_person = state
        if not (arrived and KINETIC_STARTLE_ENABLED):
            return
        if now - self._last_startle < KINETIC_STARTLE_COOLDOWN_S:
            return
        self._last_startle = now
        self._startle()

    def startle(self):
        """Public trigger (the lab's ⚡ button); arrivals call it with the
        cooldown applied in _watch_person."""
        self._startle()

    def _startle_pose(self) -> Tuple[Dict[str, float], str]:
        buckets = self.library.scan()
        if buckets.get(STARTLE_STATE):
            fn = random.choice(buckets[STARTLE_STATE])
            try:
                pose = self.library.startle_pose(fn)
                if pose:
                    return pose, fn
            except Exception as e:
                self.log(f"startle dataset {fn} failed: {e}")
        live = self._live_state()
        return {c: live[c] + d for c, d in KINETIC_STARTLE_DELTAS.items() if c in live}, "built-in deltas"

    def _startle(self):
        """Flinch -> hold -> slow release. Every servo NUDGES partway toward
        the startle pose (quick — never a full transition into it), the
        body freezes in that held tension for HOLD_S, then the supervisor
        re-enters the running dataset with the normal slow crossfade from
        the held pose. A short interruption that blends back naturally."""
        pose, source = self._startle_pose()
        self._startle_until = time.time() + KINETIC_STARTLE_HOLD_S  # claim the body BEFORE touching it (⚡ races the supervisor)
        self._stop_gens()  # mid-motion stop: the body catches its breath where it stands
        self._active_state = STARTLE_STATE  # forces the post-hold re-pick (= the slow blend back)
        live = self._live_state()
        nudge = {c: live[c] + (pose[c] - live[c]) * KINETIC_STARTLE_NUDGE for c in pose if c in live}
        if nudge:  # sent raw, bypassing the gaze lean — a flinch is absolute
            if self._ext_ease is not None:
                self._ext_ease(nudge)
            else:
                for c, v in nudge.items():
                    self.device.set_channel(c, v)
        self.log(f"startle — flinch toward {source}, hold {KINETIC_STARTLE_HOLD_S:.0f}s, slow return")
