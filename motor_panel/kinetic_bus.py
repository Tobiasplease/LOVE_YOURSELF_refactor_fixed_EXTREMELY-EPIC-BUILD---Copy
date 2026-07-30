"""Runtime kinetic bus — recorded temperament instead of blind wandering.

This is roadmap item 5 of the motor consolidation (docs/motor-panel-handover.md
§8): the practice room's markov generators lifted into the running build,
behind the mood system. The panel and the runtime are now two modes of the
same stack — same devices.py actuator layer, same arms_markov engine, same
session files.

SCOPE (v2, July 28): the bus owns the LEFTHAND device (fingers, elbow,
shoulder, wrist) AND — behind KINETIC_GANTRY — the gantry between
drawings, via a headless GantryLink (motor_panel/gantry.py) with
hook-driven port arbitration against the drawing pipeline. Pen stays UP
during generation unless KINETIC_GANTRY_PEN. Gaze and lung stay with
their own systems.

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
  - Startle (reworked July 30): the take assigned under "startle" plays
    RELATIVE — its motion, scaled by NUDGE, unfolds from wherever the
    body is (first sample = zero offset, so entry never snaps). The
    whole body flinches, gantry included; then HOLD_S of held tension
    and the slow crossfade back. Built-in deltas as fallback; cooldown
    against detector flicker; suppressed while homing/paper own the body.
  - Paper check (July 30): paper_clear() plays the take assigned under
    "paper" — both arms out of the camera's view, gantry included — and
    HOLDS until paper_release() (hooks fired by safety/paper_detection
    around the ArUco search). Same continuity rule as homing: the SAME
    dataset resumes.

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
    KINETIC_AWAKENING_MAX_WAIT_S,
    KINETIC_CROSSFADE_S,
    KINETIC_GAZE_CHOICE_K,
    KINETIC_GAZE_LEAN,
    KINETIC_GAZE_LEAN_TAU,
    KINETIC_GAZE_STRENGTH,
    KINETIC_GANTRY_PEN,
    KINETIC_GAZE_TEMPO_K,
    KINETIC_HOMING_MAX_HOLD_S,
    KINETIC_HOMING_TUCK_S,
    KINETIC_PAPER_MAX_HOLD_S,
    KINETIC_PAPER_TUCK_S,
    KINETIC_HOMING_WAIT_CLEAR,
    KINETIC_REACH_ENABLED,
    KINETIC_REACH_MAX_DEG,
    KINETIC_REACH_STRENGTH,
    KINETIC_REACH_TAU,
    KINETIC_ROTATE_S,
    KINETIC_STARTLE_COOLDOWN_S,
    KINETIC_STARTLE_DELTAS,
    KINETIC_STARTLE_ENABLED,
    KINETIC_STARTLE_HOLD_S,
    KINETIC_STARTLE_NUDGE,
    LEFT_ARM_ELBOW_LIMITS,
    LEFT_ARM_SHOULDER_LIMITS,
    LEFT_ARM_WRIST_LIMITS,
)
from motor_panel import arms_markov as engine
from motor_panel.session import SESSIONS_DIR, Session

EMOTIONS = ["energized_engaged", "alert_curious", "calm_observant", "quiet_detached", "withdrawn_distant"]
ARM_CALIB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "arm_calibration.json")
# Per-channel direction sense for the gaze-following currents (reach, lean,
# choice/tempo bias) — NOT the wire-level rev flags and NOT the recordings:
# "left" in the room must mean left for each joint, and only this layer
# knows the room. Tuned in the runtime tab, persisted here, shared by lab
# and runtime.
GAZE_DIRECTIONS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gaze_directions.json")
DIRECTION_CHANNELS = ["shoulder", "elbow", "wrist", "x", "y"]
_ARM_NEUTRALS = {"shoulder": LEFT_ARM_SHOULDER_LIMITS[2], "elbow": LEFT_ARM_ELBOW_LIMITS[2], "wrist": LEFT_ARM_WRIST_LIMITS[2]}


def _bilinear_pose(grid, u: float, v: float) -> Tuple[float, float]:
    """(shoulder, elbow) from the 9-point calibration grid — the same
    bilinear the linkage pad uses. Measured poses in, measured pose out."""
    u, v = max(0.0, min(1.0, u)), max(0.0, min(1.0, v))
    gx_, gy_ = min(u * 2, 1.999), min(v * 2, 1.999)
    ix, iy = int(gx_), int(gy_)
    fx, fy = gx_ - ix, gy_ - iy
    s = (grid[iy][ix][0] * (1 - fx) + grid[iy][ix + 1][0] * fx) * (1 - fy) + (grid[iy + 1][ix][0] * (1 - fx) + grid[iy + 1][ix + 1][0] * fx) * fy
    e = (grid[iy][ix][1] * (1 - fx) + grid[iy][ix + 1][1] * fx) * (1 - fy) + (grid[iy + 1][ix][1] * (1 - fx) + grid[iy + 1][ix + 1][1] * fx) * fy
    return s, e


DRAWING_STATE = "drawing"
STARTLE_STATE = "startle"  # recorded startle pose — interrupts, never mood-picked
HOMING_STATE = "homing"  # tucked-clear pose while the gantry homes — interrupts, never mood-picked
PAPER_STATE = "paper"  # get-clear move while the camera inspects the paper — interrupts, never mood-picked
INTERRUPT_STATES = (STARTLE_STATE, HOMING_STATE, PAPER_STATE)
STATES = [DRAWING_STATE, STARTLE_STATE, HOMING_STATE, PAPER_STATE] + EMOTIONS
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
        pool = [fn for state, fns in buckets.items() if state != DRAWING_STATE and state not in INTERRUPT_STATES for fn in fns]
        return random.choice(pool) if pool else None

    def pose_of(self, filename: str) -> Dict[str, float]:
        """A held pose from a take: per-channel MEDIAN of its samples
        (record yourself HOLDING the pose and the median IS the pose).
        Gantry and pen are excluded — poses live in the servos. Used for
        the startle flinch and the homing tuck."""
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
        gantry=None,
    ):
        self.device = device  # built lazily in enable() so tests can inject
        self.gantry = gantry  # runtime GantryLink: the right arm joins the temperament
        if owned is None and gantry is not None:
            owned = OWNED_CHANNELS | {"x", "y"} | ({"pen"} if KINETIC_GANTRY_PEN else set())
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
        self._hold_until = 0.0  # while set, an interrupt (startle/homing) owns the body
        self._home_players: List[engine.Player] = []  # the homing choreography, playing straight through
        self._home_started_at = 0.0
        self._home_token: object = None  # generation marker — a re-triggered homing invalidates the previous run
        self._startle_token: object = None  # generation marker for the relative flinch playback
        self.arm_calib_path = ARM_CALIB_PATH  # tests inject their own grid
        self._reach_amount = 0.0  # 0..1 presence ramp — the arm leans out while someone is tracked
        self._calib_cache: Tuple[float, Optional[list]] = (0.0, None)
        self.directions_path = GAZE_DIRECTIONS_PATH
        self._dir_flips: Dict[str, bool] = self._load_direction_flips()
        self._resume_bundle: Optional[str] = None  # what was playing before a homing hold
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

    # --- direction sense (gaze-following layer only) ---------------------------
    def _load_direction_flips(self) -> Dict[str, bool]:
        try:
            import json

            with open(self.directions_path) as f:
                d = json.load(f)
            return {c: bool(d.get(c, False)) for c in DIRECTION_CHANNELS}
        except Exception:
            return {c: False for c in DIRECTION_CHANNELS}

    def direction_flips(self) -> Dict[str, bool]:
        return dict(self._dir_flips)

    def set_direction_flip(self, channel: str, flipped: bool):
        """Reverse one channel's response to gaze DIRECTION — reach, lean,
        and the choice/tempo bias all follow. Recordings and wire-level rev
        flags are untouched. Persisted; runtime and lab share the file."""
        self._dir_flips[channel] = bool(flipped)
        try:
            import json

            with open(self.directions_path, "w") as f:
                json.dump(self._dir_flips, f, indent=1)
        except Exception as e:
            self.log(f"⚠ could not persist direction flips: {e}")

    def _sign(self, channel: str) -> float:
        return -1.0 if self._dir_flips.get(channel) else 1.0

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
            "reach": round(self._reach_amount, 2),
            "gantry": bool(self.gantry is not None and self.gantry.alive),
        }

    # --- lifecycle ------------------------------------------------------------
    def enable(self, await_homing: bool = False):
        if self._ext_ease is None:  # runtime mode: the bus owns the device
            if self.device is None:
                from motor_panel.devices import build_devices

                self.device = build_devices()[1]  # lefthand
            msg = self.device.connect()
            self.log(f"lefthand: {msg}")
        self._running = True
        self._bundle_since = 0.0  # re-enable picks a bundle immediately
        if await_homing:
            # THE AWAKENING: hold the body still until the startup homing
            # flow runs. The homing choreography becomes the machine's
            # first gesture and the first temperament blooms as homing
            # completes — all motors together, no left-hand solo at boot.
            self._active_state = HOMING_STATE
            self._home_started_at = time.time()  # a stale sentinel must not release us
            self._hold_until = time.time() + KINETIC_AWAKENING_MAX_WAIT_S
            self.log(f"awakening hold — still until homing (failsafe {KINETIC_AWAKENING_MAX_WAIT_S:.0f}s)")
        self._thread = threading.Thread(target=self._supervise, daemon=True)
        self._thread.start()

    def shutdown(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)
        self._stop_gens()
        for p in self._home_players:
            p.stop()
        self._home_players = []
        if self.gantry is not None:
            self.gantry.release()
        self._active_bundle = self._active_state = None
        self._hold_until = 0.0
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
        # HARD GATE: while the machine draws, the right hand belongs to the
        # GRBL execution — the bus never contests the gantry, regardless of
        # what the active dataset's chains contain.
        if self.is_drawing():
            return
        d = {c: v + self._offsets.get(c, 0.0) for c, v in d.items()}  # the lean current reaches the gantry too
        if self._ext_plan is not None:
            self._ext_plan(d, dt)
        elif self.gantry is not None and self.gantry.alive:
            self.gantry.goto(d.get("x", 0.0), d.get("y", 0.0), dt)

    def _send_step(self, d: Dict[str, float]):
        if self.is_drawing():  # the pen belongs to the drawing too
            return
        if self._ext_step is not None:
            self._ext_step(d)
        elif self.gantry is not None and self.gantry.alive and KINETIC_GANTRY_PEN and "pen" in d:
            self.gantry.pen(int(round(d["pen"])))

    # --- gantry arbitration (drawing pipeline vs temperament) ------------------
    def gantry_acquire(self):
        """Take the gantry: open the port (resets GRBL) and home — the tuck
        choreography fires with it. Called at the awakening and after every
        drawing (the drawing pipeline's port open reset GRBL anyway)."""
        if self.gantry is None:
            return
        self.gantry.on_log = self.log
        if self.gantry.connect_and_home():
            self.log("gantry acquired — right arm in the temperament")

    def gantry_release(self):
        """Drawing needs the port: pen up, close, step aside."""
        if self.gantry is not None:
            self.gantry.release()

    def _live_state(self) -> Dict[str, float]:
        if self._ext_state is not None:
            st = dict(self._ext_state())
        else:
            st = {c: float(ch.value) for c, ch in self.device.channels.items()}
        # the servo device knows nothing of x/y — the gantry's position is
        # commanded, not sensed. Seed generators with the link's truth
        # ((0,0) right after homing) or the whole owning chain dies on it.
        if self.gantry is not None and "x" in self.owned and "x" not in st:
            st["x"], st["y"] = self.gantry.position
        return st

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
            if self._active_state == HOMING_STATE:
                self._check_homing_sentinel()  # subprocess homing completes via the file
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
        if now < self._hold_until:
            return  # an interrupt pose (startle / homing tuck) owns the body
        desired = self._desired_state()
        dwell_over = now - self._bundle_since > KINETIC_ROTATE_S
        if desired == self._active_state and not dwell_over:
            return
        bundle = self.library.bundle_for(self._emotion, desired == DRAWING_STATE)
        if self._active_state in (HOMING_STATE, PAPER_STATE) and self._resume_bundle:
            # leaving the homing hold: continuity beats variety — blend back
            # into the dataset that was playing, not a fresh random pick
            stem = self._resume_bundle[len("session_") : -len(".json")]
            if stem == desired or stem.startswith(desired + "_"):
                bundle = self._resume_bundle
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
        return {c: (gx if axis == "x" else gy) * self._sign(c) for c, (axis, _deg) in KINETIC_GAZE_LEAN.items()}

    def _arm_calib(self) -> Optional[list]:
        """The 9-point arm calibration grid, mtime-cached; None until the
        user captures it in the linkage tab."""
        try:
            mtime = os.path.getmtime(self.arm_calib_path)
        except OSError:
            return None
        if self._calib_cache[0] != mtime:
            try:
                import json

                with open(self.arm_calib_path) as f:
                    self._calib_cache = (mtime, json.load(f)["grid"])
            except Exception:
                self._calib_cache = (mtime, None)
        return self._calib_cache[1]

    def _reach_pose(self, gx: float, gy: float) -> Dict[str, float]:
        """Where the arm reaches for a given look: the MEASURED calibration
        square is the IK table — gaze direction picks the point, bilinear
        over captured poses gives (shoulder, elbow). Until the arm is
        calibrated: proportional joint-space reach inside the config range."""
        grid = self._arm_calib()
        if grid:
            s, e = _bilinear_pose(grid, (gx + 1) / 2, (gy + 1) / 2)
            return {"shoulder": s, "elbow": e}
        s_lo, s_hi, s_n = LEFT_ARM_SHOULDER_LIMITS
        e_lo, e_hi, e_n = LEFT_ARM_ELBOW_LIMITS
        return {"shoulder": s_n + gx * (s_hi - s_lo) / 2 * 0.8, "elbow": e_n + gy * (e_hi - e_lo) / 2 * 0.8}

    def _update_lean(self):
        """The felt currents: ambient gaze lean on every applicable channel,
        and — while someone is being tracked — the REACH: the arm's field
        shifts partway toward the measured pose that points at them.
        Everything bounded, settling over LEAN_TAU, decaying on release. A
        current the whole body moves inside, never a snap."""
        gx, gy = self._gaze_vector()
        raw_gx, raw_gy = self.get_gaze()  # reach uses the LOOK itself, unscaled by influence
        reach_on = KINETIC_REACH_ENABLED and self.get_person() == "visible"
        f_r = min(1.0, self.SUPERVISOR_TICK / KINETIC_REACH_TAU)
        self._reach_amount += ((1.0 if reach_on else 0.0) - self._reach_amount) * f_r
        reach: Dict[str, float] = {}
        if self._reach_amount > 0.02:
            for c, pose_v in self._reach_pose(raw_gx, raw_gy).items():
                delta = (pose_v - _ARM_NEUTRALS.get(c, 90.0)) * KINETIC_REACH_STRENGTH * self._sign(c)
                reach[c] = max(-KINETIC_REACH_MAX_DEG, min(KINETIC_REACH_MAX_DEG, delta))
        f = min(1.0, self.SUPERVISOR_TICK / KINETIC_GAZE_LEAN_TAU)
        for c, (axis, deg) in KINETIC_GAZE_LEAN.items():
            ambient = (gx if axis == "x" else gy) * deg * self._sign(c)
            if c in reach:  # crossfade ambient lean -> reach as presence ramps in
                target = ambient * (1.0 - self._reach_amount) + reach[c] * self._reach_amount
            else:
                target = ambient
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

    def _startle_tracks(self):
        buckets = self.library.scan()
        if not buckets.get(STARTLE_STATE):
            return None
        fn = random.choice(buckets[STARTLE_STATE])
        try:
            session = Session.load(fn)
        except Exception as e:
            self.log(f"startle dataset {fn} failed: {e}")
            return None
        gantry_ok = self.gantry is not None or self._ext_plan is not None
        tracks = [
            t
            for t in session.tracks
            if t.has_take
            and ((set(t.channels) <= self.owned and not (set(t.channels) & {"x", "y", "pen"})) or (gantry_ok and set(t.channels) & {"x", "y"}))
        ]
        return (tracks, fn) if tracks else None

    def _startle(self):
        """The startle take plays RELATIVE: the recording's MOTION, scaled
        by NUDGE, unfolds from wherever the body is right now — the first
        sample is a zero offset, so entry is seamless by construction (no
        snap, no frozen pose). The whole body flinches, gantry included.
        After the motion: HOLD_S of held tension, then the supervisor's
        slow crossfade back. Without a take: the built-in delta nudge."""
        if self._active_state in (HOMING_STATE, PAPER_STATE) and time.time() < self._hold_until:
            self.log("startle suppressed — a safety clearing owns the body")
            return
        self._hold_until = time.time() + KINETIC_STARTLE_HOLD_S  # claim the body BEFORE touching it (⚡ races the supervisor)
        self._stop_gens()  # mid-motion stop: the body catches its breath where it stands
        self._active_state = STARTLE_STATE  # forces the post-hold re-pick (= the slow blend back)
        found = self._startle_tracks()
        live0 = self._live_state()
        if found is None:
            nudge = {c: live0[c] + d * KINETIC_STARTLE_NUDGE for c, d in KINETIC_STARTLE_DELTAS.items() if c in live0}
            if nudge:  # sent raw, bypassing the gaze lean — a flinch is absolute
                self._send_ease_raw(nudge)
            self.log(f"startle — built-in flinch, hold {KINETIC_STARTLE_HOLD_S:.0f}s, slow return")
            return
        tracks, fn = found
        motion = self._motion_end(tracks)
        self._hold_until = time.time() + motion + KINETIC_STARTLE_HOLD_S
        self._startle_token = token = object()
        for t in tracks:
            self._relative_play(t, live0, KINETIC_STARTLE_NUDGE, token, motion)
        self.log(f"startle — {fn}'s flinch from the live pose (×{KINETIC_STARTLE_NUDGE}), {motion:.1f}s + hold {KINETIC_STARTLE_HOLD_S:.0f}s")

    def _relative_play(self, track, live0: Dict[str, float], scale: float, token, cutoff: float):
        """Walk one take on the wall clock sending live0 + (sample - first)
        × scale — the recorded gesture as an offset from the live pose.
        Plan channels decimate to gantry cadence; stops at the motion end
        (the still tail IS the hold) or when superseded."""
        base = {c: float(track.samples[0][c]) for c in track.channels if c in track.samples[0] and c in live0}
        if not base:
            return
        plan = bool(set(track.channels) & {"x", "y"})

        def run():
            t0 = time.time()
            last_plan = -1.0
            for s in track.samples:
                if s["t"] > cutoff + 0.05:
                    return
                while time.time() - t0 < s["t"]:
                    time.sleep(0.01)
                    if self._startle_token is not token or self._active_state != STARTLE_STATE:
                        return
                if self._startle_token is not token or self._active_state != STARTLE_STATE:
                    return
                out = {c: live0[c] + (s[c] - base[c]) * scale for c in base if c in s}
                if not out:
                    continue
                if plan:
                    if s["t"] - last_plan >= 0.2:
                        last_plan = s["t"]
                        self._send_plan_raw(out, 0.2)
                else:
                    self._send_ease_raw(out)

        threading.Thread(target=run, daemon=True).start()

    # --- homing safety ---------------------------------------------------------
    def home_clear(self) -> float:
        """Play the RECORDED homing choreography — straight playback, no
        markov: the take IS the escape path, ending in the tucked-clear
        pose. Entry eases gently into the take's first sample (no
        snapping), the take plays through ONCE, then the body HOLDS its
        final pose until homing completes (home_release(), the
        cross-process sentinel, or the max-hold failsafe), then blends
        back through the normal crossfade. Returns the seconds the caller
        must WAIT before homing (the whole choreography must finish
        first); 0.0 when refused — without a homing dataset we REFUSE to
        guess (a wrong path could cause the collision this prevents)."""
        buckets = self.library.scan()
        if not buckets.get(HOMING_STATE):
            self.log("⚠ no homing dataset — record the get-clear movement and assign it under 'homing'")
            return 0.0
        fn = random.choice(buckets[HOMING_STATE])
        try:
            session = Session.load(fn)
        except Exception as e:
            self.log(f"⚠ homing dataset {fn} failed: {e}")
            return 0.0
        tracks = [t for t in session.tracks if t.has_take and set(t.channels) <= self.owned and not (set(t.channels) & {"x", "y", "pen"})]
        if not tracks:
            self.log(f"⚠ homing dataset {fn} has no usable takes")
            return 0.0
        # RE-TRIGGER = RESTART, never overlap: homing takes a variable time
        # and can fire from several paths in succession — a second call
        # stops the running choreography cleanly and performs it again from
        # wherever the arm is NOW. The token invalidates the previous run's
        # ramp thread and playback timer so two runs can never fight.
        if self._active_state not in (HOMING_STATE, STARTLE_STATE):
            self._resume_bundle = self._active_bundle  # the SAME dataset returns after homing — no random re-pick
        restarting = self._active_state == HOMING_STATE and bool(self._home_players)
        self._home_token = token = object()
        for p in self._home_players:
            p.stop()
        self._home_players = []
        take_len = max(t.samples[-1]["t"] for t in tracks)
        # Wait for the MOTION, not the recording: a homing take usually ends
        # with the pose held while the record pass runs out — that still
        # tail IS the hold, not something homing should wait through (a 20s
        # take with 6s of movement was stalling every homing by 14s).
        motion_end = self._motion_end(tracks)
        total = KINETIC_HOMING_TUCK_S + motion_end + 0.5
        self._home_started_at = time.time()
        self._hold_until = self._home_started_at + total + KINETIC_HOMING_MAX_HOLD_S
        self._stop_gens()
        self._active_state = HOMING_STATE
        first_pose = {}
        for t in tracks:
            for c in t.channels:
                if c in t.samples[0]:
                    first_pose[c] = float(t.samples[0][c])
        self._ease_to_pose(first_pose, KINETIC_HOMING_TUCK_S, still_valid=lambda: self._home_token is token)

        def _begin_playback():
            if self._home_token is not token or self._active_state != HOMING_STATE or time.time() > self._hold_until:
                return
            for t in tracks:
                p = engine.Player(t.samples, t.channels, send_ease=self._send_ease_raw, send_plan=lambda d, dt: None, loop=False)
                self._home_players.append(p)
                p.start()

        timer = threading.Timer(KINETIC_HOMING_TUCK_S, _begin_playback)
        timer.daemon = True
        timer.start()
        head = "homing RE-TRIGGERED — restarting the choreography" if restarting else f"homing choreography {fn}"
        if not KINETIC_HOMING_WAIT_CLEAR:
            # simultaneous mode: the dance is recorded to stay clear of the
            # gantry, so the sweep does not wait for it — they move together
            self.log(f"{head} — motion {motion_end:.1f}s, homing runs alongside")
            return 0.0
        self.log(f"{head} — motion {motion_end:.1f}s of a {take_len:.1f}s take, clearing in {total:.1f}s")
        return total

    @staticmethod
    def _motion_end(tracks) -> float:
        """Where the take's MOVEMENT ends: the last sample that still
        differs from the final pose by more than smoothing noise."""
        end = 0.0
        for t in tracks:
            final = {c: t.samples[-1].get(c) for c in t.channels}
            for s in reversed(t.samples):
                if any(c in s and final[c] is not None and abs(s[c] - final[c]) > 1.5 for c in t.channels):
                    end = max(end, s["t"])
                    break
        return end

    def _send_ease_raw(self, d: Dict[str, float]):
        """Ease without the gaze lean — safety movements are absolute."""
        if self._ext_ease is not None:
            self._ext_ease(d)
        else:
            for c, v in d.items():
                self.device.set_channel(c, v)

    def _ease_to_pose(self, pose: Dict[str, float], seconds: float, still_valid=None):
        """Gentle ramp to a pose in its own thread: interpolates from the
        LIVE pose at substep rate — no snapping, ever. Aborts if the hold
        is released/expired or `still_valid` says a newer run took over."""
        start = self._live_state()
        frm = {c: start[c] for c in pose if c in start}

        def run():
            steps = max(1, int(seconds / 0.05))
            for i in range(1, steps + 1):
                if time.time() > self._hold_until or (still_valid is not None and not still_valid()):
                    return  # released, expired, or superseded — stop pushing
                f = i / steps
                self._send_ease_raw({c: frm[c] + (pose[c] - frm[c]) * f for c in frm})
                time.sleep(0.05)

        threading.Thread(target=run, daemon=True).start()

    def _check_homing_sentinel(self):
        """Cross-process release: the idle subprocess homes the gantry and
        ensure_homed touches the sentinel on completion — a fresh mtime
        means our homing is done."""
        try:
            from utils.hooks import HOMING_SENTINEL

            if os.path.getmtime(HOMING_SENTINEL) > self._home_started_at:
                self.home_release()
        except OSError:
            pass

    def home_release(self):
        """Homing finished: stop the choreography, drop the hold; the
        supervisor blends the body back into the running dataset."""
        if self._active_state == HOMING_STATE:
            for p in self._home_players:
                p.stop()
            self._home_players = []
            self._hold_until = 0.0
            self.log("homing complete — blending back")

    # --- paper check ------------------------------------------------------------
    def _send_plan_raw(self, d: Dict[str, float], dt: float):
        """Plan without the gaze lean — safety and flinch moves are absolute."""
        if self.is_drawing():
            return
        if self._ext_plan is not None:
            self._ext_plan(d, dt)
        elif self.gantry is not None and self.gantry.alive:
            self.gantry.goto(d.get("x", 0.0), d.get("y", 0.0), dt)

    def paper_clear(self) -> float:
        """The camera is about to inspect the paper: play the recorded
        get-clear move — BOTH arms, gantry included — then hold it until
        paper_release() (or the max-hold failsafe). Returns the seconds
        until the view is clear; 0.0 without a dataset (no guessing —
        an invented path could occlude the very thing being checked)."""
        buckets = self.library.scan()
        if not buckets.get(PAPER_STATE):
            self.log("⚠ no paper dataset — record the get-clear move and assign it under 'paper'")
            return 0.0
        fn = random.choice(buckets[PAPER_STATE])
        try:
            session = Session.load(fn)
        except Exception as e:
            self.log(f"⚠ paper dataset {fn} failed: {e}")
            return 0.0
        gantry_ok = self.gantry is not None or self._ext_plan is not None
        servo_tracks = [t for t in session.tracks if t.has_take and set(t.channels) <= self.owned and not (set(t.channels) & {"x", "y", "pen"})]
        plan_tracks = [t for t in session.tracks if t.has_take and set(t.channels) & {"x", "y"}] if gantry_ok else []
        if not servo_tracks and not plan_tracks:
            self.log(f"⚠ paper dataset {fn} has no usable takes")
            return 0.0
        if self._active_state not in INTERRUPT_STATES:
            self._resume_bundle = self._active_bundle  # continuity: the SAME dataset returns after the check
        self._home_token = token = object()  # supersedes any running choreography (re-trigger = restart)
        for p in self._home_players:
            p.stop()
        self._home_players = []
        motion_end = self._motion_end(servo_tracks + plan_tracks)
        total = KINETIC_PAPER_TUCK_S + motion_end + 0.5
        self._home_started_at = time.time()
        self._hold_until = self._home_started_at + total + KINETIC_PAPER_MAX_HOLD_S
        self._stop_gens()
        self._active_state = PAPER_STATE
        first_pose = {}
        for t in servo_tracks:
            for c in t.channels:
                if c in t.samples[0]:
                    first_pose[c] = float(t.samples[0][c])
        self._ease_to_pose(first_pose, KINETIC_PAPER_TUCK_S, still_valid=lambda: self._home_token is token)

        def _begin_playback():
            if self._home_token is not token or self._active_state != PAPER_STATE or time.time() > self._hold_until:
                return
            for t in servo_tracks:
                p = engine.Player(t.samples, t.channels, send_ease=self._send_ease_raw, send_plan=lambda d, dt: None, loop=False)
                self._home_players.append(p)
                p.start()
            for t in plan_tracks:
                p = engine.Player(t.samples, t.channels, send_ease=lambda d: None, send_plan=self._send_plan_raw, loop=False)
                self._home_players.append(p)
                p.start()

        timer = threading.Timer(KINETIC_PAPER_TUCK_S, _begin_playback)
        timer.daemon = True
        timer.start()
        self.log(f"paper check — clearing the view via {fn}, clear in {total:.1f}s")
        return total

    def paper_release(self):
        """Check finished: stop the clearing move, drop the hold; the
        supervisor blends the body back into the running dataset."""
        if self._active_state == PAPER_STATE:
            for p in self._home_players:
                p.stop()
            self._home_players = []
            self._hold_until = 0.0
            self.log("paper check done — blending back")
