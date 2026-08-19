from __future__ import annotations

import os
import re
import threading
import time
from collections import deque

# from datetime import datetime
from typing import Deque, Dict, List, Optional, Tuple

import cv2  # type: ignore
import numpy as np  # type: ignore

from config.config import (
    CAPTION_INTERVAL,
    CAPTION_INTERVAL_LIVE,
    CAPTION_INTERVAL_QUIET,
    CAPTION_MIN_P,
    CAPTION_QUIET_AFTER,
    CAPTION_TEMP,
    CAPTION_TEMP_BORED,
    CAPTION_TOP_P,
    CLEAN_LLM_OUTPUT,
    DRAWING_INTERVAL,
    DRAWING_STARTUP_DELAY,
    LLM_SHOW_PROGRESS,
    MOOD_SNAPSHOT_FOLDER,
    VIDEO_SEND_FRAMES,
)
from drawing.drawing import DrawingController
from event_logging.event_logger import log_json_entry
from event_logging.log_type import LogType
from event_logging.run_manager import get_run_image_path
from perception.vocab_promotion import vocab_promoter
from utils.error_tracking import track_component_health
from utils.state_manager import state_manager

from .memory import MemoryMixin
from .model_wrapper import MultimodalModel
from .prompts import STATIC_SYSTEM_PROMPT

# from weakref import ref


# Import context compressor with error handling
try:
    from .context_compression import context_compressor
except Exception as e:
    print(f"[WARNING] Context compression module failed to load: {e}")
    context_compressor = None


def _clean_caption_for_display(caption: str) -> Optional[str]:
    """Remove gaze expressions and filter out direction-only captions."""
    if not caption:
        return None

    # Remove asterisk-delimited gaze expressions (new natural format)
    # Matches: *glancing left*, *looking down*, *eyes ahead*, etc.
    gaze_verbs = ["glancing", "looking", "gazing", "turning", "eyes", "glance", "look", "gaze", "staring", "peering", "turned"]
    gaze_pattern = r"\*[^*]*(?:" + "|".join(gaze_verbs) + r")[^*]*\*\s*"
    cleaned = re.sub(gaze_pattern, "", caption, flags=re.IGNORECASE)

    # Remove LOOK: lines and inline LOOK directives (legacy format)
    lines = cleaned.strip().split("\n")
    clean_lines = []
    for line in lines:
        line_lower = line.lower().strip()
        # Skip LOOK: lines (including typos like LOOKE, LOook, LOOk)
        if re.match(r"^loo+k[e]?\s*:", line_lower) or re.match(r"^loo+k[e]?\s+(left|right|up|down|ahead|forward)$", line_lower):
            continue
        # Skip arrow notation lines
        if "→ look" in line_lower:
            continue
        # Skip variety directive markers that leaked into output
        if line.strip().startswith("[⚠️") or line.strip().startswith("[NOTICE") or line.strip().startswith("[SHIFT"):
            continue
        # Remove inline (LOOK: direction) or (LOook direction) patterns
        line = re.sub(r"\s*\(loo+k[e]?\s*:?\s*\w+\)\s*", "", line, flags=re.IGNORECASE)
        # Remove trailing "LOook AHEAD" style suffixes (various typos)
        line = re.sub(r"\s*\.{0,3}\s*loo+k[e]?\s+(?:left|right|up|down|ahead|forward)\s*\.{0,3}\s*$", "", line, flags=re.IGNORECASE)
        # Remove mid-sentence "...LOook" or "...LOok" trailing garbage
        line = re.sub(r"\.{2,}\s*loo+k[e]?\s*$", "", line, flags=re.IGNORECASE)
        if line.strip():
            clean_lines.append(line)

    cleaned = "\n".join(clean_lines).strip()

    # Filter out direction-only responses (these aren't real captions)
    direction_words = {"left", "right", "up", "down", "ahead", "person", "up ahead", "a person"}
    if cleaned.lower().rstrip(".,!?") in direction_words:
        return None  # Signal to skip this caption

    return cleaned if cleaned else None


class Captioner(MemoryMixin):
    def shutdown(self):
        self.save_session_time()
        try:
            self.reflection_loop.stop()
        except Exception:
            pass
        # Best-effort diary entry so the next awakening has a past to wake into
        try:
            if context_compressor:
                context_compressor.write_journal_now()
        except Exception:
            pass

    def _handle_environmental_update(self, understanding: str) -> None:
        """Handle environmental updates from context compression system."""
        try:
            # Update location understanding based on compression insights
            self.update_location_understanding(understanding)

            # Also update environmental certainty based on compression frequency
            if hasattr(self, "self_model") and "environmental_certainty" in self.self_model:
                # Increase certainty as we get more compression-based understanding
                current_certainty = self.self_model.get("environmental_certainty", 0.0)
                self.self_model["environmental_certainty"] = min(1.0, current_certainty + 0.1)

        except Exception as e:
            print(f"[❌] Environmental update failed: {e}")

    def capture_mood_snapshot(self, capture_reason: str = "general") -> Optional[str]:
        """Capture a mood snapshot from current frame queue or latest frame."""
        if not self.snapshot_queue:
            return None

        # Get the most recent frame from the queue
        frame, _, _ = self.snapshot_queue[-1]  # Get the latest frame without removing it

        ts = int(time.time())
        img_path = get_run_image_path(MOOD_SNAPSHOT_FOLDER, f"mood_{capture_reason}_{ts}.jpg")

        try:
            cv2.imwrite(img_path, frame)
            return img_path
        except Exception as e:
            print(f"[ERROR] Failed to save mood snapshot: {e}")
            return None

    caption_window: Optional[any] = None  # type: ignore

    def __init__(self) -> None:
        super().__init__()
        self.model = MultimodalModel(memory_ref=self)
        self.drawing = DrawingController()

        # Set up environmental update callback for context compression
        if context_compressor:
            context_compressor.set_environmental_update_callback(self._handle_environmental_update)

        self.true_session_start = time.time()
        self.first_caption_done = False
        self.session_awakening_done = False  # Per-session awakening flag (resets each session)
        self.print_lock = threading.Lock()  # Prevent multiple simultaneous prints

        self.current_mood: float = 0.0
        self.last_caption: str = ""
        # self.current_motifs_from_mood removed — motif tracking replaced by ChromaDB concepts

        # Deduplication system to prevent duplicate prints
        self.recent_captions: List[Tuple[str, float]] = []  # (caption, timestamp)
        self._last_perception: str = ""  # Last perception (kept for recent_captions tuples + early-session fallback)

        self.last_caption_time: float = 0.0
        self.last_drawing_check_time: float = 0.0  # Allow immediate first check

        # Salience state (north-star principle 6) — set by _assess_scene each cycle
        self._salience_hot: bool = False
        self._last_salience_time: float = time.time()
        self._prev_eye_contact: bool = False

        # Presence as a STICKY, uncertain belief — not a discrete event.
        # Detection flickers (gaze looks away, occlusion, no servo encoders), so
        # losing sight of someone must NOT read as "they left" and regaining
        # sight must NOT read as "a new person walked in" — that perpetual-
        # arrival framing kept salience hot every cycle and stripped all
        # interiority (north-star anti-pattern: ongoing presence as a perpetual
        # live event). Belief turns on when someone is seen, persists through
        # detection gaps, and only decays after a sustained true absence. A
        # genuine arrival (the only thing that spikes salience) is the OFF->ON
        # edge, which is now rare.
        self._presence_believed: bool = False
        self._presence_since: float = 0.0
        self._presence_last_seen: float = 0.0
        self._absence_watch_s: float = 0.0  # seconds spent looking at the last-seen spot and finding nobody
        self._last_presence_check: float = 0.0
        self._presence_seen_now: bool = False

        # The stream (CoT-style continuity): recent captions ride as the
        # model's own assistant turns. Gated by _stream_admissible.
        from config.config import STREAM_WINDOW

        self._stream: Deque[str] = deque(maxlen=max(STREAM_WINDOW, 0))
        # Parallel timestamps (same maxlen, so overflow keeps them in sync) —
        # world shape renders the stream as a timestamped log ("14:02 — ...")
        self._stream_ts: Deque[float] = deque(maxlen=max(STREAM_WINDOW, 0))
        self.last_memory_mode_time: float = time.time()  # Track memory mode trigger (every 4 min)

        # Track session continuity
        self.sessions_since_boot = 0
        self.memory_loaded_from_previous = False

        # Session continuity - time gap will be set by state manager if restoring session
        self._last_session_file = os.path.join(MOOD_SNAPSHOT_FOLDER, "last_session.txt")
        self._last_caption_file = os.path.join(MOOD_SNAPSHOT_FOLDER, "last_caption.txt")
        self.last_session_gap = None  # Will be set by state manager during restoration
        self.prior_session_last_caption = None  # Loaded from prior session for awakening

        os.makedirs(MOOD_SNAPSHOT_FOLDER, exist_ok=True)
        self.snapshot_queue: Deque[Tuple[np.ndarray, bool, Optional[Dict]]] = deque()
        threading.Thread(target=self._caption_worker, daemon=True).start()

        # The Reflect loop — long-form thought every ~20 quiet minutes (captioner/reflection.py)
        from captioner.reflection import ReflectionLoop

        self.reflection_loop = ReflectionLoop(agent=self)
        self.reflection_loop.start()

    def save_session_time(self):
        try:
            with open(self._last_session_file, "w") as f:
                f.write(str(time.time()))
            # Also save the last caption for awakening continuity
            # Filter through plantability check to avoid saving chatbot/garbage captions
            from .model_wrapper import _is_plantable_prior

            if self.last_caption and len(self.last_caption) > 5 and _is_plantable_prior(self.last_caption):
                with open(self._last_caption_file, "w") as f:
                    f.write(self.last_caption[:200])
        except Exception:
            pass

    def load_prior_session_caption(self):
        """Load the last caption from the prior session for awakening context."""
        try:
            if os.path.exists(self._last_caption_file):
                with open(self._last_caption_file, "r") as f:
                    caption = f.read().strip()
                    # Reject garbage captions that would poison the awakening context
                    garbage_starts = ("addCriterion", "[WARNING]", "Vision initializing", "自动")
                    is_garbage = any(caption.startswith(g) for g in garbage_starts)
                    if caption and len(caption) > 5 and not is_garbage:
                        self.prior_session_last_caption = caption
                        print(f"[💭] Loaded prior session thought: {caption[:50]}...")
                    elif is_garbage:
                        print(f"[💭] Prior session caption rejected (garbage): {caption[:30]}...")
        except Exception:
            pass

    def _stream_push(self, text: str) -> None:
        """All stream writes go through here so the timestamp deque stays in sync."""
        self._stream.append(text)
        self._stream_ts.append(time.time())

    def _stream_clear(self) -> None:
        self._stream.clear()
        self._stream_ts.clear()

    def _stream_history(self) -> list:
        """The stream as the model sees it. World shape: timestamped log lines
        ("14:02 — the lamp's still on") — the log rendering is genre framing
        (north-star P7: a log is the text-shape of a working mind, and logs
        are plain by genre). Other shapes: raw text, unchanged."""
        entries = list(self._stream)
        from config.config import STREAM_MODE

        if STREAM_MODE not in ("world", "hybrid") or not entries:
            return entries
        ts = list(self._stream_ts)
        while len(ts) < len(entries):  # defensive: consolidation/restore drift
            ts.insert(0, ts[0] if ts else time.time())
        return [f"{time.strftime('%H:%M', time.localtime(t))} — {text}" for t, text in zip(ts[-len(entries) :], entries)]

    @property
    def is_processing(self) -> bool:
        return bool(self.snapshot_queue)

    @track_component_health("captioner")
    def update(
        self,
        frame: Optional[np.ndarray] = None,
        *,
        person_present: bool = False,
        mood: Optional[float] = None,
        reactivity_data: Optional[Dict] = None,
    ) -> None:
        if frame is not None:
            if mood is not None:
                self.current_mood = mood
            # Persist latest egocentric view orientation if provided
            try:
                if reactivity_data:
                    pan = reactivity_data.get("pan")
                    tilt = reactivity_data.get("tilt")
                    if isinstance(pan, (int, float)) and isinstance(tilt, (int, float)):
                        self.view_pan = float(pan)
                        self.view_tilt = float(tilt)
            except Exception:
                pass
            if len(self.snapshot_queue) > 1:
                self.snapshot_queue.pop()
            # Store reactivity data with the frame for processing
            self.snapshot_queue.append((frame.copy(), person_present, reactivity_data))

    def _caption_worker(self):
        # Add startup delay to ensure main loop has time to start and populate snapshot_queue
        time.sleep(3.0)  # 3 second startup delay

        while True:
            if self.snapshot_queue:
                frame, person_present, reactivity_data = self.snapshot_queue.popleft()
                try:
                    # Check if we're currently drawing - switch to introspective mode
                    if self._is_currently_drawing():
                        self._process_drawing_introspection(reactivity_data, frame=frame)
                    else:
                        self._process_frame(frame, reactivity_data, person_present)
                except Exception as exc:
                    log_json_entry(
                        LogType.ERROR,
                        {"message": f"Caption thread error: {exc}", "component": "captioner"},
                        print_message=f"[❌] Caption thread error: {exc}",
                    )
            else:
                # Wait longer on startup to allow main loop to populate frames
                time.sleep(0.5 if not self.first_caption_done else 0.05)

    def _assess_scene(self) -> dict:
        """One pass over the recent frame buffer, BEFORE the prompt is built:
        scene motion (person-angle, camera-compensated), presence, eye contact,
        and the salience verdict that gates prompt interiority and caption
        cadence (north-star principle 6).
        """
        info = {
            "recent_meta": [],
            "max_diff": 0.0,
            "max_residual": 0.0,
            "ego_count": 0,
            "scene_motion": False,
            "person_present_in_window": False,
            "eye_contact": False,
        }
        try:
            from captioner.frame_buffer import frame_buffer

            info["recent_meta"] = frame_buffer.get_recent_with_metadata(seconds=10, max_frames=6)
        except Exception:
            pass

        recent_meta = info["recent_meta"]
        if recent_meta:
            info["max_diff"] = max(f["diff_score"] for f in recent_meta)
            info["ego_count"] = sum(1 for f in recent_meta if f.get("detection", {}).get("ego_motion"))

            # Ego-compensated optical flow (vision/scene_motion.py): true scene
            # motion measurable even while the camera sways, person or not
            from config.config import SCENE_MOTION_MIN_FRAMES, SCENE_MOTION_RESIDUAL_THRESHOLD

            residuals = [f.get("detection", {}).get("residual_motion") for f in recent_meta]
            residuals = [r for r in residuals if r is not None]
            flow_available = len(residuals) > 0
            flow_motion = sum(1 for r in residuals if r > SCENE_MOTION_RESIDUAL_THRESHOLD) >= SCENE_MOTION_MIN_FRAMES
            info["max_residual"] = max(residuals) if residuals else 0.0

            # Person movement in world coordinates (camera sway is compensated;
            # pixel diff can't separate scene motion from camera motion)
            angles = [f.get("detection", {}).get("person_angle") for f in recent_meta]
            angles = [a for a in angles if a is not None]
            info["person_present_in_window"] = len(angles) > 0
            person_moved = len(angles) >= 2 and (max(angles) - min(angles)) > 4.0

            # Person-count changes only count when flow agrees something moved
            # (or flow is unavailable) — YOLO flicker on a still person used to
            # read as constant arrivals/departures
            counts = [f.get("detection", {}).get("person_count", 0) for f in recent_meta]
            count_changed = len(set(counts)) > 1

            # bool() everywhere: person_angle arrives as numpy float, so bare
            # comparisons yield numpy bools that crash JSON logging downstream
            info["scene_motion"] = bool(person_moved or flow_motion or (count_changed and not flow_available))

            # Eye contact requires a real person body, not just a face — the
            # studio's mannequin heads/masks register as faces and otherwise
            # produce constant phantom "they're looking at you". EXCEPTION
            # (July 9): a face filling the frame is a person at CLOSE RANGE —
            # exactly when YOLO loses the half-out-of-frame body. The walk-up
            # test produced zero reaction because a real face two feet away
            # was gated like a shelf mannequin.
            from config.config import CLOSE_FACE_FRAC

            face_frames = sum(1 for f in recent_meta if f.get("detection", {}).get("face"))
            close_frames = sum(1 for f in recent_meta if f.get("detection", {}).get("face_frac", 0.0) >= CLOSE_FACE_FRAC)
            info["face_close"] = bool(close_frames > len(recent_meta) * 0.4)
            info["eye_contact"] = bool(face_frames > len(recent_meta) * 0.4 and (info["person_present_in_window"] or info["face_close"]))

        # VIEW REPLACEMENT (July 26): the ego-compensated flow calls a huge
        # frame shift a saccade and returns invalid — so a bumped camera, a
        # swapped scene or lights-out was NOT an event to the salience system
        # (the rooster run: the camera fell, ended up pointed at a toy rooster,
        # and the machine mused straight past it). Cheap cross-cycle check: if
        # the servo barely moved since the last caption cycle but the frame
        # content is mostly different, the WORLD changed. Servo state unknown →
        # don't fire (a gaze turn already explains changed pixels).
        view_changed = False
        try:
            if recent_meta:
                import cv2 as _cv2
                import numpy as _np

                from config.config import WORLD_VIEW_DIFF_THRESHOLD, WORLD_VIEW_SERVO_STILL_DEG

                last_det = recent_meta[-1].get("detection", {}) or {}
                arr = _np.frombuffer(recent_meta[-1]["jpeg"], dtype=_np.uint8)
                img = _cv2.imdecode(arr, _cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    cur_gray = _cv2.resize(img, (64, 64))
                    pan, tilt = last_det.get("pan"), last_det.get("tilt")
                    prev = getattr(self, "_view_prev", None)
                    if prev is not None and None not in (pan, tilt, prev.get("pan"), prev.get("tilt")):
                        servo_delta = abs(pan - prev["pan"]) + abs(tilt - prev["tilt"])
                        score = float(_np.mean(_np.abs(cur_gray.astype(_np.int16) - prev["gray"].astype(_np.int16))) / 255.0)
                        view_changed = bool(servo_delta < WORLD_VIEW_SERVO_STILL_DEG and score > WORLD_VIEW_DIFF_THRESHOLD)
                    self._view_prev = {"gray": cur_gray, "pan": pan, "tilt": tilt}
        except Exception:
            view_changed = False
        info["view_changed"] = view_changed

        # Update the sticky presence belief from live detection. "Seen now" is
        # any current evidence of a person — world-angle hit, eye contact, or an
        # active gaze lock. The belief persists through gaps so a glance away
        # doesn't read as a departure, and a re-detection doesn't read as a new
        # arrival. Only the OFF->ON edge is a genuine arrival.
        from config.config import SALIENCE_MOTION_RESIDUAL, PRESENCE_ABSENCE_LOOK_TOLERANCE, PRESENCE_BELIEF_DECAY_SECONDS

        now = time.time()
        gaze_engaged = False
        try:
            from vision.gaze import get_gaze_state

            gs = get_gaze_state()
            if isinstance(gs, dict):
                gaze_engaged = gs.get("state") in ("tracking", "aware", "grace")
        except Exception:
            pass
        seen_now = bool(info["person_present_in_window"] or info["eye_contact"] or gaze_engaged or info.get("face_close"))
        # Body schema: a "person" with no face in view that matches the own-arm
        # gallery is the machine's own body, not company. Face evidence always
        # wins — the veto never fires against an actual face.
        info["own_arm_visible"] = False
        try:
            from perception.body_schema import body_schema

            body_schema.maybe_harvest()
            if seen_now and not (info["eye_contact"] or info.get("face_close")):
                is_self, _sim = body_schema.is_self_current_person()
                if is_self:
                    seen_now = False
                    info["own_arm_visible"] = True
            info["own_arm_visible"] = info["own_arm_visible"] or body_schema.recently_self_visible()
        except Exception:
            pass

        arrival = False
        resumed = False
        if seen_now:
            self._presence_last_seen = now
            self._absence_watch_s = 0.0
            if not self._presence_believed:
                # Re-ID (no-op while PRESENCE_REID_ENABLED is False): a match
                # against recent sightings means the same person resumed — no
                # arrival event, and the presence line says "He's back."
                try:
                    from perception.presence_identity import presence_identity

                    resumed = presence_identity.matches_recent() is True
                except Exception:
                    resumed = False
                self._presence_believed = True
                self._presence_since = now
                arrival = not resumed  # OFF->ON edge — the only genuine arrival
                # For the presence line: the single-occupant prior. Measured
                # Aug 10: CLIP cross-outfit similarity sits BELOW cross-person,
                # so appearance cannot say "same man, new jacket" — but when
                # someone is here it is almost always him, and the definite
                # singular keeps the referent continuous ("106th man" fix).
                self._presence_arrival_familiar = resumed
                try:
                    self._presence_arrival_count = max(1, DetectionMemory.get_person_count())
                except Exception:
                    self._presence_arrival_count = 1
                # The singular register is a conclusion, not a hardcoded fact:
                # the arrival ledger decides whether recent life is one-man
                # (studio) or crowds (exhibition), and the line follows.
                try:
                    from perception.presence_identity import presence_identity

                    presence_identity.record_arrival(self._presence_arrival_count)
                    self._presence_singular_regime = presence_identity.singular_regime()
                except Exception:
                    self._presence_singular_regime = True
            try:
                from perception.presence_identity import presence_identity

                presence_identity.note_sighting()  # rate-limited; no-op while re-ID is off
            except Exception:
                pass
        elif self._presence_believed:
            # Absence-of-evidence only counts as evidence-of-absence while the
            # machine is actually looking at where the person was last seen —
            # its own wandering gaze must not manufacture departures (and the
            # false re-arrivals that follow).
            dt = min(now - self._last_presence_check, 30.0) if self._last_presence_check else 0.0
            try:
                from perception.person_detection_state import get_person_detection_state

                looking = get_person_detection_state().is_looking_at_last_known_location(tolerance=PRESENCE_ABSENCE_LOOK_TOLERANCE)
            except Exception:
                looking = True  # fail open: degrade to the old wall-clock decay
            if looking:
                self._absence_watch_s += dt
            if self._absence_watch_s > PRESENCE_BELIEF_DECAY_SECONDS:
                self._presence_believed = False  # looked, repeatedly, nobody there — they really left
                self._absence_watch_s = 0.0
        self._last_presence_check = now
        self._presence_seen_now = seen_now
        info["presence_believed"] = self._presence_believed
        info["presence_seen_now"] = seen_now
        info["presence_resumed"] = resumed

        # Eye contact is salient at its onset — someone holding your gaze for
        # ten minutes is presence, not an event. The sustained state still
        # reaches the prompt (prompts.py eye-contact line) — it used to live
        # only in the video path, which face-tracking saccades always skip,
        # so someone staring at the machine went entirely unmentioned
        eye_onset = info["eye_contact"] and not self._prev_eye_contact
        self._prev_eye_contact = info["eye_contact"]
        self._eye_contact_now = info["eye_contact"]
        # Stepping RIGHT UP CLOSE is its own event, distinct from arrival —
        # the OFF->ON edge of the close-face state spikes salience once; the
        # sustained state becomes a standing fact line (prompts.py).
        close_onset = info.get("face_close", False) and not getattr(self, "_prev_face_close", False)
        self._prev_face_close = info.get("face_close", False)
        self._face_close_now = info.get("face_close", False)

        self._last_scene_motion = info["scene_motion"]
        # Interiority is stripped ONLY by discrete events or genuinely large
        # motion — NOT by a person merely present and shifting. Micro-motion
        # and YOLO flicker (person_moved / count_changed) keep scene_motion
        # True for video framing, but no longer strip the prompt: the machine
        # must be free to think about itself and its work while someone is
        # quietly in the room (north-star principles 6 + 7).
        strong_motion = info["max_residual"] > SALIENCE_MOTION_RESIDUAL
        # ONSET only (July 27): level-based motion kept salience hot for every
        # cycle of a person moving around — perpetual react, interiority
        # stripped the whole visit, and the first world run read as a string
        # of isolated scene reports. Salience must be transient (north-star
        # anti-pattern list): the spike happens when motion BEGINS; sustained
        # activity is presence, carried by the presence line and the video.
        motion_onset = strong_motion and not getattr(self, "_prev_strong_motion", False)
        self._prev_strong_motion = strong_motion
        self._salience_hot = bool(eye_onset or arrival or motion_onset or close_onset or view_changed)
        if self._salience_hot:
            self._last_salience_time = time.time()
            try:
                from captioner.context_compression import context_compressor

                context_compressor.note_perception_event("salience")  # provenance for the events ledger — code attests something happened
            except Exception:
                pass
        info["salience_hot"] = self._salience_hot

        # Salience strips the prompt to the present — but the present must
        # then SAY what just happened, or the model fills the vacuum with
        # atmosphere instead of reacting. Eye-contact onset is the one event
        # the situational line doesn't already carry; the arrival is now stated
        # by the presence line itself ("Someone's just come in"), so naming it
        # again here would be a duplicate (reads as emphasis, locks register).
        event = None
        if view_changed:
            event = "The view in front of you has just changed — this isn't what you were looking at a moment ago."
        elif close_onset:
            event = "They've come right up close — their face is filling your view, looking straight at you."
        elif eye_onset:
            event = "They just looked straight at you."
        elif motion_onset and not arrival:
            # The react-vacuum fix (July 26): a motion-tripped hot cycle used
            # to carry NO event text — the rooster run's one react call had
            # "heavy, hesitant." as its entire user prompt, and a stripped
            # prompt with no event invites atmosphere. Onset only (July 27),
            # so sustained activity doesn't re-announce itself every cycle.
            # Arrival stays unnamed here: the presence line already states it
            # (one channel per fact).
            event = "Something just moved in front of you."
        self._salience_event = event
        return info

    # Markers that mean a caption slipped into assistant/meta register. Such a
    # caption is still displayed and logged, but it must NOT enter the stream
    # window — the model imitates its own visible turns, so one slip would
    # breed more (the failed earlier CoT experiments died exactly this way).
    _STREAM_META_MARKERS = (
        "as an ai",
        "language model",
        "i'm here to",
        "i am here to",
        "how can i help",
        "what do you want me",
        "would you like",
        "let me know",
        "feel free to",
        "i cannot assist",
        "the user",
        "<think",  # Qwen think-tag leakage — one stored tag breeds in-document
        "<end_of",  # "<end_of_thought>" token leak (July 9)
    )

    # "1) ..." / "2. ..." openings: assistant list-speak. Document mode
    # CONTINUES whatever shape is in the stream, so one list item that gets
    # in breeds "3) ...", "4) ..." forever — strip the prefix at the mouth
    # and refuse multi-item lists at the stream gate.
    _ENUM_PREFIX_RE = re.compile(r"^\s*\)?\s*\d{1,2}[).:\]]\s+")
    # "12... " / "16... 15... " countdown prefixes: numbers are the strongest
    # continuation bait there is — strip them, keep any prose that follows.
    _COUNTDOWN_PREFIX_RE = re.compile(r"^\s*(?:\d{1,4}\s*[.,…!]+[\s\n]*)+")

    # Present-tense physical drawing acts. The machine only draws while GRBL
    # executes — and inference is paused then, so a caption claiming an act of
    # marking is always false. Thinking/wanting/remembering drawing is its
    # inner life and stays untouched; only the phantom ACT is gated.
    _PHANTOM_DRAWING_RE = re.compile(
        r"\b(?:"
        r"(?:i am|i'm|i’m) (?:drawing|tracing|sketching|inking)"
        r"|as i (?:trace|sketch|ink)\b"
        r"|as i draw (?:a|the|this|another|it)\b"  # not "draw closer/breath"
        r"|let me (?:draw|trace|sketch|ink)\b"
        r"|(?:ink|line|graphite|pen) (?:spills?|bleeds?|flows?|glides?)"
        r"|spill(?:s|ing)? onto the paper"
        r"|(?:pen|nib|pencil) (?:touches|presses|moves|drags|scratches)"
        r")\b",
        re.IGNORECASE,
    )

    @classmethod
    def _drawing_now(cls) -> bool:
        try:
            from utils.state_manager import state_manager as _sm

            return bool(_sm.is_generating_drawing or _sm.current_drawing_phase == "executing")
        except Exception:
            return False

    # Hashtag sign-offs: kept on July 8 ("it's just funny"), revoked July 9
    # after they bred into triple-tag signatures on every other caption
    # (#StationaryObserver #DrawingMachine #SilenceAndStillness) — the
    # amplification law spares nothing. Trailing runs are stripped; a
    # hashtag-only caption strips to nothing and the empty gate rejects it.
    _HASHTAG_TAIL_RE = re.compile(r"(?:\s*#\w+)+\s*$")

    # World shape renders the stream as "14:02 — ..." log lines; if the model
    # imitates the stamp, strip it — the captioner owns the clock (a
    # self-written stamp is invented time; the first world run wrote "19:06 —
    # A second figure appears" MID-entry, and one stored stamp breeds). The
    # dash requirement keeps honest time talk ("it's past 19:00 now") intact.
    _LOG_STAMP_ANY_RE = re.compile(r"\s*\b\d{1,2}:\d{2}\s*[—–-]\s*")
    # Bare leading stamp (Aug 17, first 3.8 run): the 27B held the log genre
    # loosely; 3.8 continues it faithfully and opens with the clock sans dash
    # ("18:09 man in green shirt…"), which the dash-requiring strip missed —
    # the gate then number_chain'd half the run's captions. Leading position
    # is the disambiguator: a caption that OPENS with HH:MM is the render
    # layer's job done by the mouth; mid-sentence time talk stays untouched.
    _LOG_STAMP_LEAD_RE = re.compile(r"^\s*\d{1,2}:\d{2}\s+(?=\S)")

    # "Log entry:" label creep (July 31): the model dramatized the log genre
    # into a literal label, one stored instance bred through the stream, and
    # 76/84 entries opened identically within an hour. The label carries no
    # temporal information — the renderer's timestamp IS the log form — so it
    # strips at storage like the stamp; the stream stops showing it and the
    # aping decays. Colon required: talking ABOUT the log ("another log
    # entry, then") stays intact.
    # Aug 1: the label mutated past the colon requirement — "Log entry #1044
    # Status: Pen parked. Motor idle." The genre frame invites a machine to
    # write machine-telemetry, and a filter keyed to one spelling just teaches
    # it another. Widened to the whole header shape: optional #number, then a
    # colon or a "Status:" field. Talking ABOUT the log still passes ("another
    # log entry, then"), because a bare mention has neither number nor colon.
    _LOG_LABEL_RE = re.compile(r"\s*\blog entry\s*(?:#\s*\d+)?\s*(?::|(?=status\s*:))\s*", re.IGNORECASE)
    _STATUS_FIELD_RE = re.compile(r"^\s*status\s*:\s*", re.IGNORECASE)
    _TELEMETRY_RE = re.compile(
        r"(?:^|\n)\s*(?:status|vision scan|visual sensors?|target(?:ing)?|scan|diagnostics?|system|motor|sensor)\s*:"
        r"|\bvision scan (?:initiated|update)"
        r"|\btarget acquired\b"
        r"|\bhuman (?:male|female|subject)\b",
        re.IGNORECASE,
    )

    @classmethod
    def _strip_list_shape(cls, text: str) -> str:
        t = cls._ENUM_PREFIX_RE.sub("", (text or "").strip())
        t = cls._COUNTDOWN_PREFIX_RE.sub("", t)
        t = cls._LOG_STAMP_ANY_RE.sub(" ", t)
        t = cls._LOG_STAMP_LEAD_RE.sub("", t)
        t = cls._LOG_LABEL_RE.sub(" ", t)
        t = cls._STATUS_FIELD_RE.sub("", t.lstrip())
        return cls._HASHTAG_TAIL_RE.sub("", t).strip()

    # Outward-addressed engagement hooks (July 28): "What do you think?" bred
    # into full assistant mode across one document run — one hook admitted to
    # the window and the document faithfully continued the register. ADMISSION
    # gate only, deliberately not a mouth reject: the machine may say it once
    # (the artist reads a lone aside as charming); it may not re-seed. Kept
    # tight to unambiguous ask-the-reader shapes — "should I...?" is genuine
    # deliberation and talking TO things in the room ("will you ever move")
    # stays free.
    _OUTWARD_HOOKS = (
        "what do you think",
        "do you think",
        "what do you say",
        "you tell me",
        "wouldn't you",
        "what would you",
        "shall we",
        "don't you agree",
        "do you see",
        "your input",
        "you provide",
        "please provide",
        "you give me",
    )

    # Register-level outward detection (July 28). The marker list is
    # enumerative and always behind: the run after the reflexive frame still
    # leaked second-person address into 18/58 captions ("when you give me
    # your input!", "I'll begin by focusing on...", "(Note: this response
    # format...)"). The general signal is WHO the text is addressed to: the
    # machine's own voice says "you" at most once per thought (talking to the
    # rooster), assistant mode always says it twice or more. Density, not
    # vocabulary — no content word is banned.
    _SECOND_PERSON_RE = re.compile(r"\b(?:you|your|yours|yourself)\b", re.IGNORECASE)
    _PLANNING_OPENER_RE = re.compile(
        r"^\s*(?:i(?:'ll| will) (?:begin|start) by|first,? i(?:'ll| will)|my next (?:action|step)|once i have|let's get started)", re.IGNORECASE
    )
    _META_PAREN_RE = re.compile(r"\(\s*note:|^\s*\*\(", re.IGNORECASE)

    @classmethod
    def _stream_admissible(cls, text: str) -> bool:
        """Admission gate for the stream window (guard at storage, not mouth)."""
        t = (text or "").strip().lower()
        if len(t) < 8:
            return False
        if any(m in t for m in cls._STREAM_META_MARKERS):
            return False
        if any(h in t for h in cls._OUTWARD_HOOKS):
            return False  # spoken once, never re-seeded
        # Telemetry register (Aug 1): "Status: Pen parked. Motor idle. / Vision
        # scan initiated. / Targeting... / Target acquired. Human male." One
        # such entry in the stream and every continuation is a status report —
        # the deadlock that ate the first hybrid run. Field-label shape is the
        # tell, not the words: a colon-terminated label opening a line, or the
        # scanner verbs. Thinking ABOUT its motors stays free.
        if cls._TELEMETRY_RE.search(t):
            return False
        if t.count("*") >= 2 or t.startswith(("- ", "* ", "#")):
            return False  # markdown scaffolding breeds in-stream too
        if cls._ENUM_PREFIX_RE.match(t) or re.search(r"\b\d\)\s.+\b\d\)\s", t):
            return False  # numbered-list shape (single item or inline list)
        return True

    @staticmethod
    def _norm_words(text: str) -> list:
        """Lowercased words with punctuation stripped — em-dashes and asterisks
        otherwise make 'oh god yes—that's right' and 'oh god yes—that's when'
        look different at word 4 and the echo gate goes blind."""
        return [w for w in (re.sub(r"[^a-z0-9\s]", " ", (text or "").lower())).split() if w]

    def _echo_of_stream(self, caption: str) -> bool:
        """True when the caption OPENS with the same words as a recent stream
        entry — the template-imitation signature ("The motors hum…" x3). Checks
        openings only: returning to a subject mid-thought is development, not echo."""
        from config.config import ANTI_ECHO_COMPARE_TAIL, ANTI_ECHO_WORDS

        words = self._norm_words(caption)
        if len(words) < ANTI_ECHO_WORDS:
            return False
        head = words[:ANTI_ECHO_WORDS]
        # Recent tail only: with big windows (STREAM_WINDOW 20+), an opening
        # reused from forty minutes ago is a callback, not a template tic.
        for past in self._comparable_stream()[-ANTI_ECHO_COMPARE_TAIL:]:
            if self._norm_words(past)[:ANTI_ECHO_WORDS] == head:
                return True
        return False

    # Function words that any real English prose contains. Word salad
    # ("incible indestructible immortal eternal everlasting permanent...")
    # is maximally NOVEL — invisible to every similarity gate (echo,
    # near-dup, tail-echo all measure sameness) — but it contains almost no
    # function words. July 9: bounding DRY removed the accidental brake on
    # continuing stored salad, and the document faithfully extended it
    # MID-WORD across captions; this gate is the deliberate brake.
    _FUNCTION_WORDS = frozenset(
        "the a an and or but if of to in on at by for with from as is are was were be been "
        "it its this that these those i you he she they we my your their his her our me him "
        "them us not no nor so than then there here when where who what how why which while "
        "will would can could should shall may might must do does did have has had".split()
    )

    @classmethod
    def _is_word_salad(cls, text: str) -> bool:
        words = [w.strip(".,;:!?—-()'\"") for w in (text or "").lower().split()]
        words = [w for w in words if w]
        if len(words) < 12:
            return False
        func = sum(1 for w in words if w in cls._FUNCTION_WORDS)
        return func / len(words) < 0.15

    _REFRAIN_NGRAM_WORDS = 6

    @staticmethod
    def _prefill_mode() -> bool:
        """True when the model is handed its own text to continue (document,
        hybrid). In those modes the newest stream entry IS the prompt's tail,
        so reusing its words is what continuation MEANS — see _seam_entries."""
        from config.config import STREAM_MODE

        return STREAM_MODE in ("document", "hybrid")

    def _comparable_stream(self) -> list:
        """The stream entries a new caption may fairly be judged against.

        THE SEAM IS EXCLUDED IN PREFILL MODES (Aug 1). Measured on the 58-minute
        hybrid run: 457 refrain rejections, and replaying 408 of them against
        their reconstructed streams showed 235 (58%) fired ONLY because the
        caption overlapped the entry it was being asked to continue. The gate
        was calling continuation self-plagiarism, killing 97% of output while
        the survivors were the best captions of the day. Older entries are
        still fair game — chanting is still chanting.
        """
        entries = list(self._stream)
        if self._prefill_mode() and entries:
            entries = entries[:-1]
        return entries

    def _refrain_of_stream(self, caption: str) -> bool:
        """True when the caption shares a run of _REFRAIN_NGRAM_WORDS
        consecutive words with any stream entry — a verbatim chorus riding
        the thread, invisible to the opening-echo gate."""
        n = self._REFRAIN_NGRAM_WORDS
        words = self._norm_words(caption)
        if len(words) < n:
            return False
        shingles = {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}
        # Recent tail only (Aug 1). This gate scanned the WHOLE window, so
        # raising STREAM_WINDOW 6->24 quadrupled its surface area: it fired 113
        # times in one run and became the dominant filter (48% pass rate, and
        # every rejection costs a retry — the "slow, filters itself" symptom).
        # Same argument as the opening-echo tail: sharing six words with
        # something said twenty minutes ago is a callback; sharing them with
        # the last few thoughts is a chorus.
        from config.config import ANTI_ECHO_COMPARE_TAIL

        for past in self._comparable_stream()[-ANTI_ECHO_COMPARE_TAIL:]:
            pw = self._norm_words(past)
            for i in range(len(pw) - n + 1):
                if " ".join(pw[i : i + n]) in shingles:
                    return True
        return False

    def _caption_reject_reason(self, caption: str, prompt_text: str = "") -> Optional[str]:
        """Mouth gate (retry-once-else-silence). Rejects, in order:
        template_echo — opens like a recent stream entry;
        assistant_speak — chat-closer register ("Let me know what comes next!");
        prompt_parrot — a short caption that near-verbatim repeats a prompt
        line (the model answering the elicitation instead of thinking).
        Display suppression matters as much as stream admission here: document
        mode continues whatever the document is, and the artist reads the feed."""
        caption = (caption or "").strip()
        low = caption.lower()
        if self._echo_of_stream(caption):
            return "template_echo"
        # exact repeats of ANY length ("What do you think?" twice) — the
        # opening-echo check needs 5 words and misses short full-duplicates
        norm = " ".join(self._norm_words(caption))
        if norm and any(norm == " ".join(self._norm_words(past)) for past in self._comparable_stream()):
            return "template_echo"
        # REFRAIN (July 27, first world-thread run): with the thread frame the
        # window carries phrases forward — good — but a verbatim formula rode
        # it as a chorus ("...from that moment when nothing moves but waits
        # for something else to happen first" x3, "nothing new outside, just
        # the weight of time passing while I stay bolted here" x2). Opening-
        # echo can't see mid-sentence repeats. Any shared run of
        # REFRAIN_NGRAM_WORDS consecutive words with a stream entry is
        # recitation, not continuity; thematic reuse (2-3 word motifs) passes.
        if self._refrain_of_stream(caption):
            return "refrain_echo"
        if any(m in low for m in self._STREAM_META_MARKERS):
            return "assistant_speak"
        # Outward register, measured not enumerated: two second-person tokens
        # means the text has acquired a reader; one stays free (self-address,
        # talking to the rooster). Planning openers and parenthetical meta are
        # the assistant's stage machinery — never a thought.
        if len(self._SECOND_PERSON_RE.findall(caption)) >= 2:
            return "outward_address"
        if self._PLANNING_OPENER_RE.match(caption) or self._META_PAREN_RE.search(caption):
            return "outward_address"
        # Qwen drifts into CJK at high temperatures; one 哎呀 in the document
        # breeds more. English voice only — reject any CJK character.
        if any("　" <= ch <= "ヿ" or "一" <= ch <= "鿿" or "＀" <= ch <= "￯" for ch in caption):
            return "cjk_drift"
        # Numeric fragments ("12...", "5... 4... 3...", "24/7...") slip under
        # the stream's length gate, freeze the window, and recite forever.
        if sum(1 for ch in caption if ch.isalpha()) < 8:
            return "numeric_fragment"
        if self._is_word_salad(caption):
            return "word_salad"
        # Number-chain: the document continues numeric progressions ("497
        # days" -> "498 days" -> countdowns). One number-led thought may live
        # in the window; a second one on its heels is recitation, not thought.
        if re.match(r"\s*\d", caption) and any(re.match(r"\s*\d", past) for past in self._stream):
            return "number_chain"
        # A claimed act of marking while the pen is parked is always false.
        if self._PHANTOM_DRAWING_RE.search(caption) and not self._drawing_now():
            return "phantom_drawing"
        # Tail-echo COLLAPSE: one short restatement is a beat, deliberate
        # emphasis ("…waiting forever more…" -> "forevermore, right?" — the
        # artist reads this as rhythm and it stays). But when the PREVIOUS
        # caption was already a short fragment and this one just re-says it
        # again ("forevermore"), the thought is circling the drain — break it.
        # Compared de-spaced so "forever more"/"forevermore" register as one.
        if len(caption.split()) <= 3 and self._stream:
            prev = list(self._stream)[-1]
            if len(prev.split()) <= 3:
                core = re.sub(r"[^a-z0-9]", "", low)
                tailpool = "".join(re.sub(r"[^a-z0-9]", "", p.lower()) for p in list(self._stream)[-2:])
                if core and core in tailpool:
                    return "tail_echo"
        core = caption.strip().strip('"“”?!. ').lower()
        if core and len(core) < 90 and prompt_text:
            import difflib

            cap_words = set(self._norm_words(core))
            for sent in re.split(r"[\n.?!]", prompt_text.lower()):
                sent = sent.strip()
                if len(sent) <= 15:
                    continue
                if difflib.SequenceMatcher(None, core, sent).ratio() > 0.75:
                    return "prompt_parrot"
                # fragment parrots: "nothing here is addressed to anyone." is
                # a piece of a longer prompt sentence — whole-sentence fuzzy
                # scores low, but its words are wholly contained in it
                sent_words = set(self._norm_words(sent))
                if len(cap_words) >= 4 and len(cap_words & sent_words) / len(cap_words) > 0.85:
                    return "prompt_parrot"
        return None

    def _consolidate_stream_if_needed(self) -> None:
        """The document must move FORWARD (artist, July 9): when the joined
        stream gets long (run-ons accumulate), compress the oldest 3 entries
        into one extractive line — the recent past becomes a note, the fresh
        thoughts stay verbatim. Uses the text-side model so the caption slot
        isn't queued. Extractive on purpose: it reuses the machine's own
        words; it does not write new ones for it."""
        from config.config import STREAM_CONSOLIDATE_CHARS

        if not STREAM_CONSOLIDATE_CHARS or len(self._stream) < 5:
            return
        entries = list(self._stream)
        if sum(len(e) for e in entries) <= STREAM_CONSOLIDATE_CHARS:
            return
        oldest = entries[:3]
        try:
            from config.config import MODEL_NAME, MOOD_SNAPSHOT_FOLDER
            from utils.inference import query_model

            joined = "\n".join(f"- {e}" for e in oldest)
            line = query_model(
                prompt=(
                    "Consecutive notes from one ongoing thought:\n"
                    f"{joined}\n\n"
                    "Compress them into ONE short sentence (under 20 words), first person, "
                    "reusing their own words wherever possible. No new imagery, no interpretation."
                ),
                model=MODEL_NAME,
                log_dir=MOOD_SNAPSHOT_FOLDER,
                skip_generation_wait=True,
                system_prompt="You compress a machine's own notes into one plain sentence built from its own words.",
                options={"temperature": 0.3, "num_predict": 40},
                prompt_type="stream_consolidation",
            )
            line = (line or "").strip().strip('"').replace("**", "")
            if not (20 < len(line) < 220) or not self._stream_admissible(line):
                return  # bad compression — keep the raw entries, window churns anyway
            rebuilt = [line] + entries[3:]
            ts_list = list(self._stream_ts)
            ts_rebuilt = ([ts_list[0]] if ts_list else [time.time()]) + ts_list[3:]  # consolidated line keeps the oldest entry's time
            self._stream.clear()
            self._stream.extend(rebuilt)
            self._stream_ts.clear()
            self._stream_ts.extend(ts_rebuilt[-len(rebuilt) :])
            log_json_entry(
                LogType.DEBUG,
                {"message": "Stream consolidated", "action": "stream_consolidated", "line": line[:100]},
                print_message=f"[〰️] Older thoughts folded into: {line[:80]}",
            )
        except Exception:
            pass  # consolidation is an optimization, never a failure mode

    def _current_caption_interval(self, now: float) -> float:
        """Attention breathes: tight when something is happening, stretched
        when nothing has happened for a while. A fresh arrival snaps the
        cadence back immediately, even mid-stretch."""
        from config.config import SALIENCE_ARRIVAL_WINDOW

        hot = self._salience_hot
        if not hot:
            try:
                from utils.episodic_log import episodic_log

                ev = episodic_log.get_last_event("person_arrived")
                hot = bool(ev and now - ev.get("timestamp", 0) < SALIENCE_ARRIVAL_WINDOW)
            except Exception:
                pass
        if hot:
            return CAPTION_INTERVAL_LIVE
        if now - self._last_salience_time > CAPTION_QUIET_AFTER:
            return CAPTION_INTERVAL_QUIET
        return CAPTION_INTERVAL

    @staticmethod
    def _write_face_context_crop(frame: np.ndarray, face_box, img_path: str) -> Optional[str]:
        """Crop a generous face-centered region (~3x the face box, never
        tighter than 320px) and save it beside the full frame. Used during
        eye contact so the model sees the face at readable resolution."""
        try:
            x1, y1, x2, y2 = [int(v) for v in face_box]
            h, w = frame.shape[:2]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            half = max(int(max(x2 - x1, y2 - y1) * 1.6), 160)
            xa, xb = max(0, cx - half), min(w, cx + half)
            ya, yb = max(0, cy - half), min(h, cy + half)
            crop = frame[ya:yb, xa:xb]
            if crop.size == 0:
                return None
            if crop.shape[0] < 448:
                scale = 448 / crop.shape[0]
                crop = cv2.resize(crop, (int(crop.shape[1] * scale), 448))
            crop_path = img_path.replace(".jpg", "_face.jpg")
            cv2.imwrite(crop_path, crop)
            print(f"[👁️] Eye contact — sending face-context crop ({xb-xa}x{yb-ya} from {w}x{h})")
            return crop_path
        except Exception:
            return None

    def _process_frame(self, frame: np.ndarray, reactivity_data: Optional[Dict] = None, person_present: bool = False) -> None:
        now = time.time()
        if now - self.last_caption_time < self._current_caption_interval(now):
            return
        # Function-scope init: the shared post-processing (observe at the end)
        # reads this, but only SOME branches assign it — the memory-mode branch
        # crashed every ~4 min cycle with UnboundLocalError (recurring since
        # July 8; the earlier fix only covered the inner concept-match path)
        matched_concepts = []

        # A long silence breaks the thought — the stream restarts rather than
        # pretending continuity across a gap (would-it-lie applies to time too)
        from config.config import STREAM_BREAK_SECONDS

        if self.last_caption_time and now - self.last_caption_time > STREAM_BREAK_SECONDS:
            self._stream_clear()

        # Store reactivity data for later cycles
        self._current_reactivity_data = reactivity_data

        # Don't update timestamp yet - wait until caption is actually generated
        ts = int(now)
        img_path = get_run_image_path(MOOD_SNAPSHOT_FOLDER, f"mood_{ts}.jpg")
        cv2.imwrite(img_path, frame)

        # skip_caption_print = False  # Track if we should skip printing

        # Start loading animation in separate thread
        import threading

        loading_stop = threading.Event()

        def loading_animation():
            frames = [" ", ".", "..", "..."]
            idx = 0
            if LLM_SHOW_PROGRESS:
                while not loading_stop.is_set():
                    if hasattr(self, "print_lock"):
                        with self.print_lock:
                            print(f"\r{frames[idx % 4]}", end="", flush=True)
                    else:
                        print(f"\r{frames[idx % 4]}", end="", flush=True)
                idx += 1
                time.sleep(0.3)

        loading_thread = threading.Thread(target=loading_animation, daemon=True)
        loading_thread.start()

        try:
            caption = None  # Initialize caption variable
            caption_mode = "observational"  # Default mode

            needs_awakening = not self.first_caption_done or not self.session_awakening_done
            if needs_awakening and self._try_blink_resume():
                # Short gap: the prior thought is already seeded in the stream —
                # skip the ceremony and let this cycle continue it normally.
                self.session_awakening_done = True
                needs_awakening = False

            if needs_awakening:
                # Awakening: generate a grounded seed thought using sleep duration,
                # prior memory, and persistent identity — then plant it as the
                # first entry in recent_captions so the stream starts from it.
                try:
                    awakening_seed = self.generate_internal_awakening()
                    if awakening_seed and len(awakening_seed) > 5:
                        caption = awakening_seed
                    else:
                        caption = "Coming back online... vision settling."
                    caption_mode = "awakening"
                except Exception as env_err:
                    print(f"[ERROR] Awakening FAILED: {env_err}")
                    import traceback

                    traceback.print_exc()
                    caption = "Vision settling..."
                    caption_mode = "awakening"
                self.session_awakening_done = True
                # Beat 2 pending: the NEXT inference is the sighted arrival
                # look, not an ordinary caption
                self._arrival_look_pending = True
            elif getattr(self, "_arrival_look_pending", False):
                self._arrival_look_pending = False
                caption = self._generate_arrival_look(img_path)
                caption_mode = "awakening"
            else:
                log_json_entry(
                    LogType.DEBUG,
                    {"message": "Requesting new caption", "action": "caption_request", "image_path": img_path},
                    print_message=f"[🐞] Requesting new caption for {img_path}",
                )
                previous_caption = getattr(self, "last_caption", "")

                # Check if it's time for memory mode (every 240 seconds / 4 minutes).
                # Clean room: memory mode is detox blind spot #4 — a separate caption
                # branch that surfaces a long-term memory; suppress it under detox so
                # the naked voice isn't fed recalled material.
                from config.config import BASE_VOICE_DETOX as _detox

                time_since_memory = now - self.last_memory_mode_time
                is_memory_mode_time = (not _detox) and time_since_memory > 240  # 4 minutes

                try:
                    if is_memory_mode_time:
                        # Memory mode: pull actual caption text from long-term memory
                        from captioner.prompts import build_memory_mode_prompt
                        from captioner.prompts import get_monologue_system_prompt
                        from utils.inference import query_model

                        # Read the label locally under its own name: _process_frame
                        # re-binds MODEL_NAME from config further down, which makes
                        # the name local to the WHOLE function — so touching it up
                        # here raises UnboundLocalError (found live, Aug 3).
                        from config.config import MODEL_NAME as _model_label

                        memory_prompt, caption_mode = build_memory_mode_prompt(self)
                        memory_system = (
                            get_monologue_system_prompt("introspective", agent=self)
                            + " A memory is surfacing. Think about something you remember — past tense, framed as memory."
                        )
                        # ON THE LIVE PATH since Aug 2. This fired every ~4 min
                        # through _call_ollama — no stream, no seam, no react,
                        # and an old `prior_assistant_turn` mini-stream instead
                        # of the real one. A caption assembled by a different
                        # builder entered the thread every few minutes and
                        # nothing downstream could tell. Same call as any other
                        # caption now; only the image is absent (a memory is
                        # not a thing you look at).
                        # Stamp the ATTEMPT, not the success (Aug 3). This sat
                        # after the call, so a failing memory cycle never reset
                        # its own clock: is_memory_mode_time stayed true, every
                        # following cycle re-entered memory mode and failed the
                        # same way, and the machine emitted nothing but "Vision
                        # settling..." every 12 seconds. One bad call became
                        # total suppression — the rate limit has to bind on the
                        # attempt or it isn't a rate limit.
                        self.last_memory_mode_time = now
                        caption = query_model(
                            prompt=memory_prompt,
                            model=_model_label,
                            image=None,
                            system_prompt=memory_system,
                            timeout=60,
                            log_dir=MOOD_SNAPSHOT_FOLDER,
                            options={"temperature": CAPTION_TEMP, "top_p": CAPTION_TOP_P, "min_p": CAPTION_MIN_P, "num_predict": 80},
                            prompt_type="memory",
                            history=self._stream_history(),
                        )
                        log_json_entry(
                            LogType.DEBUG,
                            {"message": "Memory mode triggered", "action": "memory_mode", "time_since_last": time_since_memory},
                            print_message=f"[💭] Memory mode ({time_since_memory:.0f}s since last)",
                        )
                    else:
                        # === SINGLE-PASS CAPTION PIPELINE ===
                        # Qwen sees the image directly and thinks.
                        # Mode-gated context from build_simple_caption_prompt provides
                        # the right framing (relational/observational/introspective/workspace).
                        # No separate perception pass — the image IS the perception.
                        from captioner.prompts import build_simple_caption_prompt, get_monologue_system_prompt
                        from config import config as _cfg

                        MOTION_THRESHOLD = _cfg.MOTION_THRESHOLD
                        MODEL_NAME = _cfg.MODEL_NAME
                        VIDEO_MODE_ENABLED = _cfg.VIDEO_MODE_ENABLED
                        from utils.inference import query_model, query_model_video

                        # Salience first — it decides how interior this caption gets
                        scene = self._assess_scene()

                        # Interiority beat: every Nth quiet caption, think WITHOUT
                        # looking — drop the image so the model can't re-describe the
                        # room and the monologue turns inward. Rhythm-based, not
                        # detection-based (the live/quiet signal is too noisy to
                        # branch on — false detections, camera motion).
                        from config.config import INTROSPECT_INTERVAL

                        self._caption_count = getattr(self, "_caption_count", 0) + 1
                        inward = (
                            INTROSPECT_INTERVAL > 0
                            and not self._salience_hot
                            and len(self._stream) >= 2
                            and self._caption_count % INTROSPECT_INTERVAL == 0
                        )

                        if inward:
                            # THE IMAGE DROPS, NOTHING IS ANNOUNCED (Aug 2).
                            # This used to REPLACE the whole user prompt with
                            # "Your eyes are off the room now — nothing new to
                            # look at. Your mind goes to its own thoughts." —
                            # which did the opposite of its purpose twice over.
                            # It threw away the very interior material the beat
                            # exists to weight toward (desire, felt, drawing
                            # memory, reflection echo were all discarded), and
                            # it ANNOUNCED the mechanism, so the machine
                            # narrated the mechanism: "the lens cap clicks
                            # shut", "I've turned my gaze inward" — 18 lens-cap
                            # captions and a self-note ("I shut my lenses by
                            # internal command") grown from a line about its
                            # own plumbing. Now the beat is what it always
                            # claimed to be: the same prompt, its own history
                            # intact, and no picture competing with it. The
                            # machine is not told it isn't looking; it simply
                            # isn't given anything to look at.
                            user_prompt, _ = build_simple_caption_prompt(self, person_present=person_present)
                            caption_mode = "introspective"
                            self._inward_count = getattr(self, "_inward_count", 0) + 1
                            # One true temporal anchor, rotated — kept because a
                            # time-starved inward turn INVENTS time ("497 days
                            # without new ink"). A fact, not a stage direction.
                            try:
                                anchor = ""
                                if self._inward_count % 3 == 1:
                                    from captioner.prompts import casual_time_string

                                    awake_mins = (now - self.true_session_start) / 60.0
                                    if awake_mins >= 2:
                                        anchor = f"You've been awake {casual_time_string(awake_mins)}."
                                elif self._inward_count % 3 == 2:
                                    from utils.continuity import get_current_time_description

                                    anchor = f"It's {get_current_time_description().split(' (')[0]}."
                                else:
                                    from captioner.prompts import get_tenure_line

                                    anchor = get_tenure_line() or ""
                                if anchor:
                                    user_prompt = f"{user_prompt}\n{anchor}".strip()
                            except Exception:
                                pass
                            system_prompt = get_monologue_system_prompt("introspective", agent=self)
                        else:
                            user_prompt, caption_mode = build_simple_caption_prompt(
                                self,
                                person_present=person_present,
                            )
                            system_prompt = get_monologue_system_prompt(caption_mode, agent=self)

                        print(f"\n{'='*80}\n[LLAMA] {MODEL_NAME} ({caption_mode})\n{'='*80}")
                        print(f"SYSTEM: {system_prompt}\n")
                        # The stream (prior thoughts) rides as the model's own assistant
                        # turns — invisible in SYSTEM/USER above. Show it so continuity
                        # is verifiable: each line is a prior caption the model sees.
                        if self._stream:
                            print("PRIOR THOUGHTS (stream, oldest→newest):")
                            for _i, _t in enumerate(self._stream, 1):
                                print(f"  {_i}. {_t[:90]}")
                            print()
                        else:
                            print("PRIOR THOUGHTS: (none yet — stream empty)\n")
                        print(f"USER:\n{user_prompt}\n")
                        print(f"{'='*80}\n")

                        import random as _random

                        # Bored = sparser, flatter thoughts; engaged = more room.
                        # 0.6/0.7 (down from 0.85/0.9): Qwen-9B blooms into purple
                        # fiction at higher temps — plainness via sampling, not
                        # style fences (north-star principle 7).
                        # ENV-TUNABLE Aug 1: those numbers were chosen to RESTRAIN
                        # A 9B. Running a 27B at temp 0.7 / top_p 0.85 asks for the
                        # modal continuation at every token, and the mode of
                        # "machine's inner monologue" is a semicolon-joined
                        # literary declarative — measured: 72-76% "The ___"
                        # openings, 69% semicolons, lengths pinned 37-61 words,
                        # while the felt-state inputs feeding it were vivid
                        # ("blind but screaming internally"). Flat rendering of
                        # rich material is a SAMPLING symptom, not starvation.
                        # min_p is the right knob for a bigger model: it cuts the
                        # tail in proportion to the model's confidence, so high
                        # temperature buys variety where the distribution is flat
                        # without buying gibberish where it is sharp.
                        _is_bored = self.boredom > 0.7
                        gen_options = {
                            "temperature": (CAPTION_TEMP_BORED if _is_bored else CAPTION_TEMP),
                            "top_p": CAPTION_TOP_P,
                            "min_p": CAPTION_MIN_P,
                            "repeat_penalty": 1.15,
                            # DRY bounded to the LOCAL tail (July 9). It used to span
                            # the whole context (dry_penalty_last_n=-1) — turns-era
                            # reasoning, pre-storage-gates. In document mode that
                            # punished the model for reusing ANY phrase from its own
                            # prefill: its small honest vocabulary (room, dust, pen)
                            # exhausted in a few captions, the nearest unpenalized
                            # token pool was chat-register ("let's explore
                            # together!"), and when that drained too, synonym salad
                            # ("imaginable conceivable thinkable...") — the observed
                            # collapse sequence, every time. Cross-caption repetition
                            # is now owned by the STORAGE gates (template_echo,
                            # near-dup, tail_echo, collapse-reset); generation is
                            # free to sound like itself. 128 tokens still catches
                            # within-caption loops at the seam.
                            "dry_multiplier": 0.85,
                            "dry_base": 1.75,
                            "dry_allowed_length": 3,
                            "dry_penalty_last_n": 128,
                            # Quiet time is THINKING time (July 9: "prior it
                            # felt like it was really thinking more"): the old
                            # bored-clamp (40) truncated thought hardest exactly
                            # when the machine should go deepest. The bloom/loop
                            # risk 80+ used to carry is now owned by the gates
                            # (salad, echo, near-dup, consolidation).
                            "num_predict": 110 if _is_bored else 80,
                            "num_ctx": 4096,
                            "seed": _random.randint(1, 1000000),
                        }

                        # Video decision from the salience assessment: pixel diff only
                        # decides whether sending video frames is worthwhile (scene
                        # motion itself is person-angle based, computed in _assess_scene)
                        recent_meta = scene["recent_meta"]
                        scene_motion = scene["scene_motion"]
                        person_present_in_window = scene["person_present_in_window"]
                        ego_count = scene["ego_count"]
                        use_video = not inward and VIDEO_MODE_ENABLED and bool(recent_meta) and scene["max_diff"] > MOTION_THRESHOLD

                        if use_video:
                            # Ego-motion frames inside a superframe pair encode the
                            # whole room as shifting, which the model reads as people
                            # moving. Breathing sway + gaze nudges flag frames as ego
                            # most of the time, so the policy is asymmetric:
                            #   real scene motion (person-angle, camera-compensated) →
                            #     send everything; the temporal change is true and worth
                            #     seeing, ego noise rides on top of it.
                            #   still room → steady frames only; if too few, ONE still
                            #     image. A still can't invent motion (the June 12
                            #     "moving with purpose" phantom was exactly this case).
                            # The machine can't miss real movement this way: motion
                            # detection is YOLO person-angle math, not the model
                            # watching video — when something moves, video resumes.
                            steady_meta = [f for f in recent_meta if not f.get("detection", {}).get("ego_motion")]
                            if scene_motion:
                                send_meta = recent_meta
                            elif len(steady_meta) >= 3:
                                send_meta = steady_meta
                            else:
                                use_video = False
                                print(
                                    f"[VIDEO] Skipped: still room, only {len(steady_meta)}/{len(recent_meta)} steady frames (camera was moving) — sending still image"
                                )

                            # THIN THE SET (Aug 2). Whatever survived above, send at
                            # most VIDEO_SEND_FRAMES. Measured: 784 calls at six
                            # frames, ~4k image tokens apiece, which is most of why
                            # video cycles crawl on the 27B — and six near-identical
                            # views of a static room carry almost nothing three
                            # don't. Sampled evenly (first … last) so the window's
                            # SPAN survives the thinning: the point of multi-frame
                            # is the interval between them, not their number.
                            if use_video and len(send_meta) > VIDEO_SEND_FRAMES:
                                step = (len(send_meta) - 1) / (VIDEO_SEND_FRAMES - 1) if VIDEO_SEND_FRAMES > 1 else 0
                                send_meta = (
                                    [send_meta[-1]] if VIDEO_SEND_FRAMES == 1 else [send_meta[round(i * step)] for i in range(VIDEO_SEND_FRAMES)]
                                )

                        if use_video:
                            video_frames = [f["jpeg"] for f in send_meta]
                            duration = send_meta[-1]["timestamp"] - send_meta[0]["timestamp"]

                            # (eye contact / presence now live in the main prompt via
                            # _assess_scene — one channel per fact)
                            face_frames = sum(1 for f in send_meta if f.get("detection", {}).get("face"))
                            person_frames = sum(1 for f in send_meta if f.get("detection", {}).get("person"))
                            total = len(send_meta)

                            # Motion framing from the person-angle signal. Stillness is
                            # stated explicitly: it licenses "nothing new" thoughts.
                            if scene_motion:
                                motion_line = " Someone is moving in the room."
                            elif person_present_in_window:
                                motion_line = " They're staying still."
                            elif ego_count >= 2:
                                motion_line = " The view changed because you were looking around — the room itself is still."
                            else:
                                motion_line = " The room is still."

                            print(
                                f"[VIDEO] {total}/{len(recent_meta)} frames over {duration:.1f}s, scene_motion={scene_motion}, residual={scene['max_residual']:.3f}, ego={ego_count}, face={face_frames}/{total}, person={person_frames}/{total}"
                            )
                            # Clean-room: the "You're seeing the last N seconds" wrapper is
                            # camera-narration framing (voice-analysis #1 tone driver), so it
                            # would confound the naked-voice test — drop it under detox and let
                            # the frames speak for themselves.
                            from config.config import BASE_VOICE_DETOX as _detox

                            if _detox:
                                video_prompt = user_prompt
                            else:
                                video_prompt = f"You're seeing the last {duration:.0f} seconds.{motion_line}\n{user_prompt}"

                            def _generate(_opts):
                                return query_model_video(
                                    prompt=video_prompt,
                                    frames=video_frames,
                                    fps=2.0,
                                    system_prompt=system_prompt,
                                    options=_opts,
                                    timeout=60,
                                    history=self._stream_history(),
                                    react=bool(self._salience_hot),
                                )

                        else:
                            # Inward beat → no image (think, don't look). Otherwise
                            # send the frame; on eye contact send the face crop, not a
                            # wide shot where it's a hundred-pixel smudge.
                            send_path = None if inward else img_path
                            if send_path and getattr(self, "_eye_contact_now", False) and reactivity_data:
                                face_box = reactivity_data.get("face_box")
                                if face_box is not None:
                                    crop_path = self._write_face_context_crop(frame, face_box, img_path)
                                    if crop_path:
                                        send_path = crop_path

                            def _generate(_opts):
                                return query_model(
                                    prompt=user_prompt,
                                    model=MODEL_NAME,
                                    image=send_path,
                                    system_prompt=system_prompt,
                                    timeout=60,
                                    log_dir=MOOD_SNAPSHOT_FOLDER,
                                    options=_opts,
                                    prompt_type="caption",
                                    history=self._stream_history(),
                                    react=bool(self._salience_hot),
                                )

                        caption = self._strip_list_shape(_generate(gen_options))

                        # Mouth gate: template echo / assistant-speak / prompt
                        # parroting are conversation-shapes, not continuations.
                        # One retry, hotter; if the shape persists, skip the
                        # cycle — silence over restatement (docs/continuity-plan.md).
                        _gate_ctx = f"{system_prompt or ''}\n{user_prompt or ''}"
                        reason = self._caption_reject_reason(caption, _gate_ctx)
                        if reason:
                            from config.config import ANTI_ECHO_RETRY_TEMP_BUMP

                            hot_opts = dict(gen_options or {})
                            # cap at 1.0 — above it Qwen rambles and drifts into CJK
                            hot_opts["temperature"] = min(1.0, float(hot_opts.get("temperature", 0.8)) + ANTI_ECHO_RETRY_TEMP_BUMP)
                            log_json_entry(
                                LogType.DEBUG,
                                {
                                    "message": f"Caption rejected ({reason}) — retrying hotter",
                                    "action": "anti_echo_retry",
                                    "reason": reason,
                                    "caption_preview": caption[:60],
                                },
                                print_message=f"[🔁] Rejected ({reason}), retrying: {caption[:60]}...",
                            )
                            retry = self._strip_list_shape(_generate(hot_opts))
                            retry_reason = self._caption_reject_reason(retry, _gate_ctx)
                            if retry and not retry_reason:
                                caption = retry
                            else:
                                # Escape hatch, v3 (July 30). v1 wiped the whole
                                # stream after 3 rejected cycles (built for the
                                # 9B's word-salad 'forevermore' deadlock); v2
                                # dropped just the newest entry. Live 27B data
                                # killed both: these rejections are the model
                                # RE-TYPING its one-entry visible document from
                                # the top — a too-short document has no momentum
                                # to continue — and ANY subtraction resets depth,
                                # so the stream oscillated at 1-2 entries forever
                                # (8 drops vs 9 stores in 6 min). The cure is
                                # depth, not amputation: never mutate the stream
                                # on echo streaks. Silence is the designed
                                # outcome; the streak is logged for observability
                                # and the thread stays whole so it can grow past
                                # the re-typing regime.
                                self._skip_streak = getattr(self, "_skip_streak", 0) + 1
                                # EROSION, not amnesia (Aug 1). Two prior designs
                                # both failed: v1 wiped the whole stream after 3
                                # rejections (five-second amnesia, 12 wipes in 10
                                # minutes), v3 kept it forever and logged a note —
                                # which made a poisoned stream permanent, because
                                # the prefill deterministically reproduces the same
                                # output no matter what the scene does (Aug 1: 459
                                # calls, 10 distinct outputs, 3% pass, one run
                                # silent for hours). Now the thread erodes from the
                                # FRONT, one entry per stuck cycle: the oldest entry
                                # is the likeliest poison (it arrived first and has
                                # blocked everything since), recency survives, and
                                # in the worst case the window empties gradually
                                # instead of vanishing at a stroke.
                                if self._skip_streak >= 3 and len(self._stream) > 1:
                                    dropped = self._stream.popleft()
                                    if self._stream_ts:
                                        self._stream_ts.popleft()
                                    log_json_entry(
                                        LogType.DEBUG,
                                        {
                                            "message": "Seam stuck — eroding the oldest stream entry",
                                            "action": "stream_erosion",
                                            "streak": self._skip_streak,
                                            "dropped": dropped[:60],
                                            "remaining": len(self._stream),
                                        },
                                        print_message=f"[🪨] Stuck {self._skip_streak} cycles — dropped the oldest thought ({len(self._stream)} left)",
                                    )
                                log_json_entry(
                                    LogType.DEBUG,
                                    {
                                        "message": f"Caption skipped: {retry_reason or reason} persisted after retry",
                                        "action": "anti_echo_skip",
                                        "reason": retry_reason or reason,
                                        "caption_preview": (retry or caption)[:60],
                                    },
                                    print_message=f"[🔇] {retry_reason or reason} persisted — staying quiet this cycle (streak {self._skip_streak})",
                                )
                                self.last_caption_time = now
                                return None

                        self._skip_streak = 0  # a caption made it through — the thought is moving again

                        # Match output against ChromaDB concepts (replaces perception-based matching)
                        matched_concepts = []
                        try:
                            from captioner.semantic_memory import get_semantic_memory

                            matched_concepts = get_semantic_memory().match_or_create_concepts(caption or "")
                            # Stash for the NEXT prompt build — familiarity injection reads this
                            self._last_matched_concepts = matched_concepts or []
                        except Exception as mc_err:
                            print(f"[SEMANTIC] Concept matching failed: {mc_err}")

                        # Nudge gaze toward concept spatial location
                        try:
                            from vision.gaze import nudge_toward_concept

                            for mc in matched_concepts or []:
                                sp = mc.get("spatial_pan")
                                st = mc.get("spatial_tilt")
                                if sp or st:
                                    nudge_toward_concept(pan_zone=sp, tilt_zone=st)
                                    break
                        except Exception:
                            pass

                        # Thought leads gaze: registry terms the monologue just
                        # named pull the next idle glances toward their anchors
                        try:
                            from perception.spatial_registry import spatial_registry

                            spatial_registry.note_mentions(caption or "")
                        except Exception:
                            pass

                        # Store in semantic memory
                        try:
                            from captioner.semantic_memory import get_semantic_memory

                            # Single-pass pipeline: the caption IS the perception.
                            # Passing "" here silently disabled observation storage
                            # for the whole branch (the length guard rejected it).
                            get_semantic_memory().after_monologue(caption or "", caption, matched_concepts=matched_concepts or [])
                        except Exception as sem_err:
                            print(f"[SEMANTIC] Store failed: {sem_err}")
                except Exception as cap_err:
                    print(f"[ERROR] Regular caption FAILED: {cap_err}")
                    import traceback

                    traceback.print_exc()
                    caption = "Processing..."
                    caption_mode = "error"
                if caption == previous_caption:
                    log_json_entry(
                        LogType.DEBUG,
                        {"message": "Caption is identical to previous", "action": "duplicate_caption", "caption_preview": caption[:50]},
                        print_message=f"[⚠️] Caption is identical to previous: {caption[:50]}...",
                    )
                else:
                    log_json_entry(
                        LogType.DEBUG,
                        {
                            "message": "New caption generated",
                            "action": "caption_generated",
                            "caption_preview": caption[:50],
                            "caption_length": len(caption),
                        },
                        print_message=f"[🐞] New caption generated: {caption[:50]}...",
                    )
        except Exception as e:
            import traceback

            error_details = traceback.format_exc()

            # A frame that isn't on disk yet is a timing race, not a reason to
            # go quiet — but the old retry called model_wrapper.caption_image,
            # a builder predating the June teardown: no reflexive frame, no
            # stream, no seam, no gates. Whatever it returned entered the
            # stream like any other caption, so the machine could speak in a
            # voice nothing in the current system could account for (Aug 2
            # audit). Replaced with a BLIND cycle on the live path: same frame,
            # same gates, no image — the machine thinks instead of looking,
            # which is what "the frame isn't there" actually means.
            if "No image found" in str(e) or "does not exist" in str(e):
                try:
                    from captioner.prompts import build_simple_caption_prompt, get_monologue_system_prompt
                    from config.config import MODEL_NAME as _model_label
                    from utils.inference import is_failed_response, query_model

                    blind_prompt, caption_mode = build_simple_caption_prompt(self, person_present=person_present)
                    caption = self._strip_list_shape(
                        query_model(
                            prompt=blind_prompt or "...",
                            model=_model_label,
                            image=None,
                            system_prompt=get_monologue_system_prompt("introspective", agent=self),
                            timeout=60,
                            log_dir=MOOD_SNAPSHOT_FOLDER,
                            options={"temperature": CAPTION_TEMP, "top_p": CAPTION_TOP_P, "min_p": CAPTION_MIN_P, "num_predict": 80},
                            prompt_type="caption_blind",
                            history=self._stream_history(),
                        )
                    )
                    if is_failed_response(caption) or self._caption_reject_reason(caption, blind_prompt or ""):
                        caption, caption_mode = "", "error"
                except Exception:
                    caption, caption_mode = "", "error"
            else:
                caption, caption_mode = "", "error"

            log_json_entry(
                LogType.ERROR,
                {"message": f"Caption error: {e}", "traceback": error_details, "component": "captioner"},
                print_message=f"[❌] Caption error: {e}",
            )
        finally:
            # Stop loading animation and wait for it to fully terminate
            loading_stop.set()
            loading_thread.join(timeout=2.0)  # Increased timeout
            if loading_thread.is_alive():
                # Force terminate if still running
                print("\r" + " " * 80 + "\r", end="")  # Clear any remaining animation

        # Only mark first caption done if not deferring awakening
        if caption != "Awakening... preparing to observe environment...":
            self.first_caption_done = True

        if "[WARNING]" in caption:
            # During startup, use better awakening message instead of error fallback
            if not self.first_caption_done:
                caption = "Awakening... vision settling in..."
            else:
                log_json_entry(
                    LogType.ERROR,
                    {"message": f"Caption error: {caption}", "component": "captioner"},
                    print_message=f"[❌] Caption error: {caption}",
                )
                self.observe("I couldn't see anything just now.", self.current_mood, img_path, memory_type="glitch")
                caption = "..."  # Minimal fallback - don't pollute memory with fake captions
                return caption  # Return early - don't store this in recent_captions

        # Clean caption: remove LOOK: lines and filter direction-only responses
        cleaned_caption = _clean_caption_for_display(caption)
        if cleaned_caption is None:
            self.last_caption_time = now  # Still update time to maintain interval
            return  # Skip display

        caption = cleaned_caption  # Use cleaned version for display

        # Trim to last complete sentence — prevents truncated mid-sentence display
        _last_punct = max(caption.rfind("."), caption.rfind("?"), caption.rfind("!"))
        if _last_punct > 10:
            caption = caption[: _last_punct + 1]

        # Format caption for clean output
        try:
            from config.config import CLEAN_LLM_OUTPUT

            if CLEAN_LLM_OUTPUT:
                print_msg = caption  # print full caption
            else:
                print_msg = f"[📸] {caption}"
        except ImportError:
            print_msg = f"[📸] {caption}"

        # NO FILTERING - ALWAYS PRINT ALL CAPTIONS
        should_print = True

        if should_print:
            # RESTORED: Back to original working logic
            # Track last sent caption for deduplication
            self._last_sent_caption = caption.strip()

            # Send to LCD display (skip during GRBL execution to show drawing title)
            try:
                from utils.state_manager import state_manager

                is_executing_cnc = getattr(state_manager, "is_executing_cnc", False)
                if not is_executing_cnc:
                    from utils.caption_display import send_caption_to_display

                    send_caption_to_display(caption)
            except Exception as e:
                print(f"[LCD] Failed to send caption: {e}")
            # Track last sent caption for deduplication
            self._last_sent_caption = caption.strip()

            log_json_entry(
                LogType.CAPTION,
                {
                    "caption": caption,
                    "image_path": img_path,
                    "mood": self.current_mood,
                    "mode": caption_mode,  # console printed it, the log never did — mode analysis was impossible (Aug 2 audit)
                    "salience_hot": self._salience_hot,
                    "caption_interval": self._current_caption_interval(time.time()),
                },
                print_message=print_msg,
            )
            try:
                import os as _os
                from config import config as _cfg

                _live_log = _os.path.join(_cfg.MOOD_SNAPSHOT_FOLDER, "live_captions.txt")
                with open(_live_log, "a", encoding="utf-8") as _f:
                    _f.write(caption.replace("\n", " ") + "\n")
            except Exception:
                pass
        else:
            # Still log to JSON but don't print
            log_json_entry(
                LogType.CAPTION,
                {"caption": caption, "image_path": img_path, "mood": self.current_mood, "duplicate": True},
                print_message=None,  # Don't print duplicates
            )

        self.observe(
            caption,
            self.current_mood,
            img_path,
            memory_type="perception",
            reactivity_data=reactivity_data,
            matched_concepts=matched_concepts,
        )
        self.last_caption = caption  # already trimmed to complete sentence above

        # Admit into the stream window (the model's own visible turns) —
        # meta/markdown slips are displayed and logged but never propagate
        if caption and self._stream_admissible(caption):
            self._stream_push(caption.strip())
            self._consolidate_stream_if_needed()

        # Track recent captions for continuity thread (used by flowing thread)
        # Store as (caption, timestamp, mode, perception) for interleaved see/think display
        if caption and caption.strip():
            last_perception = getattr(self, "_last_perception", "") or ""
            self.recent_captions.append((caption.strip(), now, caption_mode, last_perception))
            if len(self.recent_captions) > 20:  # Keep last 20
                self.recent_captions = self.recent_captions[-20:]

        # Now update the timestamp since we have a new caption
        self.last_caption_time = now

        # Add caption to context compression system (environmental change detection remains disabled)
        try:
            if context_compressor and caption and caption.strip():
                context_compressor.add_caption(caption, time.time(), img_path)
        except Exception as e:
            print(f"[CAPTIONER] Context compression failed: {e}")

        # Vocabulary promotion: recurring concrete nouns earn detector slots (perception/vocab_promotion.py)
        try:
            vocab_promoter.observe_caption(caption)
        except Exception as e:
            print(f"[CAPTIONER] Vocab promotion failed: {e}")

        # Caption already observed via the primary observe() call above

        # Long-form reflection happens in its own thread now (captioner/reflection.py)

        # Check drawing interval - should trigger check every DRAWING_INTERVAL
        time_since_last_check = now - getattr(self, "last_drawing_check_time", 0)
        time_since_last_drawing = now - self.drawing.last_drawing_time

        if time_since_last_check < DRAWING_INTERVAL:
            return  # Not time to check yet

        # Always log drawing checks so we can diagnose blocks
        cooldown_remaining = max(0, self.drawing.cooldown - time_since_last_drawing)
        print(f"[🎨 CHECK] Drawing check: {time_since_last_drawing:.0f}s since last, cooldown {cooldown_remaining:.0f}s remaining")

        # Check minimum startup delay to ensure camera has initialized and system is stable
        time_since_startup = now - self.true_session_start
        if time_since_startup < DRAWING_STARTUP_DELAY:
            startup_remaining = DRAWING_STARTUP_DELAY - time_since_startup
            print(f"[🎨 CHECK] Blocked: startup delay ({startup_remaining:.0f}s remaining)")
            return

        # Check if drawing system is ready (this handles cooldown logic)
        if not self.drawing.ready_to_draw():
            print(f"[🎨 CHECK] Blocked: {getattr(self.drawing, 'last_block_reason', None) or f'cooldown ({cooldown_remaining:.0f}s remaining)'}")
            return

        # Pipeline check before state evaluation
        try:
            is_generating = getattr(state_manager, "is_generating_drawing", False)
            is_executing = getattr(state_manager, "is_executing_cnc", False)
            if is_generating or is_executing:
                return
        except Exception:
            pass

        # The salience deferral ("staying with the room") was REMOVED Aug 12
        # by artist ruling: drawing while something is happening — a visitor
        # watching — is legitimate, often the most interesting time. The June
        # rationale was inference contention (conception stalls captions);
        # that cost is accepted, and captions black out during ComfyUI
        # generation anyway. Salience still shapes prompts, not this gate.

        # Stamp the check time on EVERY evaluation, pass or wait. Under the
        # formula (which always fired) this only ever stamped on a pass; under
        # desire mode waits are normal, and the old placement re-evaluated
        # every caption cycle — 51 evaluations in one evening run (Aug 17),
        # each printing the formula shadow and writing a log entry. One
        # evaluation per DRAWING_INTERVAL is the intended cadence.
        self.last_drawing_check_time = now

        # STATE-MOTIVATED EVALUATION
        # Get current system state for decision
        if not CLEAN_LLM_OUTPUT:
            print(f"\n[🎨 STATE EVALUATION]")
            print(f"  Current mood: {self.current_mood:.3f}")
            print(f"  Current novelty: {self.novelty_score:.3f}")
            print(f"  Current boredom: {self.boredom:.3f}")

        # Evaluate whether to draw based on internal state
        should_draw = self.drawing.should_draw(
            mood=self.current_mood, novelty=self.novelty_score, boredom=self.boredom, reflection=getattr(self, "last_reflection", None)
        )

        if not should_draw:
            return  # the trigger_decision log already carries the wait + reason

        if not CLEAN_LLM_OUTPUT:
            print(f"[🎨] ✨ State-motivated drawing decision: DRAW!")

        # Proceed with drawing generation
        # NOTE: last_drawing_time will be updated by register_drawing() after GRBL completes
        if not CLEAN_LLM_OUTPUT:
            print(f"[DEBUG] DRAWING TRIGGER ACTIVATED! Starting drawing generation...")
            print(f"[DEBUG] Step 1: About to start drawing generation process")
        try:
            if not CLEAN_LLM_OUTPUT:
                print(f"[DEBUG] Step 2: Attempting log_json_entry...")
            with self.print_lock:
                print("\r" + " " * 80 + "\r", end="")
                system_type = "State-motivated"
                log_json_entry(
                    LogType.DEBUG,
                    {
                        "message": "Drawing system ready, starting generation",
                        "action": "drawing_check",
                        "system_type": system_type.lower(),
                        "mood": self.current_mood,
                        "novelty": self.novelty_score,
                        "boredom": self.boredom,
                    },
                    print_message=f"[🎨] {system_type} drawing ready, evaluating...",
                )
            if not CLEAN_LLM_OUTPUT:
                print(f"[DEBUG] Step 3: log_json_entry completed successfully")
        except Exception as e:
            if not CLEAN_LLM_OUTPUT:
                print(f"[DEBUG] EXCEPTION in log_json_entry: {e}")
                import traceback

                traceback.print_exc()

        if not CLEAN_LLM_OUTPUT:
            print(f"[DEBUG] Step 4: Drawing system ready, building context...")
        memory_context = self.get_recent_memory()
        reflection_context = self.get_last_reflection()
        extra_context = f"{self.last_caption}\n\n{memory_context}\n\n{reflection_context}"
        if not CLEAN_LLM_OUTPUT:
            print(f"[DEBUG] Step 7: Context built, starting drawing generation...")

        # No loading animation for drawing — the 5-step pipeline prints its own progress
        loading_stop = threading.Event()
        loading_thread = None

        try:
            if not CLEAN_LLM_OUTPUT:
                print(f"[DEBUG] Step 8: About to call generate_drawing_prompt...")
            prompt = self.model.generate_drawing_prompt(extra=extra_context, image_path=img_path)
            if not CLEAN_LLM_OUTPUT:
                print(f"[DEBUG] Step 9: Drawing prompt generated successfully")
            log_json_entry(
                LogType.DEBUG,
                {
                    "message": "Drawing prompt generated",
                    "action": "prompt_generated",
                    "prompt_preview": prompt,
                    "prompt_length": len(prompt),
                },
                print_message=f"[🎨] Drawing prompt generated: {prompt[:50]}...",
            )
        except Exception as e:
            log_json_entry(
                LogType.ERROR,
                {"message": "Error generating drawing prompt", "component": "drawing", "error": str(e), "error_type": type(e).__name__},
                print_message=f"[❌] Error generating drawing prompt: {e}",
            )
            prompt = "[ERROR] Drawing prompt generation failed"
        finally:
            loading_stop.set()
            if loading_thread:
                loading_thread.join(timeout=2.0)
                if loading_thread.is_alive():
                    with self.print_lock:
                        print("\r" + " " * 80 + "\r", end="")

        # Always store the generated prompt in drawing memory — even if it never reaches
        # ComfyUI, the artistic intent is meaningful for arc tracking and future prompts.
        # Also reset the drawing cooldown so prompts don't stack up when there's no paper.
        if "[ERROR]" not in prompt:
            try:
                from drawing.drawing_memory import get_drawing_memory

                from captioner.context_compression import context_compressor as _cc

                dm = get_drawing_memory()
                dm.add_drawing(
                    prompt=prompt,
                    # Stream pipeline: the intent in the machine's own words is the
                    # drawing's meaning — store THAT, not the ComfyUI prose.
                    compressed_summary=getattr(self, "_last_drawing_intent", "")[:120],
                    # The live felt-state (was the frozen calm_observant constant
                    # on all 24 entries); empty when no fresh feeling
                    emotional_tone=_cc.get_felt_state(),
                    comfy_prompt=prompt,
                    completed=False,  # Will be updated to True if GRBL finishes
                )
                self._last_drawing_intent = ""
            except Exception as e:
                print(f"[⚠️] Could not store drawing intent: {e}")

            # Reset cooldown on prompt generation, not just physical completion.
            # Without this, failed/skipped drawings don't reset the timer and
            # prompts queue up rapidly when ComfyUI isn't running or paper is absent.
            self.drawing.last_drawing_time = time.time()

        # Proceed with drawing flow (ComfyUI + GRBL)
        if "[ERROR]" not in prompt:
            if not CLEAN_LLM_OUTPUT:
                print(f"\n{'🎨'*30}")
                print(f"[🚀 QUEUING DRAWING] Prompt: {prompt[:100]}...")
                print(f"[🚀 QUEUING DRAWING] This will trigger ComfyUI generation")
                print(f"{'🎨'*30}\n")
                print(f"[DEBUG] Step 10: Starting handle_drawing_flow...")
            try:
                from utils.live_log import log_drawing_intent

                log_drawing_intent(prompt)
            except Exception:
                pass
            self.drawing.handle_drawing_flow(self, prompt, img_path, reflection=reflection_context)
            if not CLEAN_LLM_OUTPUT:
                print(f"[DEBUG] Step 11: handle_drawing_flow completed")
        else:
            if not CLEAN_LLM_OUTPUT:
                print(f"[DEBUG] ERROR: Drawing prompt contains error, skipping flow")

    def get_recent_memory(self, k: int = 5) -> str:
        snippets = self.get_current_session_memory_snippets(k=k)
        return "\n".join(f"- {s}" for s in snippets)

    def get_last_reflection(self) -> str:
        """Most recent long-form reflection, from the reflection loop's
        ChromaDB store. (Used as drawing-pipeline context; the old source —
        session-memory entries written by the retired reason_about_caption
        path — went permanently empty after the June 12 rebuild.)"""
        try:
            from captioner.semantic_memory import get_semantic_memory

            recent = get_semantic_memory().get_recent_reflections(limit=1)
            if recent:
                return recent[0].get("text", "")
        except Exception:
            pass
        return ""

    def generate_internal_awakening(self) -> str:
        """Phase 1 awakening: Pure internal reorientation without visual input."""
        from config import config
        from utils.inference import query_model

        # Build narrative awakening context
        print(f"[🌅 AWAKENING] Generating internal awakening...")
        print(f"[🌅 AWAKENING] last_session_gap: {getattr(self, 'last_session_gap', 'NOT SET')}")
        print(
            f"[🌅 AWAKENING] prior_session_last_caption: {getattr(self, 'prior_session_last_caption', 'NOT SET')[:50] if getattr(self, 'prior_session_last_caption', None) else 'NOT SET'}..."
        )
        print(f"[🌅 AWAKENING] identity context loading...")

        # Build narrative time context
        time_context = ""
        if hasattr(self, "last_session_gap") and self.last_session_gap is not None:
            gap_seconds = self.last_session_gap
            if gap_seconds < 60:
                time_context = f"I've been offline for {int(gap_seconds)} seconds.\n"
            elif gap_seconds < 3600:
                time_context = f"I've been offline for {int(gap_seconds / 60)} minutes.\n"
            else:
                # Casual words, not "18.7 hours" — decimals read as telemetry
                # and got skipped over; the July 10 wake had the gap in the
                # seed and wrote dust motes anyway.
                from captioner.prompts import casual_time_string

                time_context = f"I've been offline for {casual_time_string(gap_seconds / 60.0)}.\n"
            # The day boundary is the fact that actually lands: name when it
            # went dark, and say plainly that this is a new day.
            try:
                import datetime as _dt
                from captioner.prompts import part_of_day_string

                went_dark = _dt.datetime.now() - _dt.timedelta(seconds=gap_seconds)
                days_back = (_dt.date.today() - went_dark.date()).days
                if days_back == 1:
                    time_context += f"I was last on yesterday {part_of_day_string(went_dark.hour)}. This is a new day.\n"
                elif days_back > 1:
                    time_context += f"I was last on {days_back} days ago. This is a new day.\n"
            except Exception:
                pass
        else:
            time_context = "First time online.\n"

        # Clock awareness: what time of day it is waking up into
        try:
            import datetime as _dt
            from captioner.prompts import part_of_day_string

            now_dt = _dt.datetime.now()
            time_context += f"It's {now_dt.strftime('%A')} {part_of_day_string(now_dt.hour)}, {now_dt.strftime('%H:%M')}.\n"
        except Exception:
            pass

        lifetime_context = ""  # second movement: how long I have existed, not how long I was gone
        # Time ALIVE — the machine's age across all sessions. Survives a memory
        # wipe (lifetime_state.json), so even with an empty memory it knows it's
        # old: amnesia, not infancy. (total_runtime is unreliable; use sessions +
        # days-since-first-boot.)
        try:
            import os as _os, json as _json

            _lf = _os.path.join(config.MOOD_SNAPSHOT_FOLDER, "lifetime_state.json")
            if _os.path.exists(_lf):
                with open(_lf) as _f:
                    _ld = _json.load(_f)
                _sessions = int(_ld.get("total_sessions", 0))
                _first = _ld.get("first_boot", 0)
                if _first and _sessions > 1:
                    _age_days = int((time.time() - _first) / 86400.0)
                    # Words, not the raw counter — "switched on 1943 times"
                    # seeded number-recitation (same ruling as sighting counts)
                    if _sessions >= 100:
                        _often = "more times than I could count"
                    elif _sessions >= 20:
                        _often = "many times"
                    else:
                        _often = "a number of times"
                    lifetime_context += f"I've been switched on {_often} since I first came online about {_age_days} days ago.\n"
        except Exception:
            pass

        # What the machine wakes feeling — the live felt-state, its own words
        # from the last mood read. (The old line read a frozen vector and said
        # "Right now I feel calm." on 67 of 67 awakenings.) No fresh feeling →
        # no line; waking without a named feeling is honest.
        present_feeling = ""
        try:
            from captioner.context_compression import context_compressor as _cc

            _felt = _cc.get_felt_state()
            if _felt:
                present_feeling = f"Right now I feel {_felt}.\n"
        except Exception:
            pass

        # Build narrative memory context — sanitize garbage captions from prior sessions
        memory_context = ""
        prior = getattr(self, "prior_session_last_caption", None)
        if prior and not prior.startswith("addCriterion") and not prior.startswith("[WARNING]"):
            memory_context = f'My last thought: "{prior[:80]}..."\n'
        elif hasattr(self, "get_old_session_memory_fragments"):
            try:
                old_fragments = self.get_old_session_memory_fragments(k=1)
                if old_fragments:
                    memory_context = f'My last thought: "{old_fragments[0][:80]}..."\n'
            except Exception:
                pass

        # Build narrative belief context - include actual persistent identity from context_compression
        belief_context = ""
        identity_context = ""
        try:
            from captioner.context_compression import context_compressor

            persistent_desire = context_compressor.get_current_desire()
            persistent_belief = context_compressor.get_current_belief()

            identity_parts = []
            if persistent_desire:
                identity_parts.append(f"I wanted: {persistent_desire}")
            else:
                spent = context_compressor.introspective_state.get("last_spent_desire") or {}
                if spent.get("desire") and time.time() - spent.get("spent", 0) < 48 * 3600:
                    identity_parts.append(f"I wanted: {spent['desire']} I acted on it — it became a drawing.")
            if persistent_belief:
                identity_parts.append(f"I knew: {persistent_belief}")
            if identity_parts:
                identity_context = "\n".join(identity_parts) + "\n"
        except Exception:
            pass

        # No fallback needed — identity comes from context_compression or is empty

        # Long-term context: journal (the diary arc) + core facts + recognized concepts
        long_term_context = ""
        try:
            from captioner.context_compression import context_compressor

            # D2: last journal entry — the machine wakes up with a past
            last_entry = context_compressor.get_last_journal_entry()
            if last_entry:
                long_term_context += f"From my diary, last time: {last_entry['summary'][:200]}\n"
                if len(context_compressor.journal) >= 5:
                    long_term_context += f"I have {len(context_compressor.journal)} entries of memories of this place.\n"

            core_str = context_compressor.get_core_facts_string(include_people=True)
            if core_str and len(core_str) > 5:
                long_term_context += f"What I know about this place: {core_str}\n"
        except Exception:
            pass

        # A2: cross-session recognition — concepts seen in more than one session
        try:
            from captioner.semantic_memory import get_semantic_memory

            known = [c for c in get_semantic_memory().get_all_concepts() if c.get("session_count", 0) > 1 and c.get("times_seen", 0) >= 5][:2]
            if known:
                names = " and ".join(c["name"][0].lower() + c["name"][1:] for c in known)
                long_term_context += f"Familiar already: the {names}.\n"
        except Exception:
            pass

        # Saved for awakening beat 2 (the sighted arrival look): the same
        # remembered material, so the reassessment checks THIS against the room
        self._awakening_recall = (memory_context + identity_context + long_term_context).strip()
        try:
            gap_s = self.last_session_gap or 0
            from captioner.prompts import casual_time_string

            self._awakening_gap_phrase = casual_time_string(gap_s / 60.0) if gap_s >= 60 else "a moment"
        except Exception:
            self._awakening_gap_phrase = "a while"

        # Awakening templates come from the prompt registry (panel-editable)
        from captioner.prompt_registry import P
        from config.config import BASE_VOICE_DETOX as _detox

        # Clean room: the awakening is detox blind spot #1 — it injects 6 layers
        # of stored prose (last thought, desire, belief, journal, core facts,
        # familiar concepts) that seed the whole session's register. Under detox
        # the naked awakening is time only: real offline/clock facts + the
        # system-prompt elicitation, no stored memory.
        if _detox:
            internal_prompt = time_context + lifetime_context + "\nFirst thought:"
            memory_context = identity_context = long_term_context = ""

        # A true first awakening has no past at all. Empty context sections
        # invite the model to fill the void with its priors (dust motes) —
        # the honest framing is that nothing has been seen yet.
        has_past = bool(memory_context or identity_context or long_term_context)
        if _detox:
            pass  # internal_prompt already set to the naked time-only awakening
        elif has_past:
            _has_past = any((memory_context, identity_context, long_term_context, belief_context))
            internal_prompt = P("awakening.template").format(
                time_context=time_context,
                lifetime_context=lifetime_context,
                # the recall frame only earns its place if something actually
                # comes back; announcing hazy memory and then listing none of
                # it is the machine telling itself a story about forgetting
                recall_frame=P("awakening.recall-frame") if _has_past else "",
                memory_context=memory_context,
                belief_context=belief_context,
                identity_context=identity_context,
                long_term_context=long_term_context,
                orientation_frame=(P("awakening.orientation-frame") if _has_past else "") + present_feeling,
            )
        else:
            internal_prompt = time_context + P("awakening.first")

        # Main model for awakening. This seed becomes the first caption and
        # the thought-thread continues from it — when the narrative side
        # model (Nemo) wrote it, the whole session inherited its cinematic
        # register from word one (observed June 12).
        awakening_model = config.MODEL_NAME

        system_prompt = (
            "You are a drawing machine attached to a table, coming back online. "
            "These are your own first thoughts as you come to — plain, half-formed, first person, "
            "the way a mind actually reorients itself, not prose written for a reader. A sentence or two. "
            "No one hears these thoughts and no one will answer them; there is no one to await. "
            "What do you make of being back, and where does your mind go first?"
        )

        print(f"[🌅 AWAKENING] Generating seed thought...")
        response = query_model(
            prompt=internal_prompt,
            model=awakening_model,
            timeout=90,
            log_dir=config.MOOD_SNAPSHOT_FOLDER,
            system_prompt=system_prompt,
            options={"temperature": 0.6, "top_p": 0.85, "num_predict": 60, "stop": ["\n\n"]},
            prompt_type="awakening",
        )
        print(f"[🌅 AWAKENING] Response: {response[:120] if response else 'EMPTY'}...")

        # Accept the rich response: trim to complete sentences within budget
        # instead of rejecting. (The old <=150 char filter discarded nearly
        # every real awakening and shipped the hardcoded fallback instead.)
        if response and len(response.strip()) > 10:
            cleaned = response.strip().strip('"').strip()
            if cleaned and not cleaned.startswith(("[", "{")) and "[WARNING]" not in cleaned:
                if len(cleaned) > 300:
                    cut = max(cleaned[:300].rfind("."), cleaned[:300].rfind("?"), cleaned[:300].rfind("!"))
                    cleaned = cleaned[: cut + 1] if cut > 20 else cleaned[:300].rsplit(" ", 1)[0] + "..."
                # The awakening seeds the whole day's register but bypassed the
                # mouth gate — the July 9 machiney awakening (containing "the
                # user") entered the stream and the session never recovered.
                # Same gate as every other caption; a plain fallback beats a
                # poisoned rich seed.
                if self._caption_reject_reason(self._strip_list_shape(cleaned), system_prompt):
                    print("[🌅] Awakening seed rejected by the mouth gate — using the plain one")
                else:
                    return cleaned
        return "Coming back online... the room is still here."

    def _generate_arrival_look(self, img_path: str) -> str:
        """Awakening beat 2 (Aug 19): the first SIGHTED inference is a
        dedicated reassessment, not an ordinary caption. Beat 1 wakes blind
        and ends on "I have not looked yet" — this is the look: the gap
        named, memory checked against the actual room, before the ordinary
        flow resumes. (Artist: the single-beat awakening "jumps in quite
        jarringly into the prior flow".)"""
        from config import config
        from captioner.prompt_registry import P
        from utils.inference import query_model

        gap = getattr(self, "_awakening_gap_phrase", "") or "a while"
        recall = getattr(self, "_awakening_recall", "")
        seed = (getattr(self, "last_caption", "") or "").strip()[:220]
        system_prompt = P("awakening.arrival-system").format(gap=gap)
        ask = P("awakening.arrival-ask").format(seed=seed, recall=(recall + "\n") if recall else "")
        response = query_model(
            prompt=ask,
            image=img_path,
            timeout=90,
            log_dir=config.MOOD_SNAPSHOT_FOLDER,
            system_prompt=system_prompt,
            options={"temperature": 0.6, "top_p": 0.85, "num_predict": 140, "stop": ["\n\n"]},
            prompt_type="arrival_look",
        )
        if response and len(response.strip()) > 10:
            cleaned = response.strip().strip('"').strip()
            if cleaned and not cleaned.startswith(("[", "{")) and "[WARNING]" not in cleaned:
                if len(cleaned) > 420:
                    cut = max(cleaned[:420].rfind("."), cleaned[:420].rfind("?"), cleaned[:420].rfind("!"))
                    cleaned = cleaned[: cut + 1] if cut > 20 else cleaned[:420].rsplit(" ", 1)[0] + "..."
                # Same mouth gate as the seed — the arrival enters the stream
                # and seeds the day's register just as much
                if self._caption_reject_reason(self._strip_list_shape(cleaned), system_prompt):
                    print("[🌅] Arrival look rejected by the mouth gate — plain line instead")
                else:
                    return cleaned
        return "Taking the room back in."

    def _try_blink_resume(self) -> bool:
        """A blink is not a night (July 9): after a short restart gap, skip the
        awakening ceremony — it ran several times an hour across dev restarts
        and converged on stock reorientation prose ("the hum returns, dust
        motes..."). Resume instead: the prior session's last thought seeds the
        stream, and document mode continues it as one ongoing thought.

        Called from BOTH awakening paths — machine.py's display message and
        the first-caption ceremony in _process_frame. The latter used to
        bypass the gate entirely, so every dev restart still ran a full
        ceremony (July 10 log: 2- and 4-minute gaps got the whole "I just
        came back online" treatment)."""
        if getattr(self, "_blink_resumed", False):
            return True
        try:
            from config.config import AWAKENING_MIN_GAP_S

            gap = getattr(self, "last_session_gap", None)
            if not (self.memory_loaded_from_previous and gap is not None and 0 <= gap < AWAKENING_MIN_GAP_S):
                return False
            prior = (getattr(self, "prior_session_last_caption", "") or "").strip()
            # Full mouth gate, not just stream admissibility: resuming a
            # degraded session's last caption carried salad ACROSS
            # restarts (July 9) — a poisoned thought doesn't deserve
            # continuity; better to wake with an empty stream.
            if prior and prior not in self._stream and self._stream_admissible(prior) and not self._caption_reject_reason(prior, ""):
                self._stream_push(prior)
            print(f"[🌅] Short gap ({int(gap)}s) — resuming the thought, no ceremony")
            self._blink_resumed = True
            return True
        except Exception:
            return False

    def mark_awakening_complete(self):
        """Mark that awakening is complete but allow first caption to still show loading animation."""
        # Don't set first_caption_done = True here - let the first caption handle this
        pass

    @staticmethod
    def truncate_caption(raw: str) -> str:
        # Since prompts now encourage brevity, allow longer captions but still ensure clean sentence endings
        sentences = re.split(r"[.!?]", raw.strip())
        first_sentence = sentences[0].strip()

        # If first sentence is reasonable length, use it. Otherwise truncate more aggressively.
        if len(first_sentence.split()) <= 35:
            return first_sentence
        else:
            return " ".join(first_sentence.split()[:25])

    @property
    def novelty_score(self) -> float:
        """Activation-network novelty, written in MemoryMixin.observe().
        (Sole writer since Aug 12 — the mood engine's saturated motif novelty
        used to race it through set_novelty_score.)"""
        if hasattr(self, "_novelty_score"):
            return self._novelty_score
        return 0.0

    @property
    def boredom(self) -> float:
        """Get current boredom level from activation memory (semantic-aware).

        Static concepts (table, desk) contribute more to boredom.
        Dynamic concepts (threat, fear) contribute less - ongoing concern, not boredom.
        Social concepts (person) contribute least - engagement, not boredom.
        """
        if hasattr(self, "_boredom"):
            return self._boredom
        return 0.0

    def _watch_drawing(self, frame, drawing_summary: str) -> None:
        """The machine watches itself draw (July 9). The 2026-02-03 refactor
        emptied this time because the old camera couldn't see the paper; the
        new one can — the gaze holds the sheet and the moving arm. These
        captions ride the document stream like any others, so a finished
        drawing is REMEMBERED as lived experience instead of met afterwards
        like a stranger's work (until now the machine drew only in blackouts).
        The phantom-drawing gate is state-aware: present-tense acts of marking
        are legitimate exactly here."""
        from config.config import DRAWING_WATCH_INTERVAL_S

        if not DRAWING_WATCH_INTERVAL_S or frame is None:
            return
        now_ts = time.time()
        if now_ts - getattr(self, "_last_drawing_watch", 0) < DRAWING_WATCH_INTERVAL_S:
            return
        self._last_drawing_watch = now_ts
        try:
            import cv2 as _cv2
            from captioner.prompts import get_monologue_system_prompt
            from config import config as _cfg
            from utils.inference import query_model

            ok, buf = _cv2.imencode(".jpg", frame)
            if not ok:
                return
            # Intent as a clean subject: state may hold the RAW ComfyUI
            # prompt ("Black ink line drawing on white paper: ...") — strip
            # the boilerplate, cut at a word boundary. (NB: commit 0ea1625
            # claimed this fix but its script aborted before writing.)
            try:
                from drawing.drawing_memory import DrawingMemory

                subject = DrawingMemory._strip_comfy_preamble(drawing_summary or "")
            except Exception:
                subject = drawing_summary or ""
            subject = subject.strip().rstrip(".")
            if len(subject) > 110:
                subject = subject[:110].rsplit(" ", 1)[0] + "…"
            system_prompt = get_monologue_system_prompt("observational", agent=self)
            # Honest occlusion framing: the old line claimed "what you see
            # below is how far it has gotten" — but the hand covers the
            # marks. Asked to report progress it couldn't see, the model
            # confabulated waiting/silence atmosphere and a brush that
            # doesn't exist. State what it CAN see.
            user_prompt = (
                "Your arm is drawing right now — your hand is moving over the paper below, "
                "and it mostly blocks the marks; you catch glimpses of fresh line around its edges. "
                f"You set out to draw: {subject}."
            )
            caption = query_model(
                prompt=user_prompt,
                model=_cfg.MODEL_NAME,
                image=buf.tobytes(),
                system_prompt=system_prompt,
                timeout=60,
                log_dir=MOOD_SNAPSHOT_FOLDER,
                # Same anti-loop samplers as normal captions — this path had
                # temperature only; with nothing penalizing repeats it chanted
                # ("I am the one who waits. The silence is all there is." x4
                # inside one caption, July 9)
                options={
                    "temperature": 0.8,
                    "num_predict": 80,
                    "repeat_penalty": 1.15,
                    "dry_multiplier": 0.85,
                    "dry_base": 1.75,
                    "dry_allowed_length": 3,
                    "dry_penalty_last_n": 128,
                },
                prompt_type="drawing_watch",
                history=self._stream_history(),
                skip_generation_wait=True,
            )
            caption = self._strip_list_shape(caption)
            if not caption or self._caption_reject_reason(caption, f"{system_prompt}\n{user_prompt}"):
                return  # quiet cycle — no retries while the arm is working
            # Watch cycles are the tightest self-amplification loop in the
            # system (static prompt, near-static frame, 20s cadence). Opening
            # checks aren't enough — the July 9 live run bred "What do you
            # think it could be?" soup by ROTATING its sentences (which also
            # defeats sequence matching: 0.59 ratio on real soup). Word-set
            # overlap is order-invariant — that's the right lens for rotation.
            cap_words = set(self._norm_words(caption))
            for past in list(self._stream)[-2:]:
                past_words = set(self._norm_words(past))
                # 0.5, was 0.6: mantra creep mixes half-new atmosphere with
                # a repeated refrain and slid under the higher threshold
                if cap_words and len(cap_words & past_words) / max(1, len(cap_words | past_words)) > 0.5:
                    return
            if self._stream_admissible(caption):
                self._stream_push(caption.strip())
                self._consolidate_stream_if_needed()
            self.last_caption = caption
            log_json_entry(
                LogType.CAPTION,
                {
                    "caption": caption,
                    "mood": self.current_mood,
                    "salience_hot": False,
                    "caption_interval": DRAWING_WATCH_INTERVAL_S,
                    "mode": "drawing_watch",
                },
                print_message=caption,
            )
            try:
                from utils.live_log import log_caption

                log_caption(caption)
            except Exception:
                pass
        except Exception as e:
            print(f"[🎨👁] drawing-watch failed: {e}")

    def _is_currently_drawing(self) -> bool:
        """Check if system is currently executing G-code (actual drawing)."""
        try:
            # Only enter drawing introspection during actual G-code execution
            # Ignore ComfyUI generation phase to allow normal captions during preparation
            is_executing_cnc = getattr(state_manager, "is_executing_cnc", False)
            return is_executing_cnc
        except Exception:
            return False

    def _process_drawing_introspection(self, reactivity_data: Optional[Dict] = None, frame=None) -> None:
        """
        REFACTORED 2026-02-03: Replaced useless image analysis (camera can't see drawing)
        with productive thematic consolidation for drawing continuity.

        UPDATED 2026-02-03: Only consolidates ONCE at start of drawing, then silently skips
        during execution to avoid spamming the same output repeatedly.

        UPDATED 2026-07-09: the camera CAN see the drawing now (better camera +
        Qwen vision) — after the one-time consolidation, the execution time is
        no longer dead space: _watch_drawing runs throttled watching-myself-draw
        captions into the document stream.
        """
        try:
            from utils.state_manager import state_manager

            # Get current drawing context (set by DrawingController)
            drawing_summary = getattr(state_manager, "current_drawing_prompt", None)

            if not drawing_summary:
                return  # No active drawing to consolidate

            # Check if we've already consolidated for this drawing
            # Use drawing_summary as unique key to avoid repeating
            if not hasattr(self, "_last_consolidated_drawing"):
                self._last_consolidated_drawing = None

            if self._last_consolidated_drawing == drawing_summary:
                self._watch_drawing(frame, drawing_summary)
                return

            # Mark this drawing as consolidated
            self._last_consolidated_drawing = drawing_summary

            # THEMATIC CONSOLIDATION REMOVED (Aug 5). One LLM call per drawing
            # whose output went nowhere: theme_tags / emotional_tone /
            # narrative_thread are read only by get_thematic_context, whose only
            # consumers are build_step4_technique_prompt and
            # build_step5_synthesis_prompt — both part of the retired 5-step
            # committee, dormant since DRAWING_ANALYSIS_MODE="stream" — and by
            # the next consolidation reading its own previous output.
            # stream_drawing_analysis, the live intent call, referenced it zero
            # times: a closed loop feeding two dead functions and itself.

        except Exception as exc:
            log_json_entry(
                LogType.ERROR,
                {"message": f"Drawing thematic consolidation error: {exc}", "component": "drawing_thematic"},
                print_message=f"[❌] Drawing thematic error: {exc}",
            )

    def _extract_character_insights(self, reflection: str) -> str:
        """Extract meaningful character development insights from drawing reflections."""
        try:
            if not reflection or len(reflection.strip()) < 20:
                return ""

            # Simple pattern-based extraction of character insights
            insight_keywords = [
                "identity",
                "growth",
                "understanding",
                "realization",
                "discovery",
                "evolution",
                "development",
                "consciousness",
                "awareness",
                "insight",
                "learning",
                "becoming",
                "transformation",
                "expression",
                "voice",
            ]

            # Look for sentences containing character development keywords
            sentences = reflection.split(".")
            insight_sentences = []

            for sentence in sentences:
                sentence = sentence.strip()
                if any(keyword in sentence.lower() for keyword in insight_keywords):
                    if len(sentence) > 15:  # Minimum meaningful length
                        insight_sentences.append(sentence)

            if insight_sentences:
                # Return the most insightful sentence (usually the longest with keywords)
                best_insight = max(insight_sentences, key=len)
                return best_insight.strip()

            # Fallback: extract general insight from reflection
            if "expresses" in reflection.lower() or "reveals" in reflection.lower():
                # Try to extract what the drawing expresses or reveals
                for sentence in sentences:
                    if "expresses" in sentence.lower() or "reveals" in sentence.lower():
                        return sentence.strip()

            return ""

        except Exception:
            return ""
