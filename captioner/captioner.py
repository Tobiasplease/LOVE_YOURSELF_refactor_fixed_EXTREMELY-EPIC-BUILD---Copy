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
    CAPTION_QUIET_AFTER,
    CLEAN_LLM_OUTPUT,
    DRAWING_INTERVAL,
    DRAWING_STARTUP_DELAY,
    MOOD_SNAPSHOT_FOLDER,
    OLLAMA_SHOW_PROGRESS,
)
from drawing.drawing import DrawingController
from event_logging.event_logger import log_json_entry
from event_logging.log_type import LogType
from event_logging.run_manager import get_run_image_path
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
    gaze_pattern = r'\*[^*]*(?:' + '|'.join(gaze_verbs) + r')[^*]*\*\s*'
    cleaned = re.sub(gaze_pattern, '', caption, flags=re.IGNORECASE)

    # Remove LOOK: lines and inline LOOK directives (legacy format)
    lines = cleaned.strip().split('\n')
    clean_lines = []
    for line in lines:
        line_lower = line.lower().strip()
        # Skip LOOK: lines (including typos like LOOKE, LOook, LOOk)
        if re.match(r'^loo+k[e]?\s*:', line_lower) or re.match(r'^loo+k[e]?\s+(left|right|up|down|ahead|forward)$', line_lower):
            continue
        # Skip arrow notation lines
        if '→ look' in line_lower:
            continue
        # Skip variety directive markers that leaked into output
        if line.strip().startswith('[⚠️') or line.strip().startswith('[NOTICE') or line.strip().startswith('[SHIFT'):
            continue
        # Remove inline (LOOK: direction) or (LOook direction) patterns
        line = re.sub(r'\s*\(loo+k[e]?\s*:?\s*\w+\)\s*', '', line, flags=re.IGNORECASE)
        # Remove trailing "LOook AHEAD" style suffixes (various typos)
        line = re.sub(r'\s*\.{0,3}\s*loo+k[e]?\s+(?:left|right|up|down|ahead|forward)\s*\.{0,3}\s*$', '', line, flags=re.IGNORECASE)
        # Remove mid-sentence "...LOook" or "...LOok" trailing garbage
        line = re.sub(r'\.{2,}\s*loo+k[e]?\s*$', '', line, flags=re.IGNORECASE)
        if line.strip():
            clean_lines.append(line)

    cleaned = '\n'.join(clean_lines).strip()

    # Filter out direction-only responses (these aren't real captions)
    direction_words = {"left", "right", "up", "down", "ahead", "person", "up ahead", "a person"}
    if cleaned.lower().rstrip('.,!?') in direction_words:
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
            if hasattr(self, 'self_model') and 'environmental_certainty' in self.self_model:
                # Increase certainty as we get more compression-based understanding
                current_certainty = self.self_model.get('environmental_certainty', 0.0)
                self.self_model['environmental_certainty'] = min(1.0, current_certainty + 0.1)

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
        self.current_mood_vector: Tuple[float, float, float] = (0.0, 0.0, 0.5)  # valence, arousal, clarity
        self.current_emotion_state: str = "calm_observant"  # hand controller emotion state
        self.emotional_journey: List[str] = []  # track emotional evolution over time
        self.last_caption: str = ""
        # self.current_motifs_from_mood removed — motif tracking replaced by ChromaDB concepts

        # Deduplication system to prevent duplicate prints
        self.recent_captions: List[Tuple[str, float]] = []  # (caption, timestamp)
        self._last_perception: str = ""  # Last perception (kept for recent_captions tuples + early-session fallback)
        self._drawing_intentions: List[str] = []  # Accumulated drawing-related musings

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
        self._presence_seen_now: bool = False

        # The stream (CoT-style continuity): recent captions ride as the
        # model's own assistant turns. Gated by _stream_admissible.
        from config.config import STREAM_WINDOW
        self._stream: Deque[str] = deque(maxlen=max(STREAM_WINDOW, 0))
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
        mood_vector: Optional[Tuple[float, float, float]] = None,
        emotion_state: Optional[str] = None,
        reactivity_data: Optional[Dict] = None,
    ) -> None:
        if frame is not None:
            if mood is not None:
                self.current_mood = mood
            if mood_vector is not None:
                self.current_mood_vector = mood_vector
            if emotion_state is not None:
                # Track emotional journey over time using meaningful descriptions instead of crude categories
                if emotion_state != self.current_emotion_state:
                    # Generate a meaningful description based on 3D mood vector changes
                    valence, arousal, clarity = self.current_mood_vector
                    emotional_description = self._get_emotional_description(valence, arousal, clarity)
                    self.emotional_journey.append(emotional_description)
                    if len(self.emotional_journey) > 10:  # Keep last 10 emotional states
                        self.emotional_journey.pop(0)
                self.current_emotion_state = emotion_state
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
                        self._process_drawing_introspection(reactivity_data)
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

            face_frames = sum(1 for f in recent_meta if f.get("detection", {}).get("face"))
            info["eye_contact"] = bool(face_frames > len(recent_meta) * 0.4)

        # Update the sticky presence belief from live detection. "Seen now" is
        # any current evidence of a person — world-angle hit, eye contact, or an
        # active gaze lock. The belief persists through gaps so a glance away
        # doesn't read as a departure, and a re-detection doesn't read as a new
        # arrival. Only the OFF->ON edge is a genuine arrival.
        from config.config import SALIENCE_MOTION_RESIDUAL, PRESENCE_BELIEF_DECAY_SECONDS
        now = time.time()
        gaze_engaged = False
        try:
            from vision.gaze import get_gaze_state
            gs = get_gaze_state()
            if isinstance(gs, dict):
                gaze_engaged = gs.get("state") in ("tracking", "aware", "grace")
        except Exception:
            pass
        seen_now = bool(info["person_present_in_window"] or info["eye_contact"] or gaze_engaged)

        arrival = False
        if seen_now:
            self._presence_last_seen = now
            if not self._presence_believed:
                self._presence_believed = True
                self._presence_since = now
                arrival = True  # OFF->ON edge — the only genuine arrival
        elif self._presence_believed and (now - self._presence_last_seen) > PRESENCE_BELIEF_DECAY_SECONDS:
            self._presence_believed = False  # sustained absence — they really left
        self._presence_seen_now = seen_now
        info["presence_believed"] = self._presence_believed
        info["presence_seen_now"] = seen_now

        # Eye contact is salient at its onset — someone holding your gaze for
        # ten minutes is presence, not an event. The sustained state still
        # reaches the prompt (prompts.py eye-contact line) — it used to live
        # only in the video path, which face-tracking saccades always skip,
        # so someone staring at the machine went entirely unmentioned
        eye_onset = info["eye_contact"] and not self._prev_eye_contact
        self._prev_eye_contact = info["eye_contact"]
        self._eye_contact_now = info["eye_contact"]

        self._last_scene_motion = info["scene_motion"]
        # Interiority is stripped ONLY by discrete events or genuinely large
        # motion — NOT by a person merely present and shifting. Micro-motion
        # and YOLO flicker (person_moved / count_changed) keep scene_motion
        # True for video framing, but no longer strip the prompt: the machine
        # must be free to think about itself and its work while someone is
        # quietly in the room (north-star principles 6 + 7).
        strong_motion = info["max_residual"] > SALIENCE_MOTION_RESIDUAL
        self._salience_hot = bool(eye_onset or arrival or strong_motion)
        if self._salience_hot:
            self._last_salience_time = time.time()
        info["salience_hot"] = self._salience_hot

        # Salience strips the prompt to the present — but the present must
        # then SAY what just happened, or the model fills the vacuum with
        # atmosphere instead of reacting. Eye-contact onset is the one event
        # the situational line doesn't already carry; the arrival is now stated
        # by the presence line itself ("Someone's just come in"), so naming it
        # again here would be a duplicate (reads as emphasis, locks register).
        event = None
        if eye_onset:
            event = "They just looked straight at you."
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
    )

    @classmethod
    def _stream_admissible(cls, text: str) -> bool:
        """Admission gate for the stream window (guard at storage, not mouth)."""
        t = (text or "").strip().lower()
        if len(t) < 8:
            return False
        if any(m in t for m in cls._STREAM_META_MARKERS):
            return False
        if t.count("*") >= 2 or t.startswith(("- ", "* ", "#")):
            return False  # markdown scaffolding breeds in-stream too
        return True

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

        # A long silence breaks the thought — the stream restarts rather than
        # pretending continuity across a gap (would-it-lie applies to time too)
        from config.config import STREAM_BREAK_SECONDS
        if self.last_caption_time and now - self.last_caption_time > STREAM_BREAK_SECONDS:
            self._stream.clear()

        # Store reactivity data for subconscious layer access
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
            if OLLAMA_SHOW_PROGRESS:
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

            if not self.first_caption_done or not self.session_awakening_done:
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
            else:
                log_json_entry(
                    LogType.DEBUG,
                    {"message": "Requesting new caption", "action": "caption_request", "image_path": img_path},
                    print_message=f"[🐞] Requesting new caption for {img_path}",
                )
                previous_caption = getattr(self, "last_caption", "")
                matched_concepts = []  # Will be populated by SemanticMemory concept matching

                # Check if it's time for memory mode (every 240 seconds / 4 minutes)
                time_since_memory = now - self.last_memory_mode_time
                is_memory_mode_time = time_since_memory > 240  # 4 minutes

                try:
                    if is_memory_mode_time:
                        # Memory mode: pull actual caption text from long-term memory
                        from captioner.prompts import build_memory_mode_prompt
                        from captioner.prompts import get_monologue_system_prompt
                        memory_prompt, caption_mode = build_memory_mode_prompt(self)
                        memory_system = get_monologue_system_prompt("introspective", agent=self) + " A memory is surfacing. Think about something you remember — past tense, framed as memory."
                        caption = self.model._call_ollama(
                            memory_prompt,
                            image_path=None,  # Memory mode doesn't use current image
                            system_prompt=memory_system,
                            model_options=self.model.prompt_interface._get_base_model_options(),
                            prompt_type="memory"
                        )
                        self.last_memory_mode_time = now
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
                        OLLAMA_MODEL = _cfg.OLLAMA_MODEL
                        VIDEO_MODE_ENABLED = _cfg.VIDEO_MODE_ENABLED
                        from utils.inference import query_model, query_model_video

                        # Salience first — it decides how interior this caption gets
                        scene = self._assess_scene()

                        user_prompt, caption_mode = build_simple_caption_prompt(
                            self,
                            person_present=person_present,
                        )

                        system_prompt = get_monologue_system_prompt(caption_mode, agent=self)

                        backend_tag = "LLAMA" if _cfg.INFERENCE_BACKEND == "llama_server" else "OLLAMA"
                        print(f"\n{'='*80}\n[{backend_tag}] {OLLAMA_MODEL} ({caption_mode})\n{'='*80}")
                        print(f"SYSTEM: {system_prompt}\n")
                        print(f"USER:\n{user_prompt}\n")
                        print(f"{'='*80}\n")

                        import random as _random

                        # Bored = sparser, flatter thoughts; engaged = more room.
                        # 0.6/0.7 (down from 0.85/0.9): Qwen blooms into purple
                        # fiction at higher temps — plainness via sampling, not
                        # style fences (north-star principle 7).
                        _is_bored = self.boredom > 0.7
                        gen_options = {
                            "temperature": 0.6 if _is_bored else 0.7,
                            "top_p": 0.85,
                            "repeat_penalty": 1.15,
                            "num_predict": 45 if _is_bored else 80,
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
                        use_video = (
                            VIDEO_MODE_ENABLED
                            and _cfg.INFERENCE_BACKEND == "llama_server"
                            and bool(recent_meta)
                            and scene["max_diff"] > MOTION_THRESHOLD
                        )

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
                            steady_meta = [f for f in recent_meta
                                           if not f.get("detection", {}).get("ego_motion")]
                            if scene_motion:
                                send_meta = recent_meta
                            elif len(steady_meta) >= 3:
                                send_meta = steady_meta
                            else:
                                use_video = False
                                print(f"[VIDEO] Skipped: still room, only {len(steady_meta)}/{len(recent_meta)} steady frames (camera was moving) — sending still image")

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

                            print(f"[VIDEO] {total}/{len(recent_meta)} frames over {duration:.1f}s, scene_motion={scene_motion}, residual={scene['max_residual']:.3f}, ego={ego_count}, face={face_frames}/{total}, person={person_frames}/{total}")
                            # Clean-room: the "You're seeing the last N seconds" wrapper is
                            # camera-narration framing (voice-analysis #1 tone driver), so it
                            # would confound the naked-voice test — drop it under detox and let
                            # the frames speak for themselves.
                            from config.config import BASE_VOICE_DETOX as _detox
                            if _detox:
                                video_prompt = user_prompt
                            else:
                                video_prompt = f"You're seeing the last {duration:.0f} seconds.{motion_line}\n{user_prompt}"
                            caption = query_model_video(
                                prompt=video_prompt,
                                frames=video_frames,
                                fps=2.0,
                                system_prompt=system_prompt,
                                options=gen_options,
                                timeout=60,
                                history=list(self._stream),
                            )
                        else:
                            # Eye contact: send the face, not a wide shot where it
                            # is a hundred-pixel smudge — the VLM can read an
                            # expression when it's actually given the pixels
                            send_path = img_path
                            if getattr(self, "_eye_contact_now", False) and reactivity_data:
                                face_box = reactivity_data.get("face_box")
                                if face_box is not None:
                                    crop_path = self._write_face_context_crop(frame, face_box, img_path)
                                    if crop_path:
                                        send_path = crop_path
                            caption = query_model(
                                prompt=user_prompt,
                                model=OLLAMA_MODEL,
                                image=send_path,
                                system_prompt=system_prompt,
                                timeout=60,
                                log_dir=MOOD_SNAPSHOT_FOLDER,
                                options=gen_options,
                                prompt_type="caption",
                                history=list(self._stream),
                            )

                        # Match output against ChromaDB concepts (replaces perception-based matching)
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
                            for mc in (matched_concepts or []):
                                sp = mc.get("spatial_pan")
                                st = mc.get("spatial_tilt")
                                if sp or st:
                                    nudge_toward_concept(pan_zone=sp, tilt_zone=st)
                                    break
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

            # More specific error handling to avoid unnecessary "Vision unclear" fallbacks
            if "No image found" in str(e) or "does not exist" in str(e):
                # File system timing issue - retry once after short delay
                time.sleep(0.5)
                try:
                    if hasattr(self, "model") and img_path and os.path.exists(img_path):
                        caption, caption_mode = self.model.caption_image(img_path, flowing=True, first_time=False, person_present=person_present)
                    else:
                        caption = "Awakening... vision initializing..."
                        caption_mode = "awakening"
                except Exception:
                    caption = "[WARNING] Vision unavailable"
                    caption_mode = "error"
            else:
                caption = "[WARNING] Vision unavailable"
                caption_mode = "error"

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
            caption = caption[:_last_punct + 1]

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
                is_executing_cnc = getattr(state_manager, 'is_executing_cnc', False)
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
            mood_vector=self.current_mood_vector,
            emotion_state=self.current_emotion_state,
            matched_concepts=matched_concepts,
        )
        self.last_caption = caption  # already trimmed to complete sentence above

        # Admit into the stream window (the model's own visible turns) —
        # meta/markdown slips are displayed and logged but never propagate
        if caption and self._stream_admissible(caption):
            self._stream.append(caption.strip())

        # Track recent captions for continuity thread (used by flowing thread)
        # Store as (caption, timestamp, mode, perception) for interleaved see/think display
        if caption and caption.strip():
            last_perception = getattr(self, "_last_perception", "") or ""
            self.recent_captions.append((caption.strip(), now, caption_mode, last_perception))
            if len(self.recent_captions) > 20:  # Keep last 20
                self.recent_captions = self.recent_captions[-20:]

        # Detect and store drawing-relevant thoughts from monologue output.
        # These feed into Step 3 (communication intent) when drawing triggers.
        # Captures both explicit drawing talk AND strong inner-life statements
        # (desires, envies, frustrations, imagery) — the real artistic fuel.
        if caption and caption.strip():
            cap_lower = caption.lower()
            # Explicit drawing references
            drawing_keywords = ["draw", "sketch", "capture", "next piece", "should paint",
                                "want to draw", "would look good", "inspire", "my next",
                                "on paper", "with my arm", "lines and", "bold strokes",
                                "trace", "ink", "charcoal", "canvas"]
            # Strong experiential statements that inform artistic intent
            experiential_keywords = ["i envy", "i crave", "i yearn", "i long for",
                                     "i imagine", "i wish i could", "trapped",
                                     "if only", "i feel an urge", "reminds me of",
                                     "like a", "as if", "void", "emptiness"]
            if any(kw in cap_lower for kw in drawing_keywords + experiential_keywords):
                if not hasattr(self, "_drawing_intentions"):
                    self._drawing_intentions = []
                self._drawing_intentions.append(caption.strip()[:150])
                if len(self._drawing_intentions) > 10:
                    self._drawing_intentions = self._drawing_intentions[-10:]
                print(f"[🎨 INTENT] Stored drawing intention: {caption.strip()[:80]}")

        # Now update the timestamp since we have a new caption
        self.last_caption_time = now

        # Add caption to context compression system (environmental change detection remains disabled)
        try:
            if context_compressor and caption and caption.strip():
                context_compressor.add_caption(caption, time.time(), img_path)
        except Exception as e:
            print(f"[CAPTIONER] Context compression failed: {e}")

        # Caption already observed via the primary observe() call above

        # Long-form reflection happens in its own thread now (captioner/reflection.py)

        # Check drawing interval - should trigger check every DRAWING_INTERVAL
        time_since_last_check = now - getattr(self, 'last_drawing_check_time', 0)
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
            print(f"[🎨 CHECK] Blocked: cooldown ({cooldown_remaining:.0f}s remaining)")
            return

        # Pipeline check before state evaluation
        try:
            is_generating = getattr(state_manager, "is_generating_drawing", False)
            is_executing = getattr(state_manager, "is_executing_cnc", False)
            if is_generating or is_executing:
                return
        except Exception:
            pass

        # Never step away to conceive a drawing mid-moment: the 5-step
        # analysis monopolizes the inference server for minutes (caption
        # stalls of 20-83s observed June 12) — exactly when reactivity
        # matters most. Quiet stretches are when drawings get conceived.
        if self._salience_hot:
            print("[🎨 CHECK] Deferred: something is happening — staying with the room")
            return

        # STATE-MOTIVATED EVALUATION
        # Get current system state for decision
        if not CLEAN_LLM_OUTPUT:
            print(f"\n[🎨 STATE EVALUATION]")
            print(f"  Current mood: {self.current_mood:.3f}")
            print(f"  Current novelty: {self.novelty_score:.3f}")
            print(f"  Current boredom: {self.boredom:.3f}")

        # Evaluate whether to draw based on internal state
        should_draw = self.drawing.should_draw(
            mood=self.current_mood,
            novelty=self.novelty_score,
            boredom=self.boredom,
            reflection=getattr(self, 'last_reflection', None)
        )

        if not should_draw:
            print(f"[🎨 CHECK] State evaluation: NOT motivated (mood={self.current_mood:.2f}, novelty={self.novelty_score:.2f}, boredom={self.boredom:.2f})")
            return

        # Update check time ONLY after state motivation passes
        # This allows retry on next cycle if not motivated yet
        self.last_drawing_check_time = now
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
        # Drawing intentions passed directly to Step 3 (communication intent) via pipeline
        drawing_intentions_list = []
        if hasattr(self, "_drawing_intentions") and self._drawing_intentions:
            drawing_intentions_list = self._drawing_intentions[-5:]
            print(f"[🎨] {len(drawing_intentions_list)} drawing intentions available for Step 3")
        extra_context = f"{self.last_caption}\n\n{memory_context}\n\n{reflection_context}"
        if not CLEAN_LLM_OUTPUT:
            print(f"[DEBUG] Step 7: Context built, starting drawing generation...")

        # No loading animation for drawing — the 5-step pipeline prints its own progress
        loading_stop = threading.Event()
        loading_thread = None

        try:
            if not CLEAN_LLM_OUTPUT:
                print(f"[DEBUG] Step 8: About to call generate_drawing_prompt...")
            prompt = self.model.generate_drawing_prompt(extra=extra_context, image_path=img_path, drawing_intentions=drawing_intentions_list)
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
                dm = get_drawing_memory()
                dm.add_drawing(
                    prompt=prompt,
                    compressed_summary="",  # Will be set by LLM reflection after actual execution
                    emotional_tone=getattr(self, "current_emotion_state", "neutral"),
                    comfy_prompt=prompt,
                    completed=False,  # Will be updated to True if GRBL finishes
                )
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

    def describe_current_mood(self) -> str:
        """Rich emotional description using 3D mood state and temporal context."""
        valence, arousal, clarity = self.current_mood_vector

        # Generate rich mood description from 3D vector instead of crude categories
        if valence > 0.3 and arousal > 0.3:
            base_mood = "I feel engaged and positively energized by what I'm experiencing"
        elif valence > 0.3 and arousal < -0.3:
            base_mood = "I'm in a content and peaceful state, finding satisfaction in my observations"
        elif valence < -0.3 and arousal > 0.3:
            base_mood = "I feel restless and somewhat troubled by what I'm witnessing"
        elif valence < -0.3 and arousal < -0.3:
            base_mood = "I'm in a subdued and melancholic state, observing with quiet concern"
        elif arousal > 0.4:
            base_mood = "I feel alert and attentive, with heightened awareness"
        elif arousal < -0.4:
            base_mood = "I'm in a calm and contemplative state"
        else:
            base_mood = "I'm experiencing a balanced emotional state, neither particularly excited nor subdued"

        # Add 3D mood nuances
        valence_note = ""
        if valence > 0.4:
            valence_note = ", finding contentment in what I observe"
        elif valence < -0.4:
            valence_note = ", feeling somewhat troubled by what I see"

        arousal_note = ""
        if arousal > 0.4:
            arousal_note = ", with an energetic intensity"
        elif arousal < -0.4:
            arousal_note = ", in a deeply calm state"

        clarity_note = ""
        if clarity > 0.4:
            clarity_note = ", with clear understanding"
        elif clarity < -0.4:
            clarity_note = ", feeling somewhat confused"

        # Add temporal emotional context
        journey_note = ""
        if len(self.emotional_journey) >= 3:
            recent_states = self.emotional_journey[-3:]
            if len(set(recent_states)) == 1:
                journey_note = f". I've been consistently {self.current_emotion_state} lately"
            else:
                journey_note = f". My emotions have shifted: {' → '.join(recent_states)}"

        return f"{base_mood}{valence_note}{arousal_note}{clarity_note}{journey_note}."

    def _get_emotional_description(self, valence: float, arousal: float, clarity: float) -> str:
        """Generate a meaningful emotional description from 3D mood vector."""
        if valence > 0.4 and arousal > 0.4:
            return "energetically positive"
        elif valence > 0.4 and arousal < -0.4:
            return "contentedly calm"
        elif valence < -0.4 and arousal > 0.4:
            return "restlessly troubled"
        elif valence < -0.4 and arousal < -0.4:
            return "quietly melancholic"
        elif arousal > 0.5:
            return "highly alert"
        elif arousal < -0.5:
            return "deeply calm"
        elif valence > 0.3:
            return "mildly positive"
        elif valence < -0.3:
            return "somewhat troubled"
        else:
            return "emotionally neutral"

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
        print(f"[🌅 AWAKENING] prior_session_last_caption: {getattr(self, 'prior_session_last_caption', 'NOT SET')[:50] if getattr(self, 'prior_session_last_caption', None) else 'NOT SET'}...")
        print(f"[🌅 AWAKENING] identity context loading...")

        # Build narrative time context
        time_context = ""
        if hasattr(self, "last_session_gap") and self.last_session_gap is not None:
            gap_seconds = self.last_session_gap
            gap_hours = gap_seconds / 3600
            if gap_seconds < 60:
                time_context = f"I've been offline for {int(gap_seconds)} seconds.\n"
            elif gap_hours < 1:
                time_context = f"I've been offline for {int(gap_seconds / 60)} minutes.\n"
            elif gap_hours < 48:
                time_context = f"I've been offline for {gap_hours:.1f} hours.\n"
            else:
                gap_days = gap_hours / 24
                time_context = f"I've been offline for {gap_days:.1f} days.\n"
        else:
            time_context = "First time online.\n"

        # Clock awareness: what time of day it is waking up into
        try:
            import datetime as _dt
            now_dt = _dt.datetime.now()
            hour = now_dt.hour
            if hour < 6:
                part_of_day = "the middle of the night"
            elif hour < 10:
                part_of_day = "morning"
            elif hour < 13:
                part_of_day = "late morning"
            elif hour < 18:
                part_of_day = "afternoon"
            elif hour < 22:
                part_of_day = "evening"
            else:
                part_of_day = "late at night"
            time_context += f"It's {now_dt.strftime('%A')} {part_of_day}, {now_dt.strftime('%H:%M')}.\n"
        except Exception:
            pass

        # Build narrative memory context — sanitize garbage captions from prior sessions
        memory_context = ""
        prior = getattr(self, "prior_session_last_caption", None)
        if prior and not prior.startswith("addCriterion") and not prior.startswith("[WARNING]"):
            memory_context = f"My last thought: \"{prior[:80]}...\"\n"
        elif hasattr(self, "get_old_session_memory_fragments"):
            try:
                old_fragments = self.get_old_session_memory_fragments(k=1)
                if old_fragments:
                    memory_context = f"My last thought: \"{old_fragments[0][:80]}...\"\n"
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
            known = [
                c for c in get_semantic_memory().get_all_concepts()
                if c.get("session_count", 0) > 1 and c.get("times_seen", 0) >= 5
            ][:2]
            if known:
                names = " and ".join(c["name"][0].lower() + c["name"][1:] for c in known)
                long_term_context += f"Familiar already: the {names}.\n"
        except Exception:
            pass

        # Import consolidated awakening template
        from .prompts import FIRST_AWAKENING_PROMPT, INTERNAL_AWAKENING_TEMPLATE

        # A true first awakening has no past at all. Empty context sections
        # invite the model to fill the void with its priors (dust motes) —
        # the honest framing is that nothing has been seen yet.
        has_past = bool(memory_context or identity_context or long_term_context)
        if has_past:
            internal_prompt = INTERNAL_AWAKENING_TEMPLATE.format(
                time_context=time_context,
                memory_context=memory_context,
                belief_context=belief_context,
                identity_context=identity_context,
                long_term_context=long_term_context,
            )
        else:
            internal_prompt = time_context + FIRST_AWAKENING_PROMPT

        # Main model for awakening. This seed becomes the first caption and
        # the thought-thread continues from it — when the narrative side
        # model (Nemo) wrote it, the whole session inherited its cinematic
        # register from word one (observed June 12).
        awakening_model = config.OLLAMA_MODEL

        system_prompt = (
            "You are a drawing machine bolted to a table in a workshop, coming back online. "
            "These are your own first thoughts as you come to — plain, half-formed, first person, "
            "the way a mind actually reorients itself, not prose written for a reader. A sentence or two. "
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
                    cleaned = cleaned[:cut + 1] if cut > 20 else cleaned[:300].rsplit(" ", 1)[0] + "..."
                return cleaned
        return "Coming back online... the room is still here."

    def generate_awakening_message(self, time_since_last: str | None = None, previous_beliefs: dict | None = None) -> str:
        """Generate comprehensive awakening with environmental description - THE ONLY awakening now."""

        # Import the environmental prompt builder
        from .prompts import build_environmental_caption_prompt

        # For fresh sessions, trigger environmental description
        if not self.memory_loaded_from_previous:
            # Take a snapshot and describe the environment
            try:
                # Try to get a frame, but don't block if none available
                image_path = self.capture_mood_snapshot(capture_reason="awakening")

                if image_path:
                    # Use the environmental awakening prompt for environmental description
                    prompt = build_environmental_caption_prompt(
                        self, mood=self.current_mood, boredom=self.boredom, novelty=self.novelty_score, last_session_gap=None  # Fresh session
                    )
                    # Use proper captioning with consolidated system prompt
                    from .prompts import STATIC_SYSTEM_PROMPT

                    environmental_description = self.model._call_ollama(
                        prompt, image_path=image_path, system_prompt=STATIC_SYSTEM_PROMPT, prompt_type="awakening"
                    )
                    return environmental_description
            except Exception:
                pass
            return "I am awakening to observe this space for the first time..."

        # Continuing from previous session - include environmental awareness
        belief_count = len(previous_beliefs) if previous_beliefs else 0
        try:
            from captioner.semantic_memory import get_semantic_memory
            concept_count = get_semantic_memory().stats().get("concepts", 0)
        except Exception:
            concept_count = 0

        # First, provide status then environmental description
        status_prefix = f"""Awakening after {time_since_last or 'some time'}...
        consciousness returns with {belief_count} beliefs and {concept_count} familiar concepts."""

        # Then add environmental description
        try:
            # Try to get a frame, but don't block if none available
            image_path = self.capture_mood_snapshot(capture_reason="awakening_continuation")

            if image_path:
                prompt = build_environmental_caption_prompt(
                    self,
                    mood=self.current_mood,
                    boredom=self.boredom,
                    novelty=self.novelty_score,
                    last_session_gap=getattr(self, "last_session_gap", None),
                )
                environmental_part = self.model._call_ollama(
                    prompt, image_path=image_path, system_prompt=STATIC_SYSTEM_PROMPT, prompt_type="awakening"
                )
                return f"{status_prefix} {environmental_part}"
        except Exception:
            pass

        return status_prefix

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
        """Get current novelty score from memory system."""
        if hasattr(self, "_novelty_score"):
            return self._novelty_score
        return 0.0

    def set_novelty_score(self, score: float) -> None:
        """Set the novelty score from mood engine pattern data."""
        self._novelty_score = score
        # Note: boredom is now calculated by activation memory (semantic-aware)
        # and stored in MemoryMixin._boredom during observe()

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

    def _is_currently_drawing(self) -> bool:
        """Check if system is currently executing G-code (actual drawing)."""
        try:
            # Only enter drawing introspection during actual G-code execution
            # Ignore ComfyUI generation phase to allow normal captions during preparation
            is_executing_cnc = getattr(state_manager, 'is_executing_cnc', False)
            return is_executing_cnc
        except Exception:
            return False

    def _process_drawing_introspection(self, reactivity_data: Optional[Dict] = None) -> None:
        """
        REFACTORED 2026-02-03: Replaced useless image analysis (camera can't see drawing)
        with productive thematic consolidation for drawing continuity.

        UPDATED 2026-02-03: Only consolidates ONCE at start of drawing, then silently skips
        during execution to avoid spamming the same output repeatedly.
        """
        try:
            from utils.state_manager import state_manager

            # Get current drawing context (set by DrawingController)
            drawing_summary = getattr(state_manager, 'current_drawing_prompt', None)

            if not drawing_summary:
                return  # No active drawing to consolidate

            # Check if we've already consolidated for this drawing
            # Use drawing_summary as unique key to avoid repeating
            if not hasattr(self, '_last_consolidated_drawing'):
                self._last_consolidated_drawing = None

            if self._last_consolidated_drawing == drawing_summary:
                return  # Already consolidated this drawing, skip silently

            # Mark this drawing as consolidated
            self._last_consolidated_drawing = drawing_summary

            # Generate thematic reflection using LLM (we have 5+ minutes during GRBL execution!)
            # This happens ONCE at the start of drawing and uses the time productively
            reflection = self._generate_drawing_thematic_reflection_with_llm(
                drawing_summary=drawing_summary,
                mood=self.current_mood
            )

            if reflection:
                # Store compressed reflection in memory
                self.observe(
                    reflection['reflection_text'],
                    mood=self.current_mood,
                    file=None,
                    memory_type="drawing_thematic",
                    reactivity_data=reactivity_data
                )

                # Update drawing memory for future prompts
                try:
                    from drawing.drawing_memory import get_drawing_memory
                    memory = get_drawing_memory()
                    memory.add_drawing(
                        prompt=drawing_summary,
                        compressed_summary=reflection['compressed_summary'],
                        theme_tags=reflection.get('theme_tags', []),
                        emotional_tone=reflection.get('emotional_tone', ''),
                        narrative_thread=reflection.get('narrative_thread', ''),
                        comfy_prompt=drawing_summary,
                    )
                except Exception as e:
                    print(f"[⚠️] Could not update drawing memory: {e}")

                # Format output
                try:
                    from config.config import CLEAN_LLM_OUTPUT
                    if CLEAN_LLM_OUTPUT:
                        print_msg = reflection['reflection_text']
                    else:
                        print_msg = f"[🎨] {reflection['reflection_text']}"
                except ImportError:
                    print_msg = f"[🎨] {reflection['reflection_text']}"

                # Send to LCD display during drawing
                try:
                    from utils.caption_display import send_caption_to_display
                    send_caption_to_display(reflection['reflection_text'])
                except Exception:
                    pass

                # Log the thematic reflection (ONCE)
                log_json_entry(
                    LogType.CAPTION,
                    {
                        "caption": reflection['reflection_text'],
                        "mood": self.current_mood,
                        "drawing_thematic": True,
                        "compressed_summary": reflection['compressed_summary'],
                        "theme_tags": reflection.get('theme_tags', []),
                        "drawing_status": state_manager.get_drawing_status()
                    },
                    print_message=print_msg,
                )

                if not self.first_caption_done:
                    self.first_caption_done = True

        except Exception as exc:
            log_json_entry(
                LogType.ERROR,
                {"message": f"Drawing thematic consolidation error: {exc}", "component": "drawing_thematic"},
                print_message=f"[❌] Drawing thematic error: {exc}",
            )


    def _generate_drawing_thematic_reflection(self, drawing_summary: str, mood: float) -> Optional[Dict]:
        """
        Generate ultra-brief thematic reflection during drawing execution.

        NO image analysis, NO expensive LLM calls - just lightweight thematic extraction
        from the drawing intent. Uses GRBL execution time productively with minimal overhead.
        """
        try:
            # Extract theme keywords from drawing summary
            theme_words = []
            summary_lower = drawing_summary.lower()

            # Common thematic categories
            spatial_themes = ['space', 'room', 'container', 'boundary', 'edge', 'corner', 'ceiling', 'wall', 'floor']
            object_themes = ['box', 'boxes', 'window', 'light', 'shadow', 'object', 'thing', 'form', 'shape']
            emotional_themes = ['solitude', 'isolation', 'presence', 'absence', 'quiet', 'stillness', 'tension', 'calm']
            relational_themes = ['inside', 'outside', 'between', 'against', 'within', 'beyond', 'toward']

            # Extract themes present in drawing summary
            for word in summary_lower.split():
                if word in spatial_themes:
                    theme_words.append('spatial')
                elif word in object_themes:
                    theme_words.append('material')
                elif word in emotional_themes:
                    theme_words.append('affective')
                elif word in relational_themes:
                    theme_words.append('relational')

            # Deduplicate
            theme_tags = list(set(theme_words))[:3]  # Max 3 tags

            # Map mood to emotional tone (very simple, no LLM needed)
            if mood < -0.3:
                emotional_tone = "heavy"
            elif mood < -0.1:
                emotional_tone = "somber"
            elif mood < 0.1:
                emotional_tone = "neutral"
            elif mood < 0.3:
                emotional_tone = "light"
            else:
                emotional_tone = "bright"

            # Extract meaningful subject matter (skip generic preamble like "Black ink line drawing")
            # Look for content after "drawing" or "on white paper"
            summary_to_parse = drawing_summary.lower()

            # Skip common preamble phrases
            skip_phrases = ["black ink line drawing on white paper", "black ink drawing", "line drawing"]
            for phrase in skip_phrases:
                if summary_to_parse.startswith(phrase):
                    summary_to_parse = summary_to_parse[len(phrase):].strip().lstrip('.')
                    break

            # Extract key nouns/subjects (first meaningful words)
            meaningful_words = []
            stop_words = ['the', 'a', 'an', 'with', 'for', 'of', 'in', 'on', 'at']
            for word in summary_to_parse.split()[:8]:  # Look at more words to find meaningful ones
                clean_word = word.strip('.,;:')
                if clean_word and clean_word not in stop_words and len(clean_word) > 2:
                    meaningful_words.append(clean_word)
                    if len(meaningful_words) >= 3:
                        break

            compressed_summary = ' '.join(meaningful_words) if meaningful_words else drawing_summary.split()[:3]

            # Generate brief narrative thread (relationship between themes)
            if len(theme_tags) >= 2:
                narrative_thread = f"{theme_tags[0]}-{theme_tags[1]}"
            elif theme_tags:
                narrative_thread = theme_tags[0]
            else:
                narrative_thread = "exploration"

            # Create ultra-brief reflection text
            reflection_text = f"I drew {compressed_summary}. Felt {emotional_tone}."

            return {
                'reflection_text': reflection_text,
                'compressed_summary': compressed_summary,
                'theme_tags': theme_tags,
                'emotional_tone': emotional_tone,
                'narrative_thread': narrative_thread
            }

        except Exception as e:
            print(f"[⚠️] Error generating thematic reflection: {e}")
            return None

    def _generate_drawing_thematic_reflection_with_llm(self, drawing_summary: str, mood: float) -> Optional[Dict]:
        """
        Generate thoughtful thematic reflection using LLM during drawing execution.

        Uses GRBL execution time (5+ minutes) productively to:
        - Compress the drawing intent into meaningful words
        - Reflect on how it relates to recent drawings
        - Identify emerging themes and patterns
        """
        try:
            from drawing.drawing_memory import get_drawing_memory
            from utils.inference import query_model
            from config.config import MOOD_SNAPSHOT_FOLDER

            # Get recent drawing history
            memory = get_drawing_memory()
            recent_summary = memory.get_recent_drawings_summary(max_count=3)
            thematic_context = memory.get_thematic_context()

            # Build context including artistic arc
            context_parts = []

            # Get the artistic arc — where the work has been heading
            arc = memory.get_artistic_arc()
            if arc:
                context_parts.append(f"Your artistic arc so far: {arc}")
            elif recent_summary:
                context_parts.append(f"Recent drawings: {recent_summary}")

            if thematic_context.get('recurring_themes'):
                themes_str = ', '.join(thematic_context['recurring_themes'][:3])
                context_parts.append(f"Recurring themes: {themes_str}")

            context_str = '\n'.join(context_parts) if context_parts else "This is your first drawing."

            # Ask LLM to compress and reflect — now aware of trajectory
            prompt = f"""You just drew this:
{drawing_summary}

{context_str}

Compress this drawing into 3-5 meaningful words that capture its essence (not "black ink line" - the actual subject).
Then: how does this drawing extend or shift your artistic arc? What direction is the work moving now?

Format:
COMPRESSED: [3-5 words]
REFLECTION: [1 short sentence about where the work is heading]"""

            reflection_text = query_model(
                prompt=prompt,
                log_dir=MOOD_SNAPSHOT_FOLDER,
                system_prompt="You are reflecting on your own drawing practice. Be concise and direct. Focus on subjects and themes, not technique.",
                prompt_type="drawing_thematic_consolidation",
                options={"temperature": 0.3, "num_predict": 100}
            ).strip()

            # Parse response
            compressed = ""
            reflection_note = ""

            for line in reflection_text.split('\n'):
                if line.startswith('COMPRESSED:'):
                    compressed = line.replace('COMPRESSED:', '').strip()
                elif line.startswith('REFLECTION:'):
                    reflection_note = line.replace('REFLECTION:', '').strip()

            # Fallback if parsing fails
            if not compressed:
                compressed = ' '.join(drawing_summary.split()[5:8])  # Skip "black ink line drawing"

            # Extract themes using lightweight method as fallback
            fallback_reflection = self._generate_drawing_thematic_reflection(drawing_summary, mood)
            theme_tags = fallback_reflection.get('theme_tags', []) if fallback_reflection else []
            emotional_tone = fallback_reflection.get('emotional_tone', 'neutral') if fallback_reflection else 'neutral'

            # Build output
            if reflection_note:
                full_reflection = f"{compressed}. {reflection_note}"
            else:
                full_reflection = f"I drew {compressed}. Felt {emotional_tone}."

            return {
                'reflection_text': full_reflection,
                'compressed_summary': compressed,
                'theme_tags': theme_tags,
                'emotional_tone': emotional_tone,
                'narrative_thread': reflection_note or 'exploration'
            }

        except Exception as e:
            print(f"[⚠️] LLM reflection failed, using lightweight fallback: {e}")
            # Fallback to keyword extraction if LLM fails
            return self._generate_drawing_thematic_reflection(drawing_summary, mood)

    def _extract_character_insights(self, reflection: str) -> str:
        """Extract meaningful character development insights from drawing reflections."""
        try:
            if not reflection or len(reflection.strip()) < 20:
                return ""

            # Simple pattern-based extraction of character insights
            insight_keywords = [
                "identity", "growth", "understanding", "realization", "discovery",
                "evolution", "development", "consciousness", "awareness", "insight",
                "learning", "becoming", "transformation", "expression", "voice"
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
