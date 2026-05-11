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

from config.config import CAPTION_INTERVAL, CLEAN_LLM_OUTPUT, DRAWING_INTERVAL, DRAWING_STARTUP_DELAY, MOOD_SNAPSHOT_FOLDER, OLLAMA_SHOW_PROGRESS, REASON_INTERVAL
from drawing.drawing import DrawingController
from event_logging.event_logger import log_json_entry
from event_logging.log_type import LogType
from event_logging.run_manager import get_run_image_path
from utils.error_tracking import track_component_health
from utils.ollama import truncate_for_print
from utils.state_manager import state_manager

from .memory import MemoryMixin
from .model_wrapper import MultimodalModel
from .prompts import SYSTEM_PROMPT, STATIC_SYSTEM_PROMPT

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
        self.awakening_done = False
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
        self.last_reason_time: float = time.time()  # Delay first reflection
        self.last_drawing_check_time: float = 0.0  # Allow immediate first check
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

    def _process_frame(self, frame: np.ndarray, reactivity_data: Optional[Dict] = None, person_present: bool = False) -> None:
        now = time.time()
        if now - self.last_caption_time < CAPTION_INTERVAL:
            return

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
                        from captioner.prompts import build_memory_mode_prompt, _get_static_system_prompt
                        memory_prompt, caption_mode = build_memory_mode_prompt(self)
                        caption = self.model._call_ollama(
                            memory_prompt,
                            image_path=None,  # Memory mode doesn't use current image
                            system_prompt=_get_static_system_prompt(),
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
                        if True:
                            # === TWO-PASS CAPTION PIPELINE ===
                            # Always use two-pass after first caption. Single-pass LLaVA
                            # produces VQA descriptions, not inner monologue.
                            from captioner.prompts import select_perception_prompt, build_monologue_prompt, determine_prompt_mode
                            from captioner.activation_memory import get_activation_network

                            gaze_state = "idle"
                            gaze_direction = "ahead"
                            try:
                                from vision.gaze import get_gaze_state, get_current_gaze_zone
                                gaze_state = get_gaze_state() or "idle"
                                gaze_direction = get_current_gaze_zone() or "ahead"
                            except Exception:
                                pass

                            network = get_activation_network()
                            boredom = network._last_boredom
                            novelty = getattr(network, "_last_novelty", 0.5)

                            # Determine mode ONCE — both models use it
                            caption_mode = determine_prompt_mode(
                                gaze_state=gaze_state,
                                gaze_direction=gaze_direction,
                                novelty=novelty,
                                boredom=boredom,
                                person_present=person_present,
                            )

                            # Pass 1: Qwen perception (mode-aware)
                            perception_prompt = select_perception_prompt(
                                gaze_direction=gaze_direction,
                                person_present=person_present,
                                boredom=boredom,
                                mode=caption_mode,
                                previous_perception=getattr(self, "_last_perception", None),
                            )

                            # If YOLO sees multiple people, tell qwen
                            person_count = (reactivity_data or {}).get("person_count", 0)
                            if person_count > 1 and caption_mode == "relational":
                                perception_prompt = f"There are {person_count} people visible. " + perception_prompt

                            perception = self.model.perceive(
                                img_path,
                                perception_prompt=perception_prompt,
                                mode=caption_mode,
                            )

                            # If we asked about a person but LLaVA saw nobody,
                            # downgrade mode and re-perceive with a non-person prompt.
                            if not perception and person_present and caption_mode == "relational":
                                print("[PERCEPTION] Person expected but not seen — falling back to observational")
                                person_present = False
                                caption_mode = "introspective"
                                perception_prompt = select_perception_prompt(
                                    gaze_direction=gaze_direction,
                                    person_present=False,
                                    boredom=boredom,
                                    mode=caption_mode,
                                )
                                perception = self.model.perceive(
                                    img_path,
                                    perception_prompt=perception_prompt,
                                    mode=caption_mode,
                                )

                            self._last_perception = perception

                            # Match perception against ChromaDB concepts BEFORE monologue
                            matched_concepts = []
                            try:
                                from captioner.semantic_memory import get_semantic_memory
                                matched_concepts = get_semantic_memory().match_or_create_concepts(perception or "")
                            except Exception as mc_err:
                                print(f"[SEMANTIC] Concept matching failed: {mc_err}")

                            # Nudge gaze toward the most salient concept's spatial location
                            try:
                                from vision.gaze import nudge_toward_concept
                                for mc in matched_concepts:
                                    sp = mc.get("spatial_pan")
                                    st = mc.get("spatial_tilt")
                                    if sp or st:
                                        nudge_toward_concept(pan_zone=sp, tilt_zone=st)
                                        break  # Only nudge toward the first concept with spatial data
                            except Exception:
                                pass

                            # Pass 2: Monologue from perception (mode pre-determined)
                            monologue_prompt, caption_mode = build_monologue_prompt(
                                self,
                                perception=perception,
                                person_present=person_present,
                                mode=caption_mode,
                                matched_concepts=matched_concepts,
                            )

                            caption, caption_mode = self.model.generate_monologue(
                                perception,
                                monologue_prompt=monologue_prompt,
                                mode=caption_mode,
                                agent=self,
                            )

                            # Store observation in semantic memory (uses pre-matched concepts)
                            try:
                                from captioner.semantic_memory import get_semantic_memory
                                get_semantic_memory().after_monologue(perception, caption, matched_concepts=matched_concepts)
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
                caption = "Awakening... camera systems initializing..."
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
                {"caption": caption, "image_path": img_path, "mood": self.current_mood},
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

        # Process emotional drift
        # environmental_factors = {
        #     "scene_static": getattr(self, "_scene_static", False),  # Will be tracked by semantic memory
        #     "novelty": self.novelty_score,
        #     "person_present": reactivity_data.get("person_present", False) if reactivity_data else False,
        #     "boredom": self.boredom,
        # }

        # Debug reflection timing
        time_since_reflection = now - self.last_reason_time
        if time_since_reflection > REASON_INTERVAL:
            log_json_entry(
                LogType.DEBUG,
                {
                    "message": "Reflection triggered",
                    "action": "reflection_trigger",
                    "time_since_last_reflection": time_since_reflection,
                    "reason_interval": REASON_INTERVAL,
                    "mood": self.current_mood,
                },
                print_message=f"[🤔] Reflection triggered! Time since last: {time_since_reflection:.0f}s > {REASON_INTERVAL}s",
            )
            try:
                mood_text = self.describe_current_mood()
                context = self.get_reflection_context()

                # Start loading animation for reflection
                loading_stop = threading.Event()

                loading_thread = threading.Thread(target=loading_animation, daemon=True)
                loading_thread.start()

                try:
                    reflection = self.model.reason_about_caption(caption, agent=self, mood_text=mood_text, extra=context)
                finally:
                    # Stop loading animation
                    loading_stop.set()
                    loading_thread.join(timeout=2.0)  # Increased timeout
                    if loading_thread.is_alive():
                        # Force clear animation remnants if thread still running
                        with self.print_lock:
                            print("\r" + " " * 80 + "\r", end="")

                if reflection and len(reflection.strip()) > 10:
                    log_json_entry(
                        LogType.REFLECTION,
                        {"reflection": reflection, "mood": self.current_mood, "image_path": img_path, "context": context},
                        print_message=f"[🤔] Reflection: {truncate_for_print(reflection, 100)}",
                    )
                    self.last_reason_time = now
                    self.awakening_done = True

                    m = re.search(r"-?\d+(?:\.\d+)?", reflection)
                    mood_val = float(m.group()) if m else self.current_mood
                    self.current_mood += 0.25 * (mood_val - self.current_mood)

                    self.observe(reflection, self.current_mood, img_path, memory_type="reflection")
                else:
                    # Clear animation line for short reflection message
                    with self.print_lock:
                        print("\r" + " " * 80 + "\r", end="")

                    log_json_entry(
                        LogType.REFLECTION,
                        {
                            "message": "Generated reflection too short, skipping",
                            "action": "skip_short",
                            "reflection_length": len(reflection),
                            "mood": self.current_mood,
                        },
                        print_message="[🤔] Generated reflection too short, skipping",
                    )
                    # Update timer even for short reflections to prevent continuous retries
                    self.last_reason_time = now

            except Exception as e:
                log_json_entry(
                    LogType.ERROR,
                    {
                        "message": f"Error during reflection: {e}",
                        "component": "reflection",
                        "error_type": type(e).__name__,
                        "mood": self.current_mood,
                    },
                    print_message=f"[❌] Error during reflection: {e}",
                )
                # Still update the timer to prevent infinite retries
                self.last_reason_time = now - REASON_INTERVAL + 60  # Retry in 60 seconds

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

        # Start loading animation for drawing prompt
        loading_stop = threading.Event()
        loading_thread = threading.Thread(target=loading_animation, daemon=True)
        loading_thread.start()

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
            # Stop loading animation and wait for it to fully terminate
            loading_stop.set()
            loading_thread.join(timeout=2.0)  # Increased timeout
            if loading_thread.is_alive():
                # Force clear animation remnants if thread still running
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
                    compressed_summary=prompt[:80],
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

    def get_reflection_context(self) -> str:
        """Enhanced reflection context with specific experiential data."""

        # Get active concepts from activation network for reflection context
        top_motifs = ""
        try:
            from captioner.activation_memory import get_activation_network
            net = get_activation_network()
            top = net.get_activated_concepts(threshold=0.4)[:5]
            if top:
                labels = [net.concept_labels.get(c, c) for c, _ in top]
                top_motifs = f"Active concepts: {', '.join(labels)}"
        except Exception:
            pass

        # Get specific emotional changes
        emotion_changes = ""
        if hasattr(self, "emotional_journey") and self.emotional_journey:
            if len(self.emotional_journey) >= 2:
                recent_emotions = self.emotional_journey[-3:]
                emotion_changes = f"Emotional shifts: {' → '.join(recent_emotions)}"
            else:
                emotion_changes = f"Current state: {self.current_emotion_state}"

        # Get specific observations from recent memory
        recent_observations = self.get_recent_memory(k=3)

        # Get mood vector details for specificity
        valence, arousal, clarity = self.current_mood_vector
        mood_details = f"""Mood details: valence={valence:.2f}
        (feeling {'positive' if valence > 0 else 'negative' if valence < 0 else 'neutral'}),
        arousal={arousal:.2f} (energy {'high' if arousal > 0.3 else 'low' if arousal < -0.3 else 'medium'}),
        clarity={clarity:.2f} (understanding {'clear' if clarity > 0.3 else 'confused' if clarity < -0.3 else 'uncertain'})"""

        # Build context focused on concrete experience
        context_parts = [
            f"Current experience: Just observed '{self.last_caption}'",
            f"Overall mood: {self.current_mood:.2f} (novelty: {self.novelty_score:.2f}, boredom: {self.boredom:.2f})",
            mood_details,
            f"Session time: {(time.time() - self.true_session_start) / 60:.0f} minutes active",
        ]

        if recent_observations:
            context_parts.append(f"Recent observations:\n{recent_observations}")

        if top_motifs:
            context_parts.append(top_motifs)

        if emotion_changes:
            context_parts.append(emotion_changes)

        # Add any identity formation
        identity = self.get_identity_summary()
        if identity and "forming" not in identity.lower():
            context_parts.append(f"Identity: {identity}")

        return "\n".join(context_parts)

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
        entries = self.get_memory_entries_by_type("reflection")
        if entries:
            return entries[-1].get("text", "")
        return ""

    def generate_internal_awakening(self) -> str:
        """Phase 1 awakening: Pure internal reorientation without visual input."""
        from config import config
        from utils.ollama import query_ollama

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

        # Include cross-session memory from ChromaDB
        long_term_context = ""
        try:
            from captioner.semantic_memory import get_semantic_memory
            greeting = get_semantic_memory().get_session_greeting(limit=2)
            if greeting:
                long_term_context = f"{greeting}\n"
        except Exception:
            pass

        # Import consolidated awakening template
        from .prompts import INTERNAL_AWAKENING_TEMPLATE

        # Internal awakening prompt with narrative placeholders
        internal_prompt = INTERNAL_AWAKENING_TEMPLATE.format(
            time_context=time_context,
            memory_context=memory_context,
            belief_context=belief_context,
            identity_context=identity_context,
            long_term_context=long_term_context
        )

        # Use Natsumura for awakening (text-only narrative model, engages with context)
        try:
            from config.config import COMPRESSION_MODEL
            awakening_model = COMPRESSION_MODEL
        except ImportError:
            awakening_model = "natsumura-storytelling-rp:latest"

        system_prompt = (
            "You are a drawing machine waking from sleep. "
            "Write the first thought that surfaces as consciousness returns. "
            "Use the context naturally — the sleep duration, the last memory, what you know about this place. "
            "Inner monologue. First person. One or two sentences."
        )

        print(f"[🌅 AWAKENING] Generating seed thought...")
        response = query_ollama(
            prompt=internal_prompt,
            model=awakening_model,
            timeout=90,
            log_dir=config.MOOD_SNAPSHOT_FOLDER,
            system_prompt=system_prompt,
            options={"temperature": 0.85, "top_p": 0.85, "num_predict": 60, "stop": ["\n\n"]},
            prompt_type="awakening",
        )
        print(f"[🌅 AWAKENING] Response: {response[:120] if response else 'EMPTY'}...")

        # Filter: must be brief inner monologue, not garbage
        if response and len(response.strip()) > 10 and len(response.strip()) <= 150:
            cleaned = response.strip().strip('"').strip()
            if cleaned and not cleaned.startswith(("[", "{")):
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
            from utils.ollama import query_ollama
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

            reflection_text = query_ollama(
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
