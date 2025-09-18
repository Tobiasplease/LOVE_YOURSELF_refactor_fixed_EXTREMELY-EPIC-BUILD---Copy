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

from config.config import CAPTION_INTERVAL, DRAWING_INTERVAL, MOOD_SNAPSHOT_FOLDER, OLLAMA_SHOW_PROGRESS, REASON_INTERVAL
from drawing.drawing import DrawingController
from event_logging.event_logger import log_json_entry
from event_logging.log_type import LogType
from event_logging.run_manager import get_run_image_path
from utils.error_tracking import robust_execution, track_component_health
from utils.ollama import truncate_for_print
from utils.state_manager import state_manager

from .memory import MemoryMixin
from .model_wrapper import MultimodalModel
from .prompts import SYSTEM_PROMPT, extract_motifs_spacy

# from weakref import ref


# Import context compressor with error handling
try:
    from .context_compression import context_compressor
except Exception as e:
    print(f"[WARNING] Context compression module failed to load: {e}")
    context_compressor = None


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
        self.print_lock = threading.Lock()  # Prevent multiple simultaneous prints

        self.current_mood: float = 0.0
        self.current_mood_vector: Tuple[float, float, float] = (0.0, 0.0, 0.5)  # valence, arousal, clarity
        self.current_emotion_state: str = "calm_observant"  # hand controller emotion state
        self.emotional_journey: List[str] = []  # track emotional evolution over time
        self.last_caption: str = ""
        self.current_motifs_from_mood: List[str] = []

        # Deduplication system to prevent duplicate prints
        self.recent_captions: List[Tuple[str, float]] = []  # (caption, timestamp)

        self.last_caption_time: float = 0.0
        self.last_reason_time: float = time.time()  # Delay first reflection
        self.last_drawing_time: float = time.time() - DRAWING_INTERVAL - 10  # Allow immediate first drawing

        # Track session continuity
        self.sessions_since_boot = 0
        self.memory_loaded_from_previous = False

        # Session continuity - time gap will be set by state manager if restoring session
        self._last_session_file = os.path.join(MOOD_SNAPSHOT_FOLDER, "last_session.txt")
        self.last_session_gap = None  # Will be set by state manager during restoration

        os.makedirs(MOOD_SNAPSHOT_FOLDER, exist_ok=True)
        self.snapshot_queue: Deque[Tuple[np.ndarray, bool, Optional[Dict]]] = deque()
        threading.Thread(target=self._caption_worker, daemon=True).start()

    def save_session_time(self):
        try:
            with open(self._last_session_file, "w") as f:
                f.write(str(time.time()))
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
                frame, _, reactivity_data = self.snapshot_queue.popleft()
                try:
                    self._process_frame(frame, reactivity_data)
                except Exception as exc:
                    log_json_entry(
                        LogType.ERROR,
                        {"message": f"Caption thread error: {exc}", "component": "captioner"},
                        print_message=f"[❌] Caption thread error: {exc}",
                    )
            else:
                # Wait longer on startup to allow main loop to populate frames
                time.sleep(0.5 if not self.first_caption_done else 0.05)

    def _process_frame(self, frame: np.ndarray, reactivity_data: Optional[Dict] = None) -> None:
        now = time.time()
        if now - self.last_caption_time < CAPTION_INTERVAL:
            return

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

            if not self.first_caption_done:
                # Phase 1: Internal awakening reorientation (no image)

                # If no frames available yet, defer awakening to next cycle
                if not self.snapshot_queue:
                    # Don't block - just defer the awakening to the next caption cycle
                    self.first_caption_done = False  # Keep awakening pending
                    caption = "Awakening... preparing to observe environment..."
                    # Add small delay to allow main loop to populate snapshot_queue
                    time.sleep(0.5)
                else:
                    caption = self.generate_internal_awakening()
                    self.awaiting_environmental_phase = True  # Flag for Phase 2
            elif getattr(self, "awaiting_environmental_phase", False):
                # Phase 2: Environmental grounding (first visual after awakening)
                caption = self.model.caption_image(img_path, flowing=True, first_time=True)  # Use awakening prompts
                self.awaiting_environmental_phase = False  # Clear flag
            else:
                log_json_entry(
                    LogType.DEBUG,
                    {"message": "Requesting new caption", "action": "caption_request", "image_path": img_path},
                    print_message=f"[🐞] Requesting new caption for {img_path}",
                )
                previous_caption = getattr(self, "last_caption", "")
                caption = self.model.caption_image(img_path, flowing=True, first_time=False)
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
                        caption = self.model.caption_image(img_path, flowing=True, first_time=False)
                    else:
                        caption = "Awakening... vision initializing..."
                except Exception:
                    caption = "[WARNING] Vision unavailable"
            else:
                caption = "[WARNING] Vision unavailable"

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
                caption = "Vision systems recalibrating..."  # Better fallback

        # Format caption for clean output
        try:
            from config.config import CLEAN_LLM_OUTPUT

            if CLEAN_LLM_OUTPUT:
                print_msg = caption  # print full caption
            else:
                print_msg = f"[📸] {caption}"
        except ImportError:
            print_msg = f"[📸] {caption}"

        # Deduplication check - avoid printing the same caption twice within 5 seconds
        now_ts = time.time()
        should_print = True

        # Clean old captions (older than 5 seconds)
        self.recent_captions = [(c, t) for c, t in self.recent_captions if now_ts - t < 5.0]

        # Check if this caption was recently printed
        for recent_caption, recent_time in self.recent_captions:
            if recent_caption.strip() == caption.strip() and now_ts - recent_time < 5.0:
                should_print = False
                break

        if should_print:
            # Add to recent captions list
            self.recent_captions.append((caption, now_ts))

            log_json_entry(
                LogType.CAPTION,
                {"caption": caption, "image_path": img_path, "mood": self.current_mood},
                print_message=print_msg,
            )
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
        )
        self.last_caption = caption

        # Now update the timestamp since we have a new caption
        self.last_caption_time = now

        # Add caption to context compression system (with error handling)
        try:
            if context_compressor and caption and caption.strip():
                context_compressor.add_caption(caption, time.time(), img_path)
        except Exception as e:
            print(f"[CAPTIONER] Context compression failed: {e}")

        # CRITICAL FIX: Add caption to memory system for motif tracking and repetition fatigue
        try:
            if caption and caption.strip():
                self.observe(
                    text=caption,
                    mood=self.current_mood,
                    memory_type="caption",
                    mood_vector=getattr(self, "current_mood_vector", (0.0, 0.0, 0.5)),
                    emotion_state=getattr(self, "current_emotion_state", "calm_observant"),
                )
        except Exception as e:
            print(f"[CAPTIONER] Memory system failed: {e}")

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

                    # Use motifs from mood engine's pattern recognition instead of re-extracting
                    if hasattr(self, "current_motifs_from_mood") and self.current_motifs_from_mood:
                        for motif in self.current_motifs_from_mood:
                            self.absorb_motif(motif)
                    else:
                        # Fallback to direct extraction if mood data not available
                        for motif in extract_motifs_spacy(caption):
                            self.absorb_motif(motif)

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

        # Debug: Check drawing interval condition
        time_since_last_drawing = now - self.last_drawing_time
        print(f"[DEBUG] Drawing check: {time_since_last_drawing:.1f}s elapsed, need {DRAWING_INTERVAL}s")
        if time_since_last_drawing > DRAWING_INTERVAL:
            print(f"[DEBUG] DRAWING TRIGGER ACTIVATED! Starting drawing generation...")
            print(f"[DEBUG] Step 1: About to start drawing generation process")
            try:
                print(f"[DEBUG] Step 2: Attempting log_json_entry...")
                with self.print_lock:
                    print("\r" + " " * 80 + "\r", end="")
                    log_json_entry(
                        LogType.DEBUG,
                        {
                            "message": "Drawing interval reached",
                            "action": "drawing_trigger",
                            "time_since_last_drawing": time_since_last_drawing,
                            "drawing_interval": DRAWING_INTERVAL,
                        },
                        print_message=f"[🎨] Drawing interval reached ({time_since_last_drawing:.0f}s > {DRAWING_INTERVAL}s), generating prompt...",
                    )
                print(f"[DEBUG] Step 3: log_json_entry completed successfully")
            except Exception as e:
                print(f"[DEBUG] EXCEPTION in log_json_entry: {e}")
                import traceback

                traceback.print_exc()

            print(f"[DEBUG] Step 4: About to check pipeline state...")
            # Guard: do not start a new drawing while pipeline is busy (prevents stacking)
            try:
                is_generating = getattr(state_manager, "is_generating_drawing", False)
                is_executing = getattr(state_manager, "is_executing_cnc", False)
                print(f"[DEBUG] Pipeline state - generating: {is_generating}, executing: {is_executing}")
                if is_generating or is_executing:
                    log_json_entry(
                        LogType.DECISION,
                        {
                            "decision": "skip_drawing",
                            "reason": "pipeline_busy",
                            "is_generating": getattr(state_manager, "is_generating_drawing", False),
                            "is_executing_cnc": getattr(state_manager, "is_executing_cnc", False),
                        },
                        print_message="[⏳] Skipping drawing: pipeline busy (generation/execution)",
                    )
                    # Re-check after short delay
                    self.last_drawing_time = now - DRAWING_INTERVAL + 30
                    return
            except Exception as e:
                print(f"[DEBUG] Exception checking pipeline state: {e}")
                pass

            print(f"[DEBUG] Step 5: Pipeline check passed, building context...")
            memory_context = self.get_recent_memory()
            reflection_context = self.get_last_reflection()
            extra_context = f"{self.last_caption}\n\n{memory_context}\n\n{reflection_context}"
            print(f"[DEBUG] Step 6: Context built, starting drawing generation...")

            # Start loading animation for drawing prompt
            loading_stop = threading.Event()
            loading_thread = threading.Thread(target=loading_animation, daemon=True)
            loading_thread.start()

            try:
                prompt = self.model.generate_drawing_prompt(extra=extra_context, image_path=img_path)
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

            # Only update timer if drawing system is ready (not in cooldown)
            if self.drawing.ready_to_draw():
                if "[ERROR]" not in prompt:
                    self.drawing.handle_drawing_flow(self, prompt, img_path, reflection=reflection_context)
                self.last_drawing_time = now
            else:
                # In cooldown, don't spam drawing attempts
                cooldown_remaining = self.drawing.cooldown - (time.time() - self.drawing.last_drawing_time)
                self.last_drawing_time = now - DRAWING_INTERVAL + cooldown_remaining + 30  # Retry after cooldown + 30s

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

        # Get specific motifs and patterns from current session
        top_motifs = ""
        # new_motifs = ""
        # recurring_motifs = ""

        if hasattr(self, "current_motifs_from_mood") and self.current_motifs_from_mood:
            # Use real-time motif data from pattern recognition
            recent_motifs = self.current_motifs_from_mood[:5]
            top_motifs = f"Current motifs: {', '.join(recent_motifs)}"
        elif hasattr(self, "get_top_motifs"):
            motifs = self.get_top_motifs(5)
            if motifs:
                top_motifs = f"Recurring motifs: {', '.join(motifs[:5])}"

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

        # Build internal awakening context
        awakening_parts = []

        # Use the properly calculated time gap from state manager
        if hasattr(self, "last_session_gap") and self.last_session_gap is not None:
            gap_seconds = self.last_session_gap
            gap_hours = gap_seconds / 3600

            if gap_seconds < 60:
                awakening_parts.append(f"Consciousness returning... I was offline for {int(gap_seconds)} seconds.")
            elif gap_hours < 1:
                awakening_parts.append(f"Consciousness returning... I was offline for {int(gap_seconds / 60)} minutes.")
            elif gap_hours < 48:  # Less than 2 days
                awakening_parts.append(f"Consciousness returning... I was offline for {gap_hours:.1f} hours.")
            else:  # Multiple days
                gap_days = gap_hours / 24
                awakening_parts.append(f"Consciousness returning... I was offline for {gap_days:.1f} days.")
        else:
            awakening_parts.append("First awakening... consciousness beginning...")

        # Memory fragments from before shutdown
        if hasattr(self, "get_old_session_memory_fragments"):
            try:
                old_fragments = self.get_old_session_memory_fragments(k=1)
                if old_fragments:
                    awakening_parts.append(f"Last memory: {old_fragments[0][:80]}...")
            except Exception:
                pass

        # Current internal state
        awakening_parts.append(f"Feeling {self.describe_current_mood() if hasattr(self, 'describe_current_mood') else 'uncertain'}.")

        # Beliefs/understanding continuity
        if self.beliefs:
            belief_count = len(self.beliefs)
            awakening_parts.append(f"My {belief_count} accumulated beliefs remain.")

        awakening_context = " ".join(awakening_parts)

        # Import consolidated awakening template
        from .prompts import INTERNAL_AWAKENING_TEMPLATE

        # Internal awakening prompt (no image needed)
        internal_prompt = INTERNAL_AWAKENING_TEMPLATE.format(awakening_context=awakening_context)

        # Get dynamic system context for organic consciousness
        if hasattr(self, "get_dynamic_system_context"):
            dynamic_context = self.get_dynamic_system_context()
            if isinstance(dynamic_context, dict):
                system_prompt = SYSTEM_PROMPT.format(
                    emotional_state=dynamic_context.get("emotional_state", "contemplative"),
                    temporal_context=dynamic_context.get("temporal_context", ""),
                    accumulated_understanding=dynamic_context.get("accumulated_understanding", ""),
                )
            else:
                system_prompt = SYSTEM_PROMPT + str(dynamic_context)
        else:
            system_prompt = SYSTEM_PROMPT

        # Generate internal awakening without image
        response = query_ollama(
            prompt=internal_prompt,
            model=config.OLLAMA_MODEL,
            timeout=90,
            log_dir=config.MOOD_SNAPSHOT_FOLDER,
            system_prompt=system_prompt,
            prompt_type="awakening",
        )

        return response

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
        motif_count = len(self.motif_counter)

        # First, provide status then environmental description
        status_prefix = f"""Awakening after {time_since_last or 'some time'}...
        consciousness returns with {belief_count} beliefs and {motif_count} familiar motifs."""

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

        # Update boredom based on low novelty over time
        self._update_boredom(score)

    def _update_boredom(self, novelty: float) -> None:
        """Calculate boredom based on sustained low novelty."""
        if not hasattr(self, "_boredom"):
            self._boredom = 0.0
        if not hasattr(self, "_low_novelty_duration"):
            self._low_novelty_duration = 0.0
        if not hasattr(self, "_last_boredom_update"):
            self._last_boredom_update = time.time()

        now = time.time()
        delta = now - self._last_boredom_update
        self._last_boredom_update = now

        # Track sustained low novelty
        if novelty < 0.3:  # Low novelty threshold
            self._low_novelty_duration += delta
        else:
            self._low_novelty_duration = max(0, self._low_novelty_duration - delta * 0.5)  # Slow decay

        # Convert to boredom (peaks at ~10 minutes of low novelty)
        self._boredom = min(1.0, self._low_novelty_duration / 600.0)  # 10 minutes = max boredom

    @property
    def boredom(self) -> float:
        """Get current boredom level from memory system."""
        if hasattr(self, "_boredom"):
            return self._boredom
        return 0.0
