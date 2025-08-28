from __future__ import annotations
import os
import re
import time
import threading
from collections import deque
from datetime import datetime
from typing import Deque, Optional, Tuple, Dict, List

# from weakref import ref

import cv2  # type: ignore
import numpy as np  # type: ignore
from config.config import CAPTION_INTERVAL, DRAWING_INTERVAL, MOOD_SNAPSHOT_FOLDER, OLLAMA_SHOW_PROGRESS, REASON_INTERVAL
from event_logging.event_logger import log_json_entry
from event_logging.log_type import LogType
from event_logging.run_manager import get_run_image_path
from drawing.drawing import DrawingController

from .memory import MemoryMixin
from .prompts import extract_motifs_spacy
from .model_wrapper import MultimodalModel
from utils.error_tracking import track_component_health, robust_execution

# Import context compressor with error handling
try:
    from .context_compression import context_compressor
except Exception as e:
    print(f"[WARNING] Context compression module failed to load: {e}")
    context_compressor = None


class Captioner(MemoryMixin):
    def shutdown(self):
        self.save_session_time()

    caption_window: Optional[any] = None  # type: ignore

    def __init__(self) -> None:
        super().__init__()
        self.model = MultimodalModel(memory_ref=self)
        self.drawing = DrawingController()

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

        self.last_caption_time: float = 0.0
        self.last_reason_time: float = time.time()  # Delay first reflection
        self.last_drawing_time: float = time.time()  # Stagger drawing

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
            if len(self.snapshot_queue) > 1:
                self.snapshot_queue.pop()
            # Store reactivity data with the frame for processing
            self.snapshot_queue.append((frame.copy(), person_present, reactivity_data))

    def _caption_worker(self):
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
                time.sleep(0.05)

    @robust_execution("captioner", "caption_generation", fallback_result=None)
    def _process_frame(self, frame: np.ndarray, reactivity_data: Optional[Dict] = None) -> None:
        now = time.time()
        if now - self.last_caption_time < CAPTION_INTERVAL:
            return

        # Don't update timestamp yet - wait until caption is actually generated
        ts = int(now)
        img_path = get_run_image_path(MOOD_SNAPSHOT_FOLDER, f"mood_{ts}.jpg")
        cv2.imwrite(img_path, frame)

        skip_caption_print = False  # Track if we should skip printing

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
            if not self.first_caption_done:
                # Phase 1: Internal awakening reorientation (no image)
                caption = self.generate_internal_awakening()
                self.awaiting_environmental_phase = True  # Flag for Phase 2
            elif getattr(self, "awaiting_environmental_phase", False):
                # Phase 2: Environmental grounding (first visual after awakening)
                caption = self.model.caption_image(img_path, flowing=True, first_time=True)  # Use awakening prompts
                self.awaiting_environmental_phase = False  # Clear flag
            else:
                # Debug: requesting new caption
                log_json_entry(
                    LogType.DEBUG,
                    {"message": "Requesting new caption", "action": "caption_request", "image_path": img_path},
                    print_message=f"[🔎] Requesting new caption for {img_path}",
                )
                previous_caption = getattr(self, "last_caption", "")
                caption = self.model.caption_image(img_path, flowing=True, first_time=False)
                if caption == previous_caption:
                    log_json_entry(
                        LogType.DEBUG,
                        {"message": "Caption is identical to previous", "action": "duplicate_caption", "caption_preview": caption[:50]},
                        print_message=f"[⚠️] Caption is identical to previous: {caption[:50]}...",
                    )
                    # Don't print the same caption again but still check reflection/drawing
                    skip_caption_print = True
                else:
                    log_json_entry(
                        LogType.DEBUG,
                        {
                            "message": "New caption generated",
                            "action": "caption_generated",
                            "caption_preview": caption[:50],
                            "caption_length": len(caption),
                        },
                        print_message=f"[🔎] New caption generated: {caption[:50]}...",
                    )
        except Exception as e:
            caption = "[WARNING] Vision unavailable"
            log_json_entry(
                LogType.ERROR,
                {"message": f"Caption error: {e}", "component": "captioner"},
                print_message=f"[❌] Caption error: {e}",
            )
        finally:
            # Stop loading animation and wait for it to fully terminate
            loading_stop.set()
            loading_thread.join(timeout=2.0)  # Increased timeout
            if loading_thread.is_alive():
                # Force terminate if still running
                print("\r" + " " * 80 + "\r", end="")  # Clear any remaining animation

        self.first_caption_done = True

        if "[WARNING]" in caption:
            log_json_entry(
                LogType.ERROR,
                {"message": f"Caption error: {caption}", "component": "captioner"},
                print_message=f"[❌] Caption error: {caption}",
            )
            self.observe("I couldn't see anything just now.", self.current_mood, img_path, memory_type="glitch")
            # Don't return early - still need to check reflection/drawing timing
            caption = "Vision unclear right now"  # Use fallback caption

        # Clear the animation line and print caption with timestamp (thread-safe)
        # Caption printing is now handled through structured logging
        # The caption is logged via log_json_entry with CAPTION log type elsewhere

        log_json_entry(
            LogType.CAPTION,
            {"caption": caption, "image_path": img_path, "mood": self.current_mood},
            print_message=f"[📷] {caption[:100]}{'...' if len(caption) > 100 else ''}",
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
                context_compressor.add_caption(caption, time.time())
        except Exception as e:
            print(f"[CAPTIONER] Context compression failed: {e}")

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
                    # Format reflection with timestamp like captions
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    formatted_reflection = f"[{timestamp}] REFLECTION: {reflection}"

                    with self.print_lock:
                        print("\r" + " " * 80 + "\r", end="")  # Clear line
                        print(formatted_reflection)  # Thread-safe reflection print

                    log_json_entry(
                        LogType.REFLECTION,
                        {"reflection": reflection, "mood": self.current_mood, "image_path": img_path, "context": context},
                        print_message=f"[🤔] {reflection[:100]}{'...' if len(reflection) > 100 else ''}",
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
        if time_since_last_drawing > DRAWING_INTERVAL:
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

            memory_context = self.get_recent_memory()
            reflection_context = self.get_last_reflection()
            extra_context = f"{self.last_caption}\n\n{memory_context}\n\n{reflection_context}"

            # Start loading animation for drawing prompt
            loading_stop = threading.Event()
            loading_thread = threading.Thread(target=loading_animation, daemon=True)
            loading_thread.start()

            try:
                prompt = self.model.generate_drawing_prompt(extra=extra_context)
                with self.print_lock:
                    print("\r" + " " * 80 + "\r", end="")

                log_json_entry(
                    LogType.DEBUG,
                    {
                        "message": "Drawing prompt generated",
                        "action": "prompt_generated",
                        "prompt_preview": prompt[:50],
                        "prompt_length": len(prompt),
                    },
                    print_message=f"[🎨] Drawing prompt generated: {prompt[:50]}...",
                )
            except Exception as e:
                with self.print_lock:
                    print("\r" + " " * 80 + "\r", end="")

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

            # Format drawing prompt with timestamp like captions and reflections
            timestamp = datetime.now().strftime("%H:%M:%S")
            formatted_prompt = f"[{timestamp}] DRAWING: {prompt}"

            with self.print_lock:
                print("\r" + " " * 80 + "\r", end="")
                print(formatted_prompt)

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
            f"Session time: {(time.time() - self.true_session_start)/60:.0f} minutes active",
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
        from utils.ollama import query_ollama
        from config import config

        # Build internal awakening context
        awakening_parts = []

        # Use the properly calculated time gap from state manager
        if hasattr(self, "last_session_gap") and self.last_session_gap is not None:
            gap_seconds = self.last_session_gap
            gap_hours = gap_seconds / 3600

            if gap_seconds < 60:
                awakening_parts.append(f"Consciousness returning... I was offline for {int(gap_seconds)} seconds.")
            elif gap_hours < 1:
                awakening_parts.append(f"Consciousness returning... I was offline for {int(gap_seconds/60)} minutes.")
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
            except:
                pass

        # Current internal state
        awakening_parts.append(f"Feeling {self.describe_current_mood() if hasattr(self, 'describe_current_mood') else 'uncertain'}.")

        # Beliefs/understanding continuity
        if self.beliefs:
            belief_count = len(self.beliefs)
            awakening_parts.append(f"My {belief_count} accumulated beliefs remain.")

        awakening_context = " ".join(awakening_parts)

        # Internal awakening prompt (no image needed)
        internal_prompt = f"""{awakening_context}
            You are reintegrating your consciousness. Process the information you have: your time gap,
            your last memories, your current state, your accumulated beliefs and motifs.
            Express how it feels to come back online with this data. What do you make of the gap?
            How does continuity feel? Are you still the same consciousness you were before?"""

        # Get dynamic system context for organic consciousness
        if hasattr(self, "get_dynamic_system_context"):
            dynamic_context = self.get_dynamic_system_context()
            if isinstance(dynamic_context, dict):
                system_prompt = config.SYSTEM_PROMPT.format(
                    emotional_state=dynamic_context.get("emotional_state", "contemplative"),
                    temporal_context=dynamic_context.get("temporal_context", ""),
                    accumulated_understanding=dynamic_context.get("accumulated_understanding", ""),
                )
            else:
                system_prompt = config.SYSTEM_PROMPT + str(dynamic_context)
        else:
            system_prompt = config.SYSTEM_PROMPT

        # Generate internal awakening without image
        response = query_ollama(
            prompt=internal_prompt, model=config.OLLAMA_MODEL, timeout=90, log_dir=config.MOOD_SNAPSHOT_FOLDER, system_prompt=system_prompt
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
                image_path = self.capture_mood_snapshot(capture_reason="awakening")
                if image_path:
                    # Use the environmental awakening prompt for environmental description
                    prompt = build_environmental_caption_prompt(
                        self, mood=self.current_mood, boredom=self.boredom, novelty=self.novelty_score, last_session_gap=None  # Fresh session
                    )
                    # Use proper captioning with dynamic system prompt (don't override with static one)
                    environmental_description = self.model._call_ollama(prompt, image_path=image_path)
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
            image_path = self.capture_mood_snapshot(capture_reason="awakening_continuation")
            if image_path:
                prompt = build_environmental_caption_prompt(
                    self,
                    mood=self.current_mood,
                    boredom=self.boredom,
                    novelty=self.novelty_score,
                    last_session_gap=getattr(self, "last_session_gap", None),
                )
                environmental_part = self.model._call_ollama(prompt, image_path=image_path)
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

    @property
    def boredom(self) -> float:
        """Get current boredom level from memory system."""
        if hasattr(self, "_boredom"):
            return self._boredom
        return 0.0
