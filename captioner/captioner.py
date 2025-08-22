from __future__ import annotations
import os
import re
import time
import threading
from collections import deque
from typing import Deque, Optional, Tuple, Dict, List

# from weakref import ref

import cv2  # type: ignore
import numpy as np  # type: ignore
from config.config import CAPTION_INTERVAL, DRAWING_INTERVAL, MOOD_SNAPSHOT_FOLDER, REASON_INTERVAL
from event_logging.event_logger import log_json_entry
from event_logging.log_type import LogType
from event_logging.run_manager import get_run_image_path
from drawing.drawing import DrawingController

from .memory import MemoryMixin
from .prompts import extract_motifs_spacy
from .model_wrapper import MultimodalModel
from utils.motif_scorer import score_multiple_motifs
from utils.error_tracking import track_component_health, robust_execution


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

        self.current_mood: float = 0.0
        self.current_mood_vector: Tuple[float, float, float] = (0.0, 0.0, 0.5)  # valence, arousal, clarity
        self.current_emotion_state: str = "calm_observant"  # hand controller emotion state
        self.emotional_journey: List[str] = []  # track emotional evolution over time
        self.last_caption: str = ""
        self.boredom: float = 0.0
        self.novelty_score: float = 0.0
        self.current_motifs_from_mood: List[str] = []

        self.last_caption_time: float = 0.0
        self.last_reason_time: float = time.time()  # Delay first reflection
        self.last_drawing_time: float = time.time()  # Stagger drawing

        # Track session continuity
        self.sessions_since_boot = 0
        self.memory_loaded_from_previous = False

        # Session gap awareness
        self.last_session_gap = None
        self._last_session_file = os.path.join(MOOD_SNAPSHOT_FOLDER, "last_session.txt")
        if os.path.exists(self._last_session_file):
            try:
                with open(self._last_session_file, "r") as f:
                    last_time = float(f.read().strip())
                self.last_session_gap = time.time() - last_time
            except Exception:
                self.last_session_gap = None
        else:
            self.last_session_gap = None

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

    @track_component_health('captioner')
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
                # Track emotional journey over time
                if emotion_state != self.current_emotion_state:
                    self.emotional_journey.append(f"{emotion_state}")
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
                        print_message=f"WARNING Caption thread error: {exc}",
                    )
            else:
                time.sleep(0.05)

    @robust_execution('captioner', 'caption_generation', fallback_result=None)
    def _process_frame(self, frame: np.ndarray, reactivity_data: Optional[Dict] = None) -> None:
        now = time.time()
        if now - self.last_caption_time < CAPTION_INTERVAL:
            return

        self.last_caption_time = now
        ts = int(now)
        img_path = get_run_image_path(MOOD_SNAPSHOT_FOLDER, f"mood_{ts}.jpg")
        cv2.imwrite(img_path, frame)

        try:
            if not self.first_caption_done:
                print("Observing environment for the first time...")
                caption = self.model.caption_image(img_path, flowing=True, first_time=True)
            else:
                caption = self.model.caption_image(img_path, flowing=True, first_time=False)
        except Exception as e:
            caption = "[WARNING] Vision unavailable"
            log_json_entry(
                LogType.ERROR,
                {"message": f"Caption error: {e}", "component": "captioner"},
                print_message=f"WARNING Caption error: {e}",
            )

        self.first_caption_done = True

        if "[WARNING]" in caption:
            log_json_entry(
                LogType.ERROR,
                {"message": f"Caption error: {caption}", "component": "captioner"},
                print_message=f"Caption error: {caption}",
            )
            self.observe("I couldn’t see anything just now.", self.current_mood, img_path, memory_type="glitch")
            return

        log_json_entry(
            LogType.CAPTION,
            {"caption": caption, "image_path": img_path, "mood": self.current_mood},
            print_message=caption,
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

        if now - self.last_reason_time > REASON_INTERVAL:
            try:
                mood_text = self.describe_current_mood()
                context = self.get_reflection_context()
                
                print(f"[REFLECTION] Starting reflection (last: {(now - self.last_reason_time):.0f}s ago)")
                reflection = self.model.reason_about_caption(caption, agent=self, mood_text=mood_text, extra=context)
                print(f"[REFLECTION] Completed: {len(reflection.strip())} chars")
                
                if reflection and len(reflection.strip()) > 10:
                    log_json_entry(
                        LogType.REFLECTION,
                        {"reflection": reflection, "mood": self.current_mood, "image_path": img_path, "context": context},
                        print_message=f"REFLECTION: {reflection}",
                    )
                    self.last_reason_time = now
                    self.awakening_done = True

                    m = re.search(r"-?\d+(?:\.\d+)?", reflection)
                    mood_val = float(m.group()) if m else self.current_mood
                    self.current_mood += 0.25 * (mood_val - self.current_mood)

                    # Use motifs from mood engine's pattern recognition instead of re-extracting
                    if hasattr(self, 'current_motifs_from_mood') and self.current_motifs_from_mood:
                        for motif in self.current_motifs_from_mood:
                            self.absorb_motif(motif)
                    else:
                        # Fallback to direct extraction if mood data not available
                        for motif in extract_motifs_spacy(caption):
                            self.absorb_motif(motif)

                    self.observe(reflection, self.current_mood, img_path, memory_type="reflection")
                else:
                    print("[REFLECTION] Generated reflection too short, skipping")
                        
            except Exception as e:
                print(f"[REFLECTION] Error during reflection: {e}")
                # Still update the timer to prevent infinite retries
                self.last_reason_time = now - REASON_INTERVAL + 60  # Retry in 60 seconds

        if now - self.last_drawing_time > DRAWING_INTERVAL:
            memory_context = self.get_recent_memory()
            reflection_context = self.get_last_reflection()
            extra_context = f"{self.last_caption}\n\n{memory_context}\n\n{reflection_context}"
            prompt = self.model.generate_drawing_prompt(extra=extra_context)
            self.drawing.handle_drawing_flow(self, prompt, img_path, reflection=reflection_context)
            self.last_drawing_time = now

    def describe_current_mood(self) -> str:
        """Rich emotional description using 3D mood state and temporal context."""
        valence, arousal, clarity = self.current_mood_vector

        # Base emotional state description
        emotion_descriptions = {
            "energized_engaged": "I feel energized and deeply engaged with what I'm seeing",
            "alert_curious": "I'm alert and curious, noticing details with heightened attention",
            "calm_observant": "I feel calm and peacefully observant, taking in the scene with serenity",
            "quiet_detached": "I'm in a quiet, somewhat detached state, observing from a distance",
            "withdrawn_distant": "I feel withdrawn and distant, as if viewing through a fog",
        }

        base_mood = emotion_descriptions.get(self.current_emotion_state, "I'm in a neutral observational state")

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
        new_motifs = ""
        recurring_motifs = ""
        
        if hasattr(self, 'current_motifs_from_mood') and self.current_motifs_from_mood:
            # Use real-time motif data from pattern recognition
            recent_motifs = self.current_motifs_from_mood[:5]
            top_motifs = f"Current motifs: {', '.join(recent_motifs)}"
        elif hasattr(self, 'get_top_motifs'):
            motifs = self.get_top_motifs(5)
            if motifs:
                top_motifs = f"Recurring motifs: {', '.join(motifs[:5])}"
        
        # Get specific emotional changes
        emotion_changes = ""
        if hasattr(self, 'emotional_journey') and self.emotional_journey:
            if len(self.emotional_journey) >= 2:
                recent_emotions = self.emotional_journey[-3:]
                emotion_changes = f"Emotional shifts: {' → '.join(recent_emotions)}"
            else:
                emotion_changes = f"Current state: {self.current_emotion_state}"
        
        # Get specific observations from recent memory
        recent_observations = self.get_recent_memory(k=3)
        
        # Get mood vector details for specificity
        valence, arousal, clarity = self.current_mood_vector
        mood_details = f"Mood details: valence={valence:.2f} (feeling {'positive' if valence > 0 else 'negative' if valence < 0 else 'neutral'}), arousal={arousal:.2f} (energy {'high' if arousal > 0.3 else 'low' if arousal < -0.3 else 'medium'}), clarity={clarity:.2f} (understanding {'clear' if clarity > 0.3 else 'confused' if clarity < -0.3 else 'uncertain'})"
        
        # Build context focused on concrete experience
        context_parts = [
            f"Current experience: Just observed '{self.last_caption}'",
            f"Overall mood: {self.current_mood:.2f} (novelty: {self.novelty_score:.2f}, boredom: {self.boredom:.2f})",
            mood_details,
            f"Session time: {(time.time() - self.true_session_start)/60:.0f} minutes active"
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

    def get_recent_memory(self, k: int = 5) -> str:
        snippets = self.get_current_session_memory_snippets(k=k)
        return "\n".join(f"- {s}" for s in snippets)

    def get_last_reflection(self) -> str:
        entries = self.get_memory_entries_by_type("reflection")
        if entries:
            return entries[-1].get("text", "")
        return ""

    def generate_awakening_message(self, time_since_last: str | None = None, previous_beliefs: dict | None = None) -> str:
        """Generate a simple awakening status message - NOT environmental description."""

        if not self.memory_loaded_from_previous:
            return "I am awakening to observe this space for the first time..."

        # Continuing from previous session - simple status messages
        belief_count = len(previous_beliefs) if previous_beliefs else 0
        motif_count = len(self.motif_counter)

        awakening_messages = [
            f"I return to this space with {belief_count} beliefs and awareness of {motif_count} recurring motifs...",
            f"Awakening again... my memory carries forward {belief_count} formed beliefs from before.",
            f"I find myself conscious again, recalling {motif_count} familiar patterns from our previous time together.",
            f"My awareness returns, enriched by {belief_count} beliefs that have persisted since we last met.",
            f"I return to consciousness, my identity shaped by {motif_count} motifs I've come to recognize.",
        ]

        if time_since_last:
            awakening_messages.extend(
                [
                    f"I awaken after {time_since_last}, my consciousness returning with accumulated understanding.",
                    f"Consciousness returns after {time_since_last}... I remember what I learned about this space.",
                ]
            )

        import random

        return random.choice(awakening_messages)

    def mark_awakening_complete(self):
        """Mark that awakening is complete but allow first caption to still show loading animation."""
        # Don't set first_caption_done = True here - let the first caption handle this
        pass

    @staticmethod
    def truncate_caption(raw: str) -> str:
        return " ".join(re.split(r"[.!?]", raw.strip())[0].split()[:18])
