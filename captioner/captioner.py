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

        # Start loading animation in separate thread
        import threading
        loading_stop = threading.Event()
        
        def loading_animation():
            frames = [" ", ".", "..", "..."]
            idx = 0
            while not loading_stop.is_set():
                print(f"\r{frames[idx % 4]}", end="", flush=True)
                idx += 1
                time.sleep(0.3)
        
        loading_thread = threading.Thread(target=loading_animation, daemon=True)
        loading_thread.start()
        
        try:
            if not self.first_caption_done:
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
        finally:
            # Stop loading animation
            loading_stop.set()
            loading_thread.join(timeout=0.5)

        self.first_caption_done = True

        if "[WARNING]" in caption:
            log_json_entry(
                LogType.ERROR,
                {"message": f"Caption error: {caption}", "component": "captioner"},
                print_message=f"Caption error: {caption}",
            )
            self.observe("I couldn’t see anything just now.", self.current_mood, img_path, memory_type="glitch")
            return

        # Clear the animation line and print caption with timestamp
        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_caption = f"[{timestamp}] {caption}"
        
        print(f"\r{formatted_caption}")
        print()  # Add blank line after caption
        
        log_json_entry(
            LogType.CAPTION,
            {"caption": caption, "image_path": img_path, "mood": self.current_mood},
            print_message=None,  # Don't double-print
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
                
                # Start loading animation for reflection
                loading_stop = threading.Event()
                
                def loading_animation():
                    frames = [" ", ".", "..", "..."]
                    idx = 0
                    while not loading_stop.is_set():
                        print(f"\r{frames[idx % 4]}", end="", flush=True)
                        idx += 1
                        time.sleep(0.3)
                
                loading_thread = threading.Thread(target=loading_animation, daemon=True)
                loading_thread.start()
                
                try:
                    reflection = self.model.reason_about_caption(caption, agent=self, mood_text=mood_text, extra=context)
                finally:
                    # Stop loading animation
                    loading_stop.set()
                    loading_thread.join(timeout=0.5)
                
                if reflection and len(reflection.strip()) > 10:
                    # Format reflection with timestamp like captions
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    formatted_reflection = f"[{timestamp}] REFLECTION: {reflection}"
                    
                    print(f"\r{formatted_reflection}")
                    print()  # Add blank line after reflection
                    
                    log_json_entry(
                        LogType.REFLECTION,
                        {"reflection": reflection, "mood": self.current_mood, "image_path": img_path, "context": context},
                        print_message=None,  # Don't double-print
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
                    # Clear animation line for short reflection message
                    print(f"\r[REFLECTION] Generated reflection too short, skipping")
                        
            except Exception as e:
                print(f"[REFLECTION] Error during reflection: {e}")
                # Still update the timer to prevent infinite retries
                self.last_reason_time = now - REASON_INTERVAL + 60  # Retry in 60 seconds

        # Debug: Check drawing interval condition
        time_since_last_drawing = now - self.last_drawing_time
        if time_since_last_drawing > DRAWING_INTERVAL:
            print(f"\r[DEBUG] Drawing interval reached ({time_since_last_drawing:.0f}s > {DRAWING_INTERVAL}s), generating prompt...")
            
            memory_context = self.get_recent_memory()
            reflection_context = self.get_last_reflection()
            extra_context = f"{self.last_caption}\n\n{memory_context}\n\n{reflection_context}"
            
            # Start loading animation for drawing prompt
            loading_stop = threading.Event()
            
            def loading_animation():
                frames = [" ", ".", "..", "..."]
                idx = 0
                while not loading_stop.is_set():
                    print(f"\r{frames[idx % 4]}", end="", flush=True)
                    idx += 1
                    time.sleep(0.3)
            
            loading_thread = threading.Thread(target=loading_animation, daemon=True)
            loading_thread.start()
            
            try:
                prompt = self.model.generate_drawing_prompt(extra=extra_context)
                print(f"\r[DEBUG] Drawing prompt generated: {prompt[:50]}...")
            except Exception as e:
                print(f"\r[DEBUG] Error generating drawing prompt: {e}")
                prompt = "[ERROR] Drawing prompt generation failed"
            finally:
                # Stop loading animation
                loading_stop.set()
                loading_thread.join(timeout=0.5)
            
            # Format drawing prompt with timestamp like captions and reflections
            timestamp = datetime.now().strftime("%H:%M:%S")
            formatted_prompt = f"[{timestamp}] DRAWING: {prompt}"
            
            print(f"\r{formatted_prompt}")
            print()  # Add blank line after drawing prompt
            
            self.drawing.handle_drawing_flow(self, prompt, img_path, reflection=reflection_context)
            self.last_drawing_time = now

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

    def generate_awakening_message(self, time_since_last: str | None = None, previous_beliefs: dict | None = None) -> str:
        """Generate comprehensive awakening with environmental description - THE ONLY awakening now."""
        
        # Import the rich prompt builder
        from .prompts import build_awakening_prompt
        
        # For fresh sessions, trigger environmental description
        if not self.memory_loaded_from_previous:
            # Take a snapshot and describe the environment
            try:
                image_path = self.capture_mood_snapshot(capture_reason="awakening")
                if image_path:
                    # Use the rich awakening prompt for environmental description
                    prompt = build_awakening_prompt(
                        self,
                        mood=self.current_mood,
                        boredom=self.boredom,
                        novelty=self.novelty_score
                    )
                    # Use proper captioning with dynamic system prompt (don't override with static one)
                    environmental_description = self.model._call_ollama(prompt, image_path=image_path)
                    return environmental_description
            except Exception as e:
                pass
            return "I am awakening to observe this space for the first time..."

        # Continuing from previous session - include environmental awareness
        belief_count = len(previous_beliefs) if previous_beliefs else 0
        motif_count = len(self.motif_counter)

        # First, provide status then environmental description
        status_prefix = f"Awakening after {time_since_last or 'some time'}... consciousness returns with {belief_count} beliefs and {motif_count} familiar motifs."
        
        # Then add environmental description
        try:
            image_path = self.capture_mood_snapshot(capture_reason="awakening_continuation")
            if image_path:
                prompt = build_awakening_prompt(
                    self,
                    mood=self.current_mood,
                    boredom=self.boredom,
                    novelty=self.novelty_score
                )
                environmental_part = self.model._call_ollama(prompt, image_path=image_path)
                return f"{status_prefix} {environmental_part}"
        except Exception as e:
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
        if hasattr(self, '_novelty_score'):
            return self._novelty_score
        return 0.0
    
    @property 
    def boredom(self) -> float:
        """Get current boredom level from memory system."""
        if hasattr(self, '_boredom'):
            return self._boredom
        return 0.0
