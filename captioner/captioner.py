from __future__ import annotations
import os
import re
import time
import threading
from collections import deque
from typing import Deque, Optional, Tuple

import cv2  # type: ignore
import numpy as np  # type: ignore
from config.config import CAPTION_INTERVAL, DRAWING_INTERVAL, MOOD_SNAPSHOT_FOLDER, REASON_INTERVAL, CLEAN_CAPTION_OUTPUT
from event_logging.event_logger import log_json_entry, LogType
from event_logging.run_manager import get_run_image_path
from drawing.drawing import DrawingController

from .memory import MemoryMixin
from .prompts import extract_motifs_spacy
from .model_wrapper import MultimodalModel
from .emotional_voice import EmotionalVoiceManager


class Captioner(MemoryMixin):
    caption_window: Optional[any] = None  # type: ignore

    def __init__(self) -> None:
        super().__init__()
        self.model = MultimodalModel(memory_ref=self)
        self.drawing = DrawingController()
        
        # Initialize emotional voice system for dynamic personality expression
        self.emotional_voice = EmotionalVoiceManager()

        self.true_session_start = time.time()
        self.first_caption_done = False
        self.awakening_done = False

        self.current_mood: float = 0.0
        self.last_caption: str = ""
        self.boredom: float = 0.0
        self.novelty_score: float = 0.0
        self.current_temporal_context: dict = {}  # Store temporal awareness

        # Person presence tracking for temporal relationships
        self.person_present: bool = False
        self.last_person_seen: float = 0.0
        self.person_absence_start: float = 0.0
        self.greeting_given_this_session: bool = False

        self.last_caption_time: float = 0.0
        self.last_reason_time: float = time.time()  # Delay first reflection
        self.last_drawing_time: float = time.time()  # Stagger drawing

        # Track session continuity
        self.sessions_since_boot = 0
        self.memory_loaded_from_previous = False

        os.makedirs(MOOD_SNAPSHOT_FOLDER, exist_ok=True)
        self.snapshot_queue: Deque[Tuple[np.ndarray, bool]] = deque()
        threading.Thread(target=self._caption_worker, daemon=True).start()

    @property
    def is_processing(self) -> bool:
        return bool(self.snapshot_queue)

    def get_pending_priority_task(self) -> str:
        """Check if any high-priority tasks are due. Returns task type or None."""
        now = time.time()
        
        # Check reflection (identity consolidation) - highest priority
        if now - self.last_reason_time > REASON_INTERVAL:
            return "reflection"
            
        # Check drawing prompt - second priority  
        if now - self.last_drawing_time > DRAWING_INTERVAL:
            return "drawing"
            
        return None

    def handle_priority_task(self, task_type: str, caption: str, img_path: str) -> bool:
        """Handle high-priority tasks with status indicators. Returns True if task was executed."""
        now = time.time()
        
        if task_type == "reflection":
            # Show status indicator
            if CLEAN_CAPTION_OUTPUT:
                print("\n-reflecting-")
            else:
                print("\n🧠 [Reflecting on recent experiences...]")
            
            mood_text = self.describe_current_mood()
            context = self.get_reflection_context()
            
            # Generate identity consolidation
            reflection = self.model.reason_about_caption(caption, agent=self, mood_text=mood_text, extra=context)
            
            if reflection and len(reflection.strip()) > 10:
                if CLEAN_CAPTION_OUTPUT:
                    # Format reflection nicely with clear sections
                    formatted_reflection = reflection.strip()
                    
                    # Add section headers if not present
                    if "CORE IDENTITY:" in formatted_reflection:
                        formatted_reflection = formatted_reflection.replace("CORE IDENTITY:", "\n🧠 CORE IDENTITY:")
                    if "CONSCIOUSNESS QUESTIONS:" in formatted_reflection:
                        formatted_reflection = formatted_reflection.replace("CONSCIOUSNESS QUESTIONS:", "\n💭 CONSCIOUSNESS QUESTIONS:")
                    if "FORWARD DIRECTION:" in formatted_reflection:
                        formatted_reflection = formatted_reflection.replace("FORWARD DIRECTION:", "\n🎯 FORWARD DIRECTION:")
                    
                    # Fix bullet point formatting more carefully
                    lines = formatted_reflection.split('\n')
                    fixed_lines = []
                    for line in lines:
                        # Fix malformed bullet points like "• Who am I. : " 
                        line = line.replace("• Who am I. :", "• Who am I:")
                        line = line.replace("• Where am I.", "• Where am I:")
                        line = line.replace("• What do", "• What do")
                        # Ensure proper spacing for bullet points
                        if line.strip().startswith("•") and not line.startswith("  •"):
                            line = "  " + line.strip()
                        fixed_lines.append(line)
                    
                    formatted_reflection = '\n'.join(fixed_lines)
                    
                    print(f"\n{formatted_reflection}\n")
                else:
                    # Show identity consolidation process with full header
                    print(f"\n🧠 [Identity Consolidation]\n{reflection}\n")

                log_json_entry(
                    LogType.REFLECTION,
                    {"identity_consolidation": reflection, "mood": self.current_mood, "image_path": img_path, "context": context},
                    MOOD_SNAPSHOT_FOLDER,
                    auto_print=not CLEAN_CAPTION_OUTPUT,
                    print_message=None,
                )
                self.last_reason_time = now
                self.awakening_done = True
                
                # Extract and update mood value if present
                m = re.search(r"-?\d+(?:\.\d+)?", reflection)
                if m:
                    try:
                        self.current_mood = float(m.group())
                    except ValueError:
                        pass
                
                # Save to memory
                self.observe(reflection, self.current_mood, img_path, memory_type="reflection")
            
            return True
            
        elif task_type == "drawing":
            # Show status indicator
            if CLEAN_CAPTION_OUTPUT:
                print("\n-thinking of a drawing-")
            else:
                print("\n🎨 [Contemplating artistic expression...]")
            
            memory_context = self.get_recent_memory()
            reflection_context = self.get_last_reflection()
            extra_context = f"{self.last_caption}\n\n{memory_context}\n\n{reflection_context}"
            prompt = self.model.generate_drawing_prompt(extra=extra_context)
            
            # Show drawing prompt in clean output with asterisks
            if CLEAN_CAPTION_OUTPUT and prompt and not prompt.startswith("[⚠️]"):
                print(f"\n*{prompt}*\n")
            
            self.drawing.handle_drawing_flow(self, prompt, img_path, reflection=reflection_context)
            self.last_drawing_time = now
            
            return True
            
        return False

    def update(self, frame: Optional[np.ndarray] = None, *, person_present: bool = False, mood: Optional[float] = None, temporal_context: Optional[dict] = None) -> None:
        if frame is not None:
            if mood is not None:
                self.current_mood = mood
            if temporal_context is not None:
                self.current_temporal_context = temporal_context
            if len(self.snapshot_queue) > 1:
                self.snapshot_queue.pop()
            self.snapshot_queue.append((frame.copy(), person_present))

    def _caption_worker(self):
        while True:
            if self.snapshot_queue:
                frame, _ = self.snapshot_queue.popleft()
                try:
                    self._process_frame(frame)
                except Exception as exc:
                    log_json_entry(
                        LogType.ERROR,
                        {"message": f"Caption thread error: {exc}", "component": "captioner"},
                        MOOD_SNAPSHOT_FOLDER,
                        auto_print=True,
                        print_message=f"⚠️ Caption thread error: {exc}",
                    )
            else:
                time.sleep(0.05)

    def _process_frame(self, frame: np.ndarray) -> None:
        now = time.time()
        if now - self.last_caption_time < CAPTION_INTERVAL:
            return

        self.last_caption_time = now
        ts = int(now)
        img_path = get_run_image_path(MOOD_SNAPSHOT_FOLDER, f"mood_{ts}.jpg")
        cv2.imwrite(img_path, frame)

        # Check for high-priority tasks first
        priority_task = self.get_pending_priority_task()
        if priority_task:
            try:
                # Generate a basic caption for context in priority tasks
                caption = self.model.caption_image(img_path, flowing=True, first_time=False, temporal_context=self.current_temporal_context)
                if "[⚠️]" not in caption:
                    # Handle priority task (this will show status indicators)
                    if self.handle_priority_task(priority_task, caption, img_path):
                        return  # Priority task completed, skip regular captioning this cycle
            except Exception as e:
                log_json_entry(
                    LogType.ERROR,
                    {"message": f"Priority task error: {e}", "component": "captioner"},
                    MOOD_SNAPSHOT_FOLDER,
                    auto_print=True,
                    print_message=f"⚠️ Priority task error: {e}",
                )

        try:
            # Regular caption generation - animation already happened during awakening
            caption = self.model.caption_image(img_path, flowing=True, first_time=False, temporal_context=self.current_temporal_context)
        except Exception as e:
            caption = "[⚠️] Vision unavailable"
            log_json_entry(
                LogType.ERROR,
                {"message": f"Caption error: {e}", "component": "captioner"},
                MOOD_SNAPSHOT_FOLDER,
                auto_print=True,
                print_message=f"⚠️ Caption error: {e}",
            )

        if "[⚠️]" in caption:
            log_json_entry(
                LogType.ERROR,
                {"message": f"Caption error: {caption}", "component": "captioner"},
                MOOD_SNAPSHOT_FOLDER,
                auto_print=True,
                print_message=f"📍 Caption error: {caption}",
            )
            self.observe("I couldn’t see anything just now.", self.current_mood, img_path, memory_type="glitch")
            return

        if CLEAN_CAPTION_OUTPUT:
            # Suppress old-style output, only print clean caption if needed
            pass

        log_json_entry(
            LogType.CAPTION,
            {"caption": caption, "image_path": img_path, "mood": self.current_mood},
            MOOD_SNAPSHOT_FOLDER,
            auto_print=not CLEAN_CAPTION_OUTPUT,
            print_message=None,
        )
        # logging mood in update_feeling_brain? dont need here?
        # if self.novelty_score > CAPTION_SAVE_THRESHOLD:
        #     log_mood(caption, self.current_mood, img_path)
        # log_json_entry(LogType.MOOD, {"caption": caption, "mood": self.current_mood, "image": img_path}, MOOD_SNAPSHOT_FOLDER)

        self.observe(caption, self.current_mood, img_path, memory_type="perception")
        self.last_caption = caption

    def describe_current_mood(self) -> str:
        if self.current_mood > 0.5:
            return "I feel quite energized and attentive."
        elif self.current_mood > 0.1:
            return "I'm calm but curious."
        elif self.current_mood > -0.1:
            return "I feel neutral and observant."
        elif self.current_mood > -0.5:
            return "I'm feeling distracted or unfocused."
        else:
            return "I feel dull, distant, and unfocused."

    def get_reflection_context(self) -> str:
        # Import the helper function to convert mood to descriptive text
        from captioner.prompts import describe_mood_state
        
        mood_description = describe_mood_state(self.current_mood, self.boredom, self.novelty_score)
        
        return f"""Emotional state: {mood_description}
                Identity: {self.get_identity_summary()}
                Recent memory: {self.get_recent_memory()}""".strip()

    def get_recent_memory(self, k: int = 5) -> str:
        snippets = self.get_clean_memory_snippets(k=k)
        return "\n".join(f"- {s}" for s in snippets)

    def get_last_reflection(self) -> str:
        entries = self.get_memory_entries_by_type("reflection")
        if entries:
            return entries[-1].get("text", "")
        return ""

    def generate_awakening_message(self, time_since_last: str | None = None, previous_beliefs: dict | None = None) -> str:
        """Generate awakening message and immediately process first environmental observation."""
        # Show animation and generate first real observation
        print("")  # Add line spacing before animation
        import threading
        import sys
        import cv2
        from event_logging.run_manager import get_run_image_path
        
        def show_awakening_animation(stop_event):
            """Display the cute ASCII awakening animation while processing."""
            from itertools import cycle
            
            dots_cycle = cycle([".", "..", "..."])
            
            while not stop_event.is_set():
                current_dots = next(dots_cycle)
                # Clear the line completely, then write the new frame
                sys.stdout.write(f"\r{' ' * 20}\r-_- {current_dots}")
                sys.stdout.flush()
                import time
                time.sleep(1.2)
            
            # Final awakening frame
            sys.stdout.write(f"\r{' ' * 20}\rO_O I am awake!")
            sys.stdout.flush()
            print("")  # New line after animation

        # Start animation
        stop_animation = threading.Event()
        animation_thread = threading.Thread(target=show_awakening_animation, args=(stop_animation,), daemon=True)
        animation_thread.start()
        
        # Wait a moment for camera to be ready, then get truly fresh frame
        import time
        time.sleep(1.0)  # Let camera stabilize
        
        # Get a fresh current frame for first observation - this should come from machine.py
        # For now, just ensure we mark as done to prevent duplication
        self.first_caption_done = True
        
        # Ensure minimum display time for animation visibility
        start_time = time.time()
        elapsed = time.time() - start_time
        if elapsed < 3.0:  # At least 3 seconds for the full animation
            time.sleep(3.0 - elapsed)

        stop_animation.set()
        animation_thread.join(timeout=0.5)
        
        return "First environmental observation will follow..."

    def mark_awakening_complete(self):
        """Mark that awakening is complete to prevent duplicate first captions."""
        self.first_caption_done = True

    def update_person_presence(self, present: bool, timestamp: float):
        """Track person presence for temporal relationship building."""
        from utils.continuity import describe_duration
        
        if present and not self.person_present:
            # Person just appeared
            if self.person_absence_start > 0:
                absence_duration = describe_duration(self.person_absence_start)
                if not self.greeting_given_this_session:
                    # First time seeing person this session
                    self.observe(f"Person returns after {absence_duration} - first encounter this session", 
                               self.current_mood, memory_type="temporal_relationship")
                    self.greeting_given_this_session = True
                elif timestamp - self.person_absence_start > 300:  # 5+ minutes
                    # Notable absence
                    self.observe(f"Person returns after {absence_duration} away", 
                               self.current_mood, memory_type="temporal_relationship")
            
            self.person_present = True
            self.last_person_seen = timestamp
            
        elif not present and self.person_present:
            # Person just left
            self.person_present = False
            self.person_absence_start = timestamp

    @staticmethod
    def truncate_caption(raw: str) -> str:
        return " ".join(re.split(r"[.!?]", raw.strip())[0].split()[:18])
