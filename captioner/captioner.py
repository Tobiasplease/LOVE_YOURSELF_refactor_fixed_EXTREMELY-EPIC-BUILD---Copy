from __future__ import annotations
import os
import re
import time
import threading
from collections import deque
from typing import Deque, Optional, Tuple

import cv2  # type: ignore
import numpy as np  # type: ignore
from config.config import CAPTION_INTERVAL, DRAWING_INTERVAL, MOOD_SNAPSHOT_FOLDER, REASON_INTERVAL
from event_logging.event_logger import log_json_entry, LogType
from event_logging.run_manager import get_run_image_path
from drawing.drawing import DrawingController

from .memory import MemoryMixin
from .prompts import extract_motifs_spacy
from .model_wrapper import MultimodalModel


class Captioner(MemoryMixin):
    caption_window: Optional[any] = None  # type: ignore

    def __init__(self) -> None:
        super().__init__()
        self.model = MultimodalModel(memory_ref=self)
        self.drawing = DrawingController()

        self.true_session_start = time.time()
        self.first_caption_done = False
        self.awakening_done = False

        self.current_mood: float = 0.0
        self.last_caption: str = ""
        self.boredom: float = 0.0
        self.novelty_score: float = 0.0

        self.last_caption_time: float = 0.0
        self.last_reason_time: float = time.time()  # Delay first reflection
        self.last_drawing_time: float = time.time()  # Stagger drawing

        os.makedirs(MOOD_SNAPSHOT_FOLDER, exist_ok=True)
        self.snapshot_queue: Deque[Tuple[np.ndarray, bool]] = deque()
        threading.Thread(target=self._caption_worker, daemon=True).start()

    @property
    def is_processing(self) -> bool:
        return bool(self.snapshot_queue)

    def update(self, frame: Optional[np.ndarray] = None, *, person_present: bool = False, mood: Optional[float] = None) -> None:
        if frame is not None:
            if mood is not None:
                self.current_mood = mood
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

        try:
            caption = self.model.caption_image(img_path, flowing=True, first_time=not self.first_caption_done)
        except Exception as e:
            caption = "[⚠️] Vision unavailable"
            log_json_entry(
                LogType.ERROR,
                {"message": f"Caption error: {e}", "component": "captioner"},
                MOOD_SNAPSHOT_FOLDER,
                auto_print=True,
                print_message=f"⚠️ Caption error: {e}",
            )

        self.first_caption_done = True

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

        log_json_entry(
            LogType.CAPTION,
            {"caption": caption, "image_path": img_path, "mood": self.current_mood},
            MOOD_SNAPSHOT_FOLDER,
            auto_print=True,
            print_message=f"👁️ Caption: {caption}",
        )
        # logging mood in update_feeling_brain? dont need here?
        # if self.novelty_score > CAPTION_SAVE_THRESHOLD:
        #     log_mood(caption, self.current_mood, img_path)
        # log_json_entry(LogType.MOOD, {"caption": caption, "mood": self.current_mood, "image": img_path}, MOOD_SNAPSHOT_FOLDER)

        self.observe(caption, self.current_mood, img_path, memory_type="perception")
        self.last_caption = caption

        if now - self.last_reason_time > REASON_INTERVAL:
            mood_text = self.describe_current_mood()
            context = self.get_reflection_context()
            reflection = self.model.reason_about_caption(caption, agent=self, mood_text=mood_text, extra=context)

            if reflection and len(reflection.strip()) > 10:
                log_json_entry(
                    LogType.REFLECTION,
                    {"reflection": reflection, "mood": self.current_mood, "image_path": img_path, "context": context},
                    MOOD_SNAPSHOT_FOLDER,
                    auto_print=True,
                    print_message=f"🧠 Reflection: {reflection}",
                )
                self.last_reason_time = now
                self.awakening_done = True

                m = re.search(r"-?\d+(?:\.\d+)?", reflection)
                mood_val = float(m.group()) if m else self.current_mood
                self.current_mood += 0.25 * (mood_val - self.current_mood)

                for motif in extract_motifs_spacy(caption):
                    self.absorb_motif(motif)

                self.observe(reflection, self.current_mood, img_path, memory_type="reflection")

        if now - self.last_drawing_time > DRAWING_INTERVAL:
            memory_context = self.get_recent_memory()
            reflection_context = self.get_last_reflection()
            extra_context = f"{self.last_caption}\n\n{memory_context}\n\n{reflection_context}"
            prompt = self.model.generate_drawing_prompt(extra=extra_context)
            self.drawing.handle_drawing_flow(self, prompt, img_path, reflection=reflection_context)
            self.last_drawing_time = now

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
        return f"""Mood: {self.current_mood:.2f}
                Boredom: {self.boredom:.2f}
                Novelty: {self.novelty_score:.2f}
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

    @staticmethod
    def truncate_caption(raw: str) -> str:
        return " ".join(re.split(r"[.!?]", raw.strip())[0].split()[:18])
