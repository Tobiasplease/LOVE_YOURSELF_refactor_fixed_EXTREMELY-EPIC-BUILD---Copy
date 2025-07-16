
from __future__ import annotations

import os
import time
import threading
import random
import base64
import re
from typing import Optional, Deque, Tuple
from collections import deque

import cv2  # type: ignore
import requests  # type: ignore
import numpy as np  # type: ignore

from config.config import MOOD_SNAPSHOT_FOLDER
from .memory import (
    MemoryMixin,
    CAPTION_SAVE_THRESHOLD,
)
from .prompts import (
    LLAVA_PROMPT,
    build_context,
    build_summary_prompt,
    build_caption_prompt,
    build_self_evaluation_prompt,
)
from mood.mood import generate_internal_note, log_mood, estimate_mood_llava

class Captioner(MemoryMixin):
    """High‑level controller that orchestrates perception, memory, and mood."""

    caption_window: Optional[any] = None

    def __init__(self) -> None:
        MemoryMixin.__init__(self)

        self.stream_interval: int = 3      # real-time
        self.summary_interval: int = 600
        self.eval_interval: int = 300      # 5 minutes

        self.session_start: float = time.time()
        self.last_stream_time: float = time.time()
        self.last_eval_time: float = time.time()

        self.current_mood: float = 0.0
        self.last_caption: str = ""
        self.last_short_caption: str = ""
        self.last_mood: float = 0.0
        self.last_person_detected: bool = False
        self.summary_history: list[str] = []
        self.evaluation_journal: list[str] = []
        self.initial_prompt_run: bool = False

        os.makedirs(MOOD_SNAPSHOT_FOLDER, exist_ok=True)

        self.snapshot_queue: Deque[Tuple[np.ndarray, bool]] = deque()
        self.caption_thread = threading.Thread(target=self._caption_worker, daemon=True)
        self.caption_thread.start()

    @property
    def is_processing(self) -> bool:
        return bool(self.snapshot_queue)

    def update(self, frame: Optional[np.ndarray] = None, *, person_present: bool = False, mood: Optional[float] = None) -> None:
        if frame is not None:
            # discard older pending frame
            if len(self.snapshot_queue) > 1:
                self.snapshot_queue.pop()
            self.snapshot_queue.append((frame.copy(), person_present))
            print("...")

        now = time.time()
        if now - self.session_start > self.summary_interval:
            threading.Thread(target=self.generate_summary, daemon=True).start()
            self.session_start = now
        if now - self.last_eval_time > self.eval_interval:
            threading.Thread(target=self.generate_self_evaluation, daemon=True).start()
            self.last_eval_time = now

    def _caption_worker(self):
        while True:
            if self.snapshot_queue:
                frame, person_present = self.snapshot_queue.popleft()
                try:
                    self._process_frame(frame, person_present)
                except Exception as e:
                    print(f"[⚠️] Caption thread error: {e}")
            else:
                time.sleep(0.05)

    def _process_frame(self, frame: np.ndarray, person_present: bool) -> None:
        timestamp = int(time.time())
        filename = f"{MOOD_SNAPSHOT_FOLDER}/mood_{timestamp}.jpg"
        cv2.imwrite(filename, frame)

        prompt = LLAVA_PROMPT if not self.initial_prompt_run else build_caption_prompt(
            self,
            current_mood=self.current_mood,
            boredom=self.boredom,
            novelty=self.novelty_score,
        )

        caption, _ = self.describe_image_with_llava(filename, prompt=prompt)
        self.initial_prompt_run = True

        self.update_meta_memory(caption)
        self.estimate_novelty()
        self.update_boredom()

        mood_val = estimate_mood_llava(caption)
        drift = 0.25
        self.current_mood += drift * (mood_val - self.current_mood)

        saw_person = "person" in caption.lower() or "individual" in caption.lower()
        note = generate_internal_note(
            caption,
            self.last_caption,
            self.current_mood,
            self.last_mood,
            saw_person,
            self.last_person_detected,
        )

        print(f"[{time.strftime('%H:%M:%S')}] {caption.strip()}")

        if self.novelty_score > CAPTION_SAVE_THRESHOLD:
            log_mood(caption, self.current_mood, filename)
            short_caption = self.truncate_caption(caption)
            self.memory_queue.append((timestamp, short_caption, self.current_mood, filename))
            self.long_memory.append((timestamp, short_caption, self.current_mood, filename))
            self.compress_long_memory()

        self.last_caption = caption
        self.last_mood = self.current_mood
        self.last_person_detected = saw_person
        self.last_short_caption = self.truncate_caption(caption)

        self.cleanup_snapshots(MOOD_SNAPSHOT_FOLDER)
        self.cleanup_snapshots(MOOD_SNAPSHOT_FOLDER)

    @staticmethod
    def describe_image_with_llava(image_path: str, *, prompt: str) -> tuple[str, str]:
        try:
            with open(image_path, "rb") as img_file:
                encoded = base64.b64encode(img_file.read()).decode("utf-8")
            payload = {
                "model": "llava",
                "prompt": prompt,
                "images": [encoded],
                "stream": False,
            }
            resp = requests.post("http://localhost:11434/api/generate", json=payload, timeout=60)
            resp.raise_for_status()
            result = resp.json()
            return result.get("response", "[No response received]"), image_path
        except Exception as exc:
            return f"[⚠️] LLaVA failed: {exc}", image_path

    @staticmethod
    def truncate_caption(raw_caption: str) -> str:
        sentence = re.split(r"[.!?]", raw_caption.strip())[0].strip().lower()
        words = sentence.split()
        return " ".join(words[:18])

    def generate_summary(self) -> None:
        if not self.memory_queue:
            return

        try:
            recent_lines = [m[1] for m in self.memory_queue[-50:]]
            prior_context = "\n".join(f"- {s}" for s in recent_lines)
            moods = [m[2] for m in self.memory_queue]
            avg_mood = round(sum(moods) / len(moods), 2) if moods else 0.0
            past_summaries = "\n".join(f"→ {s}" for s in self.summary_history[-3:])

            summary_prompt = build_summary_prompt(prior_context, avg_mood, past_summaries)
            latest_image = self.memory_queue[-1][3]
            summary, _ = self.describe_image_with_llava(latest_image, prompt=summary_prompt)
            print(f"In summary: {summary}")

            self.summary_history.append(summary)
            self.memory_queue = self.memory_queue[-3:]
        except Exception as e:
            print(f"[⚠️] Failed to generate summary: {e}")

        self.cleanup_snapshots(MOOD_SNAPSHOT_FOLDER)

    def generate_self_evaluation(self) -> None:
        try:
            mood_vals = [m[2] for m in self.memory_queue[-10:]]
            mood_delta = mood_vals[-1] - mood_vals[0] if len(mood_vals) > 1 else 0.0
            time_elapsed = int(time.time() - self.session_start)

            summary_context = "\n".join(s for s in self.summary_history[-3:])
            eval_prompt = build_self_evaluation_prompt(
                memory=self,
                mood_delta=mood_delta,
                time_elapsed=time_elapsed,
                recent_summaries=summary_context,
            )
            latest_image = self.memory_queue[-1][3] if self.memory_queue else None
            if not latest_image:
                return

            evaluation, _ = self.describe_image_with_llava(latest_image, prompt=eval_prompt)
            print(f"[🌀] Self‑evaluation: {evaluation}")
            self.evaluation_journal.append(evaluation.strip())
        except Exception as e:
            print(f"[⚠️] Failed to evaluate self: {e}")

