from __future__ import annotations

from config.config import MOOD_SNAPSHOT_FOLDER, LLAVA_TIMEOUT_SUMMARY, EVALUATION_INTERVAL, SUMMARY_INTERVAL
import os
import time
import threading
import re
from typing import Optional, Deque, Tuple
from collections import deque
import cv2  # type: ignore
import numpy as np  # type: ignore

from .memory import (
    MemoryMixin,
    CAPTION_SAVE_THRESHOLD,
)
from .prompts import (
    LLAVA_PROMPT,
    # build_context,
    build_summary_prompt,
    build_caption_prompt,
    build_self_evaluation_prompt,
)
from mood.mood import generate_internal_note, log_mood, estimate_mood_llava
from drawing.drawing import DrawingController
from event_logging.json_logger import log_json_entry
from ollama import query_ollama
from event_logging.run_manager import get_run_image_path


class Captioner(MemoryMixin):
    caption_window: Optional[any] = None  # type: ignore

    def __init__(self) -> None:
        MemoryMixin.__init__(self)

        self.stream_interval: int = 3
        self.summary_interval: int = SUMMARY_INTERVAL
        self.eval_interval: int = EVALUATION_INTERVAL

        self.true_session_start: float = time.time()
        self.session_start: float = self.true_session_start
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
        self.first_caption_done: bool = False

        os.makedirs(MOOD_SNAPSHOT_FOLDER, exist_ok=True)

        self.snapshot_queue: Deque[Tuple[np.ndarray, bool]] = deque()
        self.caption_thread = threading.Thread(target=self._caption_worker, daemon=True)
        self.caption_thread.start()

    @property
    def is_processing(self) -> bool:
        return bool(self.snapshot_queue)

    def update(self, frame: Optional[np.ndarray] = None, *, person_present: bool = False, mood: Optional[float] = None) -> None:
        if frame is not None:
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
        filename = get_run_image_path(MOOD_SNAPSHOT_FOLDER, f"mood_{timestamp}.jpg")
        cv2.imwrite(filename, frame)

        if self.first_caption_done:
            prompt = build_caption_prompt(self, self.current_mood, self.boredom, self.novelty_score)
        else:
            prompt = LLAVA_PROMPT
            self.first_caption_done = True

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
        print(note)

        print(f"[{time.strftime('%H:%M:%S')}] {caption.strip()}")

        if self.novelty_score > CAPTION_SAVE_THRESHOLD:
            # vs update_feeling_brain: double logging?
            log_mood(caption, self.current_mood, filename)
            short_caption = self.truncate_caption(caption)
            self.memory_queue.append((timestamp, short_caption, self.current_mood, filename))
            self.long_memory.append((timestamp, short_caption, self.current_mood, filename))
            self.compress_long_memory()

        self.last_caption = caption
        self.last_mood = self.current_mood
        self.last_person_detected = saw_person
        self.last_short_caption = self.truncate_caption(caption)

        # Only cleanup once per frame to avoid race conditions
        # self.cleanup_snapshots(MOOD_SNAPSHOT_FOLDER)

    @staticmethod
    def describe_image_with_llava(image_path: str, *, prompt: str) -> tuple[str, str]:
        try:
            # Check if file exists before trying to open it
            if not os.path.exists(image_path):
                error_msg = f"[⚠️] Image file not found: {image_path}"
                print(error_msg)
                return "Unable to analyze image - file not found", ""

            response_text = query_ollama(prompt=prompt, model="llava", image=image_path, timeout=LLAVA_TIMEOUT_SUMMARY, log_dir=MOOD_SNAPSHOT_FOLDER)

            # Check if response indicates an error
            if response_text.startswith("[⚠️]"):
                return response_text, image_path

            return response_text, image_path
        except Exception as exc:
            error_msg = f"[⚠️] LLaVA failed: {exc}"
            return error_msg, image_path

    @staticmethod
    def truncate_caption(raw_caption: str) -> str:
        sentence = re.split(r"[.!?]", raw_caption.strip())[0].strip().lower()
        words = sentence.split()
        return " ".join(words[:18])

    def generate_summary(self) -> None:
        if not self.memory_queue:
            return

        try:
            recent_lines = [m[1] for m in list(self.memory_queue)[-50:]]
            prior_context = "\n".join(f"- {s}" for s in recent_lines)
            moods = [m[2] for m in list(self.memory_queue)]
            avg_mood = round(sum(moods) / len(moods), 2) if moods else 0.0
            past_summaries = "\n".join(f"→ {s}" for s in (self.summary_history or [])[-3:])

            summary_prompt = build_summary_prompt(prior_context, avg_mood, past_summaries)
            latest_image = list(self.memory_queue)[-1][3]
            summary, _ = self.describe_image_with_llava(latest_image, prompt=summary_prompt)
            print(f"In summary: {summary}")

            self.summary_history.append(summary)
            self.memory_queue = deque(list(self.memory_queue)[-3:], maxlen=self.memory_queue.maxlen)
        except Exception as e:
            print(f"[⚠️] Failed to generate summary: {e}")

        # self.cleanup_snapshots(MOOD_SNAPSHOT_FOLDER)

    def generate_self_evaluation(self) -> None:
        try:
            recent = list(self.memory_queue)[-10:]
            mood_vals = [m[2] for m in recent]
            mood_delta = mood_vals[-1] - mood_vals[0] if len(mood_vals) > 1 else 0.0
            time_elapsed = int(time.time() - self.true_session_start)
            recent_summaries = "\n".join(s for s in (self.summary_history or [])[-3:])

            eval_prompt = build_self_evaluation_prompt(
                agent=self,
                mood_delta=mood_delta,
                time_elapsed=time_elapsed,
                recent_summaries=recent_summaries,
            )

            latest_image = next((m[3] for m in reversed(self.memory_queue) if isinstance(m, tuple) and len(m) > 3), None)
            if not latest_image:
                print("[⚠️] No valid image found in memory_queue")
                return

            # Check if the image file exists, try to find an alternative if not
            if not os.path.exists(latest_image):
                print(f"[⚠️] Latest image file not found: {latest_image}")
                # Try to find the most recent existing image in memory_queue
                latest_image = None
                for m in reversed(self.memory_queue):
                    if isinstance(m, tuple) and len(m) > 3 and os.path.exists(m[3]):
                        latest_image = m[3]
                        print(f"[🔄] Using alternative image: {latest_image}")
                        break

                if not latest_image:
                    print("[❌] No valid image files found in memory_queue")
                    return

            evaluation, _ = self.describe_image_with_llava(latest_image, prompt=eval_prompt)
            print(f"[🌀] Self-evaluation: {evaluation}")
            self.evaluation_journal.append(evaluation.strip())

            # Log self-evaluation in JSON format
            eval_data = {
                "evaluation": evaluation.strip(),
                "mood_delta": mood_delta,
                "time_elapsed": time_elapsed,
                "current_mood": self.current_mood,
                "last_mood": self.last_mood,
                "boredom": self.boredom,
                "novelty_score": self.novelty_score,
                "recent_summaries": recent_summaries,
            }
            log_json_entry("self_evaluation", eval_data, MOOD_SNAPSHOT_FOLDER)

            # Handle drawing flow through drawing controller
            controller = DrawingController()
            controller.handle_drawing_flow(self, evaluation, latest_image)

        except Exception as e:
            print(f"[⚠️] Failed to evaluate self: {e}")

