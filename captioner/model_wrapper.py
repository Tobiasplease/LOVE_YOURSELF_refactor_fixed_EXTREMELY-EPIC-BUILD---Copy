import os
from typing import Optional
from captioner.prompts import (
    build_awakening_prompt,
    build_caption_prompt,
    build_reflection_prompt,
    build_drawing_prompt,
)
from config.config import MOOD_SNAPSHOT_FOLDER, OLLAMA_MODEL
from utils.ollama import query_ollama


class MultimodalModel:
    def __init__(self, memory_ref: Optional[any] = None) -> None:  # type: ignore
        self.memory_ref = memory_ref
        self.model_name = OLLAMA_MODEL

    def caption_image(self, image_path: str, *, flowing: bool = True, first_time: bool = False) -> str:
        if not os.path.exists(image_path):
            return "[⚠️] No image found"

        if first_time:
            prompt = build_awakening_prompt("What do you see?")
        elif flowing and self.memory_ref:
            prompt = build_caption_prompt(
                self.memory_ref,
                mood=self.memory_ref.current_mood,
                boredom=self.memory_ref.boredom,
                novelty=self.memory_ref.novelty_score,
            )
        else:
            prompt = "Describe this image."

        return self._call_ollama(prompt, image_path=image_path)

    def reason_about_caption(
        self, caption: str, *, agent: Optional[any] = None, mood_text: Optional[str] = None, extra: Optional[str] = None  # type: ignore
    ) -> str:  # type: ignore
        prompt = build_reflection_prompt(caption, extra=extra, agent=agent)
        return self._call_ollama(prompt)

    def generate_drawing_prompt(self, *, extra: Optional[str] = None) -> str:
        if not self.memory_ref:
            return "[⚠️] No memory available for drawing prompt"

        prompt = build_drawing_prompt(self.memory_ref, extra=extra)
        return self._call_ollama(prompt)

    def _call_ollama(self, prompt: str, image_path: Optional[str] = None) -> str:
        return query_ollama(prompt=prompt, model=self.model_name, image=image_path, timeout=90, log_dir=MOOD_SNAPSHOT_FOLDER)
