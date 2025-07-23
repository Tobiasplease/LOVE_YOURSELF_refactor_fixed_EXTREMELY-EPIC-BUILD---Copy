import os
import base64
import json
import requests
from typing import Optional
from captioner.prompts import (
    build_awakening_prompt,
    build_caption_prompt,
    build_reflection_prompt,
    build_drawing_prompt,  # ✅ add this
)

OLLAMA_URL = "http://localhost:11434/api/generate"

class MultimodalModel:
    def __init__(self, memory_ref: Optional[any] = None) -> None:
        self.memory_ref = memory_ref
        self.model_name = os.getenv("MODEL_NAME", "llava:7b-v1.6-mistral-q5_1")

    def caption_image(self, image_path: str, *, flowing: bool = True, first_time: bool = False) -> str:
        if not os.path.exists(image_path):
            return "[⚠️] No image found"

        try:
            with open(image_path, "rb") as img_file:
                image_data = base64.b64encode(img_file.read()).decode("utf-8")
        except Exception as e:
            return f"[⚠️] Failed to load image: {str(e)}"

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

        return self._call_ollama(prompt, image_data=image_data)

    def reason_about_caption(
        self,
        caption: str,
        *,
        agent: Optional[any] = None,
        mood_text: Optional[str] = None,
        extra: Optional[str] = None
    ) -> str:
        prompt = build_reflection_prompt(caption, extra=extra, agent=agent)
        return self._call_ollama(prompt)

    def generate_drawing_prompt(self, *, extra: Optional[str] = None) -> str:
        if not self.memory_ref:
            return "[⚠️] No memory available for drawing prompt"

        prompt = build_drawing_prompt(self.memory_ref, extra=extra)
        return self._call_ollama(prompt)

    def _call_ollama(self, prompt: str, image_data: Optional[str] = None) -> str:
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False
        }
        if image_data:
            payload["images"] = [image_data]

        try:
            response = requests.post(OLLAMA_URL, json=payload, timeout=90)
            response.raise_for_status()
            data = response.json()
            return data.get("response", "").strip() or "[⚠️] No response"
        except requests.Timeout:
            return "[⚠️] Model timed out"
        except requests.RequestException as e:
            return f"[⚠️] Request error: {str(e)}"
