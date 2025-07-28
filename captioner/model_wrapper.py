import os
from typing import Optional
from captioner.prompts import (
    build_awakening_prompt,
    build_caption_prompt,
    build_reflection_prompt,
    build_drawing_prompt,
)
from config import config
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
            # Use awakening prompt for the first environmental description
            prompt = build_awakening_prompt("What do you see?", agent=self.memory_ref)
            return self._call_ollama(prompt, image_path=image_path, system_prompt=config.SYSTEM_PROMPT, 
                                   temperature=config.AWAKENING_TEMPERATURE)
        elif flowing and self.memory_ref:
            prompt = build_caption_prompt(
                self.memory_ref,
                mood=self.memory_ref.current_mood,
                boredom=self.memory_ref.boredom,
                novelty=self.memory_ref.novelty_score,
                previous_caption=self.memory_ref.last_caption,
            )
            return self._call_ollama(prompt, image_path=image_path, system_prompt=config.SYSTEM_PROMPT,
                                   temperature=config.CONSCIOUSNESS_TEMPERATURE)
        else:
            prompt = "Describe this image."

        # @todo SYS PROMPT HERE? SO NO "IN THIS IMAGE?"
        return self._call_ollama(prompt, image_path=image_path, system_prompt=config.SYSTEM_PROMPT,
                               temperature=config.CONSCIOUSNESS_TEMPERATURE)

    def reason_about_caption(
        self, caption: str, *, agent: Optional[any] = None, mood_text: Optional[str] = None, extra: Optional[str] = None  # type: ignore
    ) -> str:  # type: ignore
        prompt = build_reflection_prompt(caption, extra=extra, agent=agent)
        return self._call_ollama(prompt, system_prompt=config.SYSTEM_PROMPT, 
                               temperature=config.REFLECTION_TEMPERATURE)

    def generate_drawing_prompt(self, *, extra: Optional[str] = None) -> str:
        if not self.memory_ref:
            return "[⚠️] No memory available for drawing prompt"

        prompt = build_drawing_prompt(self.memory_ref, extra=extra)
        return self._call_ollama(prompt, system_prompt=config.SYSTEM_PROMPT,
                               temperature=config.CONSCIOUSNESS_TEMPERATURE)

    def _call_ollama(self, prompt: str, image_path: Optional[str] = None, system_prompt: Optional[str] = None, 
                    temperature: Optional[float] = None, stream: bool = True) -> str:
        response = query_ollama(
            prompt=prompt, model=self.model_name, image=image_path, timeout=30, 
            log_dir=MOOD_SNAPSHOT_FOLDER, system_prompt=system_prompt, temperature=temperature,
            stream=stream
        )

        # Clean up AI model leakage - remove unwanted prompt-like text
        response = self._clean_response(response)
        return response

    def _clean_response(self, response: str) -> str:
        """Remove unwanted AI-generated prompt leakage from responses."""
        # Common patterns the AI model sometimes generates
        unwanted_patterns = [
            r"\n\nFeelings:.*?\?",
            r"\n\nReflection:.*?\?",
            r"\n\nWhat do you feel\?",
            r"\n\nHow does.*?feel\?",
            r"Feelings: What do you feel\?",
            r"Reflection: How does.*?\?",
        ]

        cleaned = response
        for pattern in unwanted_patterns:
            import re

            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE | re.DOTALL)

        # Clean up extra whitespace
        cleaned = cleaned.strip()
        
        # Enforce maximum 3 sentences
        import re
        sentences = re.split(r'[.!?]+', cleaned)
        # Filter out empty sentences and take first 3
        sentences = [s.strip() for s in sentences if s.strip()]
        if len(sentences) > 3:
            sentences = sentences[:3]
        # Reconstruct with periods, handling the case where original punctuation was different
        if sentences:
            cleaned = '. '.join(sentences)
            if not cleaned.endswith(('.', '!', '?')):
                cleaned += '.'
        
        return cleaned
