import os
from typing import Optional
from captioner.prompts import (
    build_awakening_prompt,
    build_caption_prompt,
    build_environmental_caption_prompt,
    build_reflection_prompt,
    build_drawing_prompt,
    build_change_focused_caption_prompt,
)
from config import config

from config.config import (
    MOOD_SNAPSHOT_FOLDER,
    OLLAMA_MODEL,
    VISUAL_CHANGE_THRESHOLD,
    TINYLLAMA_TEMPERATURE,
    TINYLLAMA_TOP_P,
    TINYLLAMA_NUM_PREDICT,
    TINYLLAMA_TIMEOUT,
    OLLAMA_TIMEOUT_REFLECTION,
    DEBUG_VERBOSE,
)
from config.model_settings import get_model_options, get_model_system_prompt, is_qwen_model
from utils.ollama import query_ollama


class MultimodalModel:
    def __init__(self, memory_ref: Optional[any] = None) -> None:  # type: ignore
        self.memory_ref = memory_ref
        self.model_name = OLLAMA_MODEL

    def caption_image(self, image_path: str, *, flowing: bool = True, first_time: bool = False) -> str:
        if not os.path.exists(image_path):
            return "[⚠️] No image found"

        if first_time:
            # Use the same detailed environmental prompt system for fresh starts
            # This ensures first-time awakenings get proper environmental descriptions
            if self.memory_ref:
                # Get last session gap directly from captioner
                session_gap = getattr(self.memory_ref, "last_session_gap", None)

                prompt = build_environmental_caption_prompt(
                    self.memory_ref,
                    mood=self.memory_ref.current_mood,
                    boredom=self.memory_ref.boredom,
                    novelty=self.memory_ref.novelty_score,
                    last_session_gap=session_gap,  # type: ignore
                )
            else:
                # Fallback if no memory reference available
                prompt = build_awakening_prompt("What do you see?")
            return self._call_ollama(prompt, image_path=image_path, system_prompt=config.SYSTEM_PROMPT)
        elif flowing and self.memory_ref:
            # Check for significant visual change in snapshot
            visual_change_detected = self._detect_significant_visual_change()

            if visual_change_detected:
                # Use change-focused prompt to ground the AI in current reality
                prompt = build_change_focused_caption_prompt(
                    self.memory_ref,
                    mood=self.memory_ref.current_mood,
                    boredom=self.memory_ref.boredom,
                    novelty=self.memory_ref.novelty_score,
                )
            else:
                # Use normal contemplative prompt
                prompt = build_caption_prompt(
                    self.memory_ref,
                    mood=self.memory_ref.current_mood,
                    boredom=self.memory_ref.boredom,
                    novelty=self.memory_ref.novelty_score,
                )
        else:
            prompt = "Describe this image."

        # @todo SYS PROMPT HERE? SO NO "IN THIS IMAGE?"
        return self._call_ollama(prompt, image_path=image_path, system_prompt=config.SYSTEM_PROMPT)

    def reason_about_caption(
        self, caption: str, *, agent: Optional[any] = None, mood_text: Optional[str] = None, extra: Optional[str] = None  # type: ignore
    ) -> str:  # type: ignore
        try:
            prompt = build_reflection_prompt(caption, extra=extra, agent=agent)
            print(f"[🔍] Starting reflection with timeout={OLLAMA_TIMEOUT_REFLECTION}s")
            response = self._call_ollama(prompt, system_prompt=config.SYSTEM_PROMPT, timeout=OLLAMA_TIMEOUT_REFLECTION)
            print(f"[✅] Reflection completed: {len(response)} chars")
            return response
        except Exception as e:
            print(f"[❌] Reflection failed: {e}")
            import traceback

            traceback.print_exc()
            return "[⚠️] Reflection generation failed"

    def generate_drawing_prompt(self, *, extra: Optional[str] = None) -> str:
        if not self.memory_ref:
            return "[⚠️] No memory available for drawing prompt"

        prompt = build_drawing_prompt(self.memory_ref, extra=extra)
        return self._call_ollama(prompt, system_prompt=config.SYSTEM_PROMPT)

    def query_tinyllama(self, prompt: str) -> str:
        """Query TinyLlama model for motif scoring and emotional analysis."""
        # Use TinyLlama for fast, lightweight text-only queries
        tinyllama_options = {
            "temperature": TINYLLAMA_TEMPERATURE,
            "top_p": TINYLLAMA_TOP_P,
            "num_predict": TINYLLAMA_NUM_PREDICT,
        }

        try:
            response = query_ollama(
                prompt=prompt,
                model="tinyllama:latest",
                timeout=TINYLLAMA_TIMEOUT,
                log_dir=MOOD_SNAPSHOT_FOLDER,
                system_prompt="You are a number generator. Return ONLY decimal numbers. No words, no explanations, no text. Just the number.",
                options=tinyllama_options,
            )
            return response.strip()
        except Exception:
            # Fallback if TinyLlama fails
            return "0.5"

    def _detect_significant_visual_change(self) -> bool:
        """Detect if there's been a significant visual change in the snapshot (not video feed)."""
        if not self.memory_ref:
            return False

        # Get the novelty score - high novelty indicates visual change
        novelty = getattr(self.memory_ref, "novelty_score", 0.0)

        # Use configurable threshold for "significant change"
        return novelty > VISUAL_CHANGE_THRESHOLD

    def _call_ollama(self, prompt: str, image_path: Optional[str] = None, system_prompt: Optional[str] = None, timeout: int = 90) -> str:
        # Get model-specific generation options
        model_options = get_model_options(self.model_name)

        # Get model-specific system prompt if none provided
        if system_prompt is None:
            model_system_config = get_model_system_prompt(self.model_name)
            system_prompt = model_system_config["base_prompt"]

        # Add dynamic self-understanding to system prompt
        if self.memory_ref and hasattr(self.memory_ref, "get_dynamic_system_context"):
            dynamic_context = self.memory_ref.get_dynamic_system_context()
            if dynamic_context:
                system_prompt += dynamic_context

        # For Qwen models, use different prompt formatting
        if is_qwen_model(self.model_name):
            # Qwen prefers SYSTEM/USER format to prevent role confusion
            formatted_prompt = f"SYSTEM: {system_prompt}\n\nUSER: {prompt}"
            final_system_prompt = None  # Don't double-set system prompt
        else:
            # LLaVA and other models use system prompt normally
            formatted_prompt = prompt
            final_system_prompt = system_prompt

        response = query_ollama(
            prompt=formatted_prompt,
            model=self.model_name,
            image=image_path,
            timeout=timeout,
            log_dir=MOOD_SNAPSHOT_FOLDER,
            system_prompt=final_system_prompt,
            options=model_options,  # Pass model-specific options
            show_progress=True,  # Enable progress bar for captions
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
        return cleaned
